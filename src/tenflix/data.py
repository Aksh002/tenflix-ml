from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


RATING_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
}
MOVIE_DTYPES = {"movieId": "int32", "title": "string", "genres": "string"}


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_ratings(path: str | Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(path, dtype=RATING_DTYPES, chunksize=chunk_size)


def read_movies(path: str | Path) -> pd.DataFrame:
    movies = pd.read_csv(path, dtype=MOVIE_DTYPES)
    required = set(MOVIE_DTYPES)
    if set(movies.columns) != required:
        raise ValueError(f"movies.csv columns must be {sorted(required)}")
    if movies["movieId"].duplicated().any():
        raise ValueError("movies.csv contains duplicate movieId values")
    if movies[list(required)].isna().any().any():
        raise ValueError("movies.csv contains null required values")
    return movies.sort_values("movieId", kind="stable").reset_index(drop=True)


def prepare_parquet(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    ratings_path = Path(paths["ratings"])
    movies_path = Path(paths["movies"])
    output_dir = Path(paths["prepared_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ratings_output = output_dir / "ratings.parquet"
    movies_output = output_dir / "movies.parquet"
    temp_ratings = output_dir / ".ratings.parquet.tmp"
    temp_movies = output_dir / ".movies.parquet.tmp"

    movies = read_movies(movies_path)
    movie_ids = set(movies["movieId"].astype(int))
    movie_table = pa.Table.from_pandas(movies, preserve_index=False)
    pq.write_table(movie_table, temp_movies, compression="zstd")

    writer: pq.ParquetWriter | None = None
    rows = 0
    users: set[int] = set()
    rated_movies: set[int] = set()
    minimum_timestamp = np.iinfo(np.int64).max
    maximum_timestamp = np.iinfo(np.int64).min
    rating_counts: dict[str, int] = {}
    try:
        for chunk in iter_ratings(ratings_path, int(config["data"]["chunk_size"])):
            if set(chunk.columns) != set(RATING_DTYPES):
                raise ValueError(f"ratings.csv columns must be {sorted(RATING_DTYPES)}")
            if chunk.isna().any().any():
                raise ValueError("ratings.csv contains null required values")
            if not chunk["rating"].between(
                config["data"]["rating_min"], config["data"]["rating_max"]
            ).all():
                raise ValueError("ratings.csv contains a rating outside the configured range")
            unknown = set(chunk["movieId"].astype(int).unique()) - movie_ids
            if unknown:
                raise ValueError(f"ratings.csv references missing movie metadata: {sorted(unknown)[:5]}")
            rows += len(chunk)
            users.update(chunk["userId"].astype(int).unique())
            rated_movies.update(chunk["movieId"].astype(int).unique())
            minimum_timestamp = min(minimum_timestamp, int(chunk["timestamp"].min()))
            maximum_timestamp = max(maximum_timestamp, int(chunk["timestamp"].max()))
            for rating, count in chunk["rating"].value_counts().items():
                key = f"{float(rating):.1f}"
                rating_counts[key] = rating_counts.get(key, 0) + int(count)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp_ratings, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if rows == 0:
        raise ValueError("ratings.csv contains no data rows")
    _promote_file(temp_ratings, ratings_output)
    _promote_file(temp_movies, movies_output)
    manifest = {
        "schema_version": 3,
        "ratings_rows": rows,
        "users": len(users),
        "catalog_movies": len(movies),
        "rated_movies": len(rated_movies),
        "unrated_catalog_movies": len(movie_ids - rated_movies),
        "timestamp_min": pd.to_datetime(minimum_timestamp, unit="s", utc=True).isoformat(),
        "timestamp_max": pd.to_datetime(maximum_timestamp, unit="s", utc=True).isoformat(),
        "rating_counts": dict(sorted(rating_counts.items())),
        "source_hashes": {
            "ratings.csv": sha256_file(ratings_path),
            "movies.csv": sha256_file(movies_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _promote_file(source: Path, destination: Path) -> None:
    """Prefer atomic replacement, with a Windows/restricted-filesystem fallback."""
    try:
        os.replace(source, destination)
    except PermissionError:
        shutil.copy2(source, destination)
        try:
            source.unlink()
        except PermissionError:
            pass


def load_prepared(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = Path(config["paths"]["prepared_dir"])
    ratings_path = directory / "ratings.parquet"
    movies_path = directory / "movies.parquet"
    if ratings_path.exists() and movies_path.exists():
        return pd.read_parquet(ratings_path), pd.read_parquet(movies_path)
    ratings = pd.read_csv(config["paths"]["ratings"], dtype=RATING_DTYPES)
    movies = read_movies(config["paths"]["movies"])
    return ratings, movies


def validate_loaded_data(ratings: pd.DataFrame, movies: pd.DataFrame, config: dict[str, Any]) -> None:
    if ratings.empty or movies.empty:
        raise ValueError("Ratings and movies must not be empty")
    if ratings.duplicated(["userId", "movieId"]).any():
        raise ValueError("Ratings contain duplicate (userId, movieId) pairs")
    if ratings[list(RATING_DTYPES)].isna().any().any():
        raise ValueError("Ratings contain null required values")
    if not ratings["rating"].between(
        config["data"]["rating_min"], config["data"]["rating_max"]
    ).all():
        raise ValueError("Ratings contain values outside the configured range")
    unknown = set(ratings["movieId"].astype(int).unique()) - set(movies["movieId"].astype(int))
    if unknown:
        raise ValueError(f"Ratings reference missing movies: {sorted(unknown)[:5]}")


def lifecycle_stage(interactions: int, config: dict[str, Any]) -> str:
    if interactions <= 0:
        return "new"
    if interactions <= config["lifecycle"]["cold_max"]:
        return "cold"
    if interactions <= config["lifecycle"]["sparse_max"]:
        return "sparse"
    return "mature"


def temporal_split(
    ratings: pd.DataFrame,
    config: dict[str, Any],
    evaluation: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each user chronologically without exposing holdout rows to model fitting."""
    ordered = ratings.sort_values(["userId", "timestamp", "movieId"], kind="stable").copy()
    position = ordered.groupby("userId", sort=False).cumcount().to_numpy()
    size = ordered.groupby("userId", sort=False)["userId"].transform("size").to_numpy()
    if evaluation:
        old_end = np.maximum(1, np.floor(size * config["temporal"]["old_fraction"]).astype(int))
        context_end = np.maximum(
            old_end + 1,
            np.floor(
                size
                * (
                    config["temporal"]["old_fraction"]
                    + config["temporal"]["recent_context_fraction"]
                )
            ).astype(int),
        )
        context_end = np.minimum(context_end, np.maximum(1, size - 1))
        context_mask = position < context_end
        context = ordered.loc[context_mask].copy()
        holdout = ordered.loc[~context_mask].copy()
        context["time_window"] = np.where(position[context_mask] < old_end[context_mask], "old", "recent_context")
        return context.reset_index(drop=True), holdout.reset_index(drop=True)

    old_end = np.maximum(1, np.floor(size * 0.40).astype(int))
    ordered["time_window"] = np.where(position < old_end, "old", "recent_context")
    return ordered.reset_index(drop=True), ordered.iloc[0:0].copy()
