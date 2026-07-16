from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data import RATING_DTYPES, read_movies, validate_loaded_data
from .events import Movie, RatingEvent, RatingSource


YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")


@dataclass(frozen=True)
class GlobalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_cutoff: int
    validation_cutoff: int


def parse_release_year(title: str) -> int | None:
    match = YEAR_PATTERN.search(str(title))
    if not match:
        return None
    year = int(match.group(1))
    return year if 1880 <= year <= 2200 else None


def movie_records(movies: pd.DataFrame) -> list[Movie]:
    records = []
    for row in movies.itertuples(index=False):
        genre_text = str(row.genres)
        genres = () if genre_text == "(no genres listed)" else tuple(genre_text.split("|"))
        records.append(
            Movie(
                movie_id=int(row.movieId),
                title=str(row.title),
                genres=genres,
                release_year=parse_release_year(str(row.title)),
            )
        )
    return records


def load_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = Path(config["paths"]["prepared_dir"])
    ratings_path = prepared / "ratings.parquet"
    movies_path = prepared / "movies.parquet"
    if ratings_path.exists() and movies_path.exists():
        ratings, movies = pd.read_parquet(ratings_path), pd.read_parquet(movies_path)
    else:
        ratings = pd.read_csv(config["paths"]["ratings"], dtype=RATING_DTYPES)
        movies = read_movies(config["paths"]["movies"])
    if ratings.duplicated(["userId", "movieId"]).any():
        ratings = (
            ratings.sort_values(["userId", "movieId", "timestamp"], kind="stable")
            .drop_duplicates(["userId", "movieId"], keep="last")
            .reset_index(drop=True)
        )
    validate_loaded_data(ratings, movies, config)
    if (ratings["userId"] <= 0).any() or (ratings["timestamp"] <= 0).any():
        raise ValueError("Offline ratings require positive user IDs and timestamps")
    return ratings, movies


def global_time_split(ratings: pd.DataFrame, config: dict[str, Any]) -> GlobalSplit:
    timestamps = ratings["timestamp"].to_numpy(dtype=np.int64)
    train_cutoff = int(np.quantile(timestamps, config["split"]["train_quantile"]))
    validation_cutoff = int(np.quantile(timestamps, config["split"]["validation_quantile"]))
    train = ratings.loc[ratings["timestamp"] <= train_cutoff].copy()
    validation = ratings.loc[
        (ratings["timestamp"] > train_cutoff) & (ratings["timestamp"] <= validation_cutoff)
    ].copy()
    test = ratings.loc[ratings["timestamp"] > validation_cutoff].copy()
    return GlobalSplit(train, validation, test, train_cutoff, validation_cutoff)


def dataframe_events(frame: pd.DataFrame, source: RatingSource = RatingSource.LEGACY) -> list[RatingEvent]:
    return [
        RatingEvent(
            user_id=int(row.userId),
            movie_id=int(row.movieId),
            rating=float(row.rating),
            rated_at=pd.Timestamp(int(row.timestamp), unit="s", tz="UTC").to_pydatetime(),
            source=source,
        )
        for row in frame.itertuples(index=False)
    ]


def lifecycle_stage(count: int, config: dict[str, Any]) -> str:
    if count <= 0:
        return "new"
    if count <= int(config["lifecycle"]["cold_max"]):
        return "cold"
    if count <= int(config["lifecycle"]["sparse_max"]):
        return "sparse"
    return "mature"


def available_by_year(movie_years: np.ndarray, cutoff_timestamp: int) -> np.ndarray:
    cutoff_year = pd.Timestamp(cutoff_timestamp, unit="s", tz="UTC").year
    return np.isnan(movie_years) | (movie_years <= cutoff_year)
