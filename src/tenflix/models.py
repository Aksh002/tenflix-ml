from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import public_config
from .data import lifecycle_stage


def _safe_unit_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(values, axis=1)
    valid = norms > 1e-10
    result = np.zeros_like(values, dtype=np.float32)
    result[valid] = values[valid] / norms[valid, None]
    return result, valid


def rowwise_cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_unit, left_valid = _safe_unit_rows(left)
    right_unit, right_valid = _safe_unit_rows(right)
    distance = np.full(left.shape[0], np.nan, dtype=np.float32)
    valid = left_valid & right_valid
    similarity = np.sum(left_unit[valid] * right_unit[valid], axis=1)
    distance[valid] = 1.0 - np.clip(similarity, -1.0, 1.0)
    return distance


def _matrix(
    frame: pd.DataFrame,
    user_lookup: dict[int, int],
    item_lookup: dict[int, int],
    values: np.ndarray,
) -> sparse.csr_matrix:
    rows = frame["userId"].map(user_lookup).to_numpy(dtype=np.int32)
    columns = frame["movieId"].map(item_lookup).to_numpy(dtype=np.int32)
    return sparse.csr_matrix(
        (values.astype(np.float32), (rows, columns)),
        shape=(len(user_lookup), len(item_lookup)),
        dtype=np.float32,
    )


@dataclass
class ModelBundle:
    schema_version: int
    config: dict[str, Any]
    movies: pd.DataFrame
    user_ids: np.ndarray
    catalog_movie_ids: np.ndarray
    cf_movie_ids: np.ndarray
    interaction_counts: np.ndarray
    lifecycle: np.ndarray
    user_means: np.ndarray
    static_vectors: np.ndarray
    old_vectors: np.ndarray
    recent_vectors: np.ndarray
    drift_scores: np.ndarray
    drift_types: np.ndarray
    drift_thresholds: tuple[float, float]
    item_vectors: np.ndarray
    cf_catalog_indices: np.ndarray
    seen_matrix: sparse.csr_matrix
    content_profiles: sparse.csr_matrix
    content_features: sparse.csr_matrix
    vectorizer: TfidfVectorizer
    quality_scores: np.ndarray
    popularity_scores: np.ndarray
    svd: TruncatedSVD

    def __post_init__(self) -> None:
        self.user_lookup = {int(value): index for index, value in enumerate(self.user_ids)}
        self.catalog_lookup = {
            int(value): index for index, value in enumerate(self.catalog_movie_ids)
        }


def fit_model(context: pd.DataFrame, movies: pd.DataFrame, config: dict[str, Any]) -> ModelBundle:
    if context.empty:
        raise ValueError("Training context is empty")
    required_windows = {"old", "recent_context"}
    if not required_windows.issubset(set(context["time_window"].unique())):
        raise ValueError("Training context must contain old and recent_context windows")

    context = context.copy()
    user_ids = np.sort(context["userId"].unique()).astype(np.int32)
    catalog = movies.sort_values("movieId", kind="stable").reset_index(drop=True).copy()
    catalog_movie_ids = catalog["movieId"].to_numpy(dtype=np.int32)
    cf_movie_ids = np.sort(context["movieId"].unique()).astype(np.int32)
    user_lookup = {int(value): index for index, value in enumerate(user_ids)}
    catalog_lookup = {int(value): index for index, value in enumerate(catalog_movie_ids)}
    cf_lookup = {int(value): index for index, value in enumerate(cf_movie_ids)}

    counts_series = context.groupby("userId").size().reindex(user_ids)
    interaction_counts = counts_series.to_numpy(dtype=np.int32)
    lifecycle = np.asarray(
        [lifecycle_stage(int(count), config) for count in interaction_counts], dtype="U8"
    )
    means_series = context.groupby("userId")["rating"].mean().reindex(user_ids)
    user_means = means_series.to_numpy(dtype=np.float32)
    centered = (
        context["rating"].to_numpy(dtype=np.float32)
        - context["userId"].map(means_series).to_numpy(dtype=np.float32)
    )

    all_matrix = _matrix(context, user_lookup, cf_lookup, centered)
    old_mask = context["time_window"].eq("old").to_numpy()
    recent_mask = context["time_window"].eq("recent_context").to_numpy()
    old_matrix = _matrix(context.loc[old_mask], user_lookup, cf_lookup, centered[old_mask])
    recent_matrix = _matrix(
        context.loc[recent_mask], user_lookup, cf_lookup, centered[recent_mask]
    )
    maximum_factors = min(all_matrix.shape) - 1
    factors = min(int(config["model"]["factors"]), maximum_factors)
    if factors < 1:
        raise ValueError("At least two users and two rated movies are required for SVD")
    svd = TruncatedSVD(n_components=factors, random_state=int(config["seed"]))
    static_vectors = svd.fit_transform(all_matrix).astype(np.float32)
    old_vectors = svd.transform(old_matrix).astype(np.float32)
    recent_vectors = svd.transform(recent_matrix).astype(np.float32)
    item_vectors = svd.components_.T.astype(np.float32)

    drift_scores = rowwise_cosine_distance(old_vectors, recent_vectors)
    mature_valid = (lifecycle == "mature") & np.isfinite(drift_scores)
    if mature_valid.any():
        low = float(
            np.quantile(drift_scores[mature_valid], config["temporal"]["stable_quantile"])
        )
        high = float(
            np.quantile(drift_scores[mature_valid], config["temporal"]["volatile_quantile"])
        )
    else:
        low, high = 0.0, 2.0
    drift_types = np.full(len(user_ids), "unknown", dtype="U8")
    drift_types[mature_valid & (drift_scores <= low)] = "stable"
    drift_types[mature_valid & (drift_scores > low) & (drift_scores < high)] = "moderate"
    drift_types[mature_valid & (drift_scores >= high)] = "volatile"

    genre_text = catalog["genres"].fillna("").str.replace("|", " ", regex=False)
    content_config = config["content"]
    vectorizer = TfidfVectorizer(
        ngram_range=(int(content_config["ngram_min"]), int(content_config["ngram_max"])),
        min_df=int(content_config["min_df"]),
        max_df=float(content_config["max_df"]),
    )
    content_features = vectorizer.fit_transform(genre_text).astype(np.float32).tocsr()

    positive_weight = np.maximum(
        context["rating"].to_numpy(dtype=np.float32)
        - context["userId"].map(means_series).to_numpy(dtype=np.float32),
        0.0,
    )
    positive = positive_weight > 0
    profile_interactions = _matrix(
        context.loc[positive], user_lookup, catalog_lookup, positive_weight[positive]
    )
    content_profiles = normalize(profile_interactions @ content_features, norm="l2").tocsr()

    seen_values = np.ones(len(context), dtype=np.int8)
    seen_matrix = _matrix(context, user_lookup, catalog_lookup, seen_values).astype(bool)

    aggregate = context.groupby("movieId")["rating"].agg(["count", "mean"])
    counts = aggregate["count"].reindex(catalog_movie_ids, fill_value=0).to_numpy(dtype=np.float32)
    item_means = aggregate["mean"].reindex(catalog_movie_ids).to_numpy(dtype=np.float32)
    global_mean = float(context["rating"].mean())
    item_means = np.nan_to_num(item_means, nan=global_mean)
    prior_count = max(1.0, float(np.quantile(counts[counts > 0], config["model"]["quality_quantile"])))
    quality = (counts * item_means + prior_count * global_mean) / (counts + prior_count)
    quality_scores = _minmax(quality)
    popularity_scores = _minmax(np.log1p(counts))
    cf_catalog_indices = np.asarray([catalog_lookup[int(value)] for value in cf_movie_ids], dtype=np.int32)

    return ModelBundle(
        schema_version=3,
        config=public_config(config),
        movies=catalog,
        user_ids=user_ids,
        catalog_movie_ids=catalog_movie_ids,
        cf_movie_ids=cf_movie_ids,
        interaction_counts=interaction_counts,
        lifecycle=lifecycle,
        user_means=user_means,
        static_vectors=static_vectors,
        old_vectors=old_vectors,
        recent_vectors=recent_vectors,
        drift_scores=drift_scores,
        drift_types=drift_types,
        drift_thresholds=(low, high),
        item_vectors=item_vectors,
        cf_catalog_indices=cf_catalog_indices,
        seen_matrix=seen_matrix,
        content_profiles=content_profiles,
        content_features=content_features,
        vectorizer=vectorizer,
        quality_scores=quality_scores,
        popularity_scores=popularity_scores,
        svd=svd,
    )


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    if maximum - minimum <= 1e-12:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)
