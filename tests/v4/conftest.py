from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from tenflix.v4.config import DEFAULT_CONFIG
from tenflix.v4.events import RatingEvent, RatingSource
from tenflix.v4.pipeline import fit_bundle


@pytest.fixture
def v4_config(tmp_path):
    value = deepcopy(DEFAULT_CONFIG)
    value["seed"] = 9
    value["paths"]["prepared_dir"] = str(tmp_path / "prepared")
    value["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    value["lifecycle"] = {"cold_max": 2, "sparse_max": 4}
    value["model"].update(factors=3, epochs=5, early_stopping_patience=2)
    value["content"]["min_df"] = 1
    value["temporal"].update(
        minimum_sessions=3,
        minimum_lifespan_days=30,
        minimum_window_ratings=2,
        recent_window_days=90,
        recent_half_life_days=30,
    )
    value["candidates"] = {
        "collaborative": 5,
        "content": 5,
        "quality_popularity": 5,
        "recent": 5,
        "exploration": 3,
    }
    value["evaluation"].update(max_users=10, bootstrap_samples=20, minimum_coverage=0.01)
    return value


@pytest.fixture
def v4_movies():
    genres = [
        "Action|Adventure",
        "Action|Sci-Fi",
        "Drama|Romance",
        "Drama",
        "Comedy",
        "Comedy|Romance",
        "Horror|Thriller",
        "Mystery|Thriller",
        "Animation|Children",
        "Documentary",
        "Fantasy|Adventure",
        "Crime|Drama",
        "Sci-Fi|Adventure",
        "Western",
    ]
    return pd.DataFrame(
        {
            "movieId": range(1, 15),
            "title": [f"Movie {index} ({1980 + index * 3})" for index in range(1, 15)],
            "genres": genres,
        }
    )


@pytest.fixture
def v4_ratings():
    rows = []
    start = datetime(2020, 1, 1, tzinfo=UTC)
    patterns = {
        1: [1, 2, 3, 4, 5, 6, 7, 8],
        2: [3, 4, 5, 6, 7, 8, 9, 10],
        3: [5, 6, 7, 8, 9, 10, 11, 12],
        4: [1, 3, 5, 7, 9, 11, 13, 14],
        5: [2, 4, 6, 8, 10, 12, 13, 14],
    }
    for user_id, movie_ids in patterns.items():
        for offset, movie_id in enumerate(movie_ids):
            timestamp = int((start + timedelta(days=offset * 30 + user_id)).timestamp())
            rows.append(
                {
                    "userId": user_id,
                    "movieId": movie_id,
                    "rating": float(1 + ((offset + user_id) % 5)),
                    "timestamp": timestamp,
                }
            )
    return pd.DataFrame(rows).astype(
        {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    )


@pytest.fixture
def v4_bundle(v4_ratings, v4_movies, v4_config):
    cutoff = int(v4_ratings["timestamp"].max())
    return fit_bundle(v4_ratings, v4_movies, v4_config, None, cutoff, "fixture-v4")


def make_events(user_id=99, count=6, source=RatingSource.ORGANIC):
    start = datetime(2023, 1, 1, tzinfo=UTC)
    return [
        RatingEvent(
            user_id,
            index + 1,
            float(1 + index % 5),
            start + timedelta(days=index * 30),
            source=source,
        )
        for index in range(count)
    ]

