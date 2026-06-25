from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from tenflix.config import DEFAULT_CONFIG


@pytest.fixture
def config(tmp_path):
    value = deepcopy(DEFAULT_CONFIG)
    value["seed"] = 7
    value["paths"]["prepared_dir"] = str(tmp_path / "prepared")
    value["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    value["lifecycle"] = {"cold_max": 1, "sparse_max": 3}
    value["model"]["factors"] = 3
    value["content"]["min_df"] = 1
    value["evaluation"]["max_users"] = 10
    value["evaluation"]["bootstrap_samples"] = 20
    return value


@pytest.fixture
def movies():
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
    ]
    return pd.DataFrame(
        {
            "movieId": pd.Series(range(1, 13), dtype="int32"),
            "title": pd.Series([f"Movie {value} (2000)" for value in range(1, 13)], dtype="string"),
            "genres": pd.Series(genres, dtype="string"),
        }
    )


@pytest.fixture
def ratings():
    rows = []
    patterns = {
        1: [1, 2, 3, 4, 5, 6, 7, 8],
        2: [3, 4, 5, 6, 7, 8, 9, 10],
        3: [5, 6, 7, 8, 9, 10, 11, 12],
        4: [1, 3, 5, 7, 9, 10, 11, 12],
    }
    for user_id, movie_ids in patterns.items():
        for offset, movie_id in enumerate(movie_ids):
            rows.append(
                {
                    "userId": user_id,
                    "movieId": movie_id,
                    "rating": 1.0 + ((offset + user_id) % 5),
                    "timestamp": 1_600_000_000 + user_id * 100 + offset,
                }
            )
    return pd.DataFrame(rows).astype(
        {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    )

