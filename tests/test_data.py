from __future__ import annotations

import pandas as pd

from tenflix.config import DEFAULT_CONFIG
from tenflix.data import lifecycle_stage, temporal_split


def test_lifecycle_boundaries_use_one_policy(config):
    assert lifecycle_stage(0, DEFAULT_CONFIG) == "new"
    assert lifecycle_stage(1, DEFAULT_CONFIG) == "cold"
    assert lifecycle_stage(19, DEFAULT_CONFIG) == "cold"
    assert lifecycle_stage(20, DEFAULT_CONFIG) == "sparse"
    assert lifecycle_stage(49, DEFAULT_CONFIG) == "sparse"
    assert lifecycle_stage(50, DEFAULT_CONFIG) == "mature"


def test_temporal_split_is_chronological_and_disjoint(ratings, config):
    context, holdout = temporal_split(ratings.sample(frac=1, random_state=1), config, evaluation=True)
    for user_id in ratings["userId"].unique():
        user_context = context[context["userId"] == user_id]
        user_holdout = holdout[holdout["userId"] == user_id]
        assert user_context["timestamp"].max() < user_holdout["timestamp"].min()
        assert set(user_context["time_window"]) == {"old", "recent_context"}
        assert set(user_context["movieId"]).isdisjoint(set(user_holdout["movieId"]))


def test_holdout_changes_do_not_change_context(ratings, config):
    context, holdout = temporal_split(ratings, config, evaluation=True)
    altered = ratings.copy()
    holdout_keys = set(zip(holdout["userId"], holdout["movieId"]))
    holdout_mask = [
        (user_id, movie_id) in holdout_keys
        for user_id, movie_id in zip(altered["userId"], altered["movieId"])
    ]
    altered.loc[holdout_mask, "rating"] = 0.5
    context_again, _ = temporal_split(altered, config, evaluation=True)
    pd.testing.assert_frame_equal(context, context_again)
