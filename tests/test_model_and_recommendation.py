from __future__ import annotations

import numpy as np
from scipy import sparse

from tenflix.artifacts import load_bundle, load_holdout, save_run
from tenflix.data import temporal_split
from tenflix.models import fit_model
from tenflix.recommender import HybridRecommender


def test_model_uses_shared_basis_and_finite_drift(ratings, movies, config):
    context, _ = temporal_split(ratings, config, evaluation=True)
    bundle = fit_model(context, movies, config)
    assert bundle.old_vectors.shape == bundle.recent_vectors.shape == bundle.static_vectors.shape
    assert bundle.old_vectors.shape[1] == bundle.item_vectors.shape[1]
    assert np.isfinite(bundle.drift_scores).any()
    assert bundle.content_features.shape[0] == len(movies)
    assert sparse.issparse(bundle.content_features)
    assert bundle.content_features.shape != (len(movies), len(movies))


def test_recommendations_preserve_rank_and_exclude_seen(ratings, movies, config):
    context, _ = temporal_split(ratings, config, evaluation=True)
    bundle = fit_model(context, movies, config)
    recommender = HybridRecommender(bundle)
    results = recommender.recommend(user_id=1, top_k=5)
    assert [result.rank for result in results] == list(range(1, len(results) + 1))
    assert [result.score for result in results] == sorted(
        [result.score for result in results], reverse=True
    )
    seen = set(context.loc[context["userId"] == 1, "movieId"])
    assert seen.isdisjoint({result.movie_id for result in results})
    assert all(result.strategy and result.reason for result in results)


def test_unknown_user_uses_genres_and_returns_typed_results(ratings, movies, config):
    context, _ = temporal_split(ratings, config, evaluation=True)
    recommender = HybridRecommender(fit_model(context, movies, config))
    results = recommender.recommend(user_id=999, preferred_genres=["Sci-Fi"], top_k=3)
    assert len(results) == 3
    assert all(result.strategy == "new_content" for result in results)
    assert any("Sci-Fi" in result.genres for result in results)


def test_artifacts_round_trip_atomically(tmp_path, ratings, movies, config):
    context, holdout = temporal_split(ratings, config, evaluation=True)
    bundle = fit_model(context, movies, config)
    path = save_run(bundle, holdout, tmp_path, "fixture", "evaluation")
    loaded = load_bundle(path)
    loaded_holdout = load_holdout(path)
    original = HybridRecommender(bundle).ranked_movie_ids(1, 4)
    restored = HybridRecommender(loaded).ranked_movie_ids(1, 4)
    assert original == restored
    assert len(loaded_holdout) == len(holdout)
