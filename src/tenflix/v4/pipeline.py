from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data import prepare_parquet
from .artifacts import save_bundle
from .config import public_config
from .content import fit_content_model
from .data import dataframe_events, global_time_split, load_data, movie_records
from .matrix_factorization import fit_biased_mf
from .recommender import V4Bundle, V4Recommender


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    manifest = prepare_parquet(config)
    manifest["schema_version"] = 4
    manifest["event_source"] = "legacy"
    path = Path(config["paths"]["prepared_dir"]) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def fit_bundle(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    config: dict[str, Any],
    validation: pd.DataFrame | None,
    training_cutoff: int,
    model_version: str,
) -> V4Bundle:
    mf = fit_biased_mf(ratings, config, validation)
    records = movie_records(movies)
    content = fit_content_model(
        records,
        int(config["content"]["min_df"]),
        float(config["content"]["max_df"]),
        mean_prior_strength=float(config["content"].get("mean_prior_strength", 5.0)),
    )
    catalog_ids = np.asarray([movie.movie_id for movie in records], dtype=np.int32)
    aggregate = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
    counts = aggregate["count"].reindex(catalog_ids, fill_value=0).to_numpy(dtype=np.float32)
    means = aggregate["mean"].reindex(catalog_ids).to_numpy(dtype=np.float32)
    global_mean = float(ratings["rating"].mean())
    means = np.nan_to_num(means, nan=global_mean)
    prior = max(1.0, float(np.quantile(counts[counts > 0], 0.60)))
    quality = _minmax((counts * means + prior * global_mean) / (counts + prior))
    popularity = _minmax(np.log1p(counts))
    feature_statistics = {
        "predicted_rating": (float(global_mean), float(ratings["rating"].std() or 1.0)),
        "collaborative": (0.0, 1.0),
        "content": (0.0, 1.0),
        "recent_content": (0.0, 1.0),
        "negative_similarity": (0.0, 1.0),
        "quality": (float(quality.mean()), float(quality.std() or 1.0)),
        "popularity": (float(popularity.mean()), float(popularity.std() or 1.0)),
        "novelty": (float((1.0 - popularity).mean()), float((1.0 - popularity).std() or 1.0)),
        "freshness": (0.5, 0.3),
        "release_year_affinity": (0.5, 0.3),
        "exploration_jitter": (0.5, 0.29),
        "temporal_confidence": (0.0, 1.0),
        "source_agreement": (0.2, 0.2),
    }
    tuned_statistics = config.get("_tuned_feature_statistics")
    if tuned_statistics:
        feature_statistics = {
            name: (float(values[0]), max(float(values[1]), 1e-6))
            for name, values in tuned_statistics.items()
        }
    return V4Bundle(
        schema_version=4,
        model_version=model_version,
        config=public_config(config),
        catalog_movie_ids=catalog_ids,
        matrix_factorization=mf,
        content_model=content,
        quality_scores=quality,
        popularity_scores=popularity,
        feature_statistics=feature_statistics,
        reranker_weights=deepcopy(config["reranker"]["weights"]),
        training_cutoff=training_cutoff,
    )


def train_run(
    config: dict[str, Any],
    mode: str = "evaluation",
    run_id: str | None = None,
    *,
    apply_tuning: bool = True,
) -> Path:
    if mode not in {"evaluation", "production"}:
        raise ValueError("mode must be evaluation or production")
    config = _apply_tuned_model(config) if apply_tuning else deepcopy(config)
    ratings, movies = load_data(config)
    split = global_time_split(ratings, config)
    if mode == "evaluation":
        context = pd.concat([split.train, split.validation], ignore_index=True)
        test = split.test
        cutoff = split.validation_cutoff
        validation = None
    else:
        context = ratings
        test = ratings.iloc[0:0].copy()
        cutoff = int(ratings["timestamp"].max())
        validation = None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_id or f"v4-{mode}-{stamp}"
    bundle = fit_bundle(context, movies, config, validation, cutoff, run_id)
    manifest_path = Path(config["paths"]["prepared_dir"]) / "manifest.json"
    source_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    return save_bundle(
        bundle,
        test,
        config["paths"]["artifacts_dir"],
        run_id,
        mode,
        source_manifest,
    )


def tune(config: dict[str, Any]) -> dict[str, Any]:
    ratings, movies = load_data(config)
    split = global_time_split(ratings, config)
    users = np.sort(split.train["userId"].unique())
    sample_size = min(int(config["tuning"]["sample_users"]), len(users))
    rng = np.random.default_rng(int(config["seed"]))
    selected = set(map(int, rng.choice(users, sample_size, replace=False)))
    train = split.train.loc[split.train["userId"].isin(selected)]
    validation = split.validation.loc[split.validation["userId"].isin(selected)]
    trials = []
    for factors in config["tuning"]["factors"]:
        for learning_rate in config["tuning"]["learning_rates"]:
            for factor_reg in config["tuning"]["factor_regularizations"]:
                for bias_reg in config["tuning"]["bias_regularizations"]:
                    trial = deepcopy(config)
                    trial["model"].update(
                        factors=factors,
                        learning_rate=learning_rate,
                        factor_regularization=factor_reg,
                        bias_regularization=bias_reg,
                        epochs=min(6, int(config["model"]["epochs"])),
                    )
                    model = fit_biased_mf(train, trial, validation)
                    score = min(model.diagnostics.validation_rmse)
                    ndcg = _validation_ndcg(model, train, validation, int(config["model"]["min_relevant_rating"]))
                    trials.append(
                        {
                            "factors": factors,
                            "learning_rate": learning_rate,
                            "factor_regularization": factor_reg,
                            "bias_regularization": bias_reg,
                            "validation_rmse": score,
                            "validation_ndcg@10": ndcg,
                        }
                    )
    trials.sort(key=lambda value: (-value["validation_ndcg@10"], value["validation_rmse"]))
    finalists = []
    for candidate in trials[:4]:
        trial = deepcopy(config)
        trial["model"].update(
            factors=candidate["factors"],
            learning_rate=candidate["learning_rate"],
            factor_regularization=candidate["factor_regularization"],
            bias_regularization=candidate["bias_regularization"],
        )
        model = fit_biased_mf(train, trial, validation)
        finalists.append(
            {
                **candidate,
                "validation_rmse": min(model.diagnostics.validation_rmse),
                "validation_ndcg@10": _validation_ndcg(
                    model, train, validation, int(config["model"]["min_relevant_rating"])
                ),
                "stage": "finalist",
            }
        )
    finalists.sort(key=lambda value: (-value["validation_ndcg@10"], value["validation_rmse"]))
    selected_config = deepcopy(config)
    for key in ("factors", "learning_rate", "factor_regularization", "bias_regularization"):
        selected_config["model"][key] = finalists[0][key]
    tuning_bundle = fit_bundle(
        train,
        movies,
        selected_config,
        validation,
        split.train_cutoff,
        "v4-tuning",
    )
    temporal_calibration = _tune_temporal_calibration(
        tuning_bundle, train, validation, split.train_cutoff, maximum_users=200
    )
    tuning_bundle.config["temporal"].update(temporal_calibration)
    feature_statistics = _learn_feature_statistics(
        tuning_bundle, train, split.train_cutoff, maximum_users=100
    )
    tuning_bundle.feature_statistics = feature_statistics
    reranker_weights, diversity_lambda = _tune_reranker_weights(
        tuning_bundle,
        train,
        validation,
        split.train_cutoff,
        maximum_users=500,
    )
    result = {
        "best": finalists[0],
        "trials": trials,
        "finalists": finalists,
        "sample_users": sample_size,
        "reranker_weights": reranker_weights,
        "feature_statistics": feature_statistics,
        "diversity_lambda": diversity_lambda,
        "temporal_calibration": temporal_calibration,
        "config_sha256": _config_hash(config),
        "source_hashes": _prepared_source_hashes(config),
    }
    output = Path(config["paths"]["prepared_dir"]) / "v4-tuning.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _minmax(values):
    values = np.asarray(values, dtype=np.float32)
    minimum, maximum = float(values.min()), float(values.max())
    return np.zeros_like(values) if maximum <= minimum else (values - minimum) / (maximum - minimum)


def _validation_ndcg(model, train, validation, threshold, maximum_users=500):
    from ..evaluation import ndcg_at_k

    relevant = validation.loc[validation["rating"] >= threshold].groupby("userId")["movieId"].apply(set)
    users = sorted(set(map(int, relevant.index)) & set(model.user_lookup))[:maximum_users]
    seen = train.groupby("userId")["movieId"].apply(set)
    scores = []
    for user_id in users:
        predictions = model.predict_known(user_id, model.item_ids)
        order = np.argsort(predictions)[::-1]
        watched = seen.get(user_id, set())
        recommended = [int(model.item_ids[index]) for index in order if int(model.item_ids[index]) not in watched][:10]
        scores.append(ndcg_at_k(recommended, set(map(int, relevant.loc[user_id])), 10))
    return float(np.mean(scores)) if scores else 0.0


def _apply_tuned_model(config):
    config = deepcopy(config)
    if not bool(config.get("tuning", {}).get("enabled", True)):
        return config
    path = Path(config["paths"]["prepared_dir"]) / "v4-tuning.json"
    if not path.exists():
        return config
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("config_sha256") != _config_hash(config):
        raise ValueError("V4 tuning output does not match the current configuration; rerun tune")
    if value.get("source_hashes") != _prepared_source_hashes(config):
        raise ValueError("V4 tuning output does not match the prepared dataset; rerun tune")
    best = value["best"]
    for key in ("factors", "learning_rate", "factor_regularization", "bias_regularization"):
        config["model"][key] = best[key]
    if "reranker_weights" in value:
        config["reranker"]["weights"] = value["reranker_weights"]
    if "feature_statistics" in value:
        config["_tuned_feature_statistics"] = value["feature_statistics"]
    if "diversity_lambda" in value:
        config["reranker"]["diversity_lambda"] = float(value["diversity_lambda"])
    if "temporal_calibration" in value:
        config["temporal"].update(value["temporal_calibration"])
    return config


def _config_hash(config):
    payload = public_config(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepared_source_hashes(config):
    manifest = Path(config["paths"]["prepared_dir"]) / "manifest.json"
    if not manifest.exists():
        return {}
    return json.loads(manifest.read_text(encoding="utf-8")).get("source_hashes", {})


def _tune_temporal_calibration(
    bundle, train, validation, cutoff_timestamp, maximum_users=200
):
    cutoff = datetime.fromtimestamp(cutoff_timestamp, tz=UTC)
    model = bundle.matrix_factorization
    eligible = np.asarray(
        sorted(set(map(int, train["userId"].unique())) & set(map(int, validation["userId"].unique()))),
        dtype=np.int32,
    )
    rng = np.random.default_rng(int(bundle.config["seed"]))
    if len(eligible) > maximum_users:
        eligible = np.sort(rng.choice(eligible, maximum_users, replace=False))
    train_groups = {
        int(user): frame for user, frame in train.loc[train["userId"].isin(eligible)].groupby("userId")
    }
    validation_groups = {
        int(user): frame
        for user, frame in validation.loc[validation["userId"].isin(eligible)].groupby("userId")
    }
    recommender = V4Recommender(bundle)
    examples = []
    for user_id in map(int, eligible):
        profile = recommender.build_profile(
            user_id, dataframe_events(train_groups[user_id]), at=cutoff
        )
        if not profile.temporal_eligible:
            continue
        frame = validation_groups[user_id]
        mapped = np.asarray(
            [model.item_lookup.get(int(value), -1) for value in frame["movieId"]],
            dtype=np.int32,
        )
        known = mapped >= 0
        if not known.any():
            continue
        examples.append(
            (
                profile,
                mapped[known],
                frame["rating"].to_numpy(dtype=np.float32)[known],
            )
        )
    keys = (
        "sigmoid_intercept",
        "sigmoid_latent_weight",
        "sigmoid_genre_weight",
        "sigmoid_activity_weight",
    )
    parameters = {key: float(bundle.config["temporal"][key]) for key in keys}
    if not examples:
        return parameters
    for _ in range(2):
        for key in keys:
            original = parameters[key]
            step = 0.5
            candidates = (original - step, original, original + step)
            if key != "sigmoid_intercept":
                candidates = tuple(max(0.0, value) for value in candidates)
            best_value = original
            best_error = np.inf
            for candidate in candidates:
                parameters[key] = candidate
                error = _temporal_validation_rmse(bundle, examples, parameters)
                if error < best_error:
                    best_error = error
                    best_value = candidate
            parameters[key] = best_value
    return parameters


def _temporal_validation_rmse(bundle, examples, parameters):
    model = bundle.matrix_factorization
    temporal = bundle.config["temporal"]
    squared_error = 0.0
    observations = 0
    for profile, item_indices, ratings in examples:
        activity = (
            np.exp(
                -float(profile.days_since_latest)
                / float(temporal["recent_half_life_days"])
            )
            if profile.days_since_latest is not None
            else 0.0
        )
        logit = (
            parameters["sigmoid_intercept"]
            + parameters["sigmoid_latent_weight"] * profile.latent_drift
            + parameters["sigmoid_genre_weight"] * profile.genre_drift
            + parameters["sigmoid_activity_weight"] * activity
        )
        recent_weight = profile.temporal_confidence / (1.0 + np.exp(-logit))
        recent_weight = float(
            np.clip(recent_weight, 0.0, temporal["maximum_recent_weight"])
        )
        factors = (
            (1.0 - recent_weight) * profile.long_term.factors
            + recent_weight * profile.recent.factors
        )
        bias = (
            (1.0 - recent_weight) * profile.long_term.bias
            + recent_weight * profile.recent.bias
        )
        predictions = model.global_mean + bias + model.item_bias[item_indices]
        predictions += model.item_factors[item_indices] @ factors
        squared_error += float(np.square(ratings - predictions).sum())
        observations += len(ratings)
    return float(np.sqrt(squared_error / max(1, observations)))


def _learn_feature_statistics(bundle, train, cutoff_timestamp, maximum_users=100):
    cutoff = datetime.fromtimestamp(cutoff_timestamp, tz=UTC)
    available = {
        movie.movie_id
        for movie in bundle.content_model.movies
        if movie.release_year is None or movie.release_year <= cutoff.year
    }
    users = np.sort(train["userId"].unique())
    rng = np.random.default_rng(int(bundle.config["seed"]))
    if len(users) > maximum_users:
        users = np.sort(rng.choice(users, maximum_users, replace=False))
    grouped = {
        int(user): frame for user, frame in train.loc[train["userId"].isin(users)].groupby("userId")
    }
    recommender = V4Recommender(bundle)
    collected: dict[str, list[np.ndarray]] = {}
    for user_id in map(int, users):
        profile = recommender.build_profile(
            user_id, dataframe_events(grouped[user_id]), at=cutoff
        )
        candidate_set = recommender.generator.generate(profile, None, available, cutoff)
        for name, values in candidate_set.features.items():
            collected.setdefault(name, []).append(np.asarray(values, dtype=np.float64))
    result = {}
    for name, arrays in collected.items():
        values = np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float64)
        result[name] = (
            float(values.mean()) if len(values) else 0.0,
            float(values.std()) if len(values) and values.std() > 1e-6 else 1.0,
        )
    return result


def _tune_reranker_weights(bundle, train, validation, cutoff_timestamp, maximum_users=100):
    from ..evaluation import ndcg_at_k

    cutoff = datetime.fromtimestamp(cutoff_timestamp, tz=UTC)
    available = {
        movie.movie_id
        for movie in bundle.content_model.movies
        if movie.release_year is None or movie.release_year <= cutoff.year
    }
    relevant = validation.loc[
        (validation["rating"] >= float(bundle.config["model"]["min_relevant_rating"]))
        & validation["movieId"].isin(available)
    ].groupby("userId")["movieId"].apply(set)
    eligible_users = np.asarray(
        sorted(set(map(int, relevant.index)) & set(map(int, train["userId"].unique()))),
        dtype=np.int32,
    )
    rng = np.random.default_rng(int(bundle.config["seed"]))
    if len(eligible_users) > maximum_users:
        eligible_users = np.sort(rng.choice(eligible_users, maximum_users, replace=False))
    users = list(map(int, eligible_users))
    recommender = V4Recommender(bundle)
    grouped = {int(user): frame for user, frame in train.loc[train["userId"].isin(users)].groupby("userId")}
    examples = []
    for user_id in users:
        profile = recommender.build_profile(user_id, dataframe_events(grouped[user_id]), at=cutoff)
        candidates = recommender.generator.generate(profile, None, available, cutoff)
        examples.append((profile, candidates, set(map(int, relevant.loc[user_id]))))
    weights = deepcopy(bundle.reranker_weights)
    for key in weights:
        segment = [value for value in examples if recommender.reranker._weight_key(value[0]) == key]
        if not segment:
            continue
        for _ in range(2):
            for feature in list(weights[key]):
                original = float(weights[key][feature])
                candidates_values = [
                    _bounded_reranker_weight(bundle.config, feature, original - 0.05),
                    _bounded_reranker_weight(bundle.config, feature, original),
                    _bounded_reranker_weight(bundle.config, feature, original + 0.05),
                ]
                best_value = original
                best_score = (-np.inf, 0.0, 0.0)
                for candidate_value in candidates_values:
                    weights[key][feature] = candidate_value
                    recommender.reranker.weights = weights
                    score = _reranker_objective(
                        recommender,
                        segment,
                        len(available),
                        float(bundle.config["evaluation"]["minimum_coverage"]),
                        ndcg_at_k,
                    )
                    if score > best_score:
                        best_score = score
                        best_value = candidate_value
                weights[key][feature] = best_value
    recommender.reranker.weights = weights
    original_lambda = float(bundle.config["reranker"]["diversity_lambda"])
    lambda_values = sorted({0.0, 0.05, original_lambda, 0.12, 0.20})
    best_lambda = original_lambda
    best_score = (-np.inf, 0.0, 0.0)
    for candidate_lambda in lambda_values:
        recommender.reranker.config["reranker"]["diversity_lambda"] = candidate_lambda
        score = _reranker_objective(
            recommender,
            examples,
            len(available),
            float(bundle.config["evaluation"]["minimum_coverage"]),
            ndcg_at_k,
        )
        if score > best_score:
            best_score = score
            best_lambda = candidate_lambda
    recommender.reranker.config["reranker"]["diversity_lambda"] = best_lambda
    return weights, best_lambda


def _bounded_reranker_weight(config, feature, value):
    value = float(value)
    maximum_freshness = float(config["reranker"]["maximum_freshness_weight"])
    bounds = {
        "predicted_rating": (0.0, 0.70),
        "collaborative": (0.0, 0.45),
        "content": (0.0, 0.55),
        "recent_content": (0.0, 0.35),
        "negative_similarity": (-0.30, 0.0),
        "quality": (0.0, 0.30),
        "popularity": (0.0, 0.18),
        "novelty": (0.0, 0.04),
        "freshness": (0.0, maximum_freshness),
        "release_year_affinity": (0.0, 0.18),
        "source_agreement": (0.0, 0.12),
        "exploration_jitter": (0.0, 0.02),
        "temporal_confidence": (0.0, 0.10),
    }
    lower, upper = bounds.get(feature, (-1.0, 1.0))
    return float(np.clip(value, lower, upper))


def _reranker_objective(recommender, examples, catalog_size, minimum_coverage, ndcg_at_k):
    ndcg_values = []
    covered = set()
    for profile, candidate_set, truth in examples:
        recommendations = [
            int(recommender.bundle.catalog_movie_ids[result.catalog_index])
            for result in recommender.reranker.rank(candidate_set, profile, 10)
        ]
        covered.update(recommendations)
        ndcg_values.append(ndcg_at_k(recommendations, truth, 10))
    coverage = len(covered) / max(1, catalog_size)
    ndcg = float(np.mean(ndcg_values)) if ndcg_values else 0.0
    coverage_shortfall = max(0.0, minimum_coverage - coverage)
    # Coverage is important, but it is not allowed to dominate relevance.  A
    # smooth penalty nudges the reranker away from over-concentration without
    # selecting novelty-heavy settings that pass coverage by destroying NDCG.
    objective = ndcg - 0.35 * coverage_shortfall
    return (
        objective,
        coverage,
        ndcg,
    )
