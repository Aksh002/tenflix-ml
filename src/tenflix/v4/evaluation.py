from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .data import dataframe_events, load_data
from .events import EraPreference, OnboardingPreferences
from .recommender import V4Bundle, V4Recommender


def evaluate_bundle(
    bundle: V4Bundle,
    context: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    settings = bundle.config["evaluation"]
    threshold = float(bundle.config["model"]["min_relevant_rating"])
    cutoff = pd.Timestamp(bundle.training_cutoff, unit="s", tz="UTC").to_pydatetime()
    available = {
        movie.movie_id
        for movie in bundle.content_model.movies
        if movie.release_year is None or movie.release_year <= cutoff.year
    }
    relevant_frame = test.loc[
        (test["rating"] >= threshold) & test["movieId"].isin(available)
    ]
    relevant = relevant_frame.groupby("userId")["movieId"].apply(lambda values: set(map(int, values)))
    context_users = set(map(int, context["userId"].unique()))
    eligible = np.asarray(sorted(set(map(int, relevant.index)) & context_users), dtype=np.int32)
    rng = np.random.default_rng(int(bundle.config["seed"]))
    maximum = int(settings["max_users"])
    if maximum > 0 and len(eligible) > maximum:
        eligible = np.sort(rng.choice(eligible, maximum, replace=False))
    k_values = sorted(map(int, settings["k_values"]))
    maximum_k = max(k_values)
    recommender = V4Recommender(bundle)
    quality_order = np.argsort(0.65 * bundle.quality_scores + 0.35 * bundle.popularity_scores)[::-1]
    per_user = {name: {} for name in ("popularity", "static_mf", "recent_mf", "full_v4")}
    coverage = {name: set() for name in per_user}
    diversity_values = []
    decade_diversity_values = []
    novelty_values = []
    concentration_values = []
    recommendation_latencies = []
    fold_latencies = []
    temporal_users = []
    cold_full = []
    cold_popularity = []
    new_full = []
    new_popularity = []
    integrity_violations = {"rated": 0, "unavailable_or_future": 0}
    deterministic = True
    lifecycle_by_user = {}
    grouped = {int(user): frame for user, frame in context.loc[context["userId"].isin(eligible)].groupby("userId")}
    simulation_count = min(2000, len(eligible))
    cold_simulation_users = set(
        map(int, rng.choice(eligible, simulation_count, replace=False))
    ) if simulation_count else set()
    for user_id in eligible:
        frame = grouped[int(user_id)].sort_values("timestamp")
        events = dataframe_events(frame)
        truth = relevant.loc[int(user_id)]
        seen = set(map(int, frame["movieId"]))
        popularity = [
            int(bundle.catalog_movie_ids[index])
            for index in quality_order
            if int(bundle.catalog_movie_ids[index]) in available
            and int(bundle.catalog_movie_ids[index]) not in seen
        ][:maximum_k]
        start = time.perf_counter()
        profile = recommender.build_profile(int(user_id), events, at=cutoff)
        fold_latencies.append((time.perf_counter() - start) * 1000)
        static = _static_mf(bundle, profile, seen, available, maximum_k)
        recent_mf = _recent_mf(bundle, profile, seen, available, maximum_k)
        start = time.perf_counter()
        response = recommender.recommend_profile(
            profile,
            maximum_k,
            available_movie_ids=available,
            at=cutoff,
        )
        recommendation_latencies.append((time.perf_counter() - start) * 1000)
        full = [value.movie_id for value in response.recommendations]
        lifecycle_by_user[int(user_id)] = (
            f"{profile.lifecycle}_temporal_eligible"
            if profile.lifecycle == "mature" and profile.temporal_eligible
            else profile.lifecycle
        )
        integrity_violations["rated"] += len(set(full) & seen)
        integrity_violations["unavailable_or_future"] += sum(value not in available for value in full)
        if len(per_user["full_v4"]) == 0:
            repeated = recommender.recommend_profile(
                profile, maximum_k, available_movie_ids=available, at=cutoff
            )
            deterministic = full == [value.movie_id for value in repeated.recommendations]
        for name, recommendations in (
            ("popularity", popularity),
            ("static_mf", static),
            ("recent_mf", recent_mf),
            ("full_v4", full),
        ):
            coverage[name].update(recommendations)
            per_user[name][int(user_id)] = _metrics(recommendations, truth, k_values)
        diversity_values.append(_genre_diversity(full, bundle))
        decade_diversity_values.append(_decade_diversity(full, bundle))
        novelty_values.append(_novelty(full, bundle))
        concentration_values.append(_popularity_concentration(full, bundle))
        if profile.temporal_eligible:
            temporal_users.append(int(user_id))
        if int(user_id) in cold_simulation_users:
            limited = events[: min(10, len(events))]
            preferred = _preferences_from_events(limited, bundle)
            cold = recommender.recommend(
                int(user_id), limited, maximum_k, preferred, available, at=cutoff
            )
            cold_full.append(ndcg_at_k([value.movie_id for value in cold.recommendations], truth, 10))
            limited_seen = {event.movie_id for event in limited}
            genre_popularity = _genre_popularity(
                bundle, preferred, limited_seen, available, maximum_k
            )
            cold_popularity.append(ndcg_at_k(genre_popularity, truth, 10))
            new = recommender.recommend(
                int(user_id), [], maximum_k, preferred, available, at=cutoff
            )
            new_full.append(
                ndcg_at_k([value.movie_id for value in new.recommendations], truth, 10)
            )
            new_popularity.append(ndcg_at_k(genre_popularity, truth, 10))
    aggregate = {}
    for name, users in per_user.items():
        rows = list(users.values())
        aggregate[name] = (
            {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} if rows else {}
        )
        aggregate[name]["catalog_coverage"] = len(coverage[name]) / max(1, len(available))
    aggregate["full_v4"].update(
        novelty=float(np.mean(novelty_values)) if novelty_values else 0.0,
        genre_diversity=float(np.mean(diversity_values)) if diversity_values else 0.0,
        decade_diversity=(
            float(np.mean(decade_diversity_values)) if decade_diversity_values else 0.0
        ),
        popularity_concentration=(
            float(np.mean(concentration_values)) if concentration_values else 0.0
        ),
    )
    segments = _segment_metrics(per_user["full_v4"], lifecycle_by_user)
    confidence = float(settings["confidence"])
    samples = int(settings["bootstrap_samples"])
    gates = {}
    for baseline in ("popularity", "static_mf"):
        gates[f"accuracy_vs_{baseline}"] = {}
        for metric in ("ndcg@10", "recall@10"):
            interval = _paired_interval(
                [per_user["full_v4"][int(user)][metric] for user in eligible],
                [per_user[baseline][int(user)][metric] for user in eligible],
                samples,
                confidence,
                int(bundle.config["seed"]),
            )
            interval["passed"] = interval["lower"] > 0
            gates[f"accuracy_vs_{baseline}"][metric] = interval
    temporal_interval = _paired_interval(
        [per_user["recent_mf"][user]["ndcg@10"] for user in temporal_users],
        [per_user["static_mf"][user]["ndcg@10"] for user in temporal_users],
        samples,
        confidence,
        int(bundle.config["seed"]),
    )
    temporal_interval["passed"] = bool(temporal_users) and temporal_interval["lower"] > 0
    gates["temporal_ndcg"] = temporal_interval
    cold_interval = _paired_interval(
        cold_full,
        cold_popularity,
        samples,
        confidence,
        int(bundle.config["seed"]),
    )
    cold_interval["passed"] = cold_interval["lower"] > 0
    gates["cold_start_ndcg"] = cold_interval
    new_interval = _paired_interval(
        new_full,
        new_popularity,
        samples,
        confidence,
        int(bundle.config["seed"]),
    )
    new_margin = float(settings.get("new_start_noninferiority_margin", 0.0))
    new_interval["noninferiority_margin"] = new_margin
    new_interval["passed"] = new_interval["lower"] >= new_margin
    gates["new_start_ndcg"] = new_interval
    gates["coverage"] = {
        "value": aggregate["full_v4"]["catalog_coverage"],
        "minimum": float(settings["minimum_coverage"]),
        "passed": aggregate["full_v4"]["catalog_coverage"] >= float(settings["minimum_coverage"]),
    }
    diversity = float(np.mean(diversity_values)) if diversity_values else 0.0
    gates["diversity"] = {
        "value": diversity,
        "minimum": float(settings["minimum_diversity"]),
        "passed": diversity >= float(settings["minimum_diversity"]),
    }
    recommendation_p95 = float(np.percentile(recommendation_latencies, 95)) if recommendation_latencies else 0
    combined_p95 = float(np.percentile(np.asarray(recommendation_latencies) + np.asarray(fold_latencies), 95)) if recommendation_latencies else 0
    gates["latency"] = {
        "recommendation_p95_ms": recommendation_p95,
        "fold_in_plus_recommendation_p95_ms": combined_p95,
        "passed": recommendation_p95 <= float(settings["recommendation_p95_ms"])
        and combined_p95 <= float(settings["fold_in_recommendation_p95_ms"]),
    }
    gates["integrity"] = {
        "rated_violations": integrity_violations["rated"],
        "unavailable_or_future_violations": integrity_violations["unavailable_or_future"],
        "passed": not any(integrity_violations.values()),
    }
    gates["determinism"] = {"passed": bool(deterministic)}
    v3_reference = _v3_reference(settings.get("v3_reference_run"))
    validated = all(_gate_passed(value) for value in gates.values())
    return {
        "schema_version": 4,
        "model_version": bundle.model_version,
        "users_evaluated": len(eligible),
        "temporal_users": len(temporal_users),
        "cold_start_simulation_users": len(cold_full),
        "new_start_simulation_users": len(new_full),
        "aggregate": aggregate,
        "segments": segments,
        "genre_diversity": diversity,
        "latency": {
            "recommendation_p95_ms": recommendation_p95,
            "fold_in_plus_recommendation_p95_ms": combined_p95,
        },
        "gates": gates,
        "v3_reference": v3_reference,
        "validated": bool(validated),
    }


def evaluate_run(bundle: V4Bundle, test: pd.DataFrame) -> dict[str, Any]:
    ratings, _ = load_data(bundle.config)
    context = ratings.loc[ratings["timestamp"] <= bundle.training_cutoff]
    return evaluate_bundle(bundle, context, test)


def _metrics(recommendations, truth, k_values):
    result = {}
    for k in k_values:
        result.update(
            {
                f"precision@{k}": precision_at_k(recommendations, truth, k),
                f"recall@{k}": recall_at_k(recommendations, truth, k),
                f"hit_rate@{k}": hit_rate_at_k(recommendations, truth, k),
                f"map@{k}": average_precision_at_k(recommendations, truth, k),
                f"ndcg@{k}": ndcg_at_k(recommendations, truth, k),
            }
        )
    return result


def _static_mf(bundle, profile, seen, available, k):
    folded = profile.long_term
    scores = bundle.matrix_factorization.predict_folded(folded)
    order = np.argsort(scores)[::-1]
    return [
        int(bundle.matrix_factorization.item_ids[index])
        for index in order
        if int(bundle.matrix_factorization.item_ids[index]) not in seen
        and int(bundle.matrix_factorization.item_ids[index]) in available
    ][:k]


def _recent_mf(bundle, profile, seen, available, k):
    scores = (
        bundle.matrix_factorization.global_mean
        + profile.blended_bias
        + bundle.matrix_factorization.item_bias
        + bundle.matrix_factorization.item_factors @ profile.blended_factors
    )
    order = np.argsort(scores)[::-1]
    return [
        int(bundle.matrix_factorization.item_ids[index])
        for index in order
        if int(bundle.matrix_factorization.item_ids[index]) not in seen
        and int(bundle.matrix_factorization.item_ids[index]) in available
    ][:k]


def _genre_diversity(movie_ids, bundle):
    genres = [set(bundle.content_model.movies[bundle.content_model.movie_lookup[value]].genres) for value in movie_ids]
    values = []
    for left in range(len(genres)):
        for right in range(left + 1, len(genres)):
            union = genres[left] | genres[right]
            values.append(1 - len(genres[left] & genres[right]) / len(union) if union else 0)
    return float(np.mean(values)) if values else 0.0


def _decade_diversity(movie_ids, bundle):
    decades = [
        movie.release_year // 10
        for value in movie_ids
        if (movie := bundle.content_model.movies[bundle.content_model.movie_lookup[value]]).release_year
    ]
    if len(decades) < 2:
        return 0.0
    values = [decades[left] != decades[right] for left in range(len(decades)) for right in range(left)]
    return float(np.mean(values)) if values else 0.0


def _novelty(movie_ids, bundle):
    indices = [bundle.content_model.movie_lookup[value] for value in movie_ids]
    if not indices:
        return 0.0
    mass = np.asarray(bundle.popularity_scores, dtype=float) + 1e-6
    probabilities = mass / mass.sum()
    return float(np.mean(-np.log2(probabilities[indices])))


def _popularity_concentration(movie_ids, bundle):
    indices = [bundle.content_model.movie_lookup[value] for value in movie_ids]
    return float(np.mean(bundle.popularity_scores[indices])) if indices else 0.0


def _genre_popularity(bundle, preferences, seen, available, k):
    query = bundle.content_model.query_profile(preferences)
    content, _ = bundle.content_model.scores(query, bundle.config["content"]["negative_penalty"])
    scores = 0.50 * np.clip(content, 0, None) + 0.30 * bundle.quality_scores + 0.20 * bundle.popularity_scores
    order = np.argsort(scores)[::-1]
    return [
        int(bundle.catalog_movie_ids[index])
        for index in order
        if int(bundle.catalog_movie_ids[index]) in available
        and int(bundle.catalog_movie_ids[index]) not in seen
    ][:k]


def _segment_metrics(user_metrics, lifecycle_by_user):
    grouped = {}
    for user_id, metrics in user_metrics.items():
        grouped.setdefault(lifecycle_by_user[user_id], []).append(metrics)
    return {
        segment: {
            "users": len(rows),
            **{key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
        }
        for segment, rows in sorted(grouped.items())
    }


def _preferences_from_events(events, bundle):
    genres = []
    liked = []
    for event in events:
        if event.rating >= 4:
            liked.append(event.movie_id)
            index = bundle.content_model.movie_lookup.get(event.movie_id)
            if index is not None:
                genres.extend(bundle.content_model.movies[index].genres)
    common = tuple(dict.fromkeys(genres))[:3]
    return OnboardingPreferences(common, (), tuple(liked[:3]), EraPreference.BALANCED)


def _paired_interval(candidate, baseline, samples, confidence, seed):
    difference = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    if len(difference) == 0:
        return {"difference": 0.0, "lower": 0.0, "upper": 0.0}
    rng = np.random.default_rng(seed)
    estimates = [difference[rng.integers(0, len(difference), len(difference))].mean() for _ in range(samples)]
    alpha = 1 - confidence
    return {
        "difference": float(difference.mean()),
        "lower": float(np.quantile(estimates, alpha / 2)),
        "upper": float(np.quantile(estimates, 1 - alpha / 2)),
    }


def _v3_reference(path):
    if not path:
        return None
    report = Path(path) / "evaluation.json"
    if not report.exists():
        return None
    value = json.loads(report.read_text(encoding="utf-8"))
    return {"run": str(path), "aggregate": value.get("aggregate"), "validated": value.get("validated")}


def _gate_passed(value):
    if "passed" in value:
        return bool(value["passed"])
    return all(bool(child.get("passed", False)) for child in value.values())


def write_report(report, path):
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
