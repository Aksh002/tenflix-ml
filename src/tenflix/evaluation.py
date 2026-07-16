from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .models import ModelBundle
from .recommender import HybridRecommender


def precision_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    values = list(recommended)[:k]
    return len(set(values) & relevant) / k if k > 0 else 0.0


def recall_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(list(recommended)[:k]) & relevant) / len(relevant)


def hit_rate_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    return float(bool(set(list(recommended)[:k]) & relevant))


def average_precision_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for rank, movie_id in enumerate(list(recommended)[:k], start=1):
        if movie_id in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def ndcg_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    gains = [1.0 if movie_id in relevant else 0.0 for movie_id in list(recommended)[:k]]
    dcg = sum(gain / np.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return float(dcg / ideal)


def _genre_diversity(movie_ids: list[int], bundle: ModelBundle) -> float:
    if len(movie_ids) < 2:
        return 0.0
    genre_sets = []
    for movie_id in movie_ids:
        row = bundle.movies.iloc[bundle.catalog_lookup[movie_id]]
        value = str(row["genres"])
        genre_sets.append(set() if value == "(no genres listed)" else set(value.split("|")))
    distances = []
    for left in range(len(genre_sets)):
        for right in range(left + 1, len(genre_sets)):
            union = genre_sets[left] | genre_sets[right]
            similarity = len(genre_sets[left] & genre_sets[right]) / len(union) if union else 1.0
            distances.append(1.0 - similarity)
    return float(np.mean(distances))


def _paired_interval(
    candidate: np.ndarray,
    baseline: np.ndarray,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    differences = np.asarray(candidate) - np.asarray(baseline)
    if len(differences) == 0:
        return {"difference": 0.0, "lower": 0.0, "upper": 0.0}
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selection = rng.integers(0, len(differences), len(differences))
        estimates[index] = differences[selection].mean()
    alpha = 1.0 - confidence
    return {
        "difference": float(differences.mean()),
        "lower": float(np.quantile(estimates, alpha / 2)),
        "upper": float(np.quantile(estimates, 1 - alpha / 2)),
    }


def evaluate(bundle: ModelBundle, holdout: pd.DataFrame) -> dict[str, object]:
    if holdout.empty:
        raise ValueError("Evaluation requires a non-empty future holdout")
    recommender = HybridRecommender(bundle)
    evaluation_config = bundle.config["evaluation"]
    k_values = sorted(int(value) for value in evaluation_config["k_values"])
    maximum_k = max(k_values)
    threshold = float(bundle.config["model"]["min_relevant_rating"])
    relevant_frame = holdout.loc[holdout["rating"] >= threshold]
    relevant = relevant_frame.groupby("userId")["movieId"].apply(lambda values: set(map(int, values)))
    all_relevant = holdout.groupby("userId")["movieId"].apply(lambda values: set(map(int, values)))
    eligible = np.asarray(sorted(set(relevant.index) & set(map(int, bundle.user_ids))), dtype=np.int32)
    rng = np.random.default_rng(int(bundle.config["seed"]))
    maximum_users = int(evaluation_config["max_users"])
    if maximum_users > 0 and len(eligible) > maximum_users:
        eligible = np.sort(rng.choice(eligible, size=maximum_users, replace=False))
    strategies = ("popularity", "static_cf", "recent_cf", "hybrid")
    per_user: dict[str, dict[int, dict[str, float]]] = {name: {} for name in strategies}
    coverage: dict[str, set[int]] = {name: set() for name in strategies}

    popularity_probability = bundle.popularity_scores.astype(np.float64) + 1e-12
    popularity_probability /= popularity_probability.sum()
    for user_id in eligible:
        truth = relevant.loc[int(user_id)]
        for strategy in strategies:
            recommendations = recommender.ranked_movie_ids(int(user_id), maximum_k, strategy)
            coverage[strategy].update(recommendations)
            metrics: dict[str, float] = {}
            for k in k_values:
                metrics[f"precision@{k}"] = precision_at_k(recommendations, truth, k)
                metrics[f"recall@{k}"] = recall_at_k(recommendations, truth, k)
                metrics[f"hit_rate@{k}"] = hit_rate_at_k(recommendations, truth, k)
                metrics[f"map@{k}"] = average_precision_at_k(recommendations, truth, k)
                metrics[f"ndcg@{k}"] = ndcg_at_k(recommendations, truth, k)
            selected = recommendations[:maximum_k]
            indices = [bundle.catalog_lookup[value] for value in selected]
            metrics["novelty"] = float(np.mean(-np.log2(popularity_probability[indices]))) if indices else 0.0
            metrics["genre_diversity"] = _genre_diversity(selected, bundle)
            per_user[strategy][int(user_id)] = metrics

    aggregate: dict[str, dict[str, float]] = {}
    for strategy in strategies:
        rows = list(per_user[strategy].values())
        aggregate[strategy] = {
            key: float(np.mean([row[key] for row in rows])) for key in rows[0]
        } if rows else {}
        aggregate[strategy]["catalog_coverage"] = len(coverage[strategy]) / len(bundle.catalog_movie_ids)

    segment_reports: dict[str, object] = {"drift": {}, "lifecycle": {}, "interaction_bands": {}}
    user_indices = {int(user): index for index, user in enumerate(bundle.user_ids)}
    for segment in ("stable", "moderate", "volatile", "unknown"):
        segment_users = [
            int(user)
            for user in eligible
            if str(bundle.drift_types[user_indices[int(user)]]) == segment
        ]
        segment_reports["drift"][segment] = _segment_summary(per_user, segment_users)
    for segment in ("cold", "sparse", "mature"):
        segment_users = [
            int(user)
            for user in eligible
            if str(bundle.lifecycle[user_indices[int(user)]]) == segment
        ]
        segment_reports["lifecycle"][segment] = _segment_summary(per_user, segment_users)
    interaction_bands = {
        "1-19": (1, 19),
        "20-49": (20, 49),
        "50-99": (50, 99),
        "100-249": (100, 249),
        "250+": (250, np.iinfo(np.int32).max),
    }
    for label, (minimum, maximum) in interaction_bands.items():
        segment_users = [
            int(user)
            for user in eligible
            if minimum <= int(bundle.interaction_counts[user_indices[int(user)]]) <= maximum
        ]
        segment_reports["interaction_bands"][label] = _segment_summary(per_user, segment_users)

    confidence = float(evaluation_config["confidence"])
    samples = int(evaluation_config["bootstrap_samples"])
    gates: dict[str, object] = {}
    gate_pass = True
    for segment in ("moderate", "volatile"):
        users = [
            int(user)
            for user in eligible
            if str(bundle.drift_types[user_indices[int(user)]]) == segment
        ]
        gates[segment] = {}
        for metric in (f"ndcg@{k_values[0]}", f"recall@{k_values[0]}"):
            interval = _paired_interval(
                np.asarray([per_user["hybrid"][user][metric] for user in users]),
                np.asarray([per_user["popularity"][user][metric] for user in users]),
                samples,
                confidence,
                int(bundle.config["seed"]),
            )
            interval["passed"] = bool(users) and interval["lower"] > 0
            gate_pass = gate_pass and bool(interval["passed"])
            gates[segment][metric] = interval
    coverage_pass = aggregate["hybrid"]["catalog_coverage"] > aggregate["popularity"]["catalog_coverage"]
    gate_pass = gate_pass and coverage_pass
    gates["coverage"] = {
        "hybrid": aggregate["hybrid"]["catalog_coverage"],
        "popularity": aggregate["popularity"]["catalog_coverage"],
        "passed": coverage_pass,
    }

    sensitivity = _all_interactions_sensitivity(recommender, eligible, all_relevant, k_values[0])
    return {
        "schema_version": 3,
        "users_evaluated": len(eligible),
        "relevance_threshold": threshold,
        "k_values": k_values,
        "aggregate": aggregate,
        "segments": segment_reports,
        "acceptance_gates": gates,
        "validated": gate_pass,
        "all_interactions_sensitivity": sensitivity,
    }


def _segment_summary(
    per_user: dict[str, dict[int, dict[str, float]]], users: list[int]
) -> dict[str, object]:
    result: dict[str, object] = {"users": len(users), "strategies": {}}
    for strategy, values in per_user.items():
        rows = [values[user] for user in users]
        result["strategies"][strategy] = (
            {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} if rows else {}
        )
    return result


def _all_interactions_sensitivity(
    recommender: HybridRecommender,
    users: np.ndarray,
    truth: pd.Series,
    k: int,
) -> dict[str, float]:
    result = {}
    for strategy in ("popularity", "static_cf", "recent_cf", "hybrid"):
        scores = [
            recall_at_k(
                recommender.ranked_movie_ids(int(user), k, strategy), truth.loc[int(user)], k
            )
            for user in users
            if int(user) in truth.index
        ]
        result[f"{strategy}_recall@{k}"] = float(np.mean(scores)) if scores else 0.0
    return result


def write_report(report: dict[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
