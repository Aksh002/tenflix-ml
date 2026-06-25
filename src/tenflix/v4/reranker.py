from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .candidates import CandidateSet
from .events import Movie
from .profiles import UserProfile


@dataclass
class RankedCandidate:
    catalog_index: int
    score: float
    contributors: dict[str, float]
    sources: set[str]


class LinearReranker:
    def __init__(
        self,
        movies: list[Movie],
        config: dict[str, Any],
        feature_statistics: dict[str, tuple[float, float]] | None = None,
        weights: dict[str, dict[str, float]] | None = None,
    ):
        self.movies = movies
        self.config = config
        self.feature_statistics = feature_statistics or {}
        self.weights = weights or config["reranker"]["weights"]
        # MMR evaluates the same catalog metadata many times.  Cache the parsed
        # representation once so request latency does not scale with regex/set
        # construction inside the selection loop.
        self._genre_sets = [set(movie.genres) for movie in movies]
        self._decades = [movie.release_year // 10 if movie.release_year else None for movie in movies]
        self._prefixes = [_title_prefix(movie.title) for movie in movies]

    def rank(self, candidates: CandidateSet, profile: UserProfile, top_k: int) -> list[RankedCandidate]:
        key = self._weight_key(profile)
        weights = self.weights[key]
        total = np.zeros(len(candidates.indices), dtype=np.float32)
        contributions: dict[str, np.ndarray] = {}
        for name, weight in weights.items():
            values = candidates.features.get(name)
            if values is None:
                continue
            values = values * self._lifecycle_feature_scale(profile, name)
            mean, std = self.feature_statistics.get(name, (0.0, 1.0))
            normalized = (values - mean) / max(float(std), 1e-6)
            contribution = float(weight) * normalized
            contributions[name] = contribution
            total += contribution
        ordered = np.argsort(total, kind="stable")[::-1]
        # Forty to sixty strong alternatives are enough for the top-10/20 MMR
        # constraint and avoid an O(n^2) similarity pass over weak candidates.
        ordered = ordered[: min(len(ordered), max(40, top_k * 3))]
        similarities = self._similarity_matrix(candidates.indices[ordered])
        selected: list[int] = []
        selected_local: list[int] = []
        lambda_value = float(self.config["reranker"]["diversity_lambda"])
        while len(selected) < min(top_k, len(ordered)):
            best_position = None
            best_local = None
            best_score = -np.inf
            for local, position in enumerate(ordered):
                if int(position) in selected:
                    continue
                selected_similarities = similarities[local, selected_local]
                if np.count_nonzero(selected_similarities >= 0.80) >= 2:
                    continue
                similarity = float(selected_similarities.max()) if selected_local else 0.0
                score = float(total[position] - lambda_value * similarity)
                if score > best_score:
                    best_score = score
                    best_position = int(position)
                    best_local = local
            if best_position is None:
                fallback = next(
                    ((local, int(value)) for local, value in enumerate(ordered) if int(value) not in selected),
                    None,
                )
                best_local, best_position = fallback if fallback is not None else (None, None)
                if best_position is None:
                    break
            selected.append(best_position)
            selected_local.append(int(best_local))
        return [
            RankedCandidate(
                catalog_index=int(candidates.indices[position]),
                score=float(total[position]),
                contributors={name: float(values[position]) for name, values in contributions.items()},
                sources=candidates.sources[position],
            )
            for position in selected
        ]

    def _lifecycle_feature_scale(self, profile: UserProfile, feature: str) -> float:
        if profile.lifecycle != "sparse" or feature not in {"predicted_rating", "collaborative"}:
            return 1.0
        minimum = int(self.config["lifecycle"]["cold_max"]) + 1
        maximum = int(self.config["lifecycle"]["sparse_max"])
        progress = np.clip((len(profile.events) - minimum) / max(1, maximum - minimum), 0, 1)
        return float(0.25 + 0.35 * progress)

    @staticmethod
    def _weight_key(profile: UserProfile) -> str:
        if profile.lifecycle in {"new", "cold", "sparse"}:
            return profile.lifecycle
        return "mature_temporal_eligible" if profile.temporal_eligible else "mature_temporal_ineligible"

    def _similarity(self, left_index: int, right_index: int) -> float:
        left_genres = self._genre_sets[left_index]
        right_genres = self._genre_sets[right_index]
        union = left_genres | right_genres
        genre = len(left_genres & right_genres) / len(union) if union else 0.0
        decade = 1.0 if self._decades[left_index] == self._decades[right_index] is not None else 0.0
        prefix = 1.0 if self._prefixes[left_index] == self._prefixes[right_index] else 0.0
        return 0.65 * genre + 0.15 * decade + 0.20 * prefix

    def _similarity_matrix(self, catalog_indices: np.ndarray) -> np.ndarray:
        count = len(catalog_indices)
        result = np.zeros((count, count), dtype=np.float32)
        for left in range(count):
            for right in range(left):
                value = self._similarity(int(catalog_indices[left]), int(catalog_indices[right]))
                result[left, right] = value
                result[right, left] = value
        return result


def _title_prefix(title: str) -> str:
    title = re.sub(r"\(\d{4}\)\s*$", "", title).lower()
    title = re.sub(r"\b(part|chapter|episode)\s+\w+.*$", "", title)
    return re.sub(r"[^a-z0-9]+", " ", title).strip()[:24]
