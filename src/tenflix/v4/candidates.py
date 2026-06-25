from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from .content import ContentModel
from .events import EraPreference, OnboardingPreferences
from .matrix_factorization import BiasedMFModel
from .profiles import UserProfile


@dataclass
class CandidateSet:
    indices: np.ndarray
    features: dict[str, np.ndarray]
    sources: list[set[str]]


class CandidateGenerator:
    def __init__(
        self,
        model: BiasedMFModel,
        content: ContentModel,
        catalog_movie_ids: np.ndarray,
        quality: np.ndarray,
        popularity: np.ndarray,
        config: dict[str, Any],
    ):
        self.model = model
        self.content = content
        self.catalog_movie_ids = catalog_movie_ids
        self.quality = quality
        self.popularity = popularity
        self.config = config
        self.catalog_lookup = {int(value): index for index, value in enumerate(catalog_movie_ids)}
        self._release_years = np.asarray(
            [movie.release_year if movie.release_year is not None else np.nan for movie in content.movies],
            dtype=np.float32,
        )
        self._freshness_cache: dict[tuple[int, str], np.ndarray] = {}
        self.mf_catalog_indices = np.asarray(
            [self.catalog_lookup[int(value)] for value in model.item_ids], dtype=np.int32
        )
        norms = np.linalg.norm(model.item_factors, axis=1, keepdims=True)
        self.unit_item_factors = np.divide(
            model.item_factors,
            norms,
            out=np.zeros_like(model.item_factors),
            where=norms > 1e-10,
        )

    def generate(
        self,
        profile: UserProfile,
        preferences: OnboardingPreferences | None,
        available: set[int],
        at: datetime,
    ) -> CandidateSet:
        count = len(self.catalog_movie_ids)
        predicted = np.full(count, self.model.global_mean + profile.blended_bias, dtype=np.float32)
        predicted[self.mf_catalog_indices] += self.model.item_bias
        collaborative = np.zeros(count, dtype=np.float32)
        if np.linalg.norm(profile.blended_factors) > 1e-10:
            predicted[self.mf_catalog_indices] += self.model.item_factors @ profile.blended_factors
            unit = profile.blended_factors / np.linalg.norm(profile.blended_factors)
            collaborative[self.mf_catalog_indices] = self.unit_item_factors @ unit
        content_scores, negative = self.content.scores(
            profile.long_content, self.config["content"]["negative_penalty"]
        )
        recent_content, _ = self.content.scores(
            profile.recent_content, self.config["content"]["negative_penalty"]
        )
        if profile.lifecycle == "new" and preferences:
            query = self.content.query_profile(preferences)
            content_scores, negative = self.content.scores(
                query, self.config["content"]["negative_penalty"]
            )
        freshness = self._freshness(at, preferences)
        era_affinity = self._era_affinity(profile.preferred_year, preferences, at)
        novelty = (1.0 - self.popularity).astype(np.float32)
        exploration_jitter = self._exploration_jitter(profile.user_id, at)
        all_features = {
            "predicted_rating": predicted,
            "collaborative": collaborative,
            "content": content_scores.astype(np.float32),
            "recent_content": recent_content.astype(np.float32),
            "negative_similarity": negative.astype(np.float32),
            "quality": self.quality,
            "popularity": self.popularity,
            "novelty": novelty,
            "freshness": freshness,
            "release_year_affinity": era_affinity,
            "exploration_jitter": exploration_jitter,
            "temporal_confidence": np.full(count, profile.temporal_confidence, dtype=np.float32),
        }
        eligible = np.asarray(
            [
                index
                for index, movie_id in enumerate(self.catalog_movie_ids)
                if int(movie_id) in available
            ],
            dtype=np.int32,
        )
        seen = {event.movie_id for event in profile.events}
        if preferences:
            seen.update(preferences.liked_movie_ids)
        if seen:
            eligible = np.asarray(
                [index for index in eligible if int(self.catalog_movie_ids[index]) not in seen],
                dtype=np.int32,
            )
        sources_by_index: dict[int, set[str]] = {}
        limits = self.config["candidates"]
        self._add_top(sources_by_index, eligible, predicted, limits["collaborative"], "mf")
        self._add_top(sources_by_index, eligible, content_scores, limits["content"], "content")
        quality_popularity = (
            0.45 * self.quality
            + 0.25 * self.popularity
            + 0.30 * np.clip(content_scores, 0.0, None)
        )
        self._add_top(
            sources_by_index,
            eligible,
            quality_popularity,
            limits["quality_popularity"],
            "quality_popularity",
        )
        if profile.temporal_eligible:
            self._add_top(sources_by_index, eligible, recent_content, limits["recent"], "recent")
        exploration = 0.62 * self.quality + 0.28 * novelty + 0.10 * freshness
        self._add_top(sources_by_index, eligible, exploration, limits["exploration"], "exploration")
        indices = np.asarray(sorted(sources_by_index), dtype=np.int32)
        sources = [sources_by_index[int(index)] for index in indices]
        source_agreement = np.zeros(count, dtype=np.float32)
        for index, sources_for_movie in sources_by_index.items():
            source_agreement[index] = len(sources_for_movie) / 5.0
        all_features["source_agreement"] = source_agreement
        return CandidateSet(
            indices=indices,
            features={name: values[indices] for name, values in all_features.items()},
            sources=sources,
        )

    @staticmethod
    def _add_top(mapping, eligible, scores, limit, source):
        if len(eligible) == 0 or limit <= 0:
            return
        finite = eligible[np.isfinite(scores[eligible])]
        if len(finite) == 0:
            return
        values = scores[finite]
        if float(values.max() - values.min()) <= 1e-12:
            return
        amount = min(int(limit), len(finite))
        local = np.argpartition(values, -amount)[-amount:]
        for index in finite[local]:
            mapping.setdefault(int(index), set()).add(source)

    def _freshness(self, at, preferences):
        settings = self.config["reranker"]
        floor = float(settings["freshness_floor"])
        half_life = float(settings["freshness_half_life_years"])
        era = preferences.era_preference.value if preferences else EraPreference.BALANCED.value
        cache_key = (at.year, era)
        cached = self._freshness_cache.get(cache_key)
        if cached is not None:
            return cached
        age = np.maximum(0.0, at.year - self._release_years)
        freshness = floor + (1.0 - floor) * np.exp(-age / half_life)
        freshness[np.isnan(self._release_years)] = 0.5
        if preferences and preferences.era_preference == EraPreference.CLASSICS:
            freshness[:] = 0.5
        elif preferences and preferences.era_preference == EraPreference.RECENT:
            freshness = np.minimum(1.0, freshness * 1.15)
        result = freshness.astype(np.float32)
        self._freshness_cache[cache_key] = result
        return result

    def _era_affinity(self, preferred_year, preferences, at):
        if preferences and preferences.era_preference == EraPreference.CLASSICS:
            # Treat films at least 25 years old as a broad classic era, with a
            # soft decay for newer titles rather than excluding them.
            classic_boundary = at.year - 25
            age_from_boundary = np.maximum(0.0, self._release_years - classic_boundary)
            result = np.exp(-age_from_boundary / 12.0)
            result[np.isnan(self._release_years)] = 0.5
            return result.astype(np.float32)
        if preferences and preferences.era_preference == EraPreference.RECENT:
            result = np.exp(-np.abs(self._release_years - at.year) / 15.0)
            result[np.isnan(self._release_years)] = 0.5
            return result.astype(np.float32)
        if preferred_year is None:
            return np.full(len(self.content.movies), 0.5, dtype=np.float32)
        result = np.exp(-np.abs(self._release_years - preferred_year) / 15.0)
        result[np.isnan(self._release_years)] = 0.5
        return result.astype(np.float32)

    def _exploration_jitter(self, user_id: int, at: datetime) -> np.ndarray:
        # Deterministic, tiny request-scoped variety.  It only acts inside the
        # already filtered/ranked candidate set, so it cannot introduce rated,
        # unavailable, or future-release movies.  Using a sine hash avoids
        # Python's process-randomized hash seed and keeps evaluation reproducible.
        seed = int(self.config["seed"])
        day = int(at.strftime("%Y%m%d"))
        values = (
            self.catalog_movie_ids.astype(np.float64) * 12.9898
            + float(user_id) * 78.233
            + float(day + seed) * 37.719
        )
        fractional = np.sin(values) * 43758.5453
        fractional = fractional - np.floor(fractional)
        return fractional.astype(np.float32)
