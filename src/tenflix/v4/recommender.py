from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Sequence

import numpy as np

from .candidates import CandidateGenerator
from .content import ContentModel
from .events import (
    OnboardingPreferences,
    RatingEvent,
    RecommendationResponse,
    V4Recommendation,
)
from .matrix_factorization import BiasedMFModel
from .profiles import UserProfile, build_user_profile
from .reranker import LinearReranker


@dataclass
class V4Bundle:
    schema_version: int
    model_version: str
    config: dict[str, Any]
    catalog_movie_ids: np.ndarray
    matrix_factorization: BiasedMFModel
    content_model: ContentModel
    quality_scores: np.ndarray
    popularity_scores: np.ndarray
    feature_statistics: dict[str, tuple[float, float]]
    reranker_weights: dict[str, dict[str, float]]
    training_cutoff: int

    def __post_init__(self) -> None:
        if self.schema_version != 4:
            raise ValueError("V4Bundle requires artifact schema 4")


class V4Recommender:
    def __init__(self, bundle: V4Bundle):
        self.bundle = bundle
        self.generator = CandidateGenerator(
            bundle.matrix_factorization,
            bundle.content_model,
            bundle.catalog_movie_ids,
            bundle.quality_scores,
            bundle.popularity_scores,
            bundle.config,
        )
        self.reranker = LinearReranker(
            bundle.content_model.movies,
            bundle.config,
            bundle.feature_statistics,
            bundle.reranker_weights,
        )

    def build_profile(
        self,
        user_id: int,
        events: Sequence[RatingEvent],
        preferences: OnboardingPreferences | None = None,
        at: datetime | None = None,
    ) -> UserProfile:
        return build_user_profile(
            user_id,
            events,
            self.bundle.matrix_factorization,
            self.bundle.content_model,
            self.bundle.config,
            preferences,
            at,
        )

    def recommend(
        self,
        user_id: int,
        events: Sequence[RatingEvent],
        top_k: int = 10,
        preferences: OnboardingPreferences | None = None,
        available_movie_ids: set[int] | None = None,
        profile_revision: str = "preview",
        at: datetime | None = None,
    ) -> RecommendationResponse:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        at = at or datetime.now(UTC)
        profile = self.build_profile(user_id, events, preferences, at)
        return self.recommend_profile(
            profile,
            top_k,
            preferences,
            available_movie_ids,
            profile_revision,
            at,
        )

    def recommend_profile(
        self,
        profile: UserProfile,
        top_k: int = 10,
        preferences: OnboardingPreferences | None = None,
        available_movie_ids: set[int] | None = None,
        profile_revision: str = "preview",
        at: datetime | None = None,
    ) -> RecommendationResponse:
        at = at or datetime.now(UTC)
        available = (
            set(map(int, self.bundle.catalog_movie_ids))
            if available_movie_ids is None
            else available_movie_ids
        )
        candidates = self.generator.generate(profile, preferences, available, at)
        ranked = self.reranker.rank(candidates, profile, top_k)
        recommendations = []
        for rank, result in enumerate(ranked, start=1):
            movie = self.bundle.content_model.movies[result.catalog_index]
            recommendations.append(
                V4Recommendation(
                    movie_id=movie.movie_id,
                    title=movie.title,
                    genres=list(movie.genres),
                    release_year=movie.release_year,
                    score=result.score,
                    rank=rank,
                    strategy=self._strategy(profile),
                    reason=self._explanation(profile, movie, result.contributors),
                    score_contributors=result.contributors,
                )
            )
        return RecommendationResponse(
            user_id=profile.user_id,
            model_version=self.bundle.model_version,
            profile_revision=profile_revision,
            lifecycle=profile.lifecycle,
            temporal_confidence=profile.temporal_confidence,
            recommendations=recommendations,
        )

    @staticmethod
    def _strategy(profile: UserProfile) -> str:
        if profile.lifecycle == "new":
            return "new_multi_source"
        if profile.lifecycle == "cold":
            return "cold_content_fold_in"
        if profile.lifecycle == "sparse":
            return "sparse_hybrid_fold_in"
        return "mature_continuous_recency" if profile.temporal_eligible else "mature_long_term"

    def _explanation(self, profile, movie, contributors):
        positive_events = sorted(
            (event for event in profile.events if event.rating >= 4.0),
            key=lambda event: (event.preference_time, event.rating),
            reverse=True,
        )
        if positive_events and contributors.get("content", 0) > 0:
            titles = []
            for event in positive_events:
                index = self.bundle.content_model.movie_lookup.get(event.movie_id)
                if index is not None:
                    titles.append(self.bundle.content_model.movies[index].title)
                if len(titles) == 2:
                    break
            if titles:
                return f"Because you rated {' and '.join(titles)} highly"
        if profile.temporal_confidence > 0 and contributors.get("recent_content", 0) > 0:
            return f"Matches your recent interest in {', '.join(movie.genres[:2])}"
        if contributors.get("quality", 0) > 0:
            return "A highly rated match for your current profile"
        return f"Matches {', '.join(movie.genres[:2]) or 'your selected preferences'}"
