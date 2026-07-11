from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class RatingSource(StrEnum):
    ONBOARDING = "onboarding"
    ORGANIC = "organic"
    RECOMMENDATION = "recommendation"
    IMPORTED = "imported"
    LEGACY = "legacy"


class EraPreference(StrEnum):
    BALANCED = "balanced"
    RECENT = "recent"
    CLASSICS = "classics"


@dataclass(frozen=True, slots=True)
class RatingEvent:
    user_id: int
    movie_id: int
    rating: float
    rated_at: datetime
    watched_at: datetime | None = None
    source: RatingSource = RatingSource.ORGANIC

    def __post_init__(self) -> None:
        if self.user_id < 0 or self.movie_id <= 0:
            raise ValueError("user_id must be non-negative and movie_id must be positive")
        if not 0.5 <= float(self.rating) <= 5.0:
            raise ValueError("rating must be within 0.5 and 5.0")
        if not isinstance(self.source, RatingSource):
            try:
                object.__setattr__(self, "source", RatingSource(self.source))
            except ValueError as error:
                raise ValueError(f"Unsupported rating source: {self.source}") from error
        if self.rated_at.tzinfo is None:
            object.__setattr__(self, "rated_at", self.rated_at.replace(tzinfo=UTC))
        if self.watched_at is not None and self.watched_at.tzinfo is None:
            object.__setattr__(self, "watched_at", self.watched_at.replace(tzinfo=UTC))

    @property
    def preference_time(self) -> datetime:
        return self.watched_at or self.rated_at

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["rated_at"] = self.rated_at.isoformat()
        value["watched_at"] = self.watched_at.isoformat() if self.watched_at else None
        value["source"] = self.source.value
        return value


@dataclass(frozen=True, slots=True)
class OnboardingPreferences:
    preferred_genres: tuple[str, ...] = ()
    disliked_genres: tuple[str, ...] = ()
    liked_movie_ids: tuple[int, ...] = ()
    era_preference: EraPreference = EraPreference.BALANCED

    def __post_init__(self) -> None:
        if not isinstance(self.era_preference, EraPreference):
            try:
                object.__setattr__(self, "era_preference", EraPreference(self.era_preference))
            except ValueError as error:
                raise ValueError(f"Unsupported era preference: {self.era_preference}") from error
        if any(movie_id <= 0 for movie_id in self.liked_movie_ids):
            raise ValueError("liked_movie_ids must contain positive IDs")
        preferred = {value.casefold() for value in self.preferred_genres}
        disliked = {value.casefold() for value in self.disliked_genres}
        if preferred & disliked:
            raise ValueError("A genre cannot be both preferred and disliked")


@dataclass(frozen=True, slots=True)
class Movie:
    movie_id: int
    title: str
    genres: tuple[str, ...]
    release_year: int | None
    available: bool = True


@dataclass(frozen=True, slots=True)
class V4Recommendation:
    movie_id: int
    title: str
    genres: list[str]
    release_year: int | None
    score: float
    rank: int
    strategy: str
    reason: str
    score_contributors: dict[str, float]


@dataclass(frozen=True, slots=True)
class UserProfileSummary:
    user_id: int
    lifecycle: str
    rating_count: int
    temporal_confidence: float
    profile_revision: str


@dataclass(frozen=True, slots=True)
class RecommendationResponse:
    user_id: int
    model_version: str
    profile_revision: str
    lifecycle: str
    temporal_confidence: float
    recommendations: list[V4Recommendation] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
