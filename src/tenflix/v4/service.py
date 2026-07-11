from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Sequence

from .events import OnboardingPreferences, RatingEvent, RecommendationResponse, UserProfileSummary
from .recommender import V4Recommender
from .repositories import CatalogRepository, RatingRepository, RepositoryError


class RecommendationService:
    def __init__(
        self,
        recommender: V4Recommender,
        ratings: RatingRepository,
        catalog: CatalogRepository,
    ):
        self.recommender = recommender
        self.ratings = ratings
        self.catalog = catalog
        self._profile_cache = {}
        self._lock = RLock()

    def record_rating(self, event: RatingEvent) -> UserProfileSummary:
        try:
            now = datetime.now(UTC)
            if event.rated_at > now + timedelta(minutes=5):
                raise ValueError("rated_at cannot be in the future")
            if event.watched_at and event.watched_at > now + timedelta(minutes=5):
                raise ValueError("watched_at cannot be in the future")
            available = self.catalog.available_movie_ids(event.preference_time)
            if event.movie_id not in available:
                raise ValueError("movie is unknown, unavailable, or not released at rated_at")
            self.ratings.append_rating(event)
            revision = self.ratings.get_revision(event.user_id)
            with self._lock:
                self._profile_cache.pop(event.user_id, None)
            events = self.ratings.get_user_ratings(event.user_id)
            profile = self.recommender.build_profile(event.user_id, events, at=now)
            return UserProfileSummary(
                event.user_id,
                profile.lifecycle,
                len(events),
                profile.temporal_confidence,
                revision,
            )
        except (OSError, KeyError) as error:
            raise RepositoryError("Unable to persist rating") from error

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        preferences: OnboardingPreferences | None = None,
        at: datetime | None = None,
    ) -> RecommendationResponse:
        at = at or datetime.now(UTC)
        try:
            events = self.ratings.get_user_ratings(user_id)
            revision = self.ratings.get_revision(user_id)
            available = self.catalog.available_movie_ids(at)
        except (OSError, KeyError) as error:
            raise RepositoryError("Unable to load current recommendation state") from error
        cache_key = (self.recommender.bundle.model_version, user_id, revision)
        preference_key = (
            preferences.preferred_genres,
            preferences.disliked_genres,
            preferences.liked_movie_ids,
            preferences.era_preference.value,
        ) if preferences else None
        # Recency decay and temporal-window membership depend on request time.
        # A daily bucket prevents profiles from remaining stale indefinitely.
        cache_key = (*cache_key, preference_key, at.date().isoformat())
        with self._lock:
            cached = self._profile_cache.get(user_id)
        if cached is not None and cached[0] == cache_key:
            profile = cached[1]
        else:
            profile = self.recommender.build_profile(user_id, events, preferences, at)
            with self._lock:
                self._profile_cache[user_id] = (cache_key, profile)
        return self.recommender.recommend_profile(
            profile,
            top_k,
            preferences,
            available,
            revision,
            at,
        )

    def preview(
        self,
        ratings: Sequence[RatingEvent],
        top_k: int = 10,
        preferences: OnboardingPreferences | None = None,
        at: datetime | None = None,
    ) -> RecommendationResponse:
        user_id = ratings[0].user_id if ratings else 0
        if any(event.user_id != user_id for event in ratings):
            raise ValueError("All preview ratings must belong to one user")
        at = at or datetime.now(UTC)
        return self.recommender.recommend(
            user_id,
            ratings,
            top_k,
            preferences,
            self.catalog.available_movie_ids(at),
            "preview",
            at,
        )


def create_fastapi_app(service: RecommendationService):
    try:
        from fastapi import Body, FastAPI, HTTPException
        from pydantic import BaseModel, Field, ValidationError
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install TenFlix with the 'service' extra") from error

    class RatingPayload(BaseModel):
        user_id: int
        movie_id: int
        rating: float = Field(ge=0.5, le=5.0)
        rated_at: datetime
        watched_at: datetime | None = None
        source: str = "organic"

    class PreviewPayload(BaseModel):
        ratings: list[RatingPayload] = Field(default_factory=list)
        top_k: int = Field(default=10, ge=1, le=100)
        preferred_genres: list[str] = Field(default_factory=list)
        disliked_genres: list[str] = Field(default_factory=list)
        liked_movie_ids: list[int] = Field(default_factory=list)
        era_preference: str = "balanced"

    app = FastAPI(title="TenFlix V4 Recommendation Service", version="4.0.0")

    def event(payload):
        from .events import RatingSource

        return RatingEvent(
            payload.user_id,
            payload.movie_id,
            payload.rating,
            payload.rated_at,
            payload.watched_at,
            RatingSource(payload.source),
        )

    @app.post("/v1/ratings")
    def record(payload: dict = Body(...)):
        try:
            payload = RatingPayload.model_validate(payload)
            return asdict(service.record_rating(event(payload)))
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=error.errors()) from error
        except (ValueError, RepositoryError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/v1/recommendations/{user_id}")
    def recommend(user_id: int, top_k: int = 10):
        try:
            return asdict(service.recommend(user_id, top_k))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RepositoryError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/v1/recommendations/preview")
    def preview(payload: dict = Body(...)):
        from .events import EraPreference

        try:
            payload = PreviewPayload.model_validate(payload)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=error.errors()) from error
        preferences = OnboardingPreferences(
            tuple(payload.preferred_genres),
            tuple(payload.disliked_genres),
            tuple(payload.liked_movie_ids),
            EraPreference(payload.era_preference),
        )
        try:
            return asdict(
                service.preview(
                    [event(value) for value in payload.ratings], payload.top_k, preferences
                )
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RepositoryError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/v1/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/model")
    def model():
        return {
            "model_version": service.recommender.bundle.model_version,
            "schema_version": service.recommender.bundle.schema_version,
        }

    return app
