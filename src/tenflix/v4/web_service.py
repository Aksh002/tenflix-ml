from __future__ import annotations

import os
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Annotated

from .auth import AuthError, SupabaseAuthenticator
from .events import EraPreference, OnboardingPreferences, RatingEvent, RatingSource
from .recommender import V4Recommender
from .repositories import RepositoryError
from .runtime_env import load_env_file
from .service import RecommendationService
from .web_repositories import (
    SupabaseCatalogRepository,
    SupabaseRatingRepository,
    SupabaseUserRepository,
)


def create_product_fastapi_app(
    recommender: V4Recommender,
    users: SupabaseUserRepository,
    authenticator: SupabaseAuthenticator | None = None,
):
    try:
        from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field, ValidationError
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install TenFlix with the 'web' extra") from error

    authenticator = authenticator or SupabaseAuthenticator()

    class RatingPayload(BaseModel):
        movie_id: int
        rating: float = Field(ge=0.5, le=5.0)
        rated_at: datetime | None = None
        watched_at: datetime | None = None
        source: str = "organic"

    class PreviewRatingPayload(BaseModel):
        movie_id: int
        rating: float = Field(ge=0.5, le=5.0)
        rated_at: datetime | None = None
        watched_at: datetime | None = None
        source: str = "onboarding"

    class PreviewPayload(BaseModel):
        ratings: list[PreviewRatingPayload] = Field(default_factory=list)
        top_k: int = Field(default=10, ge=1, le=50)
        preferred_genres: list[str] = Field(default_factory=list)
        disliked_genres: list[str] = Field(default_factory=list)
        liked_movie_ids: list[int] = Field(default_factory=list)
        era_preference: str = "balanced"

    class PreferencesPayload(BaseModel):
        provider_region: str = Field(min_length=2, max_length=2)

    app = FastAPI(title="TenFlix Product API", version="4.1.0")
    cors_origins = _cors_origins()
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "PATCH", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

    def current_user(authorization: Annotated[str | None, Header()] = None):
        try:
            claims = authenticator.claims_from_authorization(authorization)
            return users.get_or_create_user(claims.subject, claims.email)
        except AuthError as error:
            detail = str(error)
            headers = {"WWW-Authenticate": "Bearer"}
            raise HTTPException(status_code=401, detail=detail, headers=headers) from error

    def service_for(user):
        ratings = SupabaseRatingRepository(users.db, user.app_user_id)
        catalog = SupabaseCatalogRepository(users.db, user.app_user_id)
        return RecommendationService(recommender, ratings, catalog), catalog, ratings

    def preferences(payload: PreviewPayload) -> OnboardingPreferences:
        return OnboardingPreferences(
            tuple(payload.preferred_genres),
            tuple(payload.disliked_genres),
            tuple(payload.liked_movie_ids),
            EraPreference(payload.era_preference),
        )

    @app.get("/v1/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/model")
    def model():
        return {
            "model_version": recommender.bundle.model_version,
            "schema_version": recommender.bundle.schema_version,
        }

    @app.get("/v1/me")
    def me(user=Depends(current_user)):
        return asdict(user)

    @app.patch("/v1/me/preferences")
    def update_preferences(payload: dict = Body(...), user=Depends(current_user)):
        try:
            value = PreferencesPayload.model_validate(payload)
            return asdict(users.update_preferences(user, value.provider_region))
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=error.errors()) from error

    @app.get("/v1/catalog")
    def catalog_page(
        user=Depends(current_user),
        limit: int = Query(default=40, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        search: str | None = None,
        genre: str | None = None,
        media_type: str = "movie",
        unrated_only: bool = False,
    ):
        _, catalog, _ = service_for(user)
        return asdict(
            catalog.page(
                limit=limit,
                offset=offset,
                search=search,
                genre=genre,
                media_type=media_type,
                unrated_only=unrated_only,
            )
        )

    @app.get("/v1/catalog/rows")
    def catalog_rows(user=Depends(current_user), limit: int = Query(default=14, ge=4, le=30)):
        _, catalog, _ = service_for(user)
        return {"rows": [asdict(row) for row in catalog.rows(user.provider_region, limit)]}

    @app.get("/v1/catalog/{movie_id}")
    def catalog_detail(movie_id: int, user=Depends(current_user)):
        _, catalog, _ = service_for(user)
        item = catalog.detail(movie_id, user.provider_region)
        if item is None:
            raise HTTPException(status_code=404, detail="Movie not found")
        return asdict(item)

    @app.post("/v1/ratings")
    def record_rating(payload: dict = Body(...), user=Depends(current_user)):
        try:
            value = RatingPayload.model_validate(payload)
            service, _, _ = service_for(user)
            event = RatingEvent(
                user.app_user_id,
                value.movie_id,
                value.rating,
                value.rated_at or datetime.now(UTC),
                value.watched_at,
                RatingSource(value.source),
            )
            return asdict(service.record_rating(event))
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=error.errors()) from error
        except (ValueError, RepositoryError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/v1/ratings/me")
    def ratings_me(user=Depends(current_user)):
        _, _, ratings = service_for(user)
        return {"ratings": [asdict(value) for value in ratings.current_ratings()]}

    @app.get("/v1/recommendations/me")
    def recommendations_me(user=Depends(current_user), top_k: int = Query(default=10, ge=1, le=50)):
        try:
            service, _, _ = service_for(user)
            return asdict(service.recommend(user.app_user_id, top_k))
        except RepositoryError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/v1/recommendations/preview")
    def preview(payload: dict = Body(...)):
        try:
            value = PreviewPayload.model_validate(payload)
            demo_user = 0
            catalog = SupabaseCatalogRepository(users.db)
            service = RecommendationService(
                recommender,
                SupabaseRatingRepository(users.db, demo_user),
                catalog,
            )
            now = datetime.now(UTC)
            events = [
                RatingEvent(
                    demo_user,
                    rating.movie_id,
                    rating.rating,
                    rating.rated_at or now,
                    rating.watched_at,
                    RatingSource(rating.source),
                )
                for rating in value.ratings
            ]
            return asdict(service.preview(events, value.top_k, preferences(value)))
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=error.errors()) from error
        except (ValueError, RepositoryError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/v1/enrichment/sync")
    def enrichment_sync():
        raise HTTPException(
            status_code=501,
            detail="Run `tenflix-v4 enrich-catalog` from the CLI for deterministic enrichment.",
        )

    return app


def _cors_origins() -> list[str]:
    load_env_file()
    raw = os.environ.get(
        "TENFLIX_CORS_ORIGINS",
        ",".join(
            [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:3001",
                "http://127.0.0.1:3001",
                "http://localhost:3002",
                "http://127.0.0.1:3002",
            ]
        ),
    )
    origins = [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]
    return list(dict.fromkeys(origins))
