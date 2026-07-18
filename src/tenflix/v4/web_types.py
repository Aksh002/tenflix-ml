from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class AppUser:
    app_user_id: int
    auth_user_id: str | None
    email: str | None = None
    provider_region: str = "IN"


@dataclass(frozen=True, slots=True)
class WatchProvider:
    provider_name: str
    provider_type: str
    region: str
    provider_id: int | None = None
    provider_logo_url: str | None = None
    deep_link: str | None = None
    display_priority: int | None = None


@dataclass(frozen=True, slots=True)
class WatchAction:
    action_type: str
    label: str
    url: str
    region: str | None = None


@dataclass(frozen=True, slots=True)
class CatalogItem:
    movie_id: int
    title: str
    genres: list[str]
    release_year: int | None
    media_type: str = "movie"
    imdb_id: str | None = None
    tmdb_id: int | None = None
    poster_url: str | None = None
    backdrop_url: str | None = None
    overview: str | None = None
    runtime_minutes: int | None = None
    user_rating: float | None = None
    watch_providers: list[WatchProvider] = field(default_factory=list)
    watch_actions: list[WatchAction] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CatalogPage:
    items: list[CatalogItem]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class CatalogRow:
    slug: str
    title: str
    subtitle: str
    items: list[CatalogItem]


@dataclass(frozen=True, slots=True)
class UserRatingSummary:
    movie_id: int
    title: str
    rating: float
    rated_at: datetime
    source: str


def stremio_url(imdb_id: str | None, media_type: str = "movie") -> str | None:
    if not imdb_id:
        return None
    kind = "series" if media_type == "series" else "movie"
    return f"stremio:///detail/{kind}/{imdb_id}"


def imdb_url(imdb_id: str | None) -> str | None:
    return f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None


def tmdb_url(tmdb_id: int | None, media_type: str = "movie") -> str | None:
    if tmdb_id is None:
        return None
    kind = "tv" if media_type == "series" else "movie"
    return f"https://www.themoviedb.org/{kind}/{tmdb_id}"


def compact_item(value: dict[str, Any]) -> CatalogItem:
    return CatalogItem(
        movie_id=int(value["movie_id"]),
        title=str(value["title"]),
        genres=list(value.get("genres") or []),
        release_year=value.get("release_year"),
        media_type=value.get("media_type") or "movie",
        imdb_id=value.get("imdb_id"),
        tmdb_id=value.get("tmdb_id"),
        poster_url=value.get("poster_url"),
        backdrop_url=value.get("backdrop_url"),
        overview=value.get("overview"),
        runtime_minutes=value.get("runtime_minutes"),
        user_rating=value.get("user_rating"),
    )
