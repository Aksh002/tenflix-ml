from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .events import Movie, RatingEvent, RatingSource
from .repositories import CatalogRepository, RatingRepository, RepositoryError
from .runtime_env import load_env_file
from .web_types import (
    AppUser,
    CatalogItem,
    CatalogPage,
    CatalogRow,
    UserRatingSummary,
    WatchAction,
    WatchProvider,
    imdb_url,
    stremio_url,
    tmdb_url,
)

WEB_SCHEMA_TABLES = (
    "profiles",
    "catalog_items",
    "rating_events",
    "current_ratings",
    "watch_providers",
    "watch_actions",
)


def _require_psycopg():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "The TenFlix web API requires psycopg for Supabase/PostgreSQL access. "
            'Install the web dependencies with: python -m pip install -e ".[web]" '
            'or use the full development setup: python -m pip install -e ".[dev]"'
        ) from error
    return psycopg, dict_row


class PostgresConnectionFactory:
    def __init__(self, database_url: str | None = None):
        load_env_file()
        _require_psycopg()
        self.database_url = _normalize_database_url(database_url or os.getenv("DATABASE_URL"))
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is required for the Supabase/Postgres adapter")

    @contextmanager
    def connect(self):
        psycopg, dict_row = _require_psycopg()
        with psycopg.connect(
            self.database_url,
            row_factory=dict_row,
            prepare_threshold=None,
        ) as connection:
            yield connection

    def apply_sql_path(self, path: str | os.PathLike[str]) -> list[str]:
        sql_path = Path(path)
        files = sorted(sql_path.glob("*.sql")) if sql_path.is_dir() else [sql_path]
        applied: list[str] = []
        for file in files:
            self.apply_sql_file(file)
            applied.append(str(file))
        return applied

    def apply_sql_file(self, path: str | os.PathLike[str]) -> None:
        sql = os.fspath(path)
        with open(sql, encoding="utf-8") as handle:
            statement = handle.read()
        with self.connect() as connection:
            connection.execute(statement)
            connection.commit()

    def schema_status(self) -> dict[str, bool]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                select table_name
                from information_schema.tables
                where table_schema = 'public'
                  and table_name = any(%s)
                """,
                (list(WEB_SCHEMA_TABLES),),
            ).fetchall()
        present = {str(row["table_name"]) for row in rows}
        return {table: table in present for table in WEB_SCHEMA_TABLES}


def _normalize_database_url(database_url: str | None) -> str | None:
    if not database_url:
        return database_url
    parts = urlsplit(database_url)
    if not parts.query:
        return database_url
    supported_query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() != "pgbouncer"
    ]
    query = urlencode(supported_query)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


class SupabaseUserRepository:
    def __init__(self, db: PostgresConnectionFactory):
        self.db = db

    def get_or_create_user(
        self,
        auth_user_id: str | None,
        email: str | None = None,
        provider_region: str | None = None,
    ) -> AppUser:
        with self.db.connect() as connection:
            if auth_user_id:
                row = connection.execute(
                    """
                    insert into profiles (auth_user_id, email, provider_region)
                    values (%s::uuid, %s, coalesce(%s, 'IN'))
                    on conflict (auth_user_id) do update
                    set email = coalesce(excluded.email, profiles.email)
                    returning id, auth_user_id::text, email, provider_region
                    """,
                    (auth_user_id, email, provider_region),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    insert into profiles (email, provider_region)
                    values (%s, coalesce(%s, 'IN'))
                    returning id, auth_user_id::text, email, provider_region
                    """,
                    (email, provider_region),
                ).fetchone()
            connection.commit()
        return AppUser(
            app_user_id=int(row["id"]),
            auth_user_id=row["auth_user_id"],
            email=row["email"],
            provider_region=row["provider_region"] or "IN",
        )

    def update_preferences(self, user: AppUser, provider_region: str) -> AppUser:
        with self.db.connect() as connection:
            row = connection.execute(
                """
                update profiles
                set provider_region = %s
                where id = %s
                returning id, auth_user_id::text, email, provider_region
                """,
                (provider_region.upper(), user.app_user_id),
            ).fetchone()
            connection.commit()
        return AppUser(
            app_user_id=int(row["id"]),
            auth_user_id=row["auth_user_id"],
            email=row["email"],
            provider_region=row["provider_region"] or "IN",
        )


class SupabaseRatingRepository(RatingRepository):
    def __init__(self, db: PostgresConnectionFactory, app_user_id: int):
        self.db = db
        self.app_user_id = int(app_user_id)

    def get_user_ratings(self, user_id: int) -> list[RatingEvent]:
        _assert_user(user_id, self.app_user_id)
        with self.db.connect() as connection:
            rows = connection.execute(
                """
                select movie_id, rating, rated_at, watched_at, source
                from current_ratings
                where app_user_id = %s
                order by coalesce(watched_at, rated_at), movie_id
                """,
                (self.app_user_id,),
            ).fetchall()
        return [
            RatingEvent(
                user_id=self.app_user_id,
                movie_id=int(row["movie_id"]),
                rating=float(row["rating"]),
                rated_at=_aware(row["rated_at"]),
                watched_at=_aware(row["watched_at"]) if row["watched_at"] else None,
                source=RatingSource(row["source"]),
            )
            for row in rows
        ]

    def append_rating(self, event: RatingEvent) -> None:
        _assert_user(event.user_id, self.app_user_id)
        with self.db.connect() as connection:
            connection.execute(
                """
                insert into rating_events
                  (app_user_id, movie_id, rating, rated_at, watched_at, source)
                values (%s, %s, %s, %s, %s, %s)
                """,
                (
                    self.app_user_id,
                    event.movie_id,
                    event.rating,
                    event.rated_at,
                    event.watched_at,
                    event.source.value,
                ),
            )
            connection.execute(
                """
                insert into current_ratings
                  (app_user_id, movie_id, rating, rated_at, watched_at, source, revision)
                values (%s, %s, %s, %s, %s, %s, 1)
                on conflict (app_user_id, movie_id) do update
                set rating = excluded.rating,
                    rated_at = excluded.rated_at,
                    watched_at = excluded.watched_at,
                    source = excluded.source,
                    revision = current_ratings.revision + 1,
                    updated_at = now()
                """,
                (
                    self.app_user_id,
                    event.movie_id,
                    event.rating,
                    event.rated_at,
                    event.watched_at,
                    event.source.value,
                ),
            )
            connection.commit()

    def get_revision(self, user_id: int) -> str:
        _assert_user(user_id, self.app_user_id)
        with self.db.connect() as connection:
            row = connection.execute(
                """
                select count(*) as ratings, coalesce(sum(revision), 0) as revision_sum,
                       coalesce(max(updated_at), 'epoch'::timestamptz) as latest
                from current_ratings
                where app_user_id = %s
                """,
                (self.app_user_id,),
            ).fetchone()
        payload = f"{self.app_user_id}:{row['ratings']}:{row['revision_sum']}:{row['latest']}".encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def current_ratings(self) -> list[UserRatingSummary]:
        with self.db.connect() as connection:
            rows = connection.execute(
                """
                select c.movie_id, c.title, r.rating, r.rated_at, r.source
                from current_ratings r
                join catalog_items c on c.movie_id = r.movie_id
                where r.app_user_id = %s
                order by r.rated_at desc
                """,
                (self.app_user_id,),
            ).fetchall()
        return [
            UserRatingSummary(
                movie_id=int(row["movie_id"]),
                title=row["title"],
                rating=float(row["rating"]),
                rated_at=_aware(row["rated_at"]),
                source=row["source"],
            )
            for row in rows
        ]


class SupabaseCatalogRepository(CatalogRepository):
    def __init__(self, db: PostgresConnectionFactory, app_user_id: int | None = None):
        self.db = db
        self.app_user_id = app_user_id

    def get_movies(self, movie_ids: Iterable[int] | None = None) -> list[Movie]:
        ids = list(movie_ids) if movie_ids is not None else None
        with self.db.connect() as connection:
            if ids is None:
                rows = connection.execute(
                    "select movie_id, title, genres, release_year from catalog_items order by movie_id"
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    select movie_id, title, genres, release_year
                    from catalog_items
                    where movie_id = any(%s)
                    order by movie_id
                    """,
                    (ids,),
                ).fetchall()
        return [
            Movie(
                movie_id=int(row["movie_id"]),
                title=row["title"],
                genres=tuple(row["genres"] or ()),
                release_year=row["release_year"],
            )
            for row in rows
        ]

    def available_movie_ids(self, at=None) -> set[int]:
        year = getattr(at, "year", None)
        with self.db.connect() as connection:
            if year is None:
                rows = connection.execute("select movie_id from catalog_items").fetchall()
            else:
                rows = connection.execute(
                    """
                    select movie_id from catalog_items
                    where release_year is null or release_year <= %s
                    """,
                    (year,),
                ).fetchall()
        return {int(row["movie_id"]) for row in rows}

    def page(
        self,
        *,
        limit: int = 40,
        offset: int = 0,
        search: str | None = None,
        genre: str | None = None,
        media_type: str = "movie",
        unrated_only: bool = False,
    ) -> CatalogPage:
        filters = ["c.media_type = %s"]
        params: list[Any] = [media_type]
        if search:
            filters.append("c.title ilike %s")
            params.append(f"%{search}%")
        if genre:
            filters.append("%s = any(c.genres)")
            params.append(genre)
        if unrated_only and self.app_user_id is not None:
            filters.append("r.movie_id is null")
        where = " and ".join(filters)
        join = (
            "left join current_ratings r on r.movie_id = c.movie_id and r.app_user_id = %s"
            if self.app_user_id is not None
            else "left join current_ratings r on false"
        )
        join_params = [self.app_user_id] if self.app_user_id is not None else []
        with self.db.connect() as connection:
            total = connection.execute(
                f"select count(*) as count from catalog_items c {join} where {where}",
                (*join_params, *params),
            ).fetchone()["count"]
            rows = connection.execute(
                f"""
                select c.*, r.rating as user_rating
                from catalog_items c
                {join}
                where {where}
                order by c.release_year desc nulls last, c.title
                limit %s offset %s
                """,
                (*join_params, *params, limit, offset),
            ).fetchall()
        return CatalogPage(
            items=[_catalog_item(row) for row in rows],
            total=int(total),
            limit=limit,
            offset=offset,
        )

    def detail(self, movie_id: int, region: str = "IN") -> CatalogItem | None:
        with self.db.connect() as connection:
            row = connection.execute(
                """
                select c.*, r.rating as user_rating
                from catalog_items c
                left join current_ratings r
                  on r.movie_id = c.movie_id and r.app_user_id = %s
                where c.movie_id = %s
                """,
                (self.app_user_id, movie_id),
            ).fetchone()
            if row is None:
                return None
            providers = connection.execute(
                """
                select * from watch_providers
                where movie_id = %s and region = %s
                order by display_priority nulls last, provider_name
                """,
                (movie_id, region.upper()),
            ).fetchall()
            actions = connection.execute(
                """
                select action_type, label, url, region
                from watch_actions
                where movie_id = %s and (region is null or region = %s)
                order by action_type
                """,
                (movie_id, region.upper()),
            ).fetchall()
        item = _catalog_item(row)
        return CatalogItem(
            **{
                **asdict(item),
                "watch_providers": [_provider(value) for value in providers],
                "watch_actions": [_action(value) for value in actions],
            }
        )

    def rows(self, region: str = "IN", limit: int = 14) -> list[CatalogRow]:
        definitions = [
            ("popular-starters", "Popular starters", "Familiar films to begin training taste."),
            ("recent", "Recent voltage", "Newer titles with enough signal."),
            ("classics", "Classic pressure", "Older high-signal films."),
        ]
        result = []
        with self.db.connect() as connection:
            for slug, title, subtitle in definitions:
                if slug == "recent":
                    where = "release_year >= 2005"
                    order = "release_year desc nulls last, title"
                elif slug == "classics":
                    where = "release_year < 1995"
                    order = "release_year desc nulls last, title"
                else:
                    where = "true"
                    order = "movie_id"
                rows = connection.execute(
                    f"""
                    select c.*, r.rating as user_rating
                    from catalog_items c
                    left join current_ratings r
                      on r.movie_id = c.movie_id and r.app_user_id = %s
                    where {where}
                    order by {order}
                    limit %s
                    """,
                    (self.app_user_id, limit),
                ).fetchall()
                result.append(CatalogRow(slug, title, subtitle, [_catalog_item(row) for row in rows]))
        return result


def _assert_user(requested: int, actual: int) -> None:
    if int(requested) != int(actual):
        raise RepositoryError("Authenticated user cannot access another user's ratings")


def _aware(value):
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _catalog_item(row: dict[str, Any]) -> CatalogItem:
    return CatalogItem(
        movie_id=int(row["movie_id"]),
        title=row["title"],
        genres=list(row.get("genres") or []),
        release_year=row.get("release_year"),
        media_type=row.get("media_type") or "movie",
        imdb_id=row.get("imdb_id"),
        tmdb_id=row.get("tmdb_id"),
        poster_url=row.get("poster_url"),
        backdrop_url=row.get("backdrop_url"),
        overview=row.get("overview"),
        runtime_minutes=row.get("runtime_minutes"),
        user_rating=float(row["user_rating"]) if row.get("user_rating") is not None else None,
    )


def _provider(row: dict[str, Any]) -> WatchProvider:
    return WatchProvider(
        provider_name=row["provider_name"],
        provider_type=row["provider_type"],
        region=row["region"],
        provider_id=row.get("provider_id"),
        provider_logo_url=row.get("provider_logo_url"),
        deep_link=row.get("deep_link"),
        display_priority=row.get("display_priority"),
    )


def _action(row: dict[str, Any]) -> WatchAction:
    return WatchAction(
        action_type=row["action_type"],
        label=row["label"],
        url=row["url"],
        region=row.get("region"),
    )


def external_actions(movie_id: int, imdb_id: str | None, tmdb_id: int | None, media_type: str) -> list[WatchAction]:
    actions = []
    if url := stremio_url(imdb_id, media_type):
        actions.append(WatchAction("stremio", "Open in Stremio", url))
    if url := imdb_url(imdb_id):
        actions.append(WatchAction("imdb", "IMDb", url))
    if url := tmdb_url(tmdb_id, media_type):
        actions.append(WatchAction("tmdb", "TMDb", url))
    return actions
