from __future__ import annotations

import csv
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .data import movie_records
from .runtime_env import load_env_file
from .web_repositories import PostgresConnectionFactory, external_actions


@dataclass(frozen=True, slots=True)
class EnrichmentResult:
    catalog_upserts: int
    tmdb_updates: int
    provider_updates: int
    action_updates: int
    failures: int


class TMDbClient:
    def __init__(self, api_token: str | None = None, image_base_url: str | None = None):
        load_env_file()
        self.api_token = api_token or os.getenv("TMDB_API_TOKEN")
        self.image_base_url = image_base_url or os.getenv(
            "TMDB_IMAGE_BASE_URL", "https://image.tmdb.org/t/p/w500"
        )
        if not self.api_token:
            raise RuntimeError("TMDB_API_TOKEN is required for catalog enrichment")

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Install TenFlix with the 'web' extra to use TMDb") from error
        url = f"https://api.themoviedb.org/3/{path.lstrip('/')}"
        with httpx.Client(timeout=20) as client:
            response = client.get(
                url,
                params=params or {},
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "accept": "application/json",
                },
            )
            response.raise_for_status()
            return response.json()

    def image_url(self, path: str | None) -> str | None:
        return f"{self.image_base_url}{path}" if path else None


def sync_catalog(
    db: PostgresConnectionFactory,
    movies_path: str | Path,
    links_path: str | Path | None = None,
    *,
    tmdb: TMDbClient | None = None,
    region: str = "IN",
    limit: int | None = None,
    sleep_seconds: float = 0.05,
) -> EnrichmentResult:
    movies = pd.read_csv(movies_path)
    records = movie_records(movies)
    links = _read_links(links_path)
    catalog_upserts = tmdb_updates = provider_updates = action_updates = failures = 0
    with db.connect() as connection:
        for record in records[:limit] if limit else records:
            link = links.get(record.movie_id, {})
            imdb_id = _imdb(link.get("imdbId"))
            tmdb_id = _int_or_none(link.get("tmdbId"))
            connection.execute(
                """
                insert into catalog_items
                  (movie_id, media_type, title, normalized_title, genres, release_year, imdb_id, tmdb_id)
                values (%s, 'movie', %s, %s, %s, %s, %s, %s)
                on conflict (movie_id) do update
                set title = excluded.title,
                    normalized_title = excluded.normalized_title,
                    genres = excluded.genres,
                    release_year = excluded.release_year,
                    imdb_id = coalesce(catalog_items.imdb_id, excluded.imdb_id),
                    tmdb_id = coalesce(catalog_items.tmdb_id, excluded.tmdb_id)
                """,
                (
                    record.movie_id,
                    record.title,
                    _normalize(record.title),
                    list(record.genres),
                    record.release_year,
                    imdb_id,
                    tmdb_id,
                ),
            )
            catalog_upserts += 1
        connection.commit()
    if tmdb is None:
        return EnrichmentResult(catalog_upserts, 0, 0, 0, 0)
    with db.connect() as connection:
        rows = connection.execute(
            """
            select movie_id, title, release_year, imdb_id, tmdb_id, media_type
            from catalog_items
            where enrichment_status in ('pending', 'failed')
            order by movie_id
            limit %s
            """,
            (limit or 1000000,),
        ).fetchall()
        for row in rows:
            try:
                resolved_tmdb_id = row["tmdb_id"] or _search_tmdb_id(
                    tmdb, row["title"], row["release_year"]
                )
                if not resolved_tmdb_id:
                    raise RuntimeError("No TMDb match")
                details = tmdb.get(
                    f"movie/{resolved_tmdb_id}",
                    {"append_to_response": "external_ids,watch/providers"},
                )
                imdb_id = details.get("imdb_id") or row["imdb_id"]
                connection.execute(
                    """
                    update catalog_items
                    set tmdb_id = %s,
                        imdb_id = %s,
                        poster_url = %s,
                        backdrop_url = %s,
                        overview = %s,
                        runtime_minutes = %s,
                        enrichment_status = 'ready',
                        enrichment_confidence = 1.0,
                        enrichment_error = null
                    where movie_id = %s
                    """,
                    (
                        resolved_tmdb_id,
                        imdb_id,
                        tmdb.image_url(details.get("poster_path")),
                        tmdb.image_url(details.get("backdrop_path")),
                        details.get("overview"),
                        details.get("runtime"),
                        row["movie_id"],
                    ),
                )
                tmdb_updates += 1
                provider_updates += _upsert_providers(
                    connection,
                    tmdb,
                    int(row["movie_id"]),
                    details.get("watch/providers", {}).get("results", {}).get(region.upper(), {}),
                    region.upper(),
                )
                action_updates += _upsert_actions(
                    connection,
                    int(row["movie_id"]),
                    imdb_id,
                    resolved_tmdb_id,
                    row["media_type"],
                )
                connection.commit()
                time.sleep(sleep_seconds)
            except Exception as error:  # pragma: no cover - network/API dependent
                failures += 1
                connection.execute(
                    """
                    update catalog_items
                    set enrichment_status = 'failed', enrichment_error = %s
                    where movie_id = %s
                    """,
                    (str(error)[:500], row["movie_id"]),
                )
                connection.commit()
    return EnrichmentResult(catalog_upserts, tmdb_updates, provider_updates, action_updates, failures)


def _read_links(path: str | Path | None) -> dict[int, dict[str, str]]:
    if not path or not Path(path).exists():
        return {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {int(row["movieId"]): row for row in csv.DictReader(handle)}


def _normalize(title: str) -> str:
    title = re.sub(r"\(\d{4}\)\s*$", "", title)
    return re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()


def _imdb(value: str | None) -> str | None:
    if not value or pd.isna(value):
        return None
    digits = str(value).strip()
    return digits if digits.startswith("tt") else f"tt{int(float(digits)):07d}"


def _int_or_none(value) -> int | None:
    if value is None or pd.isna(value) or value == "":
        return None
    return int(float(value))


def _search_tmdb_id(tmdb: TMDbClient, title: str, year: int | None) -> int | None:
    response = tmdb.get(
        "search/movie",
        {"query": re.sub(r"\(\d{4}\)\s*$", "", title), "year": year} if year else {"query": title},
    )
    results = response.get("results") or []
    return int(results[0]["id"]) if results else None


def _upsert_providers(connection, tmdb: TMDbClient, movie_id: int, payload: dict, region: str) -> int:
    count = 0
    for provider_type in ("flatrate", "rent", "buy"):
        for value in payload.get(provider_type, []) or []:
            connection.execute(
                """
                insert into watch_providers
                  (movie_id, region, provider_id, provider_name, provider_logo_url,
                   provider_type, display_priority)
                values (%s, %s, %s, %s, %s, %s, %s)
                on conflict (movie_id, region, provider_name, provider_type) do update
                set provider_id = excluded.provider_id,
                    provider_logo_url = excluded.provider_logo_url,
                    display_priority = excluded.display_priority,
                    updated_at = now()
                """,
                (
                    movie_id,
                    region,
                    value.get("provider_id"),
                    value.get("provider_name"),
                    tmdb.image_url(value.get("logo_path")),
                    provider_type,
                    value.get("display_priority"),
                ),
            )
            count += 1
    return count


def _upsert_actions(connection, movie_id: int, imdb_id: str | None, tmdb_id: int | None, media_type: str) -> int:
    count = 0
    for action in external_actions(movie_id, imdb_id, tmdb_id, media_type):
        connection.execute(
            """
            insert into watch_actions (movie_id, action_type, label, url, region)
            values (%s, %s, %s, %s, %s)
            on conflict (movie_id, action_type, coalesce(region, 'GLOBAL')) do update
            set label = excluded.label, url = excluded.url
            """,
            (movie_id, action.action_type, action.label, action.url, action.region),
        )
        count += 1
    return count
