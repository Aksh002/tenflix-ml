from __future__ import annotations

import hashlib
import threading
from collections import defaultdict
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import pandas as pd

from .events import Movie, RatingEvent, RatingSource


class RepositoryError(RuntimeError):
    pass


@runtime_checkable
class RatingRepository(Protocol):
    def get_user_ratings(self, user_id: int) -> list[RatingEvent]: ...

    def append_rating(self, event: RatingEvent) -> None: ...

    def get_revision(self, user_id: int) -> str: ...


@runtime_checkable
class CatalogRepository(Protocol):
    def get_movies(self, movie_ids: Sequence[int] | None = None) -> list[Movie]: ...

    def available_movie_ids(self, at=None) -> set[int]: ...


class InMemoryRatingRepository:
    def __init__(self, events: Sequence[RatingEvent] = ()):
        self._events: dict[int, list[RatingEvent]] = defaultdict(list)
        self._versions: dict[int, int] = defaultdict(int)
        self._lock = threading.RLock()
        for event in events:
            self.append_rating(event)

    def get_user_ratings(self, user_id: int) -> list[RatingEvent]:
        with self._lock:
            latest: dict[int, RatingEvent] = {}
            for event in self._events.get(user_id, []):
                previous = latest.get(event.movie_id)
                if previous is None or event.rated_at >= previous.rated_at:
                    latest[event.movie_id] = event
            return sorted(latest.values(), key=lambda event: (event.preference_time, event.movie_id))

    def append_rating(self, event: RatingEvent) -> None:
        with self._lock:
            self._events[event.user_id].append(event)
            self._versions[event.user_id] += 1

    def get_revision(self, user_id: int) -> str:
        with self._lock:
            value = f"{user_id}:{self._versions.get(user_id, 0)}".encode()
            return hashlib.sha256(value).hexdigest()[:16]

    def audit_history(self, user_id: int) -> list[RatingEvent]:
        with self._lock:
            return list(self._events.get(user_id, []))


class InMemoryCatalogRepository:
    def __init__(self, movies: Sequence[Movie]):
        self._movies = {movie.movie_id: movie for movie in movies}

    def get_movies(self, movie_ids: Sequence[int] | None = None) -> list[Movie]:
        if movie_ids is None:
            return list(self._movies.values())
        return [self._movies[value] for value in movie_ids if value in self._movies]

    def available_movie_ids(self, at=None) -> set[int]:
        year = getattr(at, "year", None)
        return {
            movie.movie_id
            for movie in self._movies.values()
            if movie.available and (year is None or movie.release_year is None or movie.release_year <= year)
        }


class ParquetRatingRepository:
    """Read-only MovieLens adapter plus in-memory live-event overlay."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._ratings = pd.read_parquet(self.path).set_index("userId", drop=False)
        self._overlay = InMemoryRatingRepository()

    def get_user_ratings(self, user_id: int) -> list[RatingEvent]:
        frame = self._user_frame(user_id)
        baseline = [
            RatingEvent(
                user_id=int(row.userId),
                movie_id=int(row.movieId),
                rating=float(row.rating),
                rated_at=pd.Timestamp(int(row.timestamp), unit="s", tz="UTC").to_pydatetime(),
                source=RatingSource.LEGACY,
            )
            for row in frame.itertuples(index=False)
        ]
        overlay = self._overlay.get_user_ratings(user_id)
        combined = InMemoryRatingRepository([*baseline, *overlay])
        return combined.get_user_ratings(user_id)

    def append_rating(self, event: RatingEvent) -> None:
        self._overlay.append_rating(event)

    def get_revision(self, user_id: int) -> str:
        count = len(self._user_frame(user_id))
        return f"file-{count}-{self._overlay.get_revision(user_id)}"

    def _user_frame(self, user_id: int) -> pd.DataFrame:
        try:
            value = self._ratings.loc[user_id]
        except KeyError:
            return self._ratings.iloc[0:0]
        if isinstance(value, pd.Series):
            return value.to_frame().T
        return value
