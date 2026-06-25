from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Recommendation:
    movie_id: int
    title: str
    genres: list[str]
    score: float
    rank: int
    strategy: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

