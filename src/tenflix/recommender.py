from __future__ import annotations

from typing import Iterable

import numpy as np

from .models import ModelBundle
from .types import Recommendation


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    result = np.zeros_like(values)
    if not finite.any():
        return result
    minimum = float(values[finite].min())
    maximum = float(values[finite].max())
    if maximum - minimum > 1e-12:
        result[finite] = (values[finite] - minimum) / (maximum - minimum)
    return result


def _unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-10)


class HybridRecommender:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self._unit_items = _unit_rows(bundle.item_vectors)

    def recommend(
        self,
        user_id: int | None = None,
        top_k: int = 10,
        preferred_genres: Iterable[str] | None = None,
    ) -> list[Recommendation]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        user_index = self.bundle.user_lookup.get(int(user_id)) if user_id is not None else None
        if user_index is None:
            relevance = self._genre_scores(preferred_genres or ())
            strategy = "new_content" if np.any(relevance) else "popularity_quality_fallback"
            reason = (
                "Matches your selected genres"
                if strategy == "new_content"
                else "Popular, highly rated catalog choice"
            )
            return self._format(self._rank(relevance, None, top_k), strategy, reason)

        lifecycle = str(self.bundle.lifecycle[user_index])
        if lifecycle == "cold":
            relevance = self._content_profile_scores(user_index)
            strategy = "cold_content"
            reason = "Matches genres from your positively rated movies"
        elif lifecycle == "sparse":
            content = self._content_profile_scores(user_index)
            collaborative = self._cf_scores(self.bundle.static_vectors[user_index])
            minimum = float(self.bundle.config["model"]["sparse_cf_weight_min"])
            maximum = float(self.bundle.config["model"]["sparse_cf_weight_max"])
            cold_max = int(self.bundle.config["lifecycle"]["cold_max"])
            sparse_max = int(self.bundle.config["lifecycle"]["sparse_max"])
            progress = (int(self.bundle.interaction_counts[user_index]) - cold_max - 1) / max(
                1, sparse_max - cold_max - 1
            )
            cf_weight = minimum + np.clip(progress, 0.0, 1.0) * (maximum - minimum)
            relevance = (1.0 - cf_weight) * content + cf_weight * collaborative
            strategy = "sparse_hybrid"
            reason = "Balances your movie genres with collaborative signals"
        else:
            relevance, strategy, reason = self._mature_scores(user_index)
        return self._format(self._rank(relevance, user_index, top_k), strategy, reason)

    def ranked_movie_ids(
        self,
        user_id: int,
        top_k: int,
        strategy: str = "hybrid",
    ) -> list[int]:
        user_index = self.bundle.user_lookup.get(int(user_id))
        if strategy == "popularity":
            relevance = np.zeros(len(self.bundle.catalog_movie_ids), dtype=np.float32)
        elif user_index is None:
            relevance = np.zeros(len(self.bundle.catalog_movie_ids), dtype=np.float32)
        elif strategy == "static_cf":
            relevance = self._cf_scores(self.bundle.static_vectors[user_index])
        elif strategy == "recent_cf":
            relevance = self._cf_scores(self.bundle.recent_vectors[user_index])
        elif strategy == "hybrid":
            lifecycle = str(self.bundle.lifecycle[user_index])
            if lifecycle == "cold":
                relevance = self._content_profile_scores(user_index)
            elif lifecycle == "sparse":
                relevance = 0.5 * self._content_profile_scores(user_index) + 0.5 * self._cf_scores(
                    self.bundle.static_vectors[user_index]
                )
            else:
                relevance = self._mature_scores(user_index)[0]
        else:
            raise ValueError(f"Unknown evaluation strategy: {strategy}")
        ranked = self._rank(relevance, user_index, top_k)
        return [int(self.bundle.catalog_movie_ids[index]) for index, _ in ranked]

    def _mature_scores(self, user_index: int) -> tuple[np.ndarray, str, str]:
        drift = str(self.bundle.drift_types[user_index])
        long_term = self.bundle.static_vectors[user_index]
        recent = self.bundle.recent_vectors[user_index]
        if drift == "stable":
            weight = float(self.bundle.config["model"]["stable_long_term_weight"])
            vector = weight * long_term + (1.0 - weight) * recent
            return self._cf_scores(vector), "mature_stable_blend", "Balances long-term and recent taste"
        if drift == "moderate":
            weight = float(self.bundle.config["model"]["moderate_long_term_weight"])
            vector = weight * long_term + (1.0 - weight) * recent
            return self._cf_scores(vector), "mature_moderate_recent", "Emphasizes your recent taste"
        if drift == "volatile":
            return self._cf_scores(recent), "mature_volatile_recent", "Reflects your latest preference shift"
        return self._cf_scores(long_term), "mature_unknown_long_term", "Uses all available history"

    def _genre_scores(self, genres: Iterable[str]) -> np.ndarray:
        query = " ".join(str(value).strip() for value in genres if str(value).strip())
        if not query:
            return np.zeros(len(self.bundle.catalog_movie_ids), dtype=np.float32)
        query_vector = self.bundle.vectorizer.transform([query])
        scores = query_vector @ self.bundle.content_features.T
        return _minmax(np.asarray(scores.toarray()).ravel())

    def _content_profile_scores(self, user_index: int) -> np.ndarray:
        profile = self.bundle.content_profiles.getrow(user_index)
        if profile.nnz == 0:
            return np.zeros(len(self.bundle.catalog_movie_ids), dtype=np.float32)
        scores = profile @ self.bundle.content_features.T
        return _minmax(np.asarray(scores.toarray()).ravel())

    def _cf_scores(self, user_vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(user_vector))
        catalog_scores = np.zeros(len(self.bundle.catalog_movie_ids), dtype=np.float32)
        if norm <= 1e-10:
            return catalog_scores
        scores = (user_vector / norm) @ self._unit_items.T
        catalog_scores[self.bundle.cf_catalog_indices] = _minmax(scores)
        return catalog_scores

    def _rank(
        self,
        relevance: np.ndarray,
        user_index: int | None,
        top_k: int,
    ) -> list[tuple[int, float]]:
        model = self.bundle.config["model"]
        scores = (
            float(model["relevance_weight"]) * _minmax(relevance)
            + float(model["quality_weight"]) * self.bundle.quality_scores
            + float(model["popularity_weight"]) * self.bundle.popularity_scores
        ).astype(np.float32)
        if user_index is not None:
            seen = self.bundle.seen_matrix.getrow(user_index).indices
            scores[seen] = -np.inf
        eligible = np.flatnonzero(np.isfinite(scores))
        count = min(int(top_k), len(eligible))
        if count == 0:
            return []
        local = np.argpartition(scores[eligible], -count)[-count:]
        selected = eligible[local]
        selected = selected[np.argsort(scores[selected], kind="stable")[::-1]]
        return [(int(index), float(scores[index])) for index in selected]

    def _format(
        self,
        ranked: list[tuple[int, float]],
        strategy: str,
        reason: str,
    ) -> list[Recommendation]:
        results: list[Recommendation] = []
        for rank, (index, score) in enumerate(ranked, start=1):
            movie = self.bundle.movies.iloc[index]
            genre_text = str(movie["genres"])
            genres = [] if genre_text == "(no genres listed)" else genre_text.split("|")
            results.append(
                Recommendation(
                    movie_id=int(movie["movieId"]),
                    title=str(movie["title"]),
                    genres=genres,
                    score=score,
                    rank=rank,
                    strategy=strategy,
                    reason=reason,
                )
            )
        return results
