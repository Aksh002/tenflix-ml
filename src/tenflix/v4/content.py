from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .events import Movie, OnboardingPreferences, RatingEvent


class MetadataFeatureProvider(Protocol):
    def documents(self, movies: Sequence[Movie]) -> Sequence[str]: ...


@dataclass
class ContentProfile:
    positive: sparse.csr_matrix
    negative: sparse.csr_matrix
    genre_distribution: np.ndarray
    preferred_year: float | None


@dataclass
class ContentModel:
    movies: list[Movie]
    vectorizer: TfidfVectorizer
    features: sparse.csr_matrix
    genre_names: tuple[str, ...]
    movie_genres: np.ndarray
    mean_prior_strength: float = 5.0

    def __post_init__(self) -> None:
        self.movie_lookup = {movie.movie_id: index for index, movie in enumerate(self.movies)}

    def profile(
        self,
        events: Sequence[RatingEvent],
        preferences: OnboardingPreferences | None = None,
        event_weights: np.ndarray | None = None,
    ) -> ContentProfile:
        # Shrink a sparse user's mean toward neutral.  Without this, one rating
        # is always equal to its own mean and contributes no content signal.
        neutral = 3.5
        prior = float(getattr(self, "mean_prior_strength", 5.0))
        mean = (
            (sum(float(event.rating) for event in events) + prior * neutral)
            / (len(events) + prior)
            if events
            else neutral
        )
        positive_rows = []
        positive_weights = []
        negative_rows = []
        negative_weights = []
        genre_distribution = np.zeros(len(self.genre_names), dtype=np.float32)
        years = []
        year_weights = []
        for position, event in enumerate(events):
            index = self.movie_lookup.get(event.movie_id)
            if index is None:
                continue
            weight = 1.0 if event_weights is None else float(event_weights[position])
            residual = float(event.rating) - mean
            if residual > 0:
                positive_rows.append(index)
                positive_weights.append(residual * weight)
                genre_distribution += self.movie_genres[index] * residual * weight
                if self.movies[index].release_year is not None:
                    years.append(self.movies[index].release_year)
                    year_weights.append(residual * weight)
            elif residual < 0:
                negative_rows.append(index)
                negative_weights.append(-residual * weight)
        if preferences:
            for genre in preferences.preferred_genres:
                if genre in self.genre_names:
                    genre_distribution[self.genre_names.index(genre)] += 1.0
            for movie_id in preferences.liked_movie_ids:
                index = self.movie_lookup.get(movie_id)
                if index is not None:
                    positive_rows.append(index)
                    positive_weights.append(1.0)
                    genre_distribution += self.movie_genres[index]
        positive = self._weighted_profile(positive_rows, positive_weights)
        negative = self._weighted_profile(negative_rows, negative_weights)
        if preferences and preferences.preferred_genres:
            preferred_query = self.vectorizer.transform(
                [" ".join(_genre_token(value) for value in preferences.preferred_genres)]
            )
            positive = normalize(positive + preferred_query).tocsr()
        if preferences and preferences.disliked_genres:
            disliked_query = self.vectorizer.transform(
                [" ".join(_genre_token(value) for value in preferences.disliked_genres)]
            )
            negative = normalize(negative + disliked_query).tocsr()
        total = float(genre_distribution.sum())
        if total > 0:
            genre_distribution /= total
        preferred_year = (
            float(np.average(years, weights=year_weights)) if years and sum(year_weights) > 0 else None
        )
        return ContentProfile(positive, negative, genre_distribution, preferred_year)

    def query_profile(self, preferences: OnboardingPreferences) -> ContentProfile:
        tokens = [_genre_token(value) for value in preferences.preferred_genres]
        positive = self.vectorizer.transform([" ".join(tokens)]).tocsr()
        liked_rows = [
            self.movie_lookup[movie_id]
            for movie_id in preferences.liked_movie_ids
            if movie_id in self.movie_lookup
        ]
        if liked_rows:
            positive = positive + self._weighted_profile(liked_rows, [1.0] * len(liked_rows))
        positive = normalize(positive).tocsr()
        negative = normalize(
            self.vectorizer.transform(
                [" ".join(_genre_token(value) for value in preferences.disliked_genres)]
            )
        ).tocsr()
        genre_distribution = np.zeros(len(self.genre_names), dtype=np.float32)
        for genre in preferences.preferred_genres:
            if genre in self.genre_names:
                genre_distribution[self.genre_names.index(genre)] = 1.0
        liked_years = []
        for index in liked_rows:
            genre_distribution += self.movie_genres[index]
            if self.movies[index].release_year is not None:
                liked_years.append(self.movies[index].release_year)
        if genre_distribution.sum() > 0:
            genre_distribution /= genre_distribution.sum()
        preferred_year = float(np.mean(liked_years)) if liked_years else None
        return ContentProfile(positive, negative, genre_distribution, preferred_year)

    def scores(self, profile: ContentProfile, negative_penalty: float) -> tuple[np.ndarray, np.ndarray]:
        positive = np.asarray((profile.positive @ self.features.T).toarray()).ravel()
        negative = np.asarray((profile.negative @ self.features.T).toarray()).ravel()
        return positive - float(negative_penalty) * negative, negative

    def _weighted_profile(self, rows: list[int], weights: list[float]) -> sparse.csr_matrix:
        if not rows:
            return sparse.csr_matrix((1, self.features.shape[1]), dtype=np.float32)
        values = np.asarray(weights, dtype=np.float32)
        profile = sparse.csr_matrix(values.reshape(1, -1)) @ self.features[rows]
        return normalize(profile, norm="l2").tocsr()


def fit_content_model(
    movies: Sequence[Movie],
    min_df: int = 2,
    max_df: float = 0.95,
    providers: Sequence[MetadataFeatureProvider] = (),
    mean_prior_strength: float = 5.0,
) -> ContentModel:
    genre_names = tuple(sorted({genre for movie in movies for genre in movie.genres}))
    documents = []
    provider_documents = [provider.documents(movies) for provider in providers]
    for index, movie in enumerate(movies):
        tokens = [_genre_token(genre) for genre in movie.genres]
        if movie.release_year is not None:
            tokens.extend([f"year_{movie.release_year}", f"decade_{movie.release_year // 10 * 10}"])
        for values in provider_documents:
            tokens.append(str(values[index]))
        documents.append(" ".join(tokens))
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w_]+\b", min_df=min_df, max_df=max_df)
    features = vectorizer.fit_transform(documents).astype(np.float32).tocsr()
    movie_genres = np.zeros((len(movies), len(genre_names)), dtype=np.float32)
    for row, movie in enumerate(movies):
        for genre in movie.genres:
            movie_genres[row, genre_names.index(genre)] = 1.0
    return ContentModel(
        list(movies), vectorizer, features, genre_names, movie_genres, mean_prior_strength
    )


def _genre_token(value: str) -> str:
    return value.replace("-", "_")


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.sum() <= 0 or right.sum() <= 0:
        return 0.0
    left /= left.sum()
    right /= right.sum()
    middle = 0.5 * (left + right)
    left_mask = left > 0
    right_mask = right > 0
    left_term = np.sum(
        left[left_mask] * np.log2(left[left_mask] / np.clip(middle[left_mask], 1e-12, None))
    )
    right_term = np.sum(
        right[right_mask] * np.log2(right[right_mask] / np.clip(middle[right_mask], 1e-12, None))
    )
    return float(np.sqrt(max(0.0, 0.5 * (left_term + right_term))))
