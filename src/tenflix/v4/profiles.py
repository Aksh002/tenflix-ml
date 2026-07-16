from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Sequence

import numpy as np

from .content import ContentModel, ContentProfile, jensen_shannon
from .data import lifecycle_stage
from .events import OnboardingPreferences, RatingEvent, RatingSource
from .matrix_factorization import BiasedMFModel, FoldedUser


TEMPORAL_SOURCES = {RatingSource.ORGANIC, RatingSource.RECOMMENDATION, RatingSource.LEGACY}


@dataclass
class UserProfile:
    user_id: int
    lifecycle: str
    events: list[RatingEvent]
    long_term: FoldedUser
    recent: FoldedUser
    blended_factors: np.ndarray
    blended_bias: float
    long_content: ContentProfile
    recent_content: ContentProfile
    temporal_eligible: bool
    temporal_confidence: float
    recent_weight: float
    latent_drift: float
    genre_drift: float
    rating_scale_change: float
    session_count: int
    lifespan_days: float
    recent_count: int
    days_since_latest: float | None
    preferred_year: float | None


def build_user_profile(
    user_id: int,
    events: Sequence[RatingEvent],
    model: BiasedMFModel,
    content: ContentModel,
    config: dict[str, Any],
    preferences: OnboardingPreferences | None = None,
    now: datetime | None = None,
) -> UserProfile:
    now = now or datetime.now(UTC)
    deduplicated = _latest_events(events)
    movie_ids = np.asarray([event.movie_id for event in deduplicated], dtype=np.int32)
    ratings = np.asarray([event.rating for event in deduplicated], dtype=np.float32)
    regularization = float(config["model"]["fold_in_regularization"])
    long_term = model.fold_in(movie_ids, ratings, regularization)
    long_content = content.profile(deduplicated, preferences)

    eligible_events = [event for event in deduplicated if event.source in TEMPORAL_SOURCES]
    eligible_events.sort(key=lambda event: event.preference_time)
    session_count = _session_count(eligible_events, float(config["temporal"]["session_gap_hours"]))
    lifespan_days = (
        (eligible_events[-1].preference_time - eligible_events[0].preference_time).total_seconds() / 86400
        if len(eligible_events) >= 2
        else 0.0
    )
    window_start = now - timedelta(days=float(config["temporal"]["recent_window_days"]))
    recent_events = [event for event in eligible_events if event.preference_time >= window_start]
    historical_events = [event for event in eligible_events if event.preference_time < window_start]
    minimum = int(config["temporal"]["minimum_window_ratings"])
    recent_weights = np.asarray(
        [
            np.exp(
                -max(0.0, (now - event.preference_time).total_seconds() / 86400)
                / float(config["temporal"]["recent_half_life_days"])
            )
            for event in recent_events
        ],
        dtype=np.float32,
    )
    recent = model.fold_in(
        np.asarray([event.movie_id for event in recent_events], dtype=np.int32),
        np.asarray([event.rating for event in recent_events], dtype=np.float32),
        regularization,
        recent_weights,
    )
    recent_content = content.profile(recent_events, event_weights=recent_weights)
    temporal_eligible = (
        session_count >= int(config["temporal"]["minimum_sessions"])
        and lifespan_days >= float(config["temporal"]["minimum_lifespan_days"])
        and len(historical_events) >= minimum
        and len(recent_events) >= minimum
        and np.linalg.norm(long_term.factors) > 1e-10
        and np.linalg.norm(recent.factors) > 1e-10
    )
    latent_drift = _cosine_distance(long_term.factors, recent.factors) if temporal_eligible else 0.0
    historical_content = content.profile(historical_events)
    genre_drift = (
        jensen_shannon(historical_content.genre_distribution, recent_content.genre_distribution)
        if temporal_eligible
        else 0.0
    )
    rating_scale_change = (
        abs(
            float(np.mean([event.rating for event in historical_events]))
            - float(np.mean([event.rating for event in recent_events]))
        )
        / 4.5
        if historical_events and recent_events
        else 0.0
    )
    confidence = _temporal_confidence(
        temporal_eligible,
        session_count,
        lifespan_days,
        len(recent_events),
        config,
    )
    activity = (
        np.exp(
            -max(0.0, (now - eligible_events[-1].preference_time).total_seconds() / 86400)
            / float(config["temporal"]["recent_half_life_days"])
        )
        if eligible_events
        else 0.0
    )
    days_since_latest = (
        max(0.0, (now - eligible_events[-1].preference_time).total_seconds() / 86400)
        if eligible_events
        else None
    )
    temporal = config["temporal"]
    logit = (
        float(temporal["sigmoid_intercept"])
        + float(temporal["sigmoid_latent_weight"]) * latent_drift
        + float(temporal["sigmoid_genre_weight"]) * genre_drift
        + float(temporal["sigmoid_activity_weight"]) * activity
    )
    recent_weight = confidence * (1.0 / (1.0 + np.exp(-logit)))
    recent_weight = float(np.clip(recent_weight, 0, temporal["maximum_recent_weight"]))
    blended_factors = (
        (1.0 - recent_weight) * long_term.factors + recent_weight * recent.factors
    ).astype(np.float32)
    blended_bias = (1.0 - recent_weight) * long_term.bias + recent_weight * recent.bias
    return UserProfile(
        user_id=user_id,
        lifecycle=lifecycle_stage(len(deduplicated), config),
        events=deduplicated,
        long_term=long_term,
        recent=recent,
        blended_factors=blended_factors,
        blended_bias=float(blended_bias),
        long_content=long_content,
        recent_content=recent_content,
        temporal_eligible=temporal_eligible,
        temporal_confidence=confidence,
        recent_weight=recent_weight,
        latent_drift=latent_drift,
        genre_drift=genre_drift,
        rating_scale_change=rating_scale_change,
        session_count=session_count,
        lifespan_days=lifespan_days,
        recent_count=len(recent_events),
        days_since_latest=days_since_latest,
        preferred_year=long_content.preferred_year,
    )


def _latest_events(events: Sequence[RatingEvent]) -> list[RatingEvent]:
    latest: dict[int, RatingEvent] = {}
    for event in events:
        previous = latest.get(event.movie_id)
        if previous is None or event.rated_at >= previous.rated_at:
            latest[event.movie_id] = event
    return sorted(latest.values(), key=lambda event: (event.preference_time, event.movie_id))


def _session_count(events: Sequence[RatingEvent], gap_hours: float) -> int:
    if not events:
        return 0
    sessions = 1
    gap = timedelta(hours=gap_hours)
    for previous, current in zip(events, events[1:]):
        if current.preference_time - previous.preference_time > gap:
            sessions += 1
    return sessions


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-10:
        return 0.0
    return float(1.0 - np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _temporal_confidence(eligible, sessions, lifespan, recent_count, config) -> float:
    if not eligible:
        return 0.0
    temporal = config["temporal"]
    session_score = min(1.0, sessions / (2 * float(temporal["minimum_sessions"])))
    lifespan_score = min(1.0, lifespan / (2 * float(temporal["minimum_lifespan_days"])))
    sample_score = min(1.0, recent_count / (2 * float(temporal["minimum_window_ratings"])))
    return float((session_score * lifespan_score * sample_score) ** (1 / 3))
