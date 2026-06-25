from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from tenflix.v4.events import RatingEvent, RatingSource
from tenflix.v4.profiles import build_user_profile
from tenflix.v4.repositories import InMemoryRatingRepository

from .conftest import make_events


def test_rating_event_validation_and_time_precedence():
    with pytest.raises(ValueError):
        RatingEvent(1, 1, 5.5, datetime.now(UTC))
    watched = datetime(2024, 1, 1, tzinfo=UTC)
    event = RatingEvent(1, 1, 4.0, watched + timedelta(days=2), watched)
    assert event.preference_time == watched


def test_onboarding_events_do_not_create_temporal_confidence(v4_bundle, v4_config):
    events = make_events(source=RatingSource.ONBOARDING)
    profile = build_user_profile(
        99,
        events,
        v4_bundle.matrix_factorization,
        v4_bundle.content_model,
        v4_config,
        now=datetime(2023, 7, 1, tzinfo=UTC),
    )
    assert not profile.temporal_eligible
    assert profile.temporal_confidence == 0
    assert profile.recent_weight == 0


def test_continuous_temporal_weight_is_bounded(v4_bundle, v4_config):
    events = make_events(count=8)
    profile = build_user_profile(
        99,
        events,
        v4_bundle.matrix_factorization,
        v4_bundle.content_model,
        v4_config,
        now=datetime(2023, 8, 15, tzinfo=UTC),
    )
    assert profile.session_count >= 3
    assert 0 <= profile.recent_weight <= v4_config["temporal"]["maximum_recent_weight"]
    assert np.isfinite(profile.blended_factors).all()


def test_latest_duplicate_rating_wins_but_audit_is_retained():
    first = RatingEvent(4, 1, 2.0, datetime(2024, 1, 1, tzinfo=UTC))
    second = RatingEvent(4, 1, 5.0, datetime(2024, 2, 1, tzinfo=UTC))
    repository = InMemoryRatingRepository([first, second])
    assert repository.get_user_ratings(4)[0].rating == 5.0
    assert len(repository.audit_history(4)) == 2
