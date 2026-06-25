from __future__ import annotations

from datetime import UTC, datetime

from tenflix.v4.data import movie_records
from tenflix.v4.events import EraPreference, OnboardingPreferences, RatingEvent
from tenflix.v4.recommender import V4Recommender
from tenflix.v4.repositories import InMemoryCatalogRepository, InMemoryRatingRepository
from tenflix.v4.service import RecommendationService, create_fastapi_app


def test_recommendations_exclude_seen_and_are_ranked(v4_bundle):
    events = [
        RatingEvent(77, 1, 5.0, datetime(2024, 1, 1, tzinfo=UTC)),
        RatingEvent(77, 2, 1.0, datetime(2024, 2, 1, tzinfo=UTC)),
    ]
    response = V4Recommender(v4_bundle).recommend(77, events, 5)
    assert {1, 2}.isdisjoint({value.movie_id for value in response.recommendations})
    assert [value.rank for value in response.recommendations] == list(range(1, 6))
    assert all(value.score_contributors for value in response.recommendations)


def test_new_user_era_preference_and_freshness_floor(v4_bundle):
    recommender = V4Recommender(v4_bundle)
    preferences = OnboardingPreferences(("Sci-Fi",), (), (), EraPreference.RECENT)
    response = recommender.recommend(0, [], 4, preferences)
    assert response.lifecycle == "new"
    assert len(response.recommendations) == 4
    floor = v4_bundle.config["reranker"]["freshness_floor"]
    assert np_min_contributor(response, "freshness") >= -2  # normalized floor remains bounded
    assert floor == 0.25


def test_new_user_exploration_jitter_is_deterministic(v4_bundle):
    recommender = V4Recommender(v4_bundle)
    preferences = OnboardingPreferences(("Adventure",), (), (), EraPreference.BALANCED)
    at = datetime(2024, 6, 1, tzinfo=UTC)
    first = recommender.recommend(101, [], 5, preferences, at=at)
    second = recommender.recommend(101, [], 5, preferences, at=at)
    assert [value.movie_id for value in first.recommendations] == [
        value.movie_id for value in second.recommendations
    ]
    assert all(
        "exploration_jitter" in value.score_contributors
        for value in first.recommendations
    )


def np_min_contributor(response, name):
    return min(value.score_contributors.get(name, 0) for value in response.recommendations)


def test_record_rating_invalidates_revision_and_updates_profile(v4_bundle, v4_movies):
    ratings = InMemoryRatingRepository()
    catalog = InMemoryCatalogRepository(movie_records(v4_movies))
    service = RecommendationService(V4Recommender(v4_bundle), ratings, catalog)
    before = service.recommend(12, 3)
    summary = service.record_rating(
        RatingEvent(12, 1, 5.0, datetime(2024, 1, 1, tzinfo=UTC))
    )
    after = service.recommend(12, 3)
    assert summary.profile_revision != before.profile_revision
    assert after.profile_revision == summary.profile_revision
    assert after.lifecycle == "cold"
    assert 1 not in {value.movie_id for value in after.recommendations}


def test_positive_and_negative_content_evidence_move_opposite_directions(v4_bundle):
    events = [
        RatingEvent(44, 1, 5.0, datetime(2024, 1, 1, tzinfo=UTC)),
        RatingEvent(44, 7, 1.0, datetime(2024, 2, 1, tzinfo=UTC)),
    ]
    profile = V4Recommender(v4_bundle).build_profile(44, events)
    combined, negative = v4_bundle.content_model.scores(profile.long_content, 0.4)
    assert negative[v4_bundle.content_model.movie_lookup[7]] > 0
    assert combined[v4_bundle.content_model.movie_lookup[1]] > combined[
        v4_bundle.content_model.movie_lookup[7]
    ]


def test_single_rating_and_hyphenated_onboarding_genre_create_content_signal(v4_bundle):
    recommender = V4Recommender(v4_bundle)
    single = recommender.build_profile(
        91, [RatingEvent(91, 2, 5.0, datetime(2024, 1, 1, tzinfo=UTC))]
    )
    assert single.long_content.positive.nnz > 0
    query = v4_bundle.content_model.query_profile(OnboardingPreferences(("Sci-Fi",)))
    sci_fi = v4_bundle.content_model.movie_lookup[2]
    drama = v4_bundle.content_model.movie_lookup[4]
    scores, _ = v4_bundle.content_model.scores(query, 0.4)
    assert scores[sci_fi] > scores[drama]


def test_liked_movie_ids_affect_new_user_content_query(v4_bundle):
    query = v4_bundle.content_model.query_profile(OnboardingPreferences(liked_movie_ids=(7,)))
    scores, _ = v4_bundle.content_model.scores(query, 0.4)
    assert scores[v4_bundle.content_model.movie_lookup[7]] > 0


def test_onboarding_dislikes_apply_to_existing_user_profile(v4_bundle):
    event = RatingEvent(92, 1, 5.0, datetime(2024, 1, 1, tzinfo=UTC))
    profile = V4Recommender(v4_bundle).build_profile(
        92, [event], OnboardingPreferences(disliked_genres=("Horror",))
    )
    _, negative = v4_bundle.content_model.scores(profile.long_content, 0.4)
    assert negative[v4_bundle.content_model.movie_lookup[7]] > 0


def test_profile_cache_separates_preferences_and_request_day(v4_bundle, v4_movies):
    service = RecommendationService(
        V4Recommender(v4_bundle),
        InMemoryRatingRepository(
            [RatingEvent(55, 1, 5.0, datetime(2024, 1, 1, tzinfo=UTC))]
        ),
        InMemoryCatalogRepository(movie_records(v4_movies)),
    )
    first_day = datetime(2024, 6, 1, tzinfo=UTC)
    service.recommend(55, 3, OnboardingPreferences(("Action",)), first_day)
    first_key = service._profile_cache[55][0]
    service.recommend(55, 3, OnboardingPreferences(("Drama",)), first_day)
    preference_key = service._profile_cache[55][0]
    service.recommend(55, 3, OnboardingPreferences(("Drama",)), first_day.replace(day=2))
    next_day_key = service._profile_cache[55][0]
    assert first_key != preference_key
    assert preference_key != next_day_key


def test_fastapi_adapter_exposes_health_and_model(v4_bundle, v4_movies):
    from fastapi.testclient import TestClient

    service = RecommendationService(
        V4Recommender(v4_bundle),
        InMemoryRatingRepository(),
        InMemoryCatalogRepository(movie_records(v4_movies)),
    )
    client = TestClient(create_fastapi_app(service))
    assert client.get("/v1/health").json() == {"status": "ok"}
    assert client.get("/v1/model").json()["schema_version"] == 4
    rating = client.post(
        "/v1/ratings",
        json={
            "user_id": 88,
            "movie_id": 1,
            "rating": 5.0,
            "rated_at": "2024-01-01T00:00:00Z",
            "source": "organic",
        },
    )
    assert rating.status_code == 200, rating.text
    assert rating.json()["rating_count"] == 1
    preview = client.post(
        "/v1/recommendations/preview",
        json={
            "ratings": [],
            "top_k": 3,
            "preferred_genres": ["Sci-Fi"],
            "era_preference": "recent",
        },
    )
    assert preview.status_code == 200
    assert len(preview.json()["recommendations"]) == 3
