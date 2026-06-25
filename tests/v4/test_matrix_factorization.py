import numpy as np

from tenflix.v4.matrix_factorization import fit_biased_mf


def test_prediction_contains_bias_and_latent_terms(v4_ratings, v4_config):
    model = fit_biased_mf(v4_ratings, v4_config)
    user = model.user_lookup[1]
    item = model.item_lookup[1]
    expected = (
        model.global_mean
        + model.user_bias[user]
        + model.item_bias[item]
        + np.dot(model.user_factors[user], model.item_factors[item])
    )
    assert model.predict_known(1, np.asarray([1]))[0] == np.float32(expected)


def test_fold_in_reduces_known_rating_error(v4_ratings, v4_config):
    model = fit_biased_mf(v4_ratings, v4_config)
    frame = v4_ratings[v4_ratings["userId"] == 1]
    movie_ids = frame["movieId"].to_numpy()
    ratings = frame["rating"].to_numpy()
    baseline = np.full(len(ratings), model.global_mean)
    folded = model.fold_in(movie_ids, ratings, 0.1)
    predictions = model.predict_folded(folded, movie_ids)
    assert np.mean((ratings - predictions) ** 2) < np.mean((ratings - baseline) ** 2)

