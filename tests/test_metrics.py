from tenflix.evaluation import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_ranking_metrics_use_movie_ids():
    recommended = [4, 2, 8, 1]
    relevant = {2, 8, 9}
    assert precision_at_k(recommended, relevant, 4) == 0.5
    assert recall_at_k(recommended, relevant, 4) == 2 / 3
    assert hit_rate_at_k(recommended, relevant, 1) == 0.0
    assert hit_rate_at_k(recommended, relevant, 2) == 1.0
    assert 0 < average_precision_at_k(recommended, relevant, 4) < 1
    assert 0 < ndcg_at_k(recommended, relevant, 4) < 1

