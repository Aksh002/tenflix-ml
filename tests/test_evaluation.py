from tenflix.data import temporal_split
from tenflix.evaluation import evaluate
from tenflix.models import fit_model


def test_evaluation_reports_validity_without_claiming_success(ratings, movies, config):
    context, holdout = temporal_split(ratings, config, evaluation=True)
    bundle = fit_model(context, movies, config)
    report = evaluate(bundle, holdout)
    assert set(report["aggregate"]) == {"popularity", "static_cf", "recent_cf", "hybrid"}
    assert "acceptance_gates" in report
    assert set(report["segments"]) == {"drift", "lifecycle", "interaction_bands"}
    assert isinstance(report["validated"], bool)
    assert report["users_evaluated"] > 0
