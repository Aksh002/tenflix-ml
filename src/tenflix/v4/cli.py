from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .artifacts import load_bundle, load_test
from .config import load_config
from .data import dataframe_events, load_data
from .evaluation import evaluate_run, write_report
from .events import EraPreference, OnboardingPreferences, RatingEvent, RatingSource
from .pipeline import prepare_data, train_run, tune
from .recommender import V4Recommender
from .repositories import InMemoryCatalogRepository, ParquetRatingRepository
from .runtime_env import load_env_file
from .service import RecommendationService, create_fastapi_app


REQUIRED_PROMOTION_GATES = {
    "accuracy_vs_popularity",
    "accuracy_vs_static_mf",
    "temporal_ndcg",
    "cold_start_ndcg",
    "new_start_ndcg",
    "coverage",
    "diversity",
    "latency",
    "integrity",
    "determinism",
}


class PromotionRejected(ValueError):
    """Expected refusal when an evaluation run has not passed every gate."""


def build_parser() -> argparse.ArgumentParser:
    load_env_file()
    parser = argparse.ArgumentParser(prog="tenflix-v4", description="TenFlix V4 pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("--config", default="configs/v4.yaml")
    tuning = commands.add_parser("tune")
    tuning.add_argument("--config", default="configs/v4.yaml")
    train = commands.add_parser("train")
    train.add_argument("--config", default="configs/v4.yaml")
    train.add_argument("--mode", choices=("evaluation",), default="evaluation")
    train.add_argument("--run-id")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--run", required=True)
    promote = commands.add_parser("promote")
    promote.add_argument("--run", required=True)
    recommend = commands.add_parser("recommend")
    recommend.add_argument("--run", required=True)
    recommend.add_argument("--user-id", type=int, required=True)
    recommend.add_argument("--top-k", type=int, default=10)
    preview = commands.add_parser("preview")
    preview.add_argument("--run", required=True)
    preview.add_argument("--ratings", required=True)
    preview.add_argument("--top-k", type=int, default=10)
    serve = commands.add_parser("serve")
    serve.add_argument("--run", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve_web = commands.add_parser("serve-web")
    serve_web.add_argument("--run", default=os.getenv("TENFLIX_V4_RUN", "artifacts/movielens-v4-tuned-eval-v4-production"))
    serve_web.add_argument("--host", default=os.getenv("TENFLIX_API_HOST", "127.0.0.1"))
    serve_web.add_argument("--port", type=int, default=int(os.getenv("TENFLIX_API_PORT", "8000")))
    enrich = commands.add_parser("enrich-catalog")
    enrich.add_argument("--movies", default="movies.csv")
    enrich.add_argument("--links", default="links.csv")
    enrich.add_argument("--region", default=os.getenv("TENFLIX_DEFAULT_PROVIDER_REGION", "IN"))
    enrich.add_argument("--limit", type=int)
    enrich.add_argument("--with-tmdb", action="store_true")
    migrate_web_db = commands.add_parser("migrate-web-db")
    migrate_web_db.add_argument("--migration", default="db/migrations")
    commands.add_parser("check-web-db")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    args = build_parser().parse_args(argv)
    if args.command == "prepare-data":
        print(json.dumps(prepare_data(load_config(args.config)), indent=2))
        return 0
    if args.command == "tune":
        print(json.dumps(tune(load_config(args.config)), indent=2))
        return 0
    if args.command == "train":
        print(train_run(load_config(args.config), args.mode, args.run_id))
        return 0
    if args.command == "evaluate":
        run = Path(args.run)
        report = evaluate_run(load_bundle(run), load_test(run))
        write_report(report, run / "evaluation.json")
        print(json.dumps(report, indent=2))
        return 0 if report["validated"] else 2
    if args.command == "promote":
        try:
            print(_promote(Path(args.run)))
            return 0
        except PromotionRejected as error:
            print(str(error), file=sys.stderr)
            return 2
    if args.command in {"recommend", "preview"}:
        bundle = load_bundle(args.run)
        recommender = V4Recommender(bundle)
        if args.command == "recommend":
            ratings, _ = load_data(bundle.config)
            frame = ratings.loc[
                (ratings["userId"] == args.user_id)
                & (ratings["timestamp"] <= bundle.training_cutoff)
            ]
            response = recommender.recommend(
                args.user_id,
                dataframe_events(frame),
                args.top_k,
                at=_cutoff_datetime(bundle.training_cutoff),
            )
        else:
            payload = json.loads(Path(args.ratings).read_text(encoding="utf-8"))
            rating_values = payload.get("ratings", []) if isinstance(payload, dict) else payload
            events = [_event_from_dict(value) for value in rating_values]
            preferences_value = payload.get("preferences", {}) if isinstance(payload, dict) else {}
            preferences = OnboardingPreferences(
                tuple(preferences_value.get("preferred_genres", ())),
                tuple(preferences_value.get("disliked_genres", ())),
                tuple(preferences_value.get("liked_movie_ids", ())),
                EraPreference(preferences_value.get("era_preference", "balanced")),
            )
            user_id = events[0].user_id if events else 0
            response = recommender.recommend(user_id, events, args.top_k, preferences)
        print(json.dumps(asdict(response), indent=2, default=str))
        return 0
    if args.command == "serve":
        import uvicorn

        run = Path(args.run)
        _assert_promoted(run)
        bundle = load_bundle(run)
        prepared = Path(bundle.config["paths"]["prepared_dir"])
        ratings = ParquetRatingRepository(prepared / "ratings.parquet")
        catalog = InMemoryCatalogRepository(bundle.content_model.movies)
        service = RecommendationService(V4Recommender(bundle), ratings, catalog)
        uvicorn.run(create_fastapi_app(service), host=args.host, port=args.port)
        return 0
    if args.command == "serve-web":
        import uvicorn

        from .auth import SupabaseAuthenticator
        from .web_repositories import PostgresConnectionFactory, SupabaseUserRepository
        from .web_service import create_product_fastapi_app

        run = Path(args.run)
        _assert_promoted(run)
        bundle = load_bundle(run)
        db = PostgresConnectionFactory()
        app = create_product_fastapi_app(
            V4Recommender(bundle),
            SupabaseUserRepository(db),
            SupabaseAuthenticator(),
        )
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    if args.command == "enrich-catalog":
        from .enrichment import TMDbClient, sync_catalog
        from .web_repositories import PostgresConnectionFactory

        tmdb = TMDbClient() if args.with_tmdb else None
        result = sync_catalog(
            PostgresConnectionFactory(),
            args.movies,
            args.links,
            tmdb=tmdb,
            region=args.region,
            limit=args.limit,
        )
        print(json.dumps(result.__dict__, indent=2))
        return 0
    if args.command == "migrate-web-db":
        from .web_repositories import PostgresConnectionFactory

        db = PostgresConnectionFactory()
        applied = db.apply_sql_path(args.migration)
        status = db.schema_status()
        print(json.dumps({"migration": args.migration, "applied": applied, "tables": status}, indent=2))
        return 0 if all(status.values()) else 2
    if args.command == "check-web-db":
        from .web_repositories import PostgresConnectionFactory

        status = PostgresConnectionFactory().schema_status()
        print(json.dumps({"tables": status}, indent=2))
        return 0 if all(status.values()) else 2
    return 1


def _promote(run: Path) -> Path:
    report_path = run / "evaluation.json"
    if not report_path.exists():
        raise ValueError("Evaluation report is missing")
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    gates = report.get("gates", {})
    if (
        not report.get("validated")
        or not REQUIRED_PROMOTION_GATES.issubset(gates)
        or not _all_gates_pass(gates)
    ):
        raise PromotionRejected(_promotion_failure_message(run, gates))
    bundle = load_bundle(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("mode") != "evaluation":
        raise ValueError("Only an evaluation artifact can be promoted")
    if report.get("schema_version") != 4 or report.get("model_version") != bundle.model_version:
        raise ValueError("Evaluation report does not belong to this V4 artifact")
    config = bundle.config
    production = train_run(
        config,
        "production",
        f"{run.name}-production",
        apply_tuning=False,
    )
    manifest_path = production / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["promotion_report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    manifest["promoted_from"] = run.name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (production / "PROMOTED").write_text("ok\n", encoding="utf-8")
    return production


def _assert_promoted(run: Path) -> None:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("mode") != "production"
        or not (run / "PROMOTED").exists()
        or not manifest.get("promotion_report_sha256")
    ):
        raise ValueError("serve requires an artifact created by promote")


def _all_gates_pass(gates) -> bool:
    if not isinstance(gates, dict) or not gates:
        return False
    for value in gates.values():
        if not isinstance(value, dict):
            return False
        if "passed" in value:
            if not bool(value["passed"]):
                return False
        elif not _all_gates_pass(value):
            return False
    return True


def _promotion_failure_message(run: Path, gates: dict) -> str:
    failures = []
    missing = sorted(REQUIRED_PROMOTION_GATES - set(gates))
    failures.extend(f"{name}: missing from evaluation report" for name in missing)
    for name in sorted(REQUIRED_PROMOTION_GATES & set(gates)):
        value = gates[name]
        if _all_gates_pass({name: value}):
            continue
        if name == "coverage":
            failures.append(
                f"coverage: {float(value.get('value', 0)):.4f} "
                f"(required >= {float(value.get('minimum', 0)):.4f})"
            )
        elif name == "new_start_ndcg":
            failures.append(
                f"new_start_ndcg: lower bound {float(value.get('lower', 0)):.4f} "
                f"(allowed margin >= {float(value.get('noninferiority_margin', 0)):.4f})"
            )
        elif name == "latency":
            failures.append(
                "latency: recommendation p95 "
                f"{float(value.get('recommendation_p95_ms', 0)):.1f} ms; "
                "fold-in + recommendation p95 "
                f"{float(value.get('fold_in_plus_recommendation_p95_ms', 0)):.1f} ms"
            )
        else:
            failures.append(f"{name}: failed")
    detail = "\n".join(f"  - {failure}" for failure in failures) or "  - validated flag is false"
    return (
        f"Promotion rejected for {run}: the evaluation run is not validated.\n"
        f"Failed gates:\n{detail}\n"
        f"Inspect {run / 'evaluation.json'}. Do not edit the report to bypass promotion."
    )


def _event_from_dict(value):
    return RatingEvent(
        int(value["user_id"]),
        int(value["movie_id"]),
        float(value["rating"]),
        datetime.fromisoformat(value["rated_at"]),
        datetime.fromisoformat(value["watched_at"]) if value.get("watched_at") else None,
        RatingSource(value.get("source", "organic")),
    )


def _cutoff_datetime(timestamp):
    from datetime import UTC

    return datetime.fromtimestamp(timestamp, tz=UTC)


if __name__ == "__main__":
    raise SystemExit(main())
