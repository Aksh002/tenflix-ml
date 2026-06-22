from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import load_bundle, load_holdout
from .config import load_config
from .evaluation import evaluate, write_report
from .pipeline import prepare_data, train_run
from .recommender import HybridRecommender


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tenflix", description="TenFlix v3 ML pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-data", help="validate CSV inputs and cache Parquet")
    prepare.add_argument("--config", default="configs/v3.yaml")

    train = commands.add_parser("train", help="train and persist an evaluation or production model")
    train.add_argument("--config", default="configs/v3.yaml")
    train.add_argument("--mode", choices=("evaluation", "production"), default="evaluation")
    train.add_argument("--run-id")

    evaluation = commands.add_parser("evaluate", help="evaluate a saved run")
    evaluation.add_argument("--run", required=True)

    recommend = commands.add_parser("recommend", help="generate ordered recommendations")
    recommend.add_argument("--run", required=True)
    recommend.add_argument("--user-id", type=int)
    recommend.add_argument("--genres", nargs="*")
    recommend.add_argument("--top-k", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "prepare-data":
        print(json.dumps(prepare_data(load_config(arguments.config)), indent=2))
        return 0
    if arguments.command == "train":
        path = train_run(
            load_config(arguments.config), mode=arguments.mode, run_id=arguments.run_id
        )
        print(path)
        return 0
    if arguments.command == "evaluate":
        run = Path(arguments.run)
        report = evaluate(load_bundle(run), load_holdout(run))
        write_report(report, run / "evaluation.json")
        print(json.dumps(report, indent=2))
        return 0 if report["validated"] else 2
    if arguments.command == "recommend":
        recommender = HybridRecommender(load_bundle(arguments.run))
        results = recommender.recommend(
            user_id=arguments.user_id,
            preferred_genres=arguments.genres,
            top_k=arguments.top_k,
        )
        print(json.dumps([result.to_dict() for result in results], indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
