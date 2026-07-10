from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import default_run_id, save_run
from .data import load_prepared, prepare_parquet, temporal_split, validate_loaded_data
from .models import fit_model


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    return prepare_parquet(config)


def train_run(
    config: dict[str, Any],
    mode: str = "evaluation",
    run_id: str | None = None,
) -> Path:
    if mode not in {"evaluation", "production"}:
        raise ValueError("mode must be evaluation or production")
    ratings, movies = load_prepared(config)
    validate_loaded_data(ratings, movies, config)
    context, holdout = temporal_split(ratings, config, evaluation=mode == "evaluation")
    bundle = fit_model(context, movies, config)
    prepared_manifest_path = Path(config["paths"]["prepared_dir"]) / "manifest.json"
    source_manifest = (
        json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
        if prepared_manifest_path.exists()
        else {}
    )
    return save_run(
        bundle=bundle,
        holdout=holdout,
        artifact_root=config["paths"]["artifacts_dir"],
        run_id=run_id or default_run_id(mode),
        mode=mode,
        source_manifest=source_manifest,
    )

