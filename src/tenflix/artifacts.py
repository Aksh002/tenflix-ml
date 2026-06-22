from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .models import ModelBundle


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def default_run_id(mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"v3-{mode}-{stamp}"


def save_run(
    bundle: ModelBundle,
    holdout: pd.DataFrame,
    artifact_root: str | Path,
    run_id: str,
    mode: str,
    source_manifest: dict[str, Any] | None = None,
) -> Path:
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_id
    temporary = root / f".{run_id}.tmp"
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Artifact run already exists: {run_id}")
    temporary.mkdir(parents=True)
    try:
        model_path = temporary / "model.joblib"
        joblib.dump(bundle, model_path, compress=0)
        holdout_path = temporary / "holdout.parquet"
        holdout.to_parquet(holdout_path, index=False, compression="zstd")
        manifest = {
            "schema_version": 3,
            "run_id": run_id,
            "mode": mode,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_sha256": _file_hash(model_path),
            "holdout_sha256": _file_hash(holdout_path),
            "holdout_rows": len(holdout),
            "drift_thresholds": list(bundle.drift_thresholds),
            "training_users": len(bundle.user_ids),
            "catalog_movies": len(bundle.catalog_movie_ids),
            "rated_training_movies": len(bundle.cf_movie_ids),
            "config": bundle.config,
            "source_manifest": source_manifest or {},
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": {
                    package: importlib.metadata.version(package)
                    for package in ("joblib", "numpy", "pandas", "pyarrow", "scikit-learn", "scipy")
                },
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        try:
            os.replace(temporary, destination)
            (destination / "COMPLETE").write_text("ok\n", encoding="utf-8")
        except PermissionError:
            shutil.copytree(temporary, destination)
            (destination / "COMPLETE").write_text("ok\n", encoding="utf-8")
            shutil.rmtree(temporary, ignore_errors=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_bundle(run_path: str | Path) -> ModelBundle:
    path = Path(run_path)
    if not (path / "COMPLETE").exists():
        raise ValueError("Artifact run is incomplete")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    model_path = path / "model.joblib"
    if _file_hash(model_path) != manifest["model_sha256"]:
        raise ValueError("Model artifact hash does not match its manifest")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, ModelBundle) or bundle.schema_version != 3:
        raise ValueError("Unsupported TenFlix artifact schema")
    return bundle


def load_holdout(run_path: str | Path) -> pd.DataFrame:
    path = Path(run_path)
    if not (path / "COMPLETE").exists():
        raise ValueError("Artifact run is incomplete")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    holdout_path = path / "holdout.parquet"
    if _file_hash(holdout_path) != manifest["holdout_sha256"]:
        raise ValueError("Holdout artifact hash does not match its manifest")
    return pd.read_parquet(holdout_path)
