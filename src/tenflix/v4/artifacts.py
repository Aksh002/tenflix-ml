from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from .recommender import V4Bundle


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def save_bundle(
    bundle: V4Bundle,
    test: pd.DataFrame,
    root: str | Path,
    run_id: str,
    mode: str,
    source_manifest: dict[str, Any] | None = None,
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_id
    temporary = root / f".{run_id}.tmp"
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Run already exists: {run_id}")
    temporary.mkdir()
    try:
        joblib.dump(bundle, temporary / "model.joblib", compress=0)
        np.save(temporary / "item_factors.npy", bundle.matrix_factorization.item_factors)
        np.save(temporary / "item_biases.npy", bundle.matrix_factorization.item_bias)
        sparse.save_npz(temporary / "content_features.npz", bundle.content_model.features)
        pd.DataFrame(
            [
                {
                    "movieId": movie.movie_id,
                    "title": movie.title,
                    "genres": "|".join(movie.genres) or "(no genres listed)",
                    "releaseYear": movie.release_year,
                }
                for movie in bundle.content_model.movies
            ]
        ).to_parquet(temporary / "catalog.parquet", index=False)
        test.to_parquet(temporary / "test.parquet", index=False)
        (temporary / "feature_statistics.json").write_text(
            json.dumps(bundle.feature_statistics, indent=2), encoding="utf-8"
        )
        (temporary / "reranker_weights.json").write_text(
            json.dumps(bundle.reranker_weights, indent=2), encoding="utf-8"
        )
        files = [
            "model.joblib",
            "item_factors.npy",
            "item_biases.npy",
            "content_features.npz",
            "catalog.parquet",
            "test.parquet",
            "feature_statistics.json",
            "reranker_weights.json",
        ]
        manifest = {
            "schema_version": 4,
            "feature_schema_version": 1,
            "repository_contract_version": 1,
            "model_version": bundle.model_version,
            "run_id": run_id,
            "mode": mode,
            "created_at": datetime.now(UTC).isoformat(),
            "training_cutoff": bundle.training_cutoff,
            "config": bundle.config,
            "source_manifest": source_manifest or {},
            "files": {name: _hash(temporary / name) for name in files},
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        try:
            os.replace(temporary, destination)
        except PermissionError:
            shutil.copytree(temporary, destination)
            shutil.rmtree(temporary, ignore_errors=True)
        (destination / "COMPLETE").write_text("ok\n", encoding="utf-8")
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_bundle(path: str | Path) -> V4Bundle:
    path = Path(path)
    if not (path / "COMPLETE").exists():
        raise ValueError("V4 artifact is incomplete")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 4:
        raise ValueError(
            f"Expected V4 artifact schema 4, received {manifest.get('schema_version')}; V3 artifacts are not migrated"
        )
    for name, expected in manifest["files"].items():
        if _hash(path / name) != expected:
            raise ValueError(f"Artifact integrity check failed: {name}")
    bundle = joblib.load(path / "model.joblib")
    if not isinstance(bundle, V4Bundle):
        raise ValueError("model.joblib does not contain a V4Bundle")
    return bundle


def load_test(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 4:
        raise ValueError("Only V4 evaluation artifacts contain a V4 test split")
    if _hash(path / "test.parquet") != manifest["files"]["test.parquet"]:
        raise ValueError("Artifact integrity check failed: test.parquet")
    return pd.read_parquet(path / "test.parquet")
