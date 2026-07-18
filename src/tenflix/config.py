from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "version": 3,
    "seed": 42,
    "paths": {
        "ratings": "ratings.csv",
        "movies": "movies.csv",
        "prepared_dir": "data/processed",
        "artifacts_dir": "artifacts",
    },
    "data": {"chunk_size": 1_000_000, "rating_min": 0.5, "rating_max": 5.0},
    "lifecycle": {"cold_max": 19, "sparse_max": 49},
    "temporal": {
        "old_fraction": 0.30,
        "recent_context_fraction": 0.40,
        "holdout_fraction": 0.30,
        "stable_quantile": 0.25,
        "volatile_quantile": 0.75,
    },
    "model": {
        "factors": 20,
        "min_relevant_rating": 4.0,
        "quality_quantile": 0.60,
        "stable_long_term_weight": 0.50,
        "moderate_long_term_weight": 0.30,
        "sparse_cf_weight_min": 0.25,
        "sparse_cf_weight_max": 0.60,
        "relevance_weight": 0.75,
        "quality_weight": 0.15,
        "popularity_weight": 0.10,
    },
    "content": {"ngram_min": 1, "ngram_max": 2, "min_df": 2, "max_df": 0.90},
    "evaluation": {
        "k_values": [10, 20],
        "max_users": 2000,
        "bootstrap_samples": 500,
        "confidence": 0.95,
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path = "configs/v3.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    override = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = _merge(DEFAULT_CONFIG, override or {})
    validate_config(config)
    config["_config_path"] = str(config_path.resolve())
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config["version"] != 3:
        raise ValueError("V3 configuration must declare version: 3")
    temporal = config["temporal"]
    total = sum(
        temporal[name]
        for name in ("old_fraction", "recent_context_fraction", "holdout_fraction")
    )
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Temporal old, recent-context, and holdout fractions must sum to 1")
    if not 0 < temporal["stable_quantile"] < temporal["volatile_quantile"] < 1:
        raise ValueError("Drift quantiles must satisfy 0 < stable < volatile < 1")
    if config["lifecycle"]["cold_max"] >= config["lifecycle"]["sparse_max"]:
        raise ValueError("cold_max must be lower than sparse_max")
    weights = config["model"]
    score_weight = sum(weights[name] for name in ("relevance_weight", "quality_weight", "popularity_weight"))
    if abs(score_weight - 1.0) > 1e-9:
        raise ValueError("Recommendation score weights must sum to 1")


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}
