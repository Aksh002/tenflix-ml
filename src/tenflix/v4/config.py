from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _read_default() -> dict[str, Any]:
    return yaml.safe_load(Path(__file__).with_name("default.yaml").read_text(encoding="utf-8"))


DEFAULT_CONFIG = _read_default()


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path = "configs/v4.yaml") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    override = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = _merge(DEFAULT_CONFIG, override or {})
    validate_config(config)
    config["_config_path"] = str(path.resolve())
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config["version"] != 4:
        raise ValueError("V4 configuration must declare version: 4")
    split = config["split"]
    if not 0 < split["train_quantile"] < split["validation_quantile"] < 1:
        raise ValueError("Split quantiles must satisfy 0 < train < validation < 1")
    if config["lifecycle"]["cold_max"] >= config["lifecycle"]["sparse_max"]:
        raise ValueError("cold_max must be lower than sparse_max")
    if config["lifecycle"]["cold_max"] < 0:
        raise ValueError("Lifecycle boundaries must be non-negative")
    if not 0 <= config["temporal"]["maximum_recent_weight"] <= 0.85:
        raise ValueError("maximum_recent_weight must be within [0, 0.85]")
    if not 0 <= config["reranker"]["freshness_floor"] <= 1:
        raise ValueError("freshness_floor must be within [0, 1]")
    if config["model"]["factors"] < 2 or config["model"]["epochs"] < 1:
        raise ValueError("Model factors must be >= 2 and epochs must be positive")
    if config["model"]["learning_rate"] <= 0 or config["model"]["fold_in_regularization"] <= 0:
        raise ValueError("Learning rate and fold-in regularization must be positive")
    if config["temporal"]["recent_half_life_days"] <= 0:
        raise ValueError("Recent half-life must be positive")
    if config["reranker"]["freshness_half_life_years"] <= 0:
        raise ValueError("Freshness half-life must be positive")
    if any(int(value) < 0 for value in config["candidates"].values()):
        raise ValueError("Candidate limits must be non-negative")
    if not any(
        int(config["candidates"][name]) > 0
        for name in ("content", "quality_popularity", "exploration")
    ):
        raise ValueError("At least one non-collaborative candidate source must be enabled")
    required_lifecycles = {
        "new",
        "cold",
        "sparse",
        "mature_temporal_ineligible",
        "mature_temporal_eligible",
    }
    if set(config["reranker"]["weights"]) != required_lifecycles:
        raise ValueError("Reranker weights must define every V4 lifecycle")
    if (
        float(config["data"]["rating_min"]) != 0.5
        or float(config["data"]["rating_max"]) != 5.0
    ):
        raise ValueError("V4 rating range is fixed at 0.5-5.0")
    maximum_freshness = float(config["reranker"]["maximum_freshness_weight"])
    if any(
        abs(float(weights.get("freshness", 0))) > maximum_freshness
        for weights in config["reranker"]["weights"].values()
    ):
        raise ValueError("Lifecycle freshness weights exceed maximum_freshness_weight")
    margin = float(config["evaluation"].get("new_start_noninferiority_margin", 0.0))
    if margin > 0:
        raise ValueError("new_start_noninferiority_margin must be zero or negative")


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}
