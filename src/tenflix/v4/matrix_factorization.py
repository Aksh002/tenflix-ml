from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only in minimal installations
    def njit(*args, **kwargs):
        def decorate(function):
            return function

        return decorate


@njit(cache=True)
def _sgd_epoch(
    users,
    items,
    ratings,
    order,
    global_mean,
    user_bias,
    item_bias,
    user_factors,
    item_factors,
    learning_rate,
    factor_regularization,
    bias_regularization,
):
    squared_error = 0.0
    for position in order:
        user = users[position]
        item = items[position]
        prediction = global_mean + user_bias[user] + item_bias[item]
        prediction += np.dot(user_factors[user], item_factors[item])
        error = ratings[position] - prediction
        squared_error += error * error
        user_bias[user] += learning_rate * (error - bias_regularization * user_bias[user])
        item_bias[item] += learning_rate * (error - bias_regularization * item_bias[item])
        old_user = user_factors[user].copy()
        user_factors[user] += learning_rate * (
            error * item_factors[item] - factor_regularization * user_factors[user]
        )
        item_factors[item] += learning_rate * (
            error * old_user - factor_regularization * item_factors[item]
        )
    return squared_error / len(order)


@dataclass
class TrainingDiagnostics:
    train_rmse: list[float] = field(default_factory=list)
    validation_rmse: list[float] = field(default_factory=list)
    stopped_epoch: int = 0


@dataclass
class FoldedUser:
    bias: float
    factors: np.ndarray
    known_ratings: int


@dataclass
class BiasedMFModel:
    global_mean: float
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_bias: np.ndarray
    item_bias: np.ndarray
    user_factors: np.ndarray
    item_factors: np.ndarray
    diagnostics: TrainingDiagnostics

    def __post_init__(self) -> None:
        self.user_lookup = {int(value): index for index, value in enumerate(self.user_ids)}
        self.item_lookup = {int(value): index for index, value in enumerate(self.item_ids)}

    def predict_known(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        user_index = self.user_lookup.get(int(user_id))
        if user_index is None:
            return np.full(len(movie_ids), self.global_mean, dtype=np.float32)
        indices = np.asarray([self.item_lookup.get(int(value), -1) for value in movie_ids])
        result = np.full(len(indices), self.global_mean + self.user_bias[user_index], dtype=np.float32)
        known = indices >= 0
        result[known] += self.item_bias[indices[known]]
        result[known] += self.item_factors[indices[known]] @ self.user_factors[user_index]
        return result

    def predict_folded(self, user: FoldedUser, movie_ids: np.ndarray | None = None) -> np.ndarray:
        if movie_ids is None:
            indices = np.arange(len(self.item_ids))
        else:
            indices = np.asarray([self.item_lookup.get(int(value), -1) for value in movie_ids])
        result = np.full(len(indices), self.global_mean + user.bias, dtype=np.float32)
        known = indices >= 0
        result[known] += self.item_bias[indices[known]]
        result[known] += self.item_factors[indices[known]] @ user.factors
        return result

    def fold_in(
        self,
        movie_ids: np.ndarray,
        ratings: np.ndarray,
        regularization: float,
        weights: np.ndarray | None = None,
    ) -> FoldedUser:
        mapped = np.asarray([self.item_lookup.get(int(value), -1) for value in movie_ids])
        known = mapped >= 0
        if not known.any():
            return FoldedUser(0.0, np.zeros(self.item_factors.shape[1], dtype=np.float32), 0)
        item_indices = mapped[known]
        targets = ratings[known].astype(np.float64) - self.global_mean - self.item_bias[item_indices]
        design = np.column_stack(
            [np.ones(len(item_indices), dtype=np.float64), self.item_factors[item_indices]]
        )
        sample_weights = (
            np.ones(len(item_indices), dtype=np.float64)
            if weights is None
            else np.asarray(weights[known], dtype=np.float64)
        )
        root = np.sqrt(np.clip(sample_weights, 1e-8, None))
        weighted_design = design * root[:, None]
        weighted_targets = targets * root
        penalty = np.eye(design.shape[1], dtype=np.float64) * float(regularization)
        penalty[0, 0] *= 0.25
        solution = np.linalg.solve(
            weighted_design.T @ weighted_design + penalty,
            weighted_design.T @ weighted_targets,
        )
        return FoldedUser(float(solution[0]), solution[1:].astype(np.float32), int(known.sum()))


def fit_biased_mf(
    train: pd.DataFrame,
    config: dict[str, Any],
    validation: pd.DataFrame | None = None,
) -> BiasedMFModel:
    if train.empty:
        raise ValueError("Cannot train matrix factorization on an empty frame")
    settings = config["model"]
    user_ids = np.sort(train["userId"].unique()).astype(np.int32)
    item_ids = np.sort(train["movieId"].unique()).astype(np.int32)
    user_lookup = {int(value): index for index, value in enumerate(user_ids)}
    item_lookup = {int(value): index for index, value in enumerate(item_ids)}
    users = train["userId"].map(user_lookup).to_numpy(dtype=np.int32)
    items = train["movieId"].map(item_lookup).to_numpy(dtype=np.int32)
    ratings = train["rating"].to_numpy(dtype=np.float32)
    factors = min(int(settings["factors"]), max(2, min(len(user_ids), len(item_ids)) - 1))
    rng = np.random.default_rng(int(config["seed"]))
    user_factors = rng.normal(0, 0.05, (len(user_ids), factors)).astype(np.float32)
    item_factors = rng.normal(0, 0.05, (len(item_ids), factors)).astype(np.float32)
    user_bias = np.zeros(len(user_ids), dtype=np.float32)
    item_bias = np.zeros(len(item_ids), dtype=np.float32)
    global_mean = float(ratings.mean())
    diagnostics = TrainingDiagnostics()
    best_state = None
    best_validation = np.inf
    stale = 0
    for epoch in range(int(settings["epochs"])):
        order = rng.permutation(len(ratings)).astype(np.int64)
        mse = _sgd_epoch(
            users,
            items,
            ratings,
            order,
            global_mean,
            user_bias,
            item_bias,
            user_factors,
            item_factors,
            float(settings["learning_rate"]),
            float(settings["factor_regularization"]),
            float(settings["bias_regularization"]),
        )
        diagnostics.train_rmse.append(float(np.sqrt(mse)))
        validation_rmse = _validation_rmse(
            validation, user_lookup, item_lookup, global_mean, user_bias, item_bias, user_factors, item_factors
        )
        diagnostics.validation_rmse.append(validation_rmse)
        if validation_rmse < best_validation - 1e-5:
            best_validation = validation_rmse
            best_state = (
                user_bias.copy(),
                item_bias.copy(),
                user_factors.copy(),
                item_factors.copy(),
            )
            stale = 0
        else:
            stale += 1
            if validation is not None and stale >= int(settings["early_stopping_patience"]):
                diagnostics.stopped_epoch = epoch + 1
                break
    if best_state is not None:
        user_bias, item_bias, user_factors, item_factors = best_state
    diagnostics.stopped_epoch = diagnostics.stopped_epoch or len(diagnostics.train_rmse)
    return BiasedMFModel(
        global_mean,
        user_ids,
        item_ids,
        user_bias,
        item_bias,
        user_factors,
        item_factors,
        diagnostics,
    )


def _validation_rmse(
    validation,
    user_lookup,
    item_lookup,
    global_mean,
    user_bias,
    item_bias,
    user_factors,
    item_factors,
) -> float:
    if validation is None or validation.empty:
        return float("inf")
    users = validation["userId"].map(user_lookup).fillna(-1).to_numpy(dtype=np.int32)
    items = validation["movieId"].map(item_lookup).fillna(-1).to_numpy(dtype=np.int32)
    ratings = validation["rating"].to_numpy(dtype=np.float32)
    known = (users >= 0) & (items >= 0)
    if not known.any():
        return float("inf")
    prediction = global_mean + user_bias[users[known]] + item_bias[items[known]]
    prediction += np.sum(user_factors[users[known]] * item_factors[items[known]], axis=1)
    return float(np.sqrt(np.mean((ratings[known] - prediction) ** 2)))
