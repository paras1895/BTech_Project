from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.adapters.lending import InterestRateModel, ReserveSnapshot


FEATURE_NAMES = [
    "utilization",
    "supply_apr",
    "borrow_apr",
    "liquidity_ratio",
    "log_price",
    "ret_1",
    "vol_6",
    "vol_24",
    "util_delta_6",
    "hour_sin",
    "hour_cos",
]


@dataclass
class ModelMetrics:
    mae: float
    rmse: float
    r2: float
    n_samples: int


@dataclass
class TrainedModel:
    name: str
    version: str
    model_type: str
    estimator: Ridge
    feature_names: list[str]
    primary_target: str
    data_label: str
    horizon_steps: int
    train_metrics: ModelMetrics
    validation_metrics: ModelMetrics
    test_metrics: ModelMetrics
    train_rows: int
    feature_importance: dict[str, float]
    notes: str
    git_commit: str | None = None
    hyperparameters: dict[str, float] = field(default_factory=dict)


def _kink_series(n: int, seed: int, irm: InterestRateModel, reserve_factor: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    utilization = np.zeros(n)
    price = np.zeros(n)
    utilization[0] = 0.45
    price[0] = 1.0
    for i in range(1, n):
        utilization[i] = np.clip(utilization[i - 1] + rng.normal(0, 0.02), 0.05, 0.95)
        price[i] = max(price[i - 1] * (1.0 + rng.normal(0, 0.01)), 0.1)
    supply = np.array([irm.supply_apr(u, reserve_factor) for u in utilization])
    borrow = np.array([irm.borrow_apr(u) for u in utilization])
    return {"utilization": utilization, "price": price, "supply_apr": supply, "borrow_apr": borrow}


def _window_std(values: np.ndarray, idx: int, window: int) -> float:
    start = max(0, idx - window + 1)
    chunk = values[start : idx + 1]
    if chunk.size < 2:
        return 0.0
    return float(chunk.std(ddof=1))


def build_feature_row(
    utilization: np.ndarray,
    supply_apr: np.ndarray,
    borrow_apr: np.ndarray,
    price: np.ndarray,
    idx: int,
    windows: tuple[int, int] = (6, 24),
) -> np.ndarray:
    w1, w2 = windows
    ret_1 = 0.0 if idx == 0 else (price[idx] - price[idx - 1]) / price[idx - 1]
    util_delta = 0.0 if idx < w1 else utilization[idx] - utilization[idx - w1]
    hour = idx % 24
    return np.array(
        [
            utilization[idx],
            supply_apr[idx],
            borrow_apr[idx],
            1.0 - utilization[idx],
            float(np.log(price[idx])),
            ret_1,
            _window_std(price, idx, w1),
            _window_std(price, idx, w2),
            util_delta,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
        ],
        dtype=float,
    )


def features_from_snapshot(
    snap: ReserveSnapshot,
    prices: list[float],
    utilization_history: list[float],
    hour: int,
) -> np.ndarray:
    price_arr = np.asarray(prices if prices else [snap.oracle_price], dtype=float)
    util_arr = np.asarray(utilization_history if utilization_history else [snap.utilization], dtype=float)
    n = price_arr.size
    supply = np.full(n, snap.supply_apr)
    borrow = np.full(n, snap.borrow_apr)
    idx = n - 1
    row = build_feature_row(util_arr[-n:], supply, borrow, price_arr, idx)
    row[0] = snap.utilization
    row[1] = snap.supply_apr
    row[2] = snap.borrow_apr
    row[3] = 1.0 - snap.utilization
    row[9] = np.sin(2 * np.pi * (hour % 24) / 24)
    row[10] = np.cos(2 * np.pi * (hour % 24) / 24)
    return row


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
    return ModelMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        n_samples=int(y_true.shape[0]),
    )


def train_ridge_on_synthetic(model_cfg: dict, irm: InterestRateModel, reserve_factor: float = 0.1, seed: int = 42) -> TrainedModel:
    n = int(model_cfg["synthetic_rows"])
    horizon = int(model_cfg["horizon_steps"])
    series = _kink_series(n, seed, irm, reserve_factor)
    xs = []
    ys = []
    for idx in range(24, n - horizon):
        xs.append(
            build_feature_row(
                series["utilization"],
                series["supply_apr"],
                series["borrow_apr"],
                series["price"],
                idx,
            )
        )
        ys.append(series["supply_apr"][idx + horizon])
    x = np.vstack(xs)
    y = np.asarray(ys, dtype=float)
    n_obs = x.shape[0]
    train_end = int(n_obs * float(model_cfg["train_fraction"]))
    val_end = train_end + int(n_obs * float(model_cfg["validation_fraction"]))
    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x[val_end:], y[val_end:]
    estimator = Ridge(alpha=float(model_cfg["alpha"]))
    estimator.fit(x_train, y_train)
    importance = {
        name: float(abs(coef))
        for name, coef in zip(FEATURE_NAMES, estimator.coef_, strict=True)
    }
    return TrainedModel(
        name=str(model_cfg["name"]),
        version=str(model_cfg["version"]),
        model_type=str(model_cfg["model_type"]),
        estimator=estimator,
        feature_names=list(FEATURE_NAMES),
        primary_target=str(model_cfg["primary_target"]),
        data_label=str(model_cfg["data_label"]),
        horizon_steps=horizon,
        train_metrics=_metrics(y_train, estimator.predict(x_train)),
        validation_metrics=_metrics(y_val, estimator.predict(x_val)),
        test_metrics=_metrics(y_test, estimator.predict(x_test)),
        train_rows=int(x_train.shape[0]),
        feature_importance=importance,
        notes=str(model_cfg.get("notes", "")).strip(),
        hyperparameters={"alpha": float(model_cfg["alpha"])},
    )


def predict_supply_apr(model: TrainedModel, features: np.ndarray) -> float:
    pred = float(model.estimator.predict(features.reshape(1, -1))[0])
    return max(pred, 0.0)
