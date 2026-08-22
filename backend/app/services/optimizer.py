from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


class AllocationError(Exception):
    pass


@dataclass(frozen=True)
class OptimizerConfig:
    lambda_risk: float = 1.0
    lambda_cost: float = 0.05
    lambda_concentration: float = 0.25
    lambda_liquidation: float = 0.40
    max_weight: float = 0.55
    min_weight: float = 0.0
    transaction_cost: float = 0.001


@dataclass
class AllocationResult:
    weights: dict[str, float]
    expected_return: float
    risk: float
    concentration: float
    liquidation_risk: float
    transaction_cost: float
    objective: float
    diagnostics: dict[str, str]


def _project_constraints(n: int, min_w: float, max_w: float) -> None:
    if n * min_w > 1.0 + 1e-9:
        raise AllocationError("min_weight infeasible for this universe")
    if max_w * n < 1.0 - 1e-9 and max_w < 1.0:
        # still feasible if max_w >= 1/n
        if max_w + 1e-12 < 1.0 / n:
            raise AllocationError("max_weight infeasible for this universe")


def allocate(
    symbols: list[str],
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    liquidation_scores: np.ndarray,
    config: OptimizerConfig,
    previous_weights: np.ndarray | None = None,
) -> AllocationResult:
    n = len(symbols)
    if n == 0:
        raise AllocationError("no assets")
    if expected_returns.shape != (n,) or liquidation_scores.shape != (n,):
        raise AllocationError("vector length mismatch")
    if covariance.shape != (n, n):
        raise AllocationError("covariance shape mismatch")
    _project_constraints(n, config.min_weight, config.max_weight)

    prev = previous_weights if previous_weights is not None else np.full(n, 1.0 / n)
    x0 = np.clip(prev, config.min_weight, config.max_weight)
    x0 = x0 / x0.sum()

    def objective(w: np.ndarray) -> float:
        ret = float(w @ expected_returns)
        risk = float(np.sqrt(max(w @ covariance @ w, 0.0)))
        conc = float(w @ w)
        liq = float(w @ liquidation_scores)
        cost = float(np.abs(w - prev).sum() * config.transaction_cost)
        return -(
            ret
            - config.lambda_risk * risk
            - config.lambda_cost * cost
            - config.lambda_concentration * conc
            - config.lambda_liquidation * liq
        )

    bounds = [(config.min_weight, config.max_weight) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-9},
    )
    if not result.success:
        weights = _fallback_weights(expected_returns, config)
        diagnostics = {"solver": "fallback_greedy", "message": result.message}
    else:
        weights = np.clip(result.x, config.min_weight, config.max_weight)
        total = weights.sum()
        if total <= 0:
            raise AllocationError("solver returned zero weights")
        weights = weights / total
        diagnostics = {"solver": "slsqp", "message": result.message}

    w = weights
    ret = float(w @ expected_returns)
    risk = float(np.sqrt(max(w @ covariance @ w, 0.0)))
    conc = float(w @ w)
    liq = float(w @ liquidation_scores)
    cost = float(np.abs(w - prev).sum() * config.transaction_cost)
    obj = ret - config.lambda_risk * risk - config.lambda_cost * cost - config.lambda_concentration * conc - config.lambda_liquidation * liq
    mapping = {symbol: float(weight) for symbol, weight in zip(symbols, w, strict=True)}
    return AllocationResult(
        weights=mapping,
        expected_return=ret,
        risk=risk,
        concentration=conc,
        liquidation_risk=liq,
        transaction_cost=cost,
        objective=obj,
        diagnostics=diagnostics,
    )


def result_from_weights(
    symbols: list[str],
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    liquidation_scores: np.ndarray,
    config: OptimizerConfig,
    previous_weights: np.ndarray | None,
    diagnostics: dict[str, str],
) -> AllocationResult:
    prev = previous_weights if previous_weights is not None else np.full(len(symbols), 1.0 / len(symbols))
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        raise AllocationError("weights must be positive")
    w = w / w.sum()
    ret = float(w @ expected_returns)
    risk = float(np.sqrt(max(w @ covariance @ w, 0.0)))
    conc = float(w @ w)
    liq = float(w @ liquidation_scores)
    cost = float(np.abs(w - prev).sum() * config.transaction_cost)
    obj = ret - config.lambda_risk * risk - config.lambda_cost * cost - config.lambda_concentration * conc - config.lambda_liquidation * liq
    return AllocationResult(
        weights={symbol: float(weight) for symbol, weight in zip(symbols, w, strict=True)},
        expected_return=ret,
        risk=risk,
        concentration=conc,
        liquidation_risk=liq,
        transaction_cost=cost,
        objective=obj,
        diagnostics=diagnostics,
    )


def _fallback_weights(expected_returns: np.ndarray, config: OptimizerConfig) -> np.ndarray:
    n = expected_returns.size
    order = np.argsort(-expected_returns)
    weights = np.zeros(n)
    remaining = 1.0
    for idx in order:
        take = min(config.max_weight, remaining)
        if take < config.min_weight:
            break
        weights[idx] = take
        remaining -= take
        if remaining <= 1e-12:
            break
    if remaining > 1e-12:
        fill = np.where(weights == 0)[0]
        if fill.size:
            even = remaining / fill.size
            weights[fill] = even
        else:
            weights[order[0]] += remaining
    return weights
