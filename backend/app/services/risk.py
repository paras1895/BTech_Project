from __future__ import annotations

import math

import numpy as np

from app.adapters.lending import ReserveSnapshot


def returns_from_prices(prices: list[float]) -> np.ndarray:
    if len(prices) < 2:
        return np.array([], dtype=float)
    arr = np.asarray(prices, dtype=float)
    prev = arr[:-1]
    safe = np.where(prev == 0, np.nan, prev)
    return (arr[1:] - prev) / safe


def volatility(prices: list[float]) -> float:
    rets = returns_from_prices(prices)
    if rets.size < 2:
        return 0.0
    return float(np.nanstd(rets, ddof=1))


def max_drawdown(prices: list[float]) -> float:
    if not prices:
        return 0.0
    arr = np.asarray(prices, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / np.where(peak == 0, 1.0, peak)
    return float(abs(dd.min()))


def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    clean = returns[np.isfinite(returns)]
    if clean.size == 0:
        return 0.0
    return float(-np.quantile(clean, alpha))


def sharpe_like(returns: np.ndarray, risk_free_per_step: float = 0.0) -> float:
    clean = returns[np.isfinite(returns)]
    if clean.size < 2:
        return 0.0
    excess = clean - risk_free_per_step
    std = float(excess.std(ddof=1))
    if std < 1e-12:
        return 0.0
    return float(excess.mean() / std)


def sortino_like(returns: np.ndarray, mar: float = 0.0) -> float:
    clean = returns[np.isfinite(returns)]
    if clean.size < 2:
        return 0.0
    downside = clean[clean < mar] - mar
    if downside.size == 0:
        return 0.0
    denom = float(np.sqrt(np.mean(downside**2)))
    if denom < 1e-12:
        return 0.0
    return float((clean.mean() - mar) / denom)


def herfindahl(weights: np.ndarray) -> float:
    return float(np.sum(np.square(weights)))


def portfolio_volatility(weights: np.ndarray, covariance: np.ndarray) -> float:
    var = float(weights @ covariance @ weights)
    return math.sqrt(max(var, 0.0))


def net_yield(supply_apr: float, borrow_apr: float, supply_weight: float, borrow_weight: float) -> float:
    return supply_weight * supply_apr - borrow_weight * borrow_apr


def liquidation_proxy(health_factor: float | None) -> float:
    if health_factor is None:
        return 0.0
    return float(min(max((1.5 - health_factor) / 1.5, 0.0), 1.0))


def reserve_risk_vector(snapshots: list[ReserveSnapshot], price_histories: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for snap in snapshots:
        prices = price_histories.get(snap.symbol, [snap.oracle_price])
        vol = volatility(prices)
        dd = max_drawdown(prices)
        rets = returns_from_prices(prices)
        out[snap.symbol] = {
            "utilization": snap.utilization,
            "liquidity": snap.available_liquidity * snap.oracle_price,
            "supply_apr": snap.supply_apr,
            "borrow_apr": snap.borrow_apr,
            "net_yield": snap.supply_apr,
            "volatility": vol,
            "drawdown": dd,
            "var_5": value_at_risk(rets) if rets.size else 0.0,
            "liquidation_threshold": snap.liquidation_threshold,
            "ltv": snap.ltv,
        }
    return out


def correlation_matrix(price_histories: dict[str, list[float]], symbols: list[str]) -> np.ndarray:
    series = []
    min_len = min((len(price_histories.get(s, [])) for s in symbols), default=0)
    if min_len < 3:
        return np.eye(len(symbols))
    for symbol in symbols:
        prices = price_histories[symbol][-min_len:]
        rets = returns_from_prices(prices)
        series.append(rets)
    stacked = np.vstack(series)
    if stacked.shape[1] < 2:
        return np.eye(len(symbols))
    corr = np.corrcoef(stacked)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr
