from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.adapters.lending import ReserveSnapshot
from app.services.optimizer import AllocationResult, OptimizerConfig, allocate, result_from_weights
from app.services.risk import correlation_matrix, volatility


class Strategy(ABC):
    name: str

    @abstractmethod
    def propose(
        self,
        snapshots: list[ReserveSnapshot],
        price_histories: dict[str, list[float]],
        config: OptimizerConfig,
        predicted_supply_apr: dict[str, float] | None = None,
        previous_weights: dict[str, float] | None = None,
    ) -> AllocationResult:
        raise NotImplementedError


def _symbols(snapshots: list[ReserveSnapshot]) -> list[str]:
    return [item.symbol for item in snapshots]


def _cov(symbols: list[str], snapshots: list[ReserveSnapshot], histories: dict[str, list[float]]) -> np.ndarray:
    vols = np.array(
        [max(volatility(histories.get(symbol, [snap.oracle_price])), 1e-4) for symbol, snap in zip(symbols, snapshots, strict=True)],
        dtype=float,
    )
    corr = correlation_matrix(histories, symbols)
    return np.outer(vols, vols) * corr


def _prev(symbols: list[str], previous: dict[str, float] | None) -> np.ndarray | None:
    if previous is None:
        return None
    return np.array([previous.get(symbol, 0.0) for symbol in symbols], dtype=float)


def _liq(snapshots: list[ReserveSnapshot]) -> np.ndarray:
    return np.array([snap.utilization * (1.0 - snap.available_liquidity / max(snap.total_supplied, 1e-9)) for snap in snapshots], dtype=float)


class EqualWeightStrategy(Strategy):
    name = "equal_weight"

    def propose(self, snapshots, price_histories, config, predicted_supply_apr=None, previous_weights=None):
        n = len(snapshots)
        mu = np.array([snap.supply_apr for snap in snapshots], dtype=float)
        symbols = _symbols(snapshots)
        weights = np.full(n, 1.0 / n)
        return result_from_weights(
            symbols,
            weights,
            mu,
            _cov(symbols, snapshots, price_histories),
            _liq(snapshots),
            config,
            _prev(symbols, previous_weights),
            {"solver": "rule", "message": "equal weights"},
        )


class HighestYieldStrategy(Strategy):
    name = "highest_yield"

    def propose(self, snapshots, price_histories, config, predicted_supply_apr=None, previous_weights=None):
        symbols = _symbols(snapshots)
        mu = np.array([snap.supply_apr for snap in snapshots], dtype=float)
        n = len(symbols)
        weights = np.zeros(n)
        order = np.argsort(-mu)
        remaining = 1.0
        for idx in order:
            take = min(config.max_weight, remaining)
            weights[idx] = take
            remaining -= take
            if remaining <= 1e-12:
                break
        if remaining > 1e-12:
            weights[order[-1]] += remaining
        return result_from_weights(
            symbols,
            weights,
            mu,
            _cov(symbols, snapshots, price_histories),
            _liq(snapshots),
            config,
            _prev(symbols, previous_weights),
            {"solver": "rule", "message": "highest current supply APR under max_weight"},
        )


class RiskAdjustedYieldStrategy(Strategy):
    name = "risk_adjusted"

    def propose(self, snapshots, price_histories, config, predicted_supply_apr=None, previous_weights=None):
        symbols = _symbols(snapshots)
        scores = []
        for snap in snapshots:
            vol = max(volatility(price_histories.get(snap.symbol, [snap.oracle_price])), 1e-4)
            scores.append(snap.supply_apr / vol)
        mu = np.array(scores, dtype=float)
        return allocate(symbols, mu, _cov(symbols, snapshots, price_histories), _liq(snapshots), config, _prev(symbols, previous_weights))


class VolatilityAwareStrategy(Strategy):
    name = "volatility_aware"

    def propose(self, snapshots, price_histories, config, predicted_supply_apr=None, previous_weights=None):
        symbols = _symbols(snapshots)
        mu = []
        for snap in snapshots:
            vol = volatility(price_histories.get(snap.symbol, [snap.oracle_price]))
            mu.append(snap.supply_apr - 2.0 * vol)
        return allocate(symbols, np.array(mu, dtype=float), _cov(symbols, snapshots, price_histories), _liq(snapshots), config, _prev(symbols, previous_weights))


class BuyAndHoldStrategy(Strategy):
    name = "buy_and_hold"

    def propose(self, snapshots, price_histories, config, predicted_supply_apr=None, previous_weights=None):
        symbols = _symbols(snapshots)
        mu = np.array([snap.supply_apr for snap in snapshots], dtype=float)
        if previous_weights:
            weights = np.array([previous_weights.get(symbol, 0.0) for symbol in symbols], dtype=float)
        else:
            weights = np.full(len(symbols), 1.0 / len(symbols))
        return result_from_weights(
            symbols,
            weights,
            mu,
            _cov(symbols, snapshots, price_histories),
            _liq(snapshots),
            config,
            weights,
            {"solver": "rule", "message": "hold previous or initial equal mix"},
        )


class MLStrategy(Strategy):
    name = "ml_optimizer"

    def propose(self, snapshots, price_histories, config, predicted_supply_apr=None, previous_weights=None):
        if not predicted_supply_apr:
            raise ValueError("ML strategy requires predictions from the model service")
        symbols = _symbols(snapshots)
        mu = np.array([predicted_supply_apr[symbol] for symbol in symbols], dtype=float)
        return allocate(symbols, mu, _cov(symbols, snapshots, price_histories), _liq(snapshots), config, _prev(symbols, previous_weights))


STRATEGIES: dict[str, Strategy] = {
    cls.name: cls()
    for cls in (
        EqualWeightStrategy,
        HighestYieldStrategy,
        RiskAdjustedYieldStrategy,
        VolatilityAwareStrategy,
        BuyAndHoldStrategy,
        MLStrategy,
    )
}
