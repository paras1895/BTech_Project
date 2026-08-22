from __future__ import annotations

import numpy as np
import pytest

from app.adapters.lending import InterestRateModel
from app.adapters.simulation_pool import SimulationError, SimulationPool
from app.core.config import load_yaml
from app.services.ml import train_ridge_on_synthetic
from app.services.optimizer import OptimizerConfig, allocate
from app.services.risk import max_drawdown, volatility
from app.services.strategies import STRATEGIES


@pytest.fixture
def pool() -> SimulationPool:
    return SimulationPool.from_asset_config(load_yaml("assets.yaml"))


def test_utilization_linked_rates_increase() -> None:
    irm = InterestRateModel(base_rate=0.02, kink=0.8, slope1=0.04, slope2=0.8)
    low = irm.borrow_apr(0.2)
    high = irm.borrow_apr(0.95)
    assert high > low
    assert irm.supply_apr(0.5, 0.1) < irm.borrow_apr(0.5)


def test_supply_borrow_repay_and_health_factor(pool: SimulationPool) -> None:
    pool.create_user("u1", "balanced", {"USDC": 10_000, "WETH": 2.0})
    pool.supply("u1", "WETH", 2.0)
    pool.borrow("u1", "USDC", 1000)
    data = pool.account_data("u1")
    assert data.debt_value > 0
    assert data.health_factor is not None and data.health_factor > 1
    pool.repay("u1", "USDC", 1000)
    assert pool.account_data("u1").debt_value < 1e-6


def test_borrow_rejected_when_unhealthy(pool: SimulationPool) -> None:
    pool.create_user("u2", "aggressive", {"USDC": 100})
    pool.supply("u2", "USDC", 100)
    with pytest.raises(SimulationError):
        pool.borrow("u2", "WETH", 50)


def test_max_drawdown() -> None:
    assert max_drawdown([100.0, 120.0, 90.0, 95.0]) == pytest.approx(0.25)


def test_allocate_sums_to_one_and_respects_cap() -> None:
    symbols = ["A", "B", "C", "D"]
    mu = np.array([0.1, 0.05, 0.02, 0.01])
    cov = np.eye(4) * 0.01
    liq = np.array([0.1, 0.2, 0.3, 0.4])
    cfg = OptimizerConfig(max_weight=0.55)
    result = allocate(symbols, mu, cov, liq, cfg)
    assert pytest.approx(sum(result.weights.values()), abs=1e-6) == 1.0
    assert max(result.weights.values()) <= 0.55 + 1e-8


def test_equal_weight_strategy(pool: SimulationPool) -> None:
    snaps = pool.snapshots()
    histories = {s.symbol: pool.reserves[s.symbol].price_history for s in snaps}
    cfg = OptimizerConfig(max_weight=0.55)
    result = STRATEGIES["equal_weight"].propose(snaps, histories, cfg)
    n = len(snaps)
    for w in result.weights.values():
        assert w == pytest.approx(1 / n, abs=1e-8)


def test_ridge_trains_without_leakage_split() -> None:
    cfg = load_yaml("model.yaml")
    irm = InterestRateModel(0.02, 0.8, 0.04, 0.8)
    model = train_ridge_on_synthetic(cfg, irm, 0.1, seed=1)
    assert model.train_metrics.n_samples > model.validation_metrics.n_samples
    assert model.test_metrics.n_samples > 0
    assert model.data_label == "synthetic_local"
    assert volatility([1.0, 1.01, 0.99]) >= 0
