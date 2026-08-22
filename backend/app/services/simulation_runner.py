from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.adapters.simulation_pool import SimulationPool
from app.services.metrics import path_metrics
from app.services.ml import TrainedModel, features_from_snapshot, predict_supply_apr
from app.services.optimizer import OptimizerConfig
from app.services.strategies import STRATEGIES


@dataclass
class SimulationOutcome:
    experiment_id: str
    scenario: str
    strategy: str
    seed: int
    steps: int
    synthetic: bool
    portfolio_values: list[float]
    weights_path: list[dict[str, float]]
    metrics: dict[str, float]
    liquidations: int
    notes: str


def apply_scenario(
    pool: SimulationPool,
    scenario_cfg: dict,
    rng: np.random.Generator,
    step_index: int,
) -> None:
    crash = float(scenario_cfg.get("crash_return", 0.0))
    vol = float(scenario_cfg["price_vol"])
    util_shock = float(scenario_cfg.get("utilization_shock", 0.0))
    liq_shock = float(scenario_cfg.get("liquidity_shock", 0.0))
    for symbol, reserve in pool.reserves.items():
        shock = rng.normal(0.0, vol)
        if crash and step_index == 1 and symbol in {"WETH", "WBTC"}:
            shock += crash
        new_price = max(reserve.oracle_price * (1.0 + shock), 0.01)
        pool.set_price(symbol, new_price)
        if util_shock:
            extra = reserve.available_liquidity * util_shock * 0.15
            reserve.total_borrowed = min(reserve.total_supplied * 0.98, reserve.total_borrowed + extra)
        if liq_shock:
            reserve.total_supplied = max(reserve.total_borrowed * 1.02, reserve.total_supplied * (1.0 - liq_shock * 0.05))


def run_simulation(
    pool: SimulationPool,
    *,
    strategy_name: str,
    scenario_name: str,
    scenario_cfg: dict,
    sim_cfg: dict,
    opt_cfg: OptimizerConfig,
    steps: int,
    seed: int,
    capital: float,
    model: TrainedModel | None,
    experiment_id: str,
) -> SimulationOutcome:
    if strategy_name not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy_name}")
    strategy = STRATEGIES[strategy_name]
    rng = np.random.default_rng(seed)
    snapshots = pool.snapshots()
    symbols = [item.symbol for item in snapshots]
    weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
    values = [capital]
    weight_path = [dict(weights)]
    turnovers: list[float] = []
    hours = float(sim_cfg["step_hours"])
    hours_year = float(sim_cfg["hours_per_year"])

    for step in range(steps):
        apply_scenario(pool, scenario_cfg, rng, step)
        pool.accrue(hours)
        snapshots = pool.snapshots()
        histories = {symbol: pool.reserves[symbol].price_history for symbol in symbols}
        predicted = None
        if strategy_name == "ml_optimizer":
            if model is None:
                raise ValueError("ml_optimizer requires a trained model")
            predicted = {}
            for snap in snapshots:
                feats = features_from_snapshot(
                    snap,
                    pool.reserves[snap.symbol].price_history,
                    [snap.utilization],
                    int(pool.time_hours),
                )
                predicted[snap.symbol] = predict_supply_apr(model, feats)
        proposed = strategy.propose(snapshots, histories, opt_cfg, predicted, weights)
        turnover = 0.5 * sum(abs(proposed.weights[s] - weights.get(s, 0.0)) for s in symbols)
        turnovers.append(turnover)
        cost = turnover * opt_cfg.transaction_cost
        weights = proposed.weights
        weight_path.append(dict(weights))
        yield_step = 0.0
        for snap in snapshots:
            yield_step += weights[snap.symbol] * snap.supply_apr * (hours / hours_year)
        price_ret = 0.0
        for symbol in symbols:
            hist = pool.reserves[symbol].price_history
            if len(hist) >= 2:
                price_ret += weights[symbol] * (hist[-1] / hist[-2] - 1.0)
        new_value = values[-1] * (1.0 + yield_step + price_ret) * (1.0 - cost)
        values.append(new_value)
        for user_id in list(pool.users):
            pool.liquidate_if_needed(user_id)

    metrics = path_metrics(values, hours, hours_year, turnovers)
    metrics["liquidations"] = float(pool.liquidations)
    return SimulationOutcome(
        experiment_id=experiment_id,
        scenario=scenario_name,
        strategy=strategy_name,
        seed=seed,
        steps=steps,
        synthetic=True,
        portfolio_values=values,
        weights_path=weight_path,
        metrics=metrics,
        liquidations=pool.liquidations,
        notes="Synthetic scenario. Not a historical market event.",
    )
