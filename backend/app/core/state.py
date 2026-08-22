from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select

from app.adapters.simulation_pool import SimulationPool
from app.core.config import load_yaml
from app.db import Base, configure_db, session_factory
from app.models.tables import Experiment, PortfolioSnapshotRow
from app.services.ml import TrainedModel, train_ridge_on_synthetic
from app.services.optimizer import AllocationResult, OptimizerConfig
from app.services.risk import reserve_risk_vector


STRATEGY_HELP = {
    "equal_weight": "Equal allocation across eligible simulated reserves.",
    "highest_yield": "Fill highest current supply APR up to max_weight.",
    "risk_adjusted": "Optimize supply APR / realized volatility via the constrained solver.",
    "volatility_aware": "Penalize assets with high price volatility, then optimize.",
    "buy_and_hold": "Keep the initial equal mix (control baseline).",
    "ml_optimizer": "Use Ridge predictions of future supply APR, then the constrained optimizer.",
}


def optimizer_config_from_yaml(raw: dict) -> OptimizerConfig:
    return OptimizerConfig(
        lambda_risk=float(raw["lambda_risk"]),
        lambda_cost=float(raw["lambda_cost"]),
        lambda_concentration=float(raw["lambda_concentration"]),
        lambda_liquidation=float(raw["lambda_liquidation"]),
        max_weight=float(raw["max_weight"]),
        min_weight=float(raw["min_weight"]),
        transaction_cost=float(raw["transaction_cost"]),
    )


def new_experiment_id() -> str:
    return str(uuid.uuid4())


@dataclass
class AppContext:
    pool: SimulationPool
    assets_cfg: dict
    sim_cfg: dict
    model_cfg: dict
    opt_cfg: OptimizerConfig
    model: TrainedModel
    last_allocation: AllocationResult | None = None
    last_strategy: str = "equal_weight"
    last_predictions: dict[str, float] = field(default_factory=dict)
    last_metrics: dict[str, float] = field(default_factory=dict)
    last_portfolio_value: float = 100000.0
    utilization_history: dict[str, list[float]] = field(default_factory=dict)

    @classmethod
    def bootstrap(cls) -> AppContext:
        engine = configure_db()
        Base.metadata.create_all(bind=engine)
        assets_cfg = load_yaml("assets.yaml")
        sim_cfg = load_yaml("simulation.yaml")
        model_cfg = load_yaml("model.yaml")
        opt_cfg = optimizer_config_from_yaml(load_yaml("optimizer.yaml"))
        pool = SimulationPool.from_asset_config(assets_cfg, hours_per_year=int(sim_cfg["hours_per_year"]))
        first = next(iter(pool.reserves.values()))
        model = train_ridge_on_synthetic(
            model_cfg,
            first.irm,
            first.reserve_factor,
            seed=int(sim_cfg["default_seed"]),
        )
        ctx = cls(
            pool=pool,
            assets_cfg=assets_cfg,
            sim_cfg=sim_cfg,
            model_cfg=model_cfg,
            opt_cfg=opt_cfg,
            model=model,
            last_portfolio_value=float(sim_cfg["default_capital"]),
            utilization_history={symbol: [res.utilization] for symbol, res in pool.reserves.items()},
        )
        ctx.seed_users(int(sim_cfg["default_users"]), int(sim_cfg["default_seed"]))
        return ctx

    def seed_users(self, n_users: int, seed: int) -> None:
        profiles = ["conservative", "balanced", "aggressive"]
        symbols = list(self.pool.reserves)
        _ = seed
        for i in range(n_users):
            profile = profiles[i % 3]
            user_id = f"sim-user-{i:04d}"
            cash = {
                symbol: 8_000.0 / self.pool.reserves[symbol].oracle_price
                for symbol in symbols
            }
            self.pool.create_user(user_id, profile, cash)
            symbol = symbols[i % len(symbols)]
            amount = cash[symbol] * 0.4
            try:
                self.pool.supply(user_id, symbol, amount)
            except Exception:
                continue

    def tvl(self) -> float:
        return sum(snap.total_supplied * snap.oracle_price for snap in self.pool.snapshots())

    def avg_utilization(self) -> float:
        snaps = self.pool.snapshots()
        if not snaps:
            return 0.0
        return sum(snap.utilization for snap in snaps) / len(snaps)

    def risk_score(self) -> float:
        snaps = self.pool.snapshots()
        histories = {snap.symbol: self.pool.reserves[snap.symbol].price_history for snap in snaps}
        risks = reserve_risk_vector(snaps, histories)
        if not risks:
            return 0.0
        vols = [item["volatility"] + item["utilization"] for item in risks.values()]
        return float(sum(vols) / len(vols))

    def persist_experiment(
        self,
        experiment_id: str,
        scenario: str,
        strategy: str,
        seed: int,
        steps: int,
        metrics: dict,
        notes: str,
    ) -> None:
        row = Experiment(
            id=experiment_id,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            scenario=scenario,
            strategy=strategy,
            seed=seed,
            steps=steps,
            data_label="synthetic_local",
            notes=notes,
            metrics_json=json.dumps(metrics),
        )
        with session_factory()() as session:
            session.add(row)
            session.commit()

    def persist_allocation(self, strategy: str, result: AllocationResult) -> None:
        row = PortfolioSnapshotRow(
            strategy=strategy,
            expected_return=result.expected_return,
            risk=result.risk,
            weights_json=json.dumps(result.weights),
        )
        with session_factory()() as session:
            session.add(row)
            session.commit()

    def list_experiments(self) -> list[Experiment]:
        with session_factory()() as session:
            stmt = select(Experiment).order_by(Experiment.created_at.desc()).limit(100)
            return list(session.scalars(stmt).all())
