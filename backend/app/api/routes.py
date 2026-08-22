from __future__ import annotations

import json

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.core.state import STRATEGY_HELP, AppContext, new_experiment_id
from app.schemas.api import (
    AllocateRequest,
    AllocateResponse,
    AssetOut,
    ExperimentOut,
    HealthOut,
    MetricsOut,
    ModelStatusOut,
    PortfolioOut,
    PredictRequest,
    PredictResponse,
    PredictionPoint,
    ReserveOut,
    SimulationRequest,
    SimulationResponse,
    StrategyOut,
)
from app.services.ml import FEATURE_NAMES, features_from_snapshot, predict_supply_apr
from app.services.optimizer import OptimizerConfig, result_from_weights
from app.services.risk import correlation_matrix, volatility
from app.services.simulation_runner import run_simulation
from app.services.strategies import STRATEGIES

router = APIRouter()


def ctx(request: Request) -> AppContext:
    return request.app.state.ctx


@router.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    state = ctx(request)
    return HealthOut(
        status="ok",
        protocol=state.pool.protocol_name,
        aave=False,
        model_ready=True,
        database="sqlite",
    )


@router.get("/assets", response_model=list[AssetOut])
def assets(request: Request) -> list[AssetOut]:
    return [
        AssetOut(
            symbol=res.symbol,
            name=res.name,
            decimals=res.decimals,
            oracle_price=res.oracle_price,
            ltv=res.ltv,
            liquidation_threshold=res.liquidation_threshold,
            simulated=True,
        )
        for res in ctx(request).pool.reserves.values()
    ]


@router.get("/reserves", response_model=list[ReserveOut])
def reserves(request: Request) -> list[ReserveOut]:
    return [
        ReserveOut(
            symbol=snap.symbol,
            name=snap.name,
            oracle_price=snap.oracle_price,
            oracle_timestamp=snap.oracle_timestamp,
            total_supplied=snap.total_supplied,
            total_borrowed=snap.total_borrowed,
            available_liquidity=snap.available_liquidity,
            utilization=snap.utilization,
            supply_apr=snap.supply_apr,
            borrow_apr=snap.borrow_apr,
            ltv=snap.ltv,
            liquidation_threshold=snap.liquidation_threshold,
            protocol=snap.protocol,
            simulated=True,
        )
        for snap in ctx(request).pool.snapshots()
    ]


@router.get("/strategies", response_model=list[StrategyOut])
def strategies() -> list[StrategyOut]:
    return [StrategyOut(name=name, description=STRATEGY_HELP[name]) for name in STRATEGIES]


@router.get("/portfolio", response_model=PortfolioOut)
def portfolio(request: Request) -> PortfolioOut:
    state = ctx(request)
    snaps = state.pool.snapshots()
    if state.last_allocation is None:
        n = len(snaps)
        weights = {snap.symbol: 1.0 / n for snap in snaps}
        expected = sum(weights[s.symbol] * s.supply_apr for s in snaps)
        risk = 0.0
        strategy = "equal_weight"
    else:
        weights = state.last_allocation.weights
        expected = state.last_allocation.expected_return
        risk = state.last_allocation.risk
        strategy = state.last_strategy
    return PortfolioOut(
        strategy=strategy,
        weights=weights,
        expected_return=expected,
        predicted_yield=state.last_predictions or None,
        risk=risk,
        portfolio_value=state.last_portfolio_value,
        simulated=True,
    )


@router.get("/metrics", response_model=MetricsOut)
def metrics(request: Request) -> MetricsOut:
    state = ctx(request)
    return MetricsOut(
        source="last_simulation_or_live_pool",
        metrics=state.last_metrics
        or {
            "tvl": state.tvl(),
            "avg_utilization": state.avg_utilization(),
            "liquidations": float(state.pool.liquidations),
        },
        tvl=state.tvl(),
        utilization_avg=state.avg_utilization(),
        risk_score=state.risk_score(),
    )


@router.get("/model/status", response_model=ModelStatusOut)
def model_status(request: Request) -> ModelStatusOut:
    model = ctx(request).model

    def pack(m) -> dict[str, float]:
        return {"mae": m.mae, "rmse": m.rmse, "r2": m.r2, "n_samples": float(m.n_samples)}

    return ModelStatusOut(
        name=model.name,
        version=model.version,
        model_type=model.model_type,
        ready=True,
        data_label=model.data_label,
        primary_target=model.primary_target,
        train_metrics=pack(model.train_metrics),
        validation_metrics=pack(model.validation_metrics),
        test_metrics=pack(model.test_metrics),
        feature_importance=model.feature_importance,
        notes=model.notes,
        hyperparameters=model.hyperparameters,
    )


@router.post("/model/predict", response_model=PredictResponse)
def model_predict(body: PredictRequest, request: Request) -> PredictResponse:
    state = ctx(request)
    snaps = state.pool.snapshots()
    wanted = set(body.symbols) if body.symbols else {s.symbol for s in snaps}
    points: list[PredictionPoint] = []
    preds: dict[str, float] = {}
    for snap in snaps:
        if snap.symbol not in wanted:
            continue
        feats = features_from_snapshot(
            snap,
            state.pool.reserves[snap.symbol].price_history,
            state.utilization_history.get(snap.symbol, [snap.utilization]),
            int(state.pool.time_hours),
        )
        pred = predict_supply_apr(state.model, feats)
        preds[snap.symbol] = pred
        points.append(
            PredictionPoint(
                symbol=snap.symbol,
                current_supply_apr=snap.supply_apr,
                predicted_supply_apr=pred,
                features={name: float(val) for name, val in zip(FEATURE_NAMES, feats, strict=True)},
            )
        )
    if not points:
        raise HTTPException(status_code=404, detail="no matching symbols")
    state.last_predictions = preds
    return PredictResponse(
        model_name=state.model.name,
        data_label=state.model.data_label,
        primary_target=state.model.primary_target,
        predictions=points,
        disclaimer="Predictions come from a Ridge model trained on synthetic local series, not live Aave data.",
    )


@router.post("/optimizer/allocate", response_model=AllocateResponse)
def optimizer_allocate(body: AllocateRequest, request: Request) -> AllocateResponse:
    state = ctx(request)
    if body.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"unknown strategy: {body.strategy}")
    cfg = state.opt_cfg
    cfg = OptimizerConfig(
        lambda_risk=body.lambda_risk if body.lambda_risk is not None else cfg.lambda_risk,
        lambda_cost=body.lambda_cost if body.lambda_cost is not None else cfg.lambda_cost,
        lambda_concentration=body.lambda_concentration if body.lambda_concentration is not None else cfg.lambda_concentration,
        lambda_liquidation=body.lambda_liquidation if body.lambda_liquidation is not None else cfg.lambda_liquidation,
        max_weight=body.max_weight if body.max_weight is not None else cfg.max_weight,
        min_weight=cfg.min_weight,
        transaction_cost=cfg.transaction_cost,
    )
    snapshots = state.pool.snapshots()
    histories = {s.symbol: state.pool.reserves[s.symbol].price_history for s in snapshots}
    predicted = None
    used_ml = body.strategy == "ml_optimizer"
    if used_ml:
        predicted = {}
        for snap in snapshots:
            feats = features_from_snapshot(
                snap,
                histories[snap.symbol],
                state.utilization_history.get(snap.symbol, [snap.utilization]),
                int(state.pool.time_hours),
            )
            predicted[snap.symbol] = predict_supply_apr(state.model, feats)
        state.last_predictions = predicted
    prev = state.last_allocation.weights if state.last_allocation else None
    try:
        result = STRATEGIES[body.strategy].propose(snapshots, histories, cfg, predicted, prev)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    state.last_allocation = result
    state.last_strategy = body.strategy
    state.persist_allocation(body.strategy, result)
    return AllocateResponse(
        strategy=body.strategy,
        weights=result.weights,
        expected_return=result.expected_return,
        risk=result.risk,
        concentration=result.concentration,
        liquidation_risk=result.liquidation_risk,
        transaction_cost=result.transaction_cost,
        objective=result.objective,
        diagnostics=result.diagnostics,
        used_ml_predictions=used_ml,
    )


@router.post("/simulation/run", response_model=SimulationResponse)
def simulation_run(body: SimulationRequest, request: Request) -> SimulationResponse:
    state = ctx(request)
    scenarios = state.sim_cfg.get("scenarios", {})
    if body.scenario not in scenarios:
        raise HTTPException(status_code=400, detail=f"unknown scenario: {body.scenario}")
    if body.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"unknown strategy: {body.strategy}")
    working = state.pool.clone()
    experiment_id = new_experiment_id()
    try:
        outcome = run_simulation(
            working,
            strategy_name=body.strategy,
            scenario_name=body.scenario,
            scenario_cfg=scenarios[body.scenario],
            sim_cfg=state.sim_cfg,
            opt_cfg=state.opt_cfg,
            steps=body.steps,
            seed=body.seed,
            capital=body.capital,
            model=state.model,
            experiment_id=experiment_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    state.last_metrics = outcome.metrics
    state.last_portfolio_value = outcome.portfolio_values[-1]
    state.last_strategy = body.strategy
    if outcome.weights_path:
        snaps = working.snapshots()
        symbols = [s.symbol for s in snaps]
        w = np.array([outcome.weights_path[-1][s] for s in symbols])
        mu = np.array([s.supply_apr for s in snaps])
        vols = np.array([max(volatility(working.reserves[s].price_history), 1e-4) for s in symbols])
        cov = np.outer(vols, vols) * correlation_matrix({s: working.reserves[s].price_history for s in symbols}, symbols)
        liq = np.array([s.utilization for s in snaps])
        state.last_allocation = result_from_weights(
            symbols,
            w,
            mu,
            cov,
            liq,
            state.opt_cfg,
            w,
            {"solver": "simulation", "message": "final sim weights"},
        )
    state.persist_experiment(
        experiment_id,
        body.scenario,
        body.strategy,
        body.seed,
        body.steps,
        outcome.metrics,
        outcome.notes,
    )
    return SimulationResponse(
        experiment_id=outcome.experiment_id,
        scenario=outcome.scenario,
        strategy=outcome.strategy,
        seed=outcome.seed,
        steps=outcome.steps,
        synthetic=True,
        metrics=outcome.metrics,
        liquidations=outcome.liquidations,
        final_value=outcome.portfolio_values[-1],
        final_weights=outcome.weights_path[-1],
        notes=outcome.notes,
    )


@router.get("/experiments", response_model=list[ExperimentOut])
def experiments(request: Request) -> list[ExperimentOut]:
    rows = ctx(request).list_experiments()
    return [
        ExperimentOut(
            id=row.id,
            created_at=row.created_at.isoformat(),
            scenario=row.scenario,
            strategy=row.strategy,
            seed=row.seed,
            steps=row.steps,
            data_label=row.data_label,
            notes=row.notes,
            metrics=json.loads(row.metrics_json),
        )
        for row in rows
    ]
