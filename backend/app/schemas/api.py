from __future__ import annotations

from pydantic import BaseModel, Field


class AssetOut(BaseModel):
    symbol: str
    name: str
    decimals: int
    oracle_price: float
    ltv: float
    liquidation_threshold: float
    simulated: bool = True


class ReserveOut(BaseModel):
    symbol: str
    name: str
    oracle_price: float
    oracle_timestamp: int
    total_supplied: float
    total_borrowed: float
    available_liquidity: float
    utilization: float
    supply_apr: float
    borrow_apr: float
    ltv: float
    liquidation_threshold: float
    protocol: str
    simulated: bool = True


class StrategyOut(BaseModel):
    name: str
    description: str


class AllocateRequest(BaseModel):
    strategy: str = "risk_adjusted"
    lambda_risk: float | None = None
    lambda_cost: float | None = None
    lambda_concentration: float | None = None
    lambda_liquidation: float | None = None
    max_weight: float | None = None


class AllocateResponse(BaseModel):
    strategy: str
    weights: dict[str, float]
    expected_return: float
    risk: float
    concentration: float
    liquidation_risk: float
    transaction_cost: float
    objective: float
    diagnostics: dict[str, str]
    used_ml_predictions: bool


class PredictRequest(BaseModel):
    symbols: list[str] | None = None


class PredictionPoint(BaseModel):
    symbol: str
    current_supply_apr: float
    predicted_supply_apr: float
    features: dict[str, float]


class PredictResponse(BaseModel):
    model_name: str
    data_label: str
    primary_target: str
    predictions: list[PredictionPoint]
    disclaimer: str


class ModelStatusOut(BaseModel):
    name: str
    version: str
    model_type: str
    ready: bool
    data_label: str
    primary_target: str
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    feature_importance: dict[str, float]
    notes: str
    hyperparameters: dict[str, float]


class SimulationRequest(BaseModel):
    steps: int = Field(default=24, ge=1, le=2000)
    scenario: str = "normal"
    strategy: str = "equal_weight"
    seed: int = 42
    n_users: int = Field(default=10, ge=0, le=1000)
    capital: float = Field(default=100000.0, gt=0)


class SimulationResponse(BaseModel):
    experiment_id: str
    scenario: str
    strategy: str
    seed: int
    steps: int
    synthetic: bool
    metrics: dict[str, float]
    liquidations: int
    final_value: float
    final_weights: dict[str, float]
    notes: str


class PortfolioOut(BaseModel):
    strategy: str
    weights: dict[str, float]
    expected_return: float
    predicted_yield: dict[str, float] | None
    risk: float
    portfolio_value: float
    simulated: bool = True


class MetricsOut(BaseModel):
    source: str
    metrics: dict[str, float]
    tvl: float
    utilization_avg: float
    risk_score: float


class ExperimentOut(BaseModel):
    id: str
    created_at: str
    scenario: str
    strategy: str
    seed: int
    steps: int
    data_label: str
    notes: str
    metrics: dict[str, float]


class HealthOut(BaseModel):
    status: str
    protocol: str
    aave: bool
    model_ready: bool
    database: str
