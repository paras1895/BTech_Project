from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class InterestRateModel:
    base_rate: float
    kink: float
    slope1: float
    slope2: float

    def borrow_apr(self, utilization: float) -> float:
        u = min(max(utilization, 0.0), 0.999)
        if u <= self.kink:
            return self.base_rate + self.slope1 * (u / self.kink if self.kink else 0.0)
        excess = (u - self.kink) / (1.0 - self.kink) if self.kink < 1.0 else 0.0
        return self.base_rate + self.slope1 + self.slope2 * excess

    def supply_apr(self, utilization: float, reserve_factor: float) -> float:
        return self.borrow_apr(utilization) * utilization * (1.0 - reserve_factor)


@dataclass
class ReserveSnapshot:
    symbol: str
    name: str
    decimals: int
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
    liquidation_bonus: float
    reserve_factor: float
    paused: bool
    protocol: str = "educational_aave_inspired_simulation"


@dataclass
class AccountData:
    user_id: str
    collateral_value: float
    debt_value: float
    health_factor: float | None
    liquidation_probability_proxy: float


class LendingPoolAdapter(Protocol):
    """Python adapter interface. Solidity fork adapter comes later."""

    def snapshot(self, symbol: str) -> ReserveSnapshot: ...

    def snapshots(self) -> list[ReserveSnapshot]: ...
