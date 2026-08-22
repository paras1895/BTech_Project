from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from app.adapters.lending import AccountData, InterestRateModel, ReserveSnapshot


class SimulationError(Exception):
    pass


@dataclass
class ReserveState:
    symbol: str
    name: str
    decimals: int
    oracle_price: float
    oracle_timestamp: int
    total_supplied: float
    total_borrowed: float
    ltv: float
    liquidation_threshold: float
    liquidation_bonus: float
    reserve_factor: float
    irm: InterestRateModel
    paused: bool = False
    price_history: list[float] = field(default_factory=list)

    @property
    def available_liquidity(self) -> float:
        return max(self.total_supplied - self.total_borrowed, 0.0)

    @property
    def utilization(self) -> float:
        if self.total_supplied <= 0:
            return 0.0
        return min(self.total_borrowed / self.total_supplied, 0.999)

    def rates(self) -> tuple[float, float]:
        u = self.utilization
        borrow = self.irm.borrow_apr(u)
        supply = self.irm.supply_apr(u, self.reserve_factor)
        return supply, borrow


@dataclass
class UserAccount:
    user_id: str
    profile: str
    supplied: dict[str, float] = field(default_factory=dict)
    borrowed: dict[str, float] = field(default_factory=dict)
    cash: dict[str, float] = field(default_factory=dict)


class SimulationPool:
    """Educational Aave-inspired pool. Not the Aave protocol."""

    protocol_name = "educational_aave_inspired_simulation"

    def __init__(self, reserves: dict[str, ReserveState], hours_per_year: int = 8760) -> None:
        self.reserves = reserves
        self.users: dict[str, UserAccount] = {}
        self.hours_per_year = hours_per_year
        self.time_hours = 0
        self.liquidations = 0

    @classmethod
    def from_asset_config(cls, assets_cfg: dict, hours_per_year: int = 8760) -> SimulationPool:
        reserves: dict[str, ReserveState] = {}
        for item in assets_cfg["assets"]:
            irm_raw = item["irm"]
            irm = InterestRateModel(
                base_rate=float(irm_raw["base_rate"]),
                kink=float(irm_raw["kink"]),
                slope1=float(irm_raw["slope1"]),
                slope2=float(irm_raw["slope2"]),
            )
            price = float(item["oracle_price"])
            reserves[item["symbol"]] = ReserveState(
                symbol=item["symbol"],
                name=item["name"],
                decimals=int(item["decimals"]),
                oracle_price=price,
                oracle_timestamp=0,
                total_supplied=float(item["initial_supplied"]),
                total_borrowed=float(item["initial_borrowed"]),
                ltv=float(item["ltv"]),
                liquidation_threshold=float(item["liquidation_threshold"]),
                liquidation_bonus=float(item["liquidation_bonus"]),
                reserve_factor=float(item["reserve_factor"]),
                irm=irm,
                price_history=[price],
            )
        return cls(reserves, hours_per_year=hours_per_year)

    def clone(self) -> SimulationPool:
        return deepcopy(self)

    def snapshot(self, symbol: str) -> ReserveSnapshot:
        reserve = self.reserves[symbol]
        supply_apr, borrow_apr = reserve.rates()
        return ReserveSnapshot(
            symbol=reserve.symbol,
            name=reserve.name,
            decimals=reserve.decimals,
            oracle_price=reserve.oracle_price,
            oracle_timestamp=reserve.oracle_timestamp,
            total_supplied=reserve.total_supplied,
            total_borrowed=reserve.total_borrowed,
            available_liquidity=reserve.available_liquidity,
            utilization=reserve.utilization,
            supply_apr=supply_apr,
            borrow_apr=borrow_apr,
            ltv=reserve.ltv,
            liquidation_threshold=reserve.liquidation_threshold,
            liquidation_bonus=reserve.liquidation_bonus,
            reserve_factor=reserve.reserve_factor,
            paused=reserve.paused,
            protocol=self.protocol_name,
        )

    def snapshots(self) -> list[ReserveSnapshot]:
        return [self.snapshot(symbol) for symbol in self.reserves]

    def create_user(self, user_id: str, profile: str, cash: dict[str, float]) -> UserAccount:
        account = UserAccount(user_id=user_id, profile=profile, cash=dict(cash))
        self.users[user_id] = account
        return account

    def supply(self, user_id: str, symbol: str, amount: float) -> None:
        if amount <= 0:
            raise SimulationError("amount must be positive")
        user = self.users[user_id]
        reserve = self.reserves[symbol]
        if reserve.paused:
            raise SimulationError("reserve paused")
        held = user.cash.get(symbol, 0.0)
        if held + 1e-12 < amount:
            raise SimulationError("insufficient cash to supply")
        user.cash[symbol] = held - amount
        user.supplied[symbol] = user.supplied.get(symbol, 0.0) + amount
        reserve.total_supplied += amount

    def withdraw(self, user_id: str, symbol: str, amount: float) -> None:
        if amount <= 0:
            raise SimulationError("amount must be positive")
        user = self.users[user_id]
        reserve = self.reserves[symbol]
        supplied = user.supplied.get(symbol, 0.0)
        if supplied + 1e-12 < amount:
            raise SimulationError("insufficient supplied balance")
        if reserve.available_liquidity + 1e-12 < amount:
            raise SimulationError("insufficient pool liquidity")
        user.supplied[symbol] = supplied - amount
        user.cash[symbol] = user.cash.get(symbol, 0.0) + amount
        reserve.total_supplied -= amount
        if self._health_factor(user) is not None and self._health_factor(user) < 1.0:
            user.supplied[symbol] = supplied
            user.cash[symbol] -= amount
            reserve.total_supplied += amount
            raise SimulationError("withdraw would liquidate the account")

    def borrow(self, user_id: str, symbol: str, amount: float) -> None:
        if amount <= 0:
            raise SimulationError("amount must be positive")
        user = self.users[user_id]
        reserve = self.reserves[symbol]
        if reserve.available_liquidity + 1e-12 < amount:
            raise SimulationError("insufficient pool liquidity")
        user.borrowed[symbol] = user.borrowed.get(symbol, 0.0) + amount
        user.cash[symbol] = user.cash.get(symbol, 0.0) + amount
        reserve.total_borrowed += amount
        hf = self._health_factor(user)
        max_ltv_ok = self._loan_to_value(user) <= self._max_ltv(user) + 1e-9
        if hf is not None and (hf < 1.0 or not max_ltv_ok):
            user.borrowed[symbol] -= amount
            user.cash[symbol] -= amount
            reserve.total_borrowed -= amount
            raise SimulationError("borrow violates health factor or LTV")

    def repay(self, user_id: str, symbol: str, amount: float) -> None:
        if amount <= 0:
            raise SimulationError("amount must be positive")
        user = self.users[user_id]
        debt = user.borrowed.get(symbol, 0.0)
        pay = min(amount, debt)
        held = user.cash.get(symbol, 0.0)
        if held + 1e-12 < pay:
            raise SimulationError("insufficient cash to repay")
        user.cash[symbol] = held - pay
        user.borrowed[symbol] = debt - pay
        self.reserves[symbol].total_borrowed -= pay

    def account_data(self, user_id: str) -> AccountData:
        user = self.users[user_id]
        collateral, debt = self._values(user)
        hf = self._health_factor(user)
        if hf is None:
            liq = 0.0
        else:
            liq = float(min(max((1.5 - hf) / 1.5, 0.0), 1.0))
        return AccountData(
            user_id=user_id,
            collateral_value=collateral,
            debt_value=debt,
            health_factor=hf,
            liquidation_probability_proxy=liq,
        )

    def accrue(self, hours: float) -> None:
        if hours <= 0:
            return
        fraction = hours / self.hours_per_year
        for reserve in self.reserves.values():
            supply_apr, borrow_apr = reserve.rates()
            reserve.total_supplied *= 1.0 + supply_apr * fraction
            reserve.total_borrowed *= 1.0 + borrow_apr * fraction
            for user in self.users.values():
                if reserve.symbol in user.supplied:
                    user.supplied[reserve.symbol] *= 1.0 + supply_apr * fraction
                if reserve.symbol in user.borrowed:
                    user.borrowed[reserve.symbol] *= 1.0 + borrow_apr * fraction
        self.time_hours += hours
        for reserve in self.reserves.values():
            reserve.oracle_timestamp = int(self.time_hours)

    def set_price(self, symbol: str, price: float) -> None:
        if price <= 0:
            raise SimulationError("price must be positive")
        reserve = self.reserves[symbol]
        reserve.oracle_price = price
        reserve.oracle_timestamp = int(self.time_hours)
        reserve.price_history.append(price)

    def liquidate_if_needed(self, user_id: str) -> bool:
        user = self.users[user_id]
        hf = self._health_factor(user)
        if hf is None or hf >= 1.0:
            return False
        debt_symbol = max(
            user.borrowed,
            key=lambda symbol: user.borrowed[symbol] * self.reserves[symbol].oracle_price,
            default=None,
        )
        coll_symbol = max(
            user.supplied,
            key=lambda symbol: user.supplied[symbol] * self.reserves[symbol].oracle_price,
            default=None,
        )
        if debt_symbol is None or coll_symbol is None:
            return False
        close_fraction = 0.5
        repay_amount = user.borrowed[debt_symbol] * close_fraction
        debt_value = repay_amount * self.reserves[debt_symbol].oracle_price
        bonus = self.reserves[coll_symbol].liquidation_bonus
        seized_value = debt_value * (1.0 + bonus)
        seized_amount = seized_value / self.reserves[coll_symbol].oracle_price
        seized_amount = min(seized_amount, user.supplied.get(coll_symbol, 0.0))
        user.borrowed[debt_symbol] -= repay_amount
        self.reserves[debt_symbol].total_borrowed -= repay_amount
        user.supplied[coll_symbol] -= seized_amount
        self.reserves[coll_symbol].total_supplied -= seized_amount
        self.liquidations += 1
        return True

    def _values(self, user: UserAccount) -> tuple[float, float]:
        collateral = sum(
            amount * self.reserves[symbol].oracle_price
            for symbol, amount in user.supplied.items()
        )
        debt = sum(
            amount * self.reserves[symbol].oracle_price
            for symbol, amount in user.borrowed.items()
        )
        return collateral, debt

    def _health_factor(self, user: UserAccount) -> float | None:
        threshold_value = sum(
            amount
            * self.reserves[symbol].oracle_price
            * self.reserves[symbol].liquidation_threshold
            for symbol, amount in user.supplied.items()
        )
        _, debt = self._values(user)
        if debt <= 1e-12:
            return None
        return threshold_value / debt

    def _loan_to_value(self, user: UserAccount) -> float:
        collateral, debt = self._values(user)
        if collateral <= 1e-12:
            return 0.0 if debt <= 1e-12 else float("inf")
        return debt / collateral

    def _max_ltv(self, user: UserAccount) -> float:
        collateral, _ = self._values(user)
        if collateral <= 1e-12:
            return 0.0
        weighted = sum(
            amount
            * self.reserves[symbol].oracle_price
            * self.reserves[symbol].ltv
            for symbol, amount in user.supplied.items()
        )
        return weighted / collateral
