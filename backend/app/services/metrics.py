from __future__ import annotations

import math

import numpy as np

from app.services.risk import max_drawdown, sharpe_like, sortino_like


def path_metrics(values: list[float], hours_per_step: float, hours_per_year: float, turnovers: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "turnover": 0.0,
        }
    rets = (arr[1:] - arr[:-1]) / arr[:-1]
    steps_per_year = hours_per_year / max(hours_per_step, 1e-9)
    cumulative = float(arr[-1] / arr[0] - 1.0)
    ann = float((1.0 + cumulative) ** (steps_per_year / max(len(rets), 1)) - 1.0) if arr[0] > 0 else 0.0
    vol = float(rets.std(ddof=1) * math.sqrt(steps_per_year)) if rets.size > 1 else 0.0
    sharpe = sharpe_like(rets) * math.sqrt(steps_per_year) if rets.size > 1 else 0.0
    sortino = sortino_like(rets) * math.sqrt(steps_per_year) if rets.size > 1 else 0.0
    mdd = max_drawdown(values)
    calmar = ann / mdd if mdd > 1e-12 else 0.0
    return {
        "cumulative_return": cumulative,
        "annualized_return": ann,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "turnover": float(np.mean(turnovers)) if turnovers else 0.0,
    }
