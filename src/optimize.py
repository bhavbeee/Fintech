"""Portfolio construction: equal-weight, max-Sharpe (Markowitz), and risk
parity — under simple, realistic constraints (no shorting, per-name cap,
cash buffer).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


@dataclass
class Constraints:
    allow_short: bool = False
    max_weight: float = 0.30  # cap per single name
    cash_weight: float = 0.0  # fraction held uninvested


def equal_weight(n: int, cash_weight: float = 0.0) -> np.ndarray:
    invested = 1 - cash_weight
    return np.full(n, invested / n)


def _bounds(n: int, c: Constraints):
    lo = -c.max_weight if c.allow_short else 0.0
    return [(lo, c.max_weight) for _ in range(n)]


def _base_constraints(c: Constraints):
    invested = 1 - c.cash_weight
    return [{"type": "eq", "fun": lambda w, inv=invested: np.sum(w) - inv}]


def max_sharpe(
    mu: pd.Series, cov: pd.DataFrame, tickers: list[str], constraints: Constraints, rf_annual: float = 0.04
) -> np.ndarray:
    n = len(tickers)
    x0 = equal_weight(n, constraints.cash_weight)
    mu_ann = mu.values * TRADING_DAYS
    cov_ann = cov.values * TRADING_DAYS

    def neg_sharpe(w):
        ret = w @ mu_ann
        vol = np.sqrt(max(w @ cov_ann @ w, 1e-12))
        return -(ret - rf_annual) / vol

    res = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=_bounds(n, constraints),
        constraints=_base_constraints(constraints),
        options={"maxiter": 500, "ftol": 1e-10},
    )
    return res.x if res.success else x0


def risk_parity(cov: pd.DataFrame, tickers: list[str], constraints: Constraints) -> np.ndarray:
    """Equalize each asset's contribution to total portfolio variance."""
    n = len(tickers)
    x0 = equal_weight(n, constraints.cash_weight)
    cov_v = cov.values * TRADING_DAYS  # annualize for well-scaled SLSQP objective

    def objective(w):
        port_var = w @ cov_v @ w
        marginal = cov_v @ w
        rc = w * marginal
        target = port_var / n
        return np.sum((rc - target) ** 2)

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=_bounds(n, constraints),
        constraints=_base_constraints(constraints),
        options={"maxiter": 1000, "ftol": 1e-16},
    )
    return res.x if res.success else x0


def risk_contributions(weights: np.ndarray, cov: pd.DataFrame) -> pd.Series:
    cov_v = cov.values
    port_var = weights @ cov_v @ weights
    marginal = cov_v @ weights
    rc = weights * marginal / port_var
    return pd.Series(rc, index=cov.columns)
