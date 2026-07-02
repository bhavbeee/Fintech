"""Portfolio construction: equal-weight, min-variance / max-Sharpe (Markowitz),
and risk parity — all under realistic constraints (no shorting, per-name cap,
sector cap, cash buffer, optional turnover budget).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


@dataclass
class Constraints:
    allow_short: bool = False
    max_weight: float = 0.30           # cap per single name
    cash_weight: float = 0.0           # fraction held uninvested
    sector_map: dict[str, str] | None = None
    sector_cap: float = 1.0            # max weight per sector
    prev_weights: np.ndarray | None = None
    max_turnover: float | None = None  # cap on sum(|w - w_prev|) at a rebalance


def equal_weight(n: int, cash_weight: float = 0.0) -> np.ndarray:
    invested = 1 - cash_weight
    return np.full(n, invested / n)


def _bounds(n: int, c: Constraints):
    lo = -c.max_weight if c.allow_short else 0.0
    return [(lo, c.max_weight) for _ in range(n)]


def _base_constraints(tickers: list[str], c: Constraints):
    invested = 1 - c.cash_weight
    cons = [{"type": "eq", "fun": lambda w, inv=invested: np.sum(w) - inv}]

    if c.sector_map:
        sectors = sorted(set(c.sector_map.get(t, "Unclassified") for t in tickers))
        for sec in sectors:
            idx = [i for i, t in enumerate(tickers) if c.sector_map.get(t, "Unclassified") == sec]
            cons.append(
                {"type": "ineq", "fun": lambda w, idx=idx, cap=c.sector_cap: cap - np.sum(w[idx])}
            )

    if c.max_turnover is not None and c.prev_weights is not None:
        prev = c.prev_weights
        cons.append(
            {"type": "ineq", "fun": lambda w, prev=prev, mt=c.max_turnover: mt - np.sum(np.abs(w - prev))}
        )
    return cons


def min_variance(cov: pd.DataFrame, tickers: list[str], constraints: Constraints) -> np.ndarray:
    n = len(tickers)
    x0 = equal_weight(n, constraints.cash_weight)
    cov_ann = cov.values * TRADING_DAYS  # annualize for well-scaled SLSQP objective
    res = minimize(
        lambda w: w @ cov_ann @ w,
        x0,
        method="SLSQP",
        bounds=_bounds(n, constraints),
        constraints=_base_constraints(tickers, constraints),
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return res.x if res.success else x0


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
        constraints=_base_constraints(tickers, constraints),
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
        constraints=_base_constraints(tickers, constraints),
        options={"maxiter": 1000, "ftol": 1e-16},
    )
    return res.x if res.success else x0


def efficient_frontier(
    mu: pd.Series, cov: pd.DataFrame, tickers: list[str], constraints: Constraints, n_points: int = 25
) -> pd.DataFrame:
    n = len(tickers)
    mu_ann = mu.values * TRADING_DAYS
    cov_ann = cov.values * TRADING_DAYS
    target_returns = np.linspace(mu_ann.min(), mu_ann.max(), n_points)

    rows = []
    for target in target_returns:
        x0 = equal_weight(n, constraints.cash_weight)
        cons = _base_constraints(tickers, constraints) + [
            {"type": "eq", "fun": lambda w, t=target: w @ mu_ann - t}
        ]
        res = minimize(
            lambda w: w @ cov_ann @ w,
            x0,
            method="SLSQP",
            bounds=_bounds(n, constraints),
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if res.success:
            vol = np.sqrt(res.x @ cov_ann @ res.x)
            rows.append({"target_return": target, "volatility": vol, "weights": res.x})
    return pd.DataFrame(rows)


def risk_contributions(weights: np.ndarray, cov: pd.DataFrame) -> pd.Series:
    cov_v = cov.values
    port_var = weights @ cov_v @ weights
    marginal = cov_v @ weights
    rc = weights * marginal / port_var
    return pd.Series(rc, index=cov.columns)
