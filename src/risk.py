"""Risk measurement: VaR/CVaR (historical + parametric), rolling risk, and
Ledoit-Wolf shrinkage covariance estimation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    weights = np.asarray(weights)
    return (returns * weights).sum(axis=1)


# ---------------------------------------------------------------------------
# VaR / CVaR — each returns a POSITIVE number = fraction of portfolio value
# expected to be lost at the given confidence level.
# ---------------------------------------------------------------------------

def var_historical(port_returns: pd.Series, alpha: float = 0.05) -> float:
    return -np.percentile(port_returns.dropna(), alpha * 100)


def cvar_historical(port_returns: pd.Series, alpha: float = 0.05) -> float:
    r = port_returns.dropna()
    var = -np.percentile(r, alpha * 100)
    tail = r[r <= -var]
    return -tail.mean() if len(tail) else var


def var_parametric(port_returns: pd.Series, alpha: float = 0.05) -> float:
    from scipy.stats import norm

    mu, sigma = port_returns.mean(), port_returns.std()
    z = norm.ppf(alpha)
    return -(mu + z * sigma)


def cvar_parametric(port_returns: pd.Series, alpha: float = 0.05) -> float:
    from scipy.stats import norm

    mu, sigma = port_returns.mean(), port_returns.std()
    z = norm.ppf(alpha)
    es_z = -norm.pdf(z) / alpha
    return -(mu + sigma * es_z)


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def shrinkage_cov(returns: pd.DataFrame, annualize: bool = False) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage covariance — pulls the noisy sample covariance
    toward a structured target, standard practice vs. raw sample cov on
    short/noisy windows."""
    lw = LedoitWolf().fit(returns.values)
    cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    return cov * TRADING_DAYS if annualize else cov


# ---------------------------------------------------------------------------
# Rolling risk metrics — closer to what a production risk system reports
# daily rather than a single point-in-time number.
# ---------------------------------------------------------------------------

def rolling_risk_metrics(port_returns: pd.Series, window: int = 250, alpha: float = 0.05) -> pd.DataFrame:
    out = pd.DataFrame(index=port_returns.index)
    out["rolling_vol_annualized"] = port_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    out["rolling_var"] = port_returns.rolling(window).apply(
        lambda x: -np.percentile(x, alpha * 100), raw=True
    )

    def _es(x):
        var = -np.percentile(x, alpha * 100)
        tail = x[x <= -var]
        return -tail.mean() if len(tail) else var

    out["rolling_es"] = port_returns.rolling(window).apply(_es, raw=True)
    return out.dropna()


def performance_summary_stats(returns: pd.Series, rf_annual: float = 0.04) -> dict:
    ann_return = (1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - rf_annual) / ann_vol if ann_vol > 0 else np.nan
    downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (ann_return - rf_annual) / downside if downside and downside > 0 else np.nan
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan
    return {
        "CAGR": ann_return,
        "Ann. Volatility": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
    }
