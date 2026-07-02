"""Risk measurement: VaR/CVaR (3 methods), rolling risk, covariance estimation,
and PCA-based factor risk decomposition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

TRADING_DAYS = 252


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    weights = np.asarray(weights)
    return (returns * weights).sum(axis=1)


# ---------------------------------------------------------------------------
# VaR / CVaR — three methods, each returns a POSITIVE number = fraction of
# portfolio value expected to be lost at the given confidence level.
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


def var_monte_carlo(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
    n_sims: int = 20_000,
    horizon: int = 1,
    seed: int = 42,
) -> tuple[float, float]:
    """Simulate correlated daily returns via Cholesky, aggregate to `horizon`
    days, return (VaR, CVaR) as positive loss fractions."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov + np.eye(len(cov)) * 1e-12)
    sims = np.zeros(n_sims)
    for h in range(horizon):
        z = rng.standard_normal((n_sims, len(mu)))
        daily = mu + z @ L.T
        sims += daily @ weights
    var = -np.percentile(sims, alpha * 100)
    tail = sims[sims <= -var]
    cvar = -tail.mean() if len(tail) else var
    return var, cvar


def dollar_var(pct_var: float, portfolio_value: float) -> float:
    return pct_var * portfolio_value


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def sample_cov(returns: pd.DataFrame, annualize: bool = False) -> pd.DataFrame:
    cov = returns.cov()
    return cov * TRADING_DAYS if annualize else cov


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


# ---------------------------------------------------------------------------
# PCA factor risk decomposition
# ---------------------------------------------------------------------------

def pca_factor_decomposition(
    returns: pd.DataFrame, weights: np.ndarray, n_factors: int = 3
) -> pd.DataFrame:
    """Decompose total portfolio variance into contributions from the top
    `n_factors` principal components of asset returns plus a residual bucket.

    The first PC of an equity return panel is almost always a broad "market"
    factor (all loadings same sign); later PCs pick up sector/style tilts.
    This mirrors a statistical-factor risk model (Barra-style) without
    requiring external factor data.
    """
    weights = np.asarray(weights)
    n_factors = min(n_factors, returns.shape[1] - 1)

    pca = PCA(n_components=n_factors)
    pca.fit(returns.values)

    total_var = float(weights @ returns.cov().values @ weights)

    rows = []
    explained = 0.0
    for k in range(n_factors):
        loading = float(weights @ pca.components_[k])
        factor_var = float(pca.explained_variance_[k])
        contribution = loading**2 * factor_var
        explained += contribution
        rows.append({"factor": f"PC{k + 1}", "variance_contribution": contribution})

    residual = max(total_var - explained, 0.0)
    rows.append({"factor": "Residual / Idiosyncratic", "variance_contribution": residual})

    df = pd.DataFrame(rows)
    df["pct_of_total_risk"] = df["variance_contribution"] / total_var
    df["volatility_contribution_annualized"] = np.sqrt(
        np.maximum(df["variance_contribution"], 0) * TRADING_DAYS
    )
    return df


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
