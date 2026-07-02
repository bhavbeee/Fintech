"""Equity derivatives overlay: Black-Scholes pricing/Greeks, a rolling
protective-put hedge book, and a Monte Carlo comparison of portfolio VaR
before vs. after the hedge.

Data note: free historical options-chain data isn't available, so hedge
pricing uses Black-Scholes with trailing realized volatility as the sigma
input (standard practice for theoretical/backtest pricing when live chains
aren't accessible). This is stated explicitly rather than silently assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "put") -> float:
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0) if option_type == "put" else max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "put") -> dict:
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1 vol point (1%)

    if option_type == "put":
        delta = norm.cdf(d1) - 1
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        ) / TRADING_DAYS
    else:
        delta = norm.cdf(d1)
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        ) / TRADING_DAYS

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def realized_vol(returns: pd.DataFrame, window: int = 30) -> pd.Series:
    return returns.tail(window).std() * np.sqrt(TRADING_DAYS)


def build_protective_put_book(
    tickers: list[str],
    spot: pd.Series,
    dollar_alloc: pd.Series,
    sigmas: pd.Series,
    moneyness: float = 0.95,
    tenor_days: int = 30,
    r: float = 0.04,
) -> pd.DataFrame:
    """One protective put per name, struck at `moneyness` * spot, covering
    the shares implied by each name's dollar allocation."""
    T = tenor_days / TRADING_DAYS
    rows = []
    for t in tickers:
        S0 = float(spot[t])
        K = S0 * moneyness
        sigma = float(sigmas[t])
        shares = dollar_alloc[t] / S0
        premium = bs_price(S0, K, T, r, sigma, "put")
        g = bs_greeks(S0, K, T, r, sigma, "put")
        rows.append(
            {
                "ticker": t,
                "spot": S0,
                "strike": K,
                "sigma": sigma,
                "shares": shares,
                "premium_per_share": premium,
                "premium_cost": premium * shares,
                "delta_$": g["delta"] * shares * S0,
                "gamma_$": g["gamma"] * shares * S0**2 / 100,
                "vega_$": g["vega"] * shares,
                "theta_$": g["theta"] * shares,
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def portfolio_greeks_summary(stock_delta_dollar: pd.Series, put_book: pd.DataFrame) -> dict:
    """Aggregate book-level Greeks: stock legs contribute delta=$notional only;
    puts contribute all four Greeks."""
    total_delta = stock_delta_dollar.sum() + put_book["delta_$"].sum()
    return {
        "Delta ($)": total_delta,
        "Gamma ($)": put_book["gamma_$"].sum(),
        "Vega ($ per 1 vol pt)": put_book["vega_$"].sum(),
        "Theta ($ per day)": put_book["theta_$"].sum(),
    }


def simulate_hedge_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    put_book: pd.DataFrame,
    tenor_days: int = 30,
    alpha: float = 0.05,
    n_sims: int = 20_000,
    seed: int = 42,
) -> dict:
    """Monte Carlo the portfolio `tenor_days` ahead, compare unhedged vs.
    protective-put-hedged P&L distributions -> VaR/CVaR before and after."""
    tickers = list(returns.columns)
    mu = returns.mean().values
    cov = returns.cov().values
    dollar_alloc = weights * portfolio_value
    spot = put_book["spot"].reindex(tickers).values
    strikes = put_book["strike"].reindex(tickers).values
    shares = put_book["shares"].reindex(tickers).values
    premium_cost = put_book["premium_cost"].sum()

    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov + np.eye(len(cov)) * 1e-12)

    cum_returns = np.zeros((n_sims, len(tickers)))
    for _ in range(tenor_days):
        z = rng.standard_normal((n_sims, len(tickers)))
        daily = mu + z @ L.T
        cum_returns += daily
    terminal_spot = spot * (1 + cum_returns)

    unhedged_terminal = (terminal_spot * shares).sum(axis=1)
    unhedged_pnl = unhedged_terminal - portfolio_value

    put_payoff = np.maximum(strikes - terminal_spot, 0) * shares
    hedged_pnl = unhedged_pnl + put_payoff.sum(axis=1) - premium_cost

    def var_cvar(pnl):
        loss = -pnl
        var = np.percentile(loss, (1 - alpha) * 100)
        tail = loss[loss >= var]
        cvar = tail.mean() if len(tail) else var
        return var, cvar

    var_u, cvar_u = var_cvar(unhedged_pnl)
    var_h, cvar_h = var_cvar(hedged_pnl)

    return {
        "unhedged_pnl": unhedged_pnl,
        "hedged_pnl": hedged_pnl,
        "VaR_unhedged": var_u,
        "CVaR_unhedged": cvar_u,
        "VaR_hedged": var_h,
        "CVaR_hedged": cvar_h,
        "premium_cost": premium_cost,
        "var_reduction_pct": (var_u - var_h) / var_u if var_u else 0.0,
    }
