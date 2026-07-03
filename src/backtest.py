"""Walk-forward backtest engine.

Critical design choice: at each rebalance date, weights are computed using
ONLY the trailing lookback window of returns (no lookahead). Weights are then
held fixed until the next rebalance, and turnover/transaction costs are
charged at each rebalance based on the change in weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import optimize as opt
from . import risk

STRATEGIES = ["Equal Weight", "Max Sharpe", "Risk Parity"]


def _target_weights(strategy: str, mu: pd.Series, cov: pd.DataFrame, tickers: list[str], constraints: opt.Constraints, rf_annual: float) -> np.ndarray:
    if strategy == "Equal Weight":
        return opt.equal_weight(len(tickers), constraints.cash_weight)
    if strategy == "Max Sharpe":
        return opt.max_sharpe(mu, cov, tickers, constraints, rf_annual)
    if strategy == "Risk Parity":
        return opt.risk_parity(cov, tickers, constraints)
    raise ValueError(f"Unknown strategy: {strategy}")


def walk_forward_backtest(
    prices: pd.DataFrame,
    strategy: str,
    rebalance_freq: str = "QE",
    lookback_days: int = 252,
    cost_bps: float = 5.0,
    constraints: opt.Constraints | None = None,
    rf_annual: float = 0.04,
) -> dict:
    constraints = constraints or opt.Constraints()
    tickers = list(prices.columns)
    returns = prices.pct_change().dropna()

    rebalance_dates = pd.date_range(returns.index[0], returns.index[-1], freq=rebalance_freq)
    rebalance_dates = [d for d in rebalance_dates if d in returns.index or True]
    # snap each rebalance date to the nearest available trading date on/after it
    snapped = []
    for d in rebalance_dates:
        idx = returns.index[returns.index >= d]
        if len(idx):
            snapped.append(idx[0])
    rebalance_dates = sorted(set(snapped))
    rebalance_dates = [d for d in rebalance_dates if len(returns.loc[:d]) >= lookback_days]

    if not rebalance_dates:
        raise ValueError("Not enough history for the requested lookback window.")

    daily_returns = pd.Series(index=returns.index, dtype=float)
    weights_history = {}
    turnover_history = {}
    prev_weights = np.zeros(len(tickers))

    for i, reb_date in enumerate(rebalance_dates):
        window = returns.loc[:reb_date].tail(lookback_days)
        cov = risk.shrinkage_cov(window)
        mu = window.mean()

        weights = _target_weights(strategy, mu, cov, tickers, constraints, rf_annual)

        turnover = float(np.sum(np.abs(weights - prev_weights)))
        cost = turnover * (cost_bps / 10_000)

        period_end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else returns.index[-1]
        period_mask = (returns.index > reb_date) if i > 0 else (returns.index >= reb_date)
        period_mask &= returns.index <= period_end

        period_returns = returns.loc[period_mask] @ weights
        if len(period_returns):
            period_returns.iloc[0] -= cost  # charge transaction cost on the rebalance day
            daily_returns.loc[period_returns.index] = period_returns.values

        weights_history[reb_date] = weights
        turnover_history[reb_date] = turnover
        prev_weights = weights

    daily_returns = daily_returns.dropna()
    weights_df = pd.DataFrame(weights_history, index=tickers).T
    turnover_s = pd.Series(turnover_history, name="turnover")

    return {
        "returns": daily_returns,
        "weights": weights_df,
        "turnover": turnover_s,
        "avg_turnover": turnover_s.mean(),
        "total_cost_drag": (turnover_s * cost_bps / 10_000).sum(),
    }


def compare_strategies(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series | None = None,
    strategies: list[str] | None = None,
    **kwargs,
) -> dict:
    strategies = strategies or STRATEGIES
    results = {}
    cum_returns = {}
    stats = {}

    for strat in strategies:
        res = walk_forward_backtest(prices, strat, **kwargs)
        results[strat] = res
        cum_returns[strat] = (1 + res["returns"]).cumprod()
        stats[strat] = risk.performance_summary_stats(res["returns"])
        stats[strat]["Avg Turnover / Rebalance"] = res["avg_turnover"]
        stats[strat]["Total Cost Drag"] = res["total_cost_drag"]

    if benchmark_prices is not None:
        bench_returns = benchmark_prices.pct_change().dropna()
        bench_returns = bench_returns.loc[bench_returns.index.isin(cum_returns[strategies[0]].index) | True]
        common_idx = cum_returns[strategies[0]].index.intersection(bench_returns.index)
        bench_returns = bench_returns.loc[common_idx]
        cum_returns["Benchmark"] = (1 + bench_returns).cumprod()
        bench_stats = risk.performance_summary_stats(bench_returns)
        bench_stats["Avg Turnover / Rebalance"] = 0.0
        bench_stats["Total Cost Drag"] = 0.0
        stats["Benchmark"] = bench_stats

    cum_df = pd.DataFrame(cum_returns)
    stats_df = pd.DataFrame(stats).T
    return {"results": results, "cumulative_returns": cum_df, "stats": stats_df}


def drawdown_series(returns: pd.Series) -> pd.Series:
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    return cum / running_max - 1
