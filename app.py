"""Portfolio Risk & Optimization Engine — Streamlit dashboard.

Run with: streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import backtest as bt
from src import data as dt
from src import options_overlay as ov
from src import optimize as opt
from src import risk

st.set_page_config(page_title="Portfolio Risk & Optimization Engine", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Pulling prices...")
def load_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    return dt.fetch_prices(list(tickers), start, end)


@st.cache_data(ttl=3600, show_spinner="Pulling benchmark...")
def load_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    return dt.fetch_benchmark(ticker, start, end)


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.title("Configuration")

tickers_input = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,MSFT,NVDA,GOOG,JPM")
tickers = tuple(t.strip().upper() for t in tickers_input.split(",") if t.strip())

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start", pd.Timestamp.today() - pd.DateOffset(years=5))
end_date = col2.date_input("End", pd.Timestamp.today())

benchmark_ticker = st.sidebar.selectbox("Benchmark", ["SPY", "QQQ", "^NSEI", "DIA"], index=0)
portfolio_value = st.sidebar.number_input("Portfolio value ($)", value=500_000, step=50_000)

st.sidebar.markdown("---")
st.sidebar.subheader("Constraints")
allow_short = st.sidebar.checkbox("Allow short selling", value=False)
max_weight = st.sidebar.slider("Max weight per stock", 0.10, 1.0, 0.40, 0.05)
cash_weight = st.sidebar.slider("Cash allocation", 0.0, 0.50, 0.0, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest")
rebalance_label = st.sidebar.selectbox("Rebalance frequency", ["Monthly", "Quarterly", "Annually"], index=1)
rebalance_freq = {"Monthly": "ME", "Quarterly": "QE", "Annually": "YE"}[rebalance_label]
lookback_days = st.sidebar.slider("Lookback window (days)", 60, 500, 252, 20)
cost_bps = st.sidebar.slider("Transaction cost (bps per rebalance turnover)", 0, 50, 5, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk")
alpha = 1 - st.sidebar.slider("VaR / CVaR confidence level", 0.90, 0.99, 0.95, 0.01)
rolling_window = st.sidebar.slider("Rolling risk window (days)", 60, 500, 250, 10)

if len(tickers) < 2:
    st.warning("Enter at least 2 tickers.")
    st.stop()

prices = load_prices(tickers, str(start_date), str(end_date))
returns = dt.compute_returns(prices)
bench_prices = load_benchmark(benchmark_ticker, str(start_date), str(end_date))

constraints = opt.Constraints(allow_short=allow_short, max_weight=max_weight, cash_weight=cash_weight)

cov = risk.shrinkage_cov(returns)
mu = returns.mean()

st.title("Portfolio Risk & Optimization Engine")
st.caption(
    "Multi-asset risk system: VaR/CVaR, rolling risk monitoring, walk-forward backtesting with "
    "transaction costs, and an equity-derivatives protective-put overlay."
)

tab_overview, tab_risk, tab_backtest, tab_options = st.tabs(
    ["Overview", "Risk Metrics", "Backtest Comparison", "Options Overlay"]
)

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tab_overview:
    strategy_choice = st.selectbox("Portfolio construction method", bt.STRATEGIES, index=0)
    if strategy_choice == "Equal Weight":
        weights = opt.equal_weight(len(tickers), cash_weight)
    elif strategy_choice == "Max Sharpe":
        weights = opt.max_sharpe(mu, cov, list(tickers), constraints)
    else:
        weights = opt.risk_parity(cov, list(tickers), constraints)

    w_series = pd.Series(weights, index=tickers)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Allocation")
        fig = go.Figure(data=[go.Pie(labels=w_series.index, values=w_series.values, hole=0.4)])
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.subheader("Normalized price performance")
        norm_prices = prices / prices.iloc[0]
        fig2 = go.Figure()
        for t in tickers:
            fig2.add_trace(go.Scatter(x=norm_prices.index, y=norm_prices[t], name=t))
        st.plotly_chart(fig2, width='stretch')

    st.subheader("Correlation matrix")
    corr = returns.corr()
    fig3 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0))
    st.plotly_chart(fig3, width='stretch')

# --------------------------------------------------------------------------
# Risk Metrics
# --------------------------------------------------------------------------
with tab_risk:
    port_returns = risk.portfolio_returns(returns, weights)

    st.subheader(f"VaR / CVaR at {int((1-alpha)*100)}% confidence")
    st.caption("Historical (empirical percentile) vs. Parametric (Normal-distribution assumption) — comparing the two shows how much the Normal assumption understates or overstates tail risk.")
    var_h, cvar_h = risk.var_historical(port_returns, alpha), risk.cvar_historical(port_returns, alpha)
    var_p, cvar_p = risk.var_parametric(port_returns, alpha), risk.cvar_parametric(port_returns, alpha)

    method_df = pd.DataFrame(
        {
            "Method": ["Historical", "Parametric"],
            "VaR (%)": [var_h, var_p],
            "CVaR / ES (%)": [cvar_h, cvar_p],
        }
    )
    method_df["VaR ($)"] = method_df["VaR (%)"] * portfolio_value
    method_df["CVaR ($)"] = method_df["CVaR / ES (%)"] * portfolio_value
    st.dataframe(method_df.style.format({"VaR (%)": "{:.2%}", "CVaR / ES (%)": "{:.2%}", "VaR ($)": "${:,.0f}", "CVaR ($)": "${:,.0f}"}), width='stretch')

    st.subheader("Rolling risk (production-style daily monitoring)")
    roll = risk.rolling_risk_metrics(port_returns, window=rolling_window, alpha=alpha)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=roll.index, y=roll["rolling_var"], name="Rolling VaR"))
    fig4.add_trace(go.Scatter(x=roll.index, y=roll["rolling_es"], name="Rolling ES / CVaR"))
    st.plotly_chart(fig4, width='stretch')

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=roll.index, y=roll["rolling_vol_annualized"], name="Rolling Ann. Vol"))
    st.plotly_chart(fig5, width='stretch')

    st.subheader("Risk contribution by asset (current weights)")
    st.caption("Ledoit-Wolf shrinkage covariance feeds this and every optimizer — it pulls the noisy sample covariance toward a structured target, more stable than raw sample covariance on realistic lookback windows.")
    rc = opt.risk_contributions(weights, cov)
    fig7 = go.Figure(data=[go.Bar(x=rc.index, y=rc.values * 100)])
    fig7.update_layout(yaxis_title="% of total portfolio variance")
    st.plotly_chart(fig7, width='stretch')

# --------------------------------------------------------------------------
# Backtest Comparison
# --------------------------------------------------------------------------
with tab_backtest:
    st.subheader("Walk-forward strategy comparison (no lookahead)")
    st.caption(
        f"Weights re-optimized at each {rebalance_label.lower()} rebalance using only the trailing "
        f"{lookback_days}-day window, then held until the next rebalance. Transaction costs = turnover x {cost_bps}bps."
    )

    with st.spinner("Running walk-forward backtests..."):
        cmp = bt.compare_strategies(
            prices,
            bench_prices,
            rebalance_freq=rebalance_freq,
            lookback_days=lookback_days,
            cost_bps=cost_bps,
            constraints=constraints,
        )

    fig8 = go.Figure()
    for col in cmp["cumulative_returns"].columns:
        fig8.add_trace(go.Scatter(x=cmp["cumulative_returns"].index, y=cmp["cumulative_returns"][col], name=col))
    fig8.update_layout(yaxis_title="Cumulative growth of $1")
    st.plotly_chart(fig8, width='stretch')

    st.subheader("Drawdowns")
    fig9 = go.Figure()
    for strat, res in cmp["results"].items():
        dd = bt.drawdown_series(res["returns"])
        fig9.add_trace(go.Scatter(x=dd.index, y=dd.values * 100, name=strat))
    fig9.update_layout(yaxis_title="Drawdown (%)")
    st.plotly_chart(fig9, width='stretch')

    st.subheader("Performance statistics")
    stats_fmt = cmp["stats"].copy()
    for c in ["CAGR", "Ann. Volatility", "Max Drawdown", "Avg Turnover / Rebalance", "Total Cost Drag"]:
        stats_fmt[c] = stats_fmt[c].map("{:.2%}".format)
    for c in ["Sharpe", "Sortino", "Calmar"]:
        stats_fmt[c] = stats_fmt[c].map("{:.2f}".format)
    st.dataframe(stats_fmt, width='stretch')

# --------------------------------------------------------------------------
# Options Overlay
# --------------------------------------------------------------------------
with tab_options:
    st.subheader("Protective put overlay — equity derivatives risk management")
    st.info(
        "Free historical options-chain data isn't available, so hedge pricing uses Black-Scholes "
        "with trailing realized volatility as the sigma input — standard when live chains aren't accessible."
    )

    oc1, oc2, oc3 = st.columns(3)
    moneyness = oc1.slider("Put strike moneyness (% of spot)", 0.85, 1.00, 0.97, 0.01)
    tenor_days = oc2.slider("Tenor (trading days)", 5, 90, 30, 5)
    vol_window = oc3.slider("Realized vol window (days)", 10, 90, 30, 5)

    spot = prices.iloc[-1]
    sigmas = ov.realized_vol(returns, window=vol_window)
    dollar_alloc = pd.Series(weights * portfolio_value, index=tickers)

    put_book = ov.build_protective_put_book(list(tickers), spot, dollar_alloc, sigmas, moneyness, tenor_days)
    greeks_summary = ov.portfolio_greeks_summary(dollar_alloc, put_book)

    st.subheader("Put positions")
    st.dataframe(
        put_book.style.format({
            "spot": "${:,.2f}", "strike": "${:,.2f}", "sigma": "{:.1%}", "shares": "{:,.1f}",
            "premium_per_share": "${:,.2f}", "premium_cost": "${:,.0f}",
            "delta_$": "${:,.0f}", "gamma_$": "${:,.0f}", "vega_$": "${:,.0f}", "theta_$": "${:,.0f}",
        }),
        width='stretch',
    )

    st.subheader("Portfolio-level Greeks (stock + puts)")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Delta ($)", f"${greeks_summary['Delta ($)']:,.0f}")
    g2.metric("Gamma ($)", f"${greeks_summary['Gamma ($)']:,.0f}")
    g3.metric("Vega ($/vol pt)", f"${greeks_summary['Vega ($ per 1 vol pt)']:,.0f}")
    g4.metric("Theta ($/day)", f"${greeks_summary['Theta ($ per day)']:,.0f}")

    st.subheader(f"VaR / CVaR before vs. after hedge ({tenor_days}-day horizon, Monte Carlo)")
    with st.spinner("Simulating hedge P&L distribution..."):
        hedge_res = ov.simulate_hedge_var(returns, weights, portfolio_value, put_book, tenor_days, alpha)

    hc1, hc2, hc3 = st.columns(3)
    hc1.metric("VaR — Unhedged", f"${hedge_res['VaR_unhedged']:,.0f}")
    hc2.metric(
        "VaR — Hedged",
        f"${hedge_res['VaR_hedged']:,.0f}",
        delta=f"{-hedge_res['var_reduction_pct']:.1%}",
        delta_color="inverse",
    )
    hc3.metric("Premium cost", f"${hedge_res['premium_cost']:,.0f}")

    hc4, hc5 = st.columns(2)
    hc4.metric("CVaR / ES — Unhedged", f"${hedge_res['CVaR_unhedged']:,.0f}")
    hc5.metric("CVaR / ES — Hedged", f"${hedge_res['CVaR_hedged']:,.0f}")

    fig11 = go.Figure()
    fig11.add_trace(go.Histogram(x=hedge_res["unhedged_pnl"], name="Unhedged P&L", opacity=0.6, nbinsx=80))
    fig11.add_trace(go.Histogram(x=hedge_res["hedged_pnl"], name="Hedged P&L", opacity=0.6, nbinsx=80))
    fig11.update_layout(barmode="overlay", xaxis_title="P&L ($)", yaxis_title="Frequency")
    st.plotly_chart(fig11, width='stretch')

    st.caption(
        "Near-ATM puts (moneyness closer to 1.0) cut both VaR and CVaR but cost more premium; "
        "far-OTM puts are cheap but can leave VaR roughly unchanged since they only pay off beyond "
        "the strike — the premium drag can outweigh the thin protection at the 95% level. This "
        "trade-off is exactly what a derivatives desk prices when structuring a hedge."
    )
