# Portfolio Risk & Optimization Engine

A multi-asset portfolio risk and construction system covering the core
toolkit used on institutional risk/QIS/derivatives desks: VaR/CVaR,
Ledoit-Wolf shrinkage covariance, walk-forward-backtested portfolio
construction with transaction costs, and an equity-derivatives
protective-put overlay with Greeks and Monte Carlo hedge-effectiveness
analysis.

Live dashboard built with Streamlit. All data is free/public (`yfinance`).

## Why this exists

Most portfolio-optimizer projects stop at "here are the optimal weights."
This one is built around the questions a risk desk actually asks:

- Not just "portfolio VaR is $12k" — does the **Historical** (empirical
  percentile) and **Parametric** (Normal-distribution) method agree, and if
  not, what does that say about the return distribution?
- Not just a backtest-fitted "optimal" weight vector — what happens if you
  **actually trade this strategy** for 5 years, quarter by quarter, paying
  transaction costs, without looking into the future?
- Not just portfolio-level VaR — what does it look like **after a
  derivatives desk hedges it** with protective puts, and how does that
  change with strike selection?

## Methodology & key design decisions

**Walk-forward backtesting, not lookahead-fitted weights.** At each
rebalance date, portfolio weights are computed using *only* the trailing
lookback window of returns, then held fixed until the next rebalance.
Fitting weights on the full historical window and "backtesting" them on the
same window is a common mistake — it silently leaks future information into
the allocation decision.

**Ledoit-Wolf shrinkage covariance**, not raw sample covariance, feeds every
optimizer. Sample covariance is noisy on realistic lookback windows (e.g. 1
year of daily data for 5-10 assets); shrinkage pulls it toward a structured
target and is standard practice for portfolio construction in production.

**Risk parity actually equalizes risk contribution**, not dollar weight —
the optimizer minimizes the dispersion of each asset's contribution to
total portfolio variance, verified against the closed-form risk-contribution
identity (`w_i * (Σw)_i / w'Σw`).

**The options overlay uses Black-Scholes with trailing realized volatility**,
not real historical options chains, because free historical chain data isn't
available. This is stated explicitly in the app rather than assumed
silently. Strike selection is exposed as a slider specifically to show the
real trade-off a derivatives desk prices: near-ATM puts cut VaR/CVaR more
but cost more premium; far-OTM puts are cheap but can leave headline VaR
roughly unchanged since they only pay off beyond the strike.

## Features

| Module | What it does |
|---|---|
| `src/data.py` | Price/return fetching (yfinance) with disk caching |
| `src/risk.py` | VaR & CVaR (historical, parametric), rolling risk metrics, Ledoit-Wolf shrinkage covariance, performance stats (CAGR, Sharpe, Sortino, Max DD, Calmar) |
| `src/optimize.py` | Equal-weight, max-Sharpe, risk-parity optimizers under no-short / per-name cap / cash-buffer constraints |
| `src/backtest.py` | Walk-forward rebalancing engine with transaction costs, benchmark comparison |
| `src/options_overlay.py` | Black-Scholes pricing & Greeks, protective-put book construction, portfolio-level Greek aggregation, Monte Carlo VaR/CVaR before vs. after hedge |
| `app.py` | Streamlit dashboard tying it all together |

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter any tickers, set constraints (max weight, cash buffer), choose a
rebalance frequency and transaction cost assumption, and the dashboard runs
the full pipeline: allocation → risk metrics → walk-forward backtest
comparison vs. a benchmark → options hedge overlay.

## Assumptions & limitations

- Weights are held fixed between rebalances for return calculation (no
  intra-period share-count drift tracking) — a standard simplification for
  this scale of backtest.
- Transaction costs are modeled as `turnover × cost_bps` at each rebalance,
  not a full market-impact model.
- The options overlay assumes European exercise, no dividends in the BS
  formula, and constant volatility over the option's life (trailing realized
  vol as the input) — a reasonable approximation, not a production pricing
  library.
- Free yfinance data occasionally has gaps/rate limits; the data layer caches
  to `data/cache/` to reduce repeated calls.
