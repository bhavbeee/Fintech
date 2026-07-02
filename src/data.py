"""Price/return data access for the risk & portfolio engine.

All downloads are cached to disk (data/cache/*.csv) so repeated runs of the
Streamlit app don't re-hit yfinance and don't break when offline.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE_RATE_ANNUAL = 0.04  # flat proxy, used where a risk-free rate is needed


def _cache_key(tickers: list[str], start: str, end: str) -> Path:
    raw = ",".join(sorted(tickers)) + start + end
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    return CACHE_DIR / f"prices_{h}.csv"


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Daily adjusted close prices, tickers as columns, dates as index."""
    tickers = list(tickers)
    cache_path = _cache_key(tickers, start, end)
    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if set(tickers).issubset(set(df.columns)):
            return df[tickers]

    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # single ticker -> flat columns
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all").ffill().dropna()
    prices.to_csv(cache_path)
    return prices[tickers]


def fetch_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    df = fetch_prices([ticker], start, end)
    return df[ticker]


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def align(*series_or_frames: pd.DataFrame | pd.Series):
    """Inner-join align a set of frames/series on their date index."""
    combined = pd.concat(series_or_frames, axis=1, join="inner")
    n_cols = [s.shape[1] if isinstance(s, pd.DataFrame) else 1 for s in series_or_frames]
    out, i = [], 0
    for n in n_cols:
        out.append(combined.iloc[:, i] if n == 1 else combined.iloc[:, i : i + n])
        i += n
    return tuple(out)
