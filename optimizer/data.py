"""
Handles pulling price data and turning it into the statistical inputs
(expected returns vector and covariance matrix) that the Markowitz
optimizer needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS_PER_YEAR = 252


def fetch_price_history(
    tickers: list[str],
    lookback_years: float = 5.0,
) -> pd.DataFrame:
    """
    Return a DataFrame indexed by date, one column per ticker.
    Tickers that fail to download or have almost no history are dropped.
    """
    period = f"{int(lookback_years * 365)}d"

    raw = yf.download(
        tickers,
        period=period,
        auto_adjust=True,  # adjusts for splits/dividends automatically
        progress=False,
    )

    if raw.empty:
        raise ValueError("No price data returned for any ticker.")

    # yfinance returns a MultiIndex column structure when given multiple
    # tickers, and a flat structure for a single ticker. Normalize both
    # to a simple DataFrame of close prices, one column per ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    # Drop tickers with too little usable history to compute stable stats.
    min_required_days = int(TRADING_DAYS_PER_YEAR * 0.5)  # at least ~6 months
    valid_counts = prices.count()
    dropped = valid_counts[valid_counts < min_required_days].index.tolist()
    if dropped:
        prices = prices.drop(columns=dropped)

    if prices.empty:
        raise ValueError(
            "All requested tickers had insufficient price history."
        )

    # Forward-fill small gaps (holidays/missing days), then drop any
    # remaining rows with NaNs (e.g. leading rows before a ticker existed).
    prices = prices.ffill().dropna()

    return prices, dropped


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a price DataFrame into daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a price DataFrame into daily simple (arithmetic) returns.

    Simple returns are used (instead of log returns) for portfolio-level
    backtesting: a portfolio's simple return each day is the weighted sum
    of each asset's simple return, which is not true for log returns.
    This is what drawdown calculations need.
    """
    return prices.pct_change().dropna()


def compute_annualized_stats(
    returns: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Turn daily log returns into annualized expected returns (mu) and
    an annualized covariance matrix (Sigma).
    """
    mu = returns.mean() * TRADING_DAYS_PER_YEAR
    sigma = returns.cov() * TRADING_DAYS_PER_YEAR
    return mu, sigma


def get_portfolio_inputs(
    tickers: list[str],
    lookback_years: float = 5.0,
) -> dict:

    prices, dropped = fetch_price_history(tickers, lookback_years)
    returns = compute_log_returns(prices)
    mu, sigma = compute_annualized_stats(returns)
    simple_returns = compute_simple_returns(prices)

    return {
        "mu": mu,
        "sigma": sigma,
        "tickers": list(prices.columns),
        "dropped_tickers": dropped,
        "prices": prices,
        "simple_returns": simple_returns,
    }
