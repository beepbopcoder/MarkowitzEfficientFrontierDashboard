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


def _fetch_one_ticker(ticker: str, period: str, retries: int = 1):
    """Fetch a single ticker's close prices, retrying once on failure.
    Returns (series_or_None, reason_string)."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            data = yf.download(
                ticker,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if data.empty:
                last_error = "no data returned for this date range"
                continue
            close = data["Close"]
            if isinstance(close, pd.DataFrame):  # occasional single-col MultiIndex
                close = close.iloc[:, 0]
            close.name = ticker
            return close, None
        except Exception as exc:  # network hiccup, throttling, bad symbol, etc.
            last_error = str(exc)
    return None, last_error


def fetch_price_history(
    tickers: list[str],
    lookback_years: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Return a DataFrame indexed by date, one column per ticker, plus a dict
    mapping any dropped ticker to the reason it was dropped.

    Tickers are fetched individually (rather than in one batch yf.download
    call) so a single ticker's transient failure — a throttled request, a
    dropped connection, etc. — doesn't silently take it out without
    explanation, and doesn't get masked by retrying the whole batch.
    """
    period = f"{int(lookback_years * 365)}d"
    min_required_days = int(TRADING_DAYS_PER_YEAR * 0.5)  # at least ~6 months

    series_list = []
    dropped: dict[str, str] = {}

    for ticker in tickers:
        series, reason = _fetch_one_ticker(ticker, period)
        if series is None:
            dropped[ticker] = reason or "unknown error"
            continue
        valid_days = series.count()
        if valid_days < min_required_days:
            dropped[ticker] = f"only {valid_days} valid trading days returned"
            continue
        series_list.append(series)

    if not series_list:
        raise ValueError("No price data returned for any ticker.")

    prices = pd.concat(series_list, axis=1)

    # Forward-fill small gaps (holidays/missing days), then drop any
    # remaining rows with NaNs (e.g. leading rows before a ticker existed).
    prices = prices.ffill().dropna()

    if prices.empty:
        raise ValueError(
            "All requested tickers had insufficient price history."
        )

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
