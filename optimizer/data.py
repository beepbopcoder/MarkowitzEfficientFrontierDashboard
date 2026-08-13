"""
optimizer/data.py

Handles pulling price data and turning it into the statistical inputs
(expected returns vector and covariance matrix) that the Markowitz
optimizer needs.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS_PER_YEAR = 252
MIN_REQUIRED_DAYS = int(TRADING_DAYS_PER_YEAR * 0.5)  # at least ~6 months


def _fetch_one_ticker(
    ticker: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
    retries: int = 1,
) -> tuple[pd.Series | None, str | None]:
    """
    Fetch a single ticker's adjusted close prices, retrying once on failure.

    Fetching one ticker at a time with threads=False avoids a failure mode
    in batched multi-ticker yf.download calls: Yahoo's endpoint can
    consistently drop one specific ticker's data in a threaded batch
    request (returning an all-NaN column) even though that ticker is
    perfectly valid on its own. Per-ticker fetching sidesteps this and
    also lets us report the *actual* reason a ticker failed, rather than
    lumping every failure into a generic "insufficient history" message.

    Returns (series_or_None, reason_string_or_None).
    """
    last_error = None
    for _ in range(retries + 1):
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
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
            close = close.dropna()
            if close.shape[0] < MIN_REQUIRED_DAYS:
                last_error = f"only {close.shape[0]} valid trading days returned"
                continue
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
    Fetch adjusted close prices for the given tickers, one ticker at a time.

    Returns (prices, drop_reasons):
        prices: DataFrame indexed by date, one column per successfully
            fetched ticker, aligned on common trading days.
        drop_reasons: dict of ticker -> reason string, for any ticker
            that failed to fetch or had insufficient history.
    """
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=int(lookback_years * 365))

    series_list = []
    drop_reasons: dict[str, str] = {}
    for ticker in tickers:
        series, reason = _fetch_one_ticker(ticker, start_date, end_date)
        if series is None:
            drop_reasons[ticker] = reason or "unknown error"
        else:
            series_list.append(series)

    if not series_list:
        raise ValueError("No price data returned for any ticker.")

    prices = pd.concat(series_list, axis=1)
    # Forward-fill small gaps (holidays/missing days), then drop any
    # remaining rows with NaNs (e.g. leading rows before a ticker existed).
    prices = prices.ffill().dropna()

    if prices.empty:
        raise ValueError(
            "All requested tickers had insufficient price history.")

    return prices, drop_reasons


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
    """
    End-to-end convenience function: given a list of tickers, return
    everything the optimizer needs.

    Returns a dict with:
        mu: pd.Series of annualized expected returns, indexed by ticker
        sigma: pd.DataFrame annualized covariance matrix
        tickers: list of tickers actually used (after dropping bad ones)
        dropped_tickers: dict of ticker -> reason, for tickers excluded
        prices: raw price DataFrame (useful for the UI, e.g. plotting)
        simple_returns: daily simple returns, used for drawdown analysis
    """
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
