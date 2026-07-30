"""
optimizer/risk.py

Historical risk analytics that operate on realized price/return data,
as opposed to optimization.py which works with the forward-looking
mu/sigma statistical inputs.

Currently covers: portfolio drawdown (largest historical peak-to-trough
decline for a given set of weights).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_growth_curve(
    weights: np.ndarray,
    simple_returns: pd.DataFrame,
) -> pd.Series:
    """
    Given portfolio weights and a DataFrame of daily simple returns
    (one column per asset, same order as weights), compute the growth
    of a hypothetical $1 invested at the start of the period.

    Portfolio daily return = weighted sum of each asset's daily return.
    This only works correctly with simple returns, not log returns,
    since simple returns are what combine linearly across assets.
    """
    portfolio_daily_returns = simple_returns.dot(weights)
    growth = (1.0 + portfolio_daily_returns).cumprod()
    return growth


def drawdown_series(growth_curve: pd.Series) -> pd.Series:
    """
    Given a cumulative growth curve, compute the drawdown at each point
    in time: the percentage drop from the running peak so far.

    A drawdown of -0.20 means the portfolio was 20% below its
    highest-ever value at that point.
    """
    running_peak = growth_curve.cummax()
    drawdown = growth_curve / running_peak - 1.0
    return drawdown


def max_drawdown(growth_curve: pd.Series) -> float:
    """
    The single worst peak-to-trough decline over the period, as a
    negative fraction (e.g. -0.35 for a 35% drawdown).
    """
    return float(drawdown_series(growth_curve).min())


def portfolio_drawdown_analysis(
    weights: np.ndarray,
    simple_returns: pd.DataFrame,
) -> dict:
    """
    Convenience function: given weights and historical simple returns,
    return the growth curve, drawdown series, and max drawdown together.
    """
    growth = portfolio_growth_curve(weights, simple_returns)
    dd = drawdown_series(growth)
    return {
        "growth_curve": growth,
        "drawdown_series": dd,
        "max_drawdown": float(dd.min()),
    }
