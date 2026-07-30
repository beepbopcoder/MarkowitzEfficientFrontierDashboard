"""
Core Markowitz mean-variance optimization:
    - portfolio performance (expected return, volatility, Sharpe)
    - minimum-variance portfolio for a given target return
    - global minimum-variance portfolio
    - max-Sharpe ("tangency") portfolio
    - full efficient frontier (a curve of min-variance portfolios
      swept across a range of target returns)

All weights are long-only (w_i >= 0) and fully invested (sum(w) == 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_performance(
    weights: np.ndarray,
    mu: pd.Series,
    sigma: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> tuple[float, float, float]:
    """
    Given portfolio weights, return (expected_return, volatility, sharpe_ratio).
    """
    expected_return = float(np.dot(weights, mu))
    variance = float(np.dot(weights.T, np.dot(sigma.values, weights)))
    volatility = float(np.sqrt(variance))
    sharpe = (expected_return - risk_free_rate) / \
        volatility if volatility > 0 else 0.0
    return expected_return, volatility, sharpe


def _n_assets(mu: pd.Series) -> int:
    return len(mu)


def _base_constraints(n: int) -> list[dict]:
    """Weights must sum to 1 (fully invested, no leverage/cash)."""
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]


def _base_bounds(n: int, max_weight: float = 1.0) -> tuple:
    """Long-only: each weight between 0 and max_weight (default 1.0 = uncapped)."""
    return tuple((0.0, max_weight) for _ in range(n))


def _equal_weight_guess(n: int) -> np.ndarray:
    return np.repeat(1.0 / n, n)


def minimize_volatility_for_target_return(
    mu: pd.Series,
    sigma: pd.DataFrame,
    target_return: float,
    max_weight: float = 1.0,
) -> dict:
    """
    Find the lowest-volatility portfolio that achieves at least the
    target expected return. This is one point on the efficient frontier.

    max_weight caps how much can be allocated to any single asset
    (e.g. 0.4 = no more than 40% in one stock). Must satisfy
    max_weight * n_assets >= 1, or the fully-invested constraint becomes
    infeasible.
    """
    n = _n_assets(mu)

    def volatility(w):
        return np.sqrt(np.dot(w.T, np.dot(sigma.values, w)))

    constraints = _base_constraints(n) + [
        {"type": "eq", "fun": lambda w: np.dot(w, mu.values) - target_return}
    ]

    result = minimize(
        volatility,
        x0=_equal_weight_guess(n),
        method="SLSQP",
        bounds=_base_bounds(n, max_weight),
        constraints=constraints,
    )

    return {"weights": result.x, "success": result.success}


def global_minimum_variance_portfolio(
    mu: pd.Series, sigma: pd.DataFrame, max_weight: float = 1.0
) -> dict:
    """
    Find the absolute lowest-risk portfolio, with no return target
    """
    n = _n_assets(mu)

    def variance(w):
        return np.dot(w.T, np.dot(sigma.values, w))

    result = minimize(
        variance,
        x0=_equal_weight_guess(n),
        method="SLSQP",
        bounds=_base_bounds(n, max_weight),
        constraints=_base_constraints(n),
    )

    weights = result.x
    ret, vol, sharpe = portfolio_performance(weights, mu, sigma)
    return {
        "weights": weights,
        "expected_return": ret,
        "volatility": vol,
        "sharpe": sharpe,
        "success": result.success,
    }


def max_sharpe_portfolio(
    mu: pd.Series,
    sigma: pd.DataFrame,
    risk_free_rate: float = 0.02,
    max_weight: float = 1.0,
) -> dict:
    """
    Find the portfolio that maximizes the Sharpe ratio (tangent point to risk-free rate)
    """
    n = _n_assets(mu)

    def negative_sharpe(w):
        ret, vol, _ = portfolio_performance(w, mu, sigma, risk_free_rate)
        if vol == 0:
            return 0.0
        return -(ret - risk_free_rate) / vol

    result = minimize(
        negative_sharpe,
        x0=_equal_weight_guess(n),
        method="SLSQP",
        bounds=_base_bounds(n, max_weight),
        constraints=_base_constraints(n),
    )

    weights = result.x
    ret, vol, sharpe = portfolio_performance(
        weights, mu, sigma, risk_free_rate)
    return {
        "weights": weights,
        "expected_return": ret,
        "volatility": vol,
        "sharpe": sharpe,
        "success": result.success,
    }


def compute_efficient_frontier(
    mu: pd.Series,
    sigma: pd.DataFrame,
    n_points: int = 50,
    max_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
    target_return, volatility, sharpe, and one weight column per asset.
    """
    gmv = global_minimum_variance_portfolio(mu, sigma, max_weight)
    lowest_return = gmv["expected_return"]
    highest_return = float(mu.max())

    target_returns = np.linspace(lowest_return, highest_return, n_points)

    rows = []
    for target in target_returns:
        result = minimize_volatility_for_target_return(
            mu, sigma, target, max_weight)
        if not result["success"]:
            continue
        weights = result["weights"]
        ret, vol, sharpe = portfolio_performance(weights, mu, sigma)
        row = {"target_return": ret, "volatility": vol, "sharpe": sharpe}
        row.update({f"weight_{ticker}": w for ticker,
                   w in zip(mu.index, weights)})
        rows.append(row)

    return pd.DataFrame(rows)
