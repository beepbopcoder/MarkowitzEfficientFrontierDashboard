"""
app.py

Streamlit UI for PortfolioMaximizer.

Lets a user enter a list of tickers, pulls price history, computes the
Markowitz efficient frontier, and displays the frontier chart plus the
recommended (max-Sharpe) portfolio and its allocation.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from optimizer.data import get_portfolio_inputs
from optimizer.optimization import (
    compute_efficient_frontier,
    global_minimum_variance_portfolio,
    max_sharpe_portfolio,
    portfolio_performance,
)
from optimizer.risk import portfolio_drawdown_analysis

st.set_page_config(page_title="PortfolioMaximizer", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_portfolio_inputs(tickers: list[str], lookback_years: float) -> dict:
    """
    Cached wrapper around get_portfolio_inputs. Streamlit hashes the
    function's arguments to build the cache key, so identical tickers +
    lookback within the same hour will skip the yfinance download entirely.
    ttl=3600 (1 hour) keeps prices from going stale for too long while
    still avoiding redundant downloads during a single working session.
    """
    return get_portfolio_inputs(tickers, lookback_years)


st.title("PortfolioMaximizer")
st.caption(
    "Markowitz mean-variance portfolio optimizer: enter a set of tickers "
    "and find the efficient frontier, the best risk-adjusted portfolio, and a lower risk alternative."
)

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Inputs")

    default_tickers = "AAPL, MSFT, GOOGL, AMZN, JPM"
    tickers_raw = st.text_input(
        "Tickers (comma-separated)",
        value=default_tickers,
    )
    tickers = sorted({t.strip().upper()
                     for t in tickers_raw.split(",") if t.strip()})

    lookback_years = st.slider(
        "Lookback period (years)", min_value=1.0, max_value=10.0, value=5.0, step=0.5
    )

    risk_free_rate_pct = st.number_input(
        "Risk-free rate (%)",
        min_value=0.0,
        max_value=15.0,
        value=2.0,
        step=0.25,
        format="%.2f",
    )
    risk_free_rate = risk_free_rate_pct / 100.0

    n_frontier_points = st.slider(
        "Frontier resolution (points)", min_value=20, max_value=100, value=50
    )

    max_weight_pct = st.slider(
        "Max weight per asset (%)",
        min_value=10,
        max_value=100,
        value=100,
        step=5,
        help="Caps how much of the portfolio can go into any single stock. "
        "100% means uncapped (standard Markowitz).",
    )
    max_weight = max_weight_pct / 100.0

    run_button = st.button("Run optimization", type="primary")

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
if run_button:
    if len(tickers) < 2:
        st.error("Enter at least 2 tickers to build a diversified portfolio.")
        st.stop()

    with st.spinner("Fetching price history..."):
        try:
            inputs = load_portfolio_inputs(tickers, lookback_years)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    mu, sigma = inputs["mu"], inputs["sigma"]
    used_tickers = inputs["tickers"]

    if inputs["dropped_tickers"]:
        st.warning(
            f"Dropped due to insufficient history: {', '.join(inputs['dropped_tickers'])}"
        )

    if len(used_tickers) < 2:
        st.error("Fewer than 2 tickers had usable data. Try different tickers.")
        st.stop()

    if max_weight * len(used_tickers) < 1.0:
        st.error(
            f"Max weight of {max_weight_pct}% per asset is too restrictive for "
            f"{len(used_tickers)} assets — it's mathematically impossible to "
            f"reach 100% invested. Raise the max weight or add more tickers."
        )
        st.stop()

    with st.spinner("Solving Markowitz optimization..."):
        frontier_df = compute_efficient_frontier(
            mu, sigma, n_points=n_frontier_points, max_weight=max_weight
        )
        max_sharpe = max_sharpe_portfolio(
            mu, sigma, risk_free_rate, max_weight)
        gmv = global_minimum_variance_portfolio(mu, sigma, max_weight)

    # -----------------------------------------------------------------------
    # Efficient frontier chart
    # -----------------------------------------------------------------------
    st.subheader("Efficient Frontier")

    fig = go.Figure()

    # The frontier curve itself, colored by Sharpe ratio.
    fig.add_trace(
        go.Scatter(
            x=frontier_df["volatility"],
            y=frontier_df["target_return"],
            mode="markers",
            marker=dict(
                size=6,
                color=frontier_df["sharpe"],
                colorscale="Viridis",
                colorbar=dict(title="Sharpe"),
            ),
            name="Efficient Frontier",
            hovertemplate=(
                "Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>"
            ),
        )
    )

    # Individual assets, for context.
    for ticker in used_tickers:
        asset_vol = float(np.sqrt(sigma.loc[ticker, ticker]))
        asset_ret = float(mu.loc[ticker])
        fig.add_trace(
            go.Scatter(
                x=[asset_vol],
                y=[asset_ret],
                mode="markers+text",
                marker=dict(size=10, color="gray", symbol="diamond"),
                text=[ticker],
                textposition="top center",
                name=ticker,
                showlegend=False,
            )
        )

    # Highlight max-Sharpe portfolio.
    fig.add_trace(
        go.Scatter(
            x=[max_sharpe["volatility"]],
            y=[max_sharpe["expected_return"]],
            mode="markers",
            marker=dict(size=15, color="red", symbol="star"),
            name="Max Sharpe (recommended)",
        )
    )

    # Highlight global min-variance portfolio.
    fig.add_trace(
        go.Scatter(
            x=[gmv["volatility"]],
            y=[gmv["expected_return"]],
            mode="markers",
            marker=dict(size=14, color="blue", symbol="square"),
            name="Min Volatility",
        )
    )

    fig.update_layout(
        xaxis_title="Volatility (annualized std. dev.)",
        yaxis_title="Expected Return (annualized)",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # Recommended portfolio summary
    # -----------------------------------------------------------------------
    st.subheader("Recommended Portfolio (Max Sharpe)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Annual Return",
                f"{max_sharpe['expected_return']:.2%}")
    col2.metric("Annual Volatility", f"{max_sharpe['volatility']:.2%}")
    col3.metric("Sharpe Ratio", f"{max_sharpe['sharpe']:.2f}")

    weights_df = pd.DataFrame(
        {
            "Ticker": used_tickers,
            "Weight": max_sharpe["weights"],
        }
    ).sort_values("Weight", ascending=False)
    weights_df = weights_df[weights_df["Weight"] > 0.001]  # hide near-zero
    weights_df["Weight"] = weights_df["Weight"].map(lambda w: f"{w:.1%}")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.dataframe(weights_df, hide_index=True, use_container_width=True)
    with col_right:
        pie_fig = go.Figure(
            data=[
                go.Pie(
                    labels=weights_df["Ticker"],
                    values=[
                        float(w.strip("%")) for w in weights_df["Weight"]
                    ],
                    hole=0.4,
                )
            ]
        )
        pie_fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(pie_fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # Min volatility portfolio, for comparison
    # -----------------------------------------------------------------------
    with st.expander("Min Volatility Portfolio (lower risk alternative)"):
        gcol1, gcol2, gcol3 = st.columns(3)
        gcol1.metric("Expected Annual Return", f"{gmv['expected_return']:.2%}")
        gcol2.metric("Annual Volatility", f"{gmv['volatility']:.2%}")
        gcol3.metric("Sharpe Ratio", f"{gmv['sharpe']:.2f}")

        gmv_weights_df = pd.DataFrame(
            {"Ticker": used_tickers, "Weight": gmv["weights"]}
        ).sort_values("Weight", ascending=False)
        gmv_weights_df = gmv_weights_df[gmv_weights_df["Weight"] > 0.001]
        gmv_weights_df["Weight"] = gmv_weights_df["Weight"].map(
            lambda w: f"{w:.1%}")
        st.dataframe(gmv_weights_df, hide_index=True, use_container_width=True)

    # -----------------------------------------------------------------------
    # Historical drawdown
    # -----------------------------------------------------------------------
    st.subheader("Historical Drawdown")

    simple_returns = inputs["simple_returns"]
    max_sharpe_dd = portfolio_drawdown_analysis(
        max_sharpe["weights"], simple_returns)
    gmv_dd = portfolio_drawdown_analysis(gmv["weights"], simple_returns)

    dd_col1, dd_col2 = st.columns(2)
    dd_col1.metric("Recommended Portfolio Max Drawdown",
                   f"{max_sharpe_dd['max_drawdown']:.1%}")
    dd_col2.metric("Min Volatility Portfolio Max Drawdown",
                   f"{gmv_dd['max_drawdown']:.1%}")

    dd_fig = go.Figure()
    dd_fig.add_trace(
        go.Scatter(
            x=max_sharpe_dd["drawdown_series"].index,
            y=max_sharpe_dd["drawdown_series"].values,
            mode="lines",
            fill="tozeroy",
            name="Max Sharpe drawdown",
            line=dict(color="red"),
        )
    )
    dd_fig.add_trace(
        go.Scatter(
            x=gmv_dd["drawdown_series"].index,
            y=gmv_dd["drawdown_series"].values,
            mode="lines",
            fill="tozeroy",
            name="Min Volatility drawdown",
            line=dict(color="blue"),
            opacity=0.6,
        )
    )
    dd_fig.update_layout(
        yaxis_title="Drawdown from peak",
        yaxis_tickformat=".0%",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(dd_fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # Raw price history, for reference
    # -----------------------------------------------------------------------
    with st.expander("Raw price history used"):
        st.line_chart(inputs["prices"])

else:
    st.info("Enter tickers in the sidebar and click **Run optimization** to begin.")
