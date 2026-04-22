"""
Chart Utilities
================
Reusable Plotly chart builders for all dashboard pages.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

COLORS = {
    "green":       "#00C896",
    "red":         "#FF4B6E",
    "gold":        "#FFD166",
    "blue":        "#3D8EFF",
    "purple":      "#9B59B6",
    "bg":          "#0D1117",
    "surface":     "#161B22",
    "border":      "#30363D",
    "text":        "#E6EDF3",
    "muted":       "#8B949E",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["surface"],
    font=dict(family="'IBM Plex Mono', monospace", color=COLORS["text"], size=11),
    xaxis=dict(gridcolor=COLORS["border"], zeroline=False),
    yaxis=dict(gridcolor=COLORS["border"], zeroline=False),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"]),
)


def candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure(go.Candlestick(
        x=df["date"],
        open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
        name=symbol,
    ))
    fig.update_layout(
        title=f"{symbol} — Price History",
        xaxis_rangeslider_visible=False,
        **LAYOUT_DEFAULTS,
    )
    return fig


def technical_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.15, 0.20, 0.15],
        vertical_spacing=0.03,
        subplot_titles=["Price & MAs", "Volume", "RSI", "MACD"],
    )

    # ── Row 1: Candlestick + SMA lines ──────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
        name="OHLC", showlegend=False,
    ), row=1, col=1)

    for col, color, name in [
        ("sma_20", COLORS["gold"], "SMA20"),
        ("sma_50", COLORS["blue"], "SMA50"),
        ("bb_upper", COLORS["muted"], "BB Upper"),
        ("bb_lower", COLORS["muted"], "BB Lower"),
    ]:
        if col in df.columns:
            dash = "dot" if "bb" in col else "solid"
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                line=dict(color=color, width=1.2, dash=dash),
                name=name,
            ), row=1, col=1)

    # ── Row 2: Volume ─────────────────────────────────────────────────────
    colors_vol = [COLORS["green"] if c >= o else COLORS["red"]
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"],
        marker_color=colors_vol, name="Volume", showlegend=False,
    ), row=2, col=1)

    # ── Row 3: RSI ────────────────────────────────────────────────────────
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rsi"],
            line=dict(color=COLORS["purple"], width=1.5),
            name="RSI", showlegend=False,
        ), row=3, col=1)
        fig.add_hline(y=70, line_color=COLORS["red"],   line_dash="dash", row=3, col=1)
        fig.add_hline(y=30, line_color=COLORS["green"], line_dash="dash", row=3, col=1)

    # ── Row 4: MACD ───────────────────────────────────────────────────────
    if "macd" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["macd"],
            line=dict(color=COLORS["blue"], width=1.5),
            name="MACD", showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["macd_signal"],
            line=dict(color=COLORS["gold"], width=1.2),
            name="Signal", showlegend=False,
        ), row=4, col=1)
        hist_colors = [COLORS["green"] if v >= 0 else COLORS["red"]
                       for v in df.get("macd_hist", [])]
        fig.add_trace(go.Bar(
            x=df["date"], y=df.get("macd_hist", []),
            marker_color=hist_colors, name="Histogram", showlegend=False,
        ), row=4, col=1)

    fig.update_layout(
        title=f"{symbol} — Technical Analysis",
        height=700,
        xaxis_rangeslider_visible=False,
        **LAYOUT_DEFAULTS,
    )
    return fig


def monte_carlo_paths_chart(mc: dict, symbol: str) -> go.Figure:
    """Plots a subset of simulated price paths."""
    paths = mc.get("paths", np.array([]))
    if paths.size == 0:
        return go.Figure()

    n_days = mc["n_days"]
    days   = list(range(n_days))
    current_price = mc["current_price"]

    fig = go.Figure()

    # Plot 200 sample paths (transparent)
    sample_paths = paths[:200] if len(paths) > 200 else paths
    for path in sample_paths:
        fig.add_trace(go.Scatter(
            x=days, y=path,
            mode="lines",
            line=dict(color="rgba(61,142,255,0.07)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Percentile bands
    p5  = np.percentile(paths, 5,  axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    med = np.median(paths, axis=0)

    fig.add_trace(go.Scatter(
        x=days + days[::-1],
        y=list(p95) + list(p5[::-1]),
        fill="toself",
        fillcolor="rgba(61,142,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90% Range",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=med,
        line=dict(color=COLORS["gold"], width=2.5, dash="dash"),
        name="Median Path",
    ))
    fig.add_hline(
        y=current_price,
        line_color=COLORS["muted"],
        line_dash="dot",
        annotation_text="Current Price",
    )

    fig.update_layout(
        title=f"{symbol} — Monte Carlo ({mc['n_simulations']:,} simulations, {n_days}-day horizon)",
        xaxis_title="Trading Days",
        yaxis_title="Price (EGP)",
        height=460,
        **LAYOUT_DEFAULTS,
    )
    return fig


def monte_carlo_distribution_chart(mc: dict, symbol: str) -> go.Figure:
    """Histogram of final simulated prices."""
    final_prices = mc.get("final_prices", np.array([]))
    if final_prices.size == 0:
        return go.Figure()

    current = mc["current_price"]
    colors = [COLORS["green"] if p > current else COLORS["red"] for p in final_prices]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=final_prices,
        nbinsx=60,
        marker_color=COLORS["blue"],
        opacity=0.8,
        name="Final Price Distribution",
    ))
    fig.add_vline(
        x=current,
        line_color=COLORS["gold"],
        line_dash="dash",
        annotation_text=f"Current: {current:.2f}",
        annotation_font_color=COLORS["gold"],
    )
    fig.add_vline(
        x=mc["stats"]["mean"],
        line_color=COLORS["green"],
        line_dash="dot",
        annotation_text=f"Mean: {mc['stats']['mean']:.2f}",
        annotation_font_color=COLORS["green"],
    )

    fig.update_layout(
        title=f"{symbol} — Distribution of Simulated Prices (Day {mc['n_days']})",
        xaxis_title="Price (EGP)",
        yaxis_title="Frequency",
        height=350,
        **LAYOUT_DEFAULTS,
    )
    return fig


def portfolio_pie_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=df["Symbol"],
        values=df["Current Value"],
        hole=0.45,
        marker=dict(
            colors=px.colors.qualitative.Bold,
            line=dict(color=COLORS["bg"], width=2),
        ),
        textfont=dict(size=12),
    ))
    fig.update_layout(
        title="Portfolio Allocation",
        height=380,
        **LAYOUT_DEFAULTS,
    )
    return fig


def portfolio_performance_chart(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["color"] = df["Unrealized P&L"].apply(
        lambda x: COLORS["green"] if x >= 0 else COLORS["red"]
    )
    fig = go.Figure(go.Bar(
        x=df["Symbol"],
        y=df["Return %"],
        marker_color=df["color"],
        text=df["Return %"].apply(lambda x: f"{x:+.1f}%"),
        textposition="outside",
    ))
    fig.update_layout(
        title="Portfolio Returns by Stock",
        xaxis_title="",
        yaxis_title="Return %",
        height=350,
        **LAYOUT_DEFAULTS,
    )
    return fig


def sentiment_bar_chart(sentiment_dict: dict) -> go.Figure:
    if not sentiment_dict:
        return go.Figure()
    companies = list(sentiment_dict.keys())
    scores    = [v["avg_score"] for v in sentiment_dict.values()]
    colors    = [COLORS["green"] if s > 0.1 else (COLORS["red"] if s < -0.1 else COLORS["muted"])
                 for s in scores]

    fig = go.Figure(go.Bar(
        x=companies, y=scores,
        marker_color=colors,
        text=[f"{s:+.2f}" for s in scores],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color=COLORS["border"])
    fig.update_layout(
        title="News Sentiment by Company",
        yaxis_title="Sentiment Score",
        height=380,
        **LAYOUT_DEFAULTS,
    )
    return fig
