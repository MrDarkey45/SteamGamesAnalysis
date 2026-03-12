"""
utils.py - Shared helper functions for Steam Games Analysis

Contains reusable plotting helpers and data formatting functions
used across multiple notebooks.
"""

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import STEAM_COLORS, CHART_COLORS


def setup_matplotlib_style():
    """
    Set up a consistent, portfolio-ready matplotlib style.
    Call this at the top of each notebook.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.facecolor": "#1b2838",
        "axes.facecolor": "#2a475e",
        "axes.edgecolor": "#c7d5e0",
        "axes.labelcolor": "#c7d5e0",
        "text.color": "#c7d5e0",
        "xtick.color": "#c7d5e0",
        "ytick.color": "#c7d5e0",
        "legend.facecolor": "#2a475e",
        "legend.edgecolor": "#c7d5e0",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
    })


def get_plotly_layout(title, xaxis_title="", yaxis_title=""):
    """
    Return a consistent Plotly layout dict with Steam-themed dark styling.

    Args:
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
    """
    return go.Layout(
        title=dict(
            text=title,
            font=dict(size=20, color="#c7d5e0"),
            x=0.5,
        ),
        paper_bgcolor="#1b2838",
        plot_bgcolor="#2a475e",
        font=dict(color="#c7d5e0", size=12),
        xaxis=dict(
            title=xaxis_title,
            gridcolor="rgba(198, 213, 224, 0.15)",
            zerolinecolor="rgba(198, 213, 224, 0.3)",
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor="rgba(198, 213, 224, 0.15)",
            zerolinecolor="rgba(198, 213, 224, 0.3)",
        ),
        legend=dict(
            bgcolor="rgba(42, 71, 94, 0.8)",
            bordercolor="#c7d5e0",
            borderwidth=1,
        ),
        margin=dict(l=60, r=30, t=60, b=60),
    )


def format_large_number(num):
    """
    Format large numbers for display (e.g., 1500000 -> '1.5M').
    Makes chart labels cleaner and easier to read.
    """
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(int(num))


def create_bar_chart(data, x_col, y_col, title, color=None, horizontal=False):
    """
    Create a styled Plotly bar chart.

    Args:
        data: DataFrame with the chart data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Chart title
        color: Bar color (defaults to Steam light blue)
        horizontal: If True, make horizontal bars
    """
    if color is None:
        color = STEAM_COLORS["light_blue"]

    if horizontal:
        fig = go.Figure(go.Bar(
            y=data[x_col],
            x=data[y_col],
            orientation="h",
            marker_color=color,
        ))
        layout = get_plotly_layout(title, xaxis_title=y_col, yaxis_title=x_col)
    else:
        fig = go.Figure(go.Bar(
            x=data[x_col],
            y=data[y_col],
            marker_color=color,
        ))
        layout = get_plotly_layout(title, xaxis_title=x_col, yaxis_title=y_col)

    fig.update_layout(layout)
    return fig
