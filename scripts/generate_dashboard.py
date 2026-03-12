"""
generate_dashboard.py - Build an interactive HTML dashboard with user filters.

Filters: Genre multi-select, Year range slider, Publisher tier toggles.
All charts update reactively via client-side JavaScript + Plotly.

Run from the project root:
    python scripts/generate_dashboard.py

Produces: output/dashboard.html
"""

import sys, os, json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings("ignore")

from src.data_loader import (
    load_applications, load_genres, load_application_genres,
    build_games_with_genres, build_games_with_publishers,
    STEAM_COLORS, CHART_COLORS,
)

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "dashboard.html")


# ── Helpers ──────────────────────────────────────────────────────────────────

def min_max_normalize(series):
    """Scale a pandas Series to the 0-1 range."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


# ── Data loading ─────────────────────────────────────────────────────────────

print("Loading data...")
games = load_applications()
games_genres = build_games_with_genres(games)
games_pubs = build_games_with_publishers(games)

# Add publisher tier to games_pubs
pub_counts = games_pubs.groupby("publisher_name")["appid"].nunique().reset_index()
pub_counts.columns = ["publisher_name", "game_count"]
pub_counts["tier"] = pub_counts["game_count"].apply(
    lambda x: "AAA" if x >= 50 else ("Mid-tier" if x >= 6 else "Indie"))
games_pubs = games_pubs.merge(pub_counts[["publisher_name", "tier"]],
                               on="publisher_name", how="left")

# Build a combined dataset: games with genres AND tier
# Each row = one game-genre pair, with the tier from the first publisher
game_tier_map = games_pubs.drop_duplicates(subset="appid")[["appid", "tier"]]
games_full = games_genres.merge(game_tier_map, on="appid", how="left")
games_full["tier"] = games_full["tier"].fillna("Indie")


# ══════════════════════════════════════════════════════════════════════════════
# PRE-AGGREGATE DATA FOR EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

# Keep only known English genre names for the filter UI
english_genres = sorted(games_full["genre_name"].dropna().unique().tolist())
# Filter to genres with at least 100 games to keep the picker manageable
genre_game_counts = games_full.groupby("genre_name")["appid"].nunique()
valid_genres = genre_game_counts[genre_game_counts >= 100].index.tolist()
valid_genres = sorted(valid_genres)

# Default selected genres (the most interesting ones)
default_genres = ["Indie", "Action", "Adventure", "Casual", "Simulation",
                  "Strategy", "RPG", "Sports", "Racing", "Puzzle"]

print(f"Genres available for filtering: {len(valid_genres)}")

# -- Dataset 1: yearly stats by genre and tier --
# Used for: releases per year, KPI cards, genre treemap
print("Aggregating yearly stats...")
games_dedup = games_full.drop_duplicates(subset=["appid", "genre_name"])
games_dedup = games_dedup[games_dedup["genre_name"].isin(valid_genres)]
games_dedup["release_year"] = games_dedup["release_year"].fillna(0).astype(int)

yearly_agg = (games_dedup
    .groupby(["release_year", "genre_name", "tier"])
    .agg(
        game_count=("appid", "nunique"),
        free_count=("is_free", "sum"),
        avg_price=("price_dollars", "mean"),
        median_price=("price_dollars", "median"),
        avg_recs=("recommendations_total", "mean"),
        median_recs=("recommendations_total", "median"),
        metacritic_count=("metacritic_score", lambda x: (x > 0).sum()),
        avg_metacritic=("metacritic_score", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
    )
    .reset_index())

# Filter to valid years
yearly_agg = yearly_agg[(yearly_agg["release_year"] >= 2005) &
                         (yearly_agg["release_year"] <= 2025)]

# Convert to JSON-friendly records
yearly_data = yearly_agg.to_dict(orient="records")
# Clean NaN values for JSON
for row in yearly_data:
    for k, v in row.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            row[k] = 0

# -- Dataset 2: price bracket stats by genre and tier --
print("Aggregating price bracket stats...")
paid = games_dedup[(games_dedup["is_free"] == False) &
                    (games_dedup["price_dollars"] > 0)].copy()
bracket_bins = [0, 5, 10, 15, 20, 30, 50, 100, float("inf")]
bracket_labels = ["$0-5", "$5-10", "$10-15", "$15-20",
                   "$20-30", "$30-50", "$50-100", "$100+"]
paid["price_bracket"] = pd.cut(paid["price_dollars"], bins=bracket_bins,
                                labels=bracket_labels, right=True)

price_agg = (paid
    .groupby(["genre_name", "tier", "price_bracket"], observed=False)
    .agg(
        avg_recs=("recommendations_total", "mean"),
        game_count=("appid", "nunique"),
    )
    .reset_index())
price_agg["price_bracket"] = price_agg["price_bracket"].astype(str)

price_data = price_agg.to_dict(orient="records")
for row in price_data:
    for k, v in row.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            row[k] = 0

# -- Dataset 3: top 300 games for the "top games" chart --
print("Building top games list...")
games_dedup_single = games_full.drop_duplicates(subset="appid")
top_n = (games_dedup_single
    .nlargest(300, "recommendations_total")
    [["appid", "name", "recommendations_total", "price_dollars", "is_free",
      "tier", "metacritic_score", "release_year"]]
    .copy())

# Attach genres as a list for each game — only keep genres in valid_genres
# (filters out non-English genre names like Arabic ones)
game_genre_lists = (games_full[games_full["appid"].isin(top_n["appid"])]
    .groupby("appid")["genre_name"]
    .apply(lambda x: [g for g in x.dropna().unique() if g in valid_genres])
    .reset_index())
game_genre_lists.columns = ["appid", "genres"]
top_n = top_n.merge(game_genre_lists, on="appid", how="left")
top_n["genres"] = top_n["genres"].apply(lambda x: x if isinstance(x, list) else [])

top_games_data = []
for _, row in top_n.iterrows():
    year_val = int(row["release_year"]) if pd.notna(row["release_year"]) else 0
    # Clamp future-dated games to 2025 so they pass the year filter
    if year_val > 2025:
        year_val = 2025
    top_games_data.append({
        "name": str(row["name"]),  # ensure it's a string
        "recs": int(row["recommendations_total"]) if pd.notna(row["recommendations_total"]) else 0,
        "price": round(float(row["price_dollars"]), 2) if pd.notna(row["price_dollars"]) else 0,
        "is_free": bool(row["is_free"]),
        "tier": str(row["tier"]) if pd.notna(row["tier"]) else "Indie",
        "metacritic": int(row["metacritic_score"]) if pd.notna(row["metacritic_score"]) and row["metacritic_score"] > 0 else 0,
        "year": year_val,
        "genres": row["genres"],
    })

# Games with no matching English genres should still appear (use empty list = matches all)
for g in top_games_data:
    if len(g["genres"]) == 0:
        g["genres"] = list(valid_genres)  # match any genre filter

# -- Dataset 4: genre forecast data --
print("Computing genre forecasts...")
df_fc = games_genres[
    (games_genres["release_year"] >= 2010) &
    (games_genres["release_year"] <= 2025)       # exclude junk dates (year 9998, etc.)
].copy()
df_fc = df_fc.dropna(subset=["genre_name"])
df_fc["year_month_str"] = df_fc["release_year_month"].astype(str)

monthly_genre = df_fc.groupby(["year_month_str", "genre_name"]).agg(
    game_count=("appid", "nunique"),
    avg_recommendations=("recommendations_total", "mean"),
    avg_price=("price_dollars", "mean"),
).reset_index()
monthly_genre["avg_recommendations"] = monthly_genre["avg_recommendations"].fillna(0)
monthly_genre["avg_price"] = monthly_genre["avg_price"].fillna(0)

monthly_genre["popularity_score"] = (
    min_max_normalize(monthly_genre["game_count"]) * 0.4
    + min_max_normalize(monthly_genre["avg_recommendations"]) * 0.4
    + (1 - min_max_normalize(monthly_genre["avg_price"])) * 0.2
)

# Compute forecasts for the top 15 genres (so users can pick from more)
genre_avg_pop = monthly_genre.groupby("genre_name")["popularity_score"].mean().sort_values(ascending=False)
forecast_genres = genre_avg_pop.head(15).index.tolist()

all_periods = sorted(monthly_genre["year_month_str"].unique())
period_to_num = {p: i for i, p in enumerate(all_periods)}
num_periods = len(all_periods)

print(f"  Forecast periods: {all_periods[0]} to {all_periods[-1]} ({num_periods} months)")

last_period = pd.Period(all_periods[-1], freq="M")
future_labels = [(last_period + i + 1).strftime("%Y-%m") for i in range(12)]
recent_periods = all_periods[-18:]  # show 18 months of actuals for context

forecast_data = {}
for genre in forecast_genres:
    gdf = monthly_genre[monthly_genre["genre_name"] == genre].sort_values("year_month_str")
    X = gdf["year_month_str"].map(period_to_num).values.reshape(-1, 1)
    y = gdf["popularity_score"].values
    poly = PolynomialFeatures(degree=2)
    model = LinearRegression().fit(poly.fit_transform(X), y)
    future_X = np.arange(num_periods, num_periods + 12).reshape(-1, 1)
    pred = np.clip(model.predict(poly.transform(future_X)), 0, None)

    # Get actual scores for recent periods
    recent = gdf[gdf["year_month_str"].isin(recent_periods)]
    actual_scores = []
    for p in recent_periods:
        match = recent[recent["year_month_str"] == p]["popularity_score"]
        actual_scores.append(round(float(match.values[0]), 4) if len(match) > 0 else None)

    forecast_data[genre] = {
        "actual_periods": recent_periods,
        "actual_scores": actual_scores,
        "forecast_periods": future_labels,
        "forecast_scores": [round(float(v), 4) for v in pred],
    }

# -- Dataset 5: sentiment (pre-computed, static — no filtering needed) --
sentiment_data = {
    "distribution": {"positive": 29526, "neutral": 13377, "negative": 10118},
    "thumbs": {"up": 0.156, "down": -0.065},
    "agreement_rate": 71.9,
}


# ══════════════════════════════════════════════════════════════════════════════
# BUILD THE HTML
# ══════════════════════════════════════════════════════════════════════════════
print("Assembling interactive dashboard...")

# Serialize data to JSON for embedding
embedded_data = {
    "yearly": yearly_data,
    "priceBrackets": price_data,
    "topGames": top_games_data,
    "forecast": forecast_data,
    "sentiment": sentiment_data,
    "validGenres": valid_genres,
    "defaultGenres": default_genres,
    "forecastGenres": forecast_genres,
    "chartColors": CHART_COLORS,
    "steamColors": STEAM_COLORS,
    "bracketLabels": bracket_labels,
}

# Custom encoder to handle numpy types that json.dumps can't serialize
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

data_json = json.dumps(embedded_data, ensure_ascii=False, cls=NumpyEncoder)

html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Steam Games Analysis — Interactive Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #1b2838;
            color: #c7d5e0;
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
        }
        header {
            text-align: center;
            padding: 24px 20px 14px;
            border-bottom: 2px solid #2a475e;
        }
        header h1 { font-size: 2em; color: #66c0f4; margin-bottom: 4px; }
        header p {
            font-size: 1em; opacity: 0.8;
            max-width: 680px; margin: 0 auto;
        }

        /* ── Filter bar ── */
        .filter-bar {
            background: #171d25;
            border-bottom: 2px solid #2a475e;
            padding: 14px 20px;
            position: sticky; top: 0; z-index: 100;
        }
        .filter-bar-inner {
            max-width: 1200px; margin: 0 auto;
        }
        /* Row 1: genre tags (full width) */
        .filter-row-genres {
            margin-bottom: 10px;
        }
        /* Row 2: year + tier side by side */
        .filter-row-controls {
            display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-end;
        }
        .filter-group-label {
            display: block; font-size: 0.8em; color: #66c0f4;
            font-weight: 600; margin-bottom: 5px; text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Genre tag picker */
        .genre-picker {
            display: flex; flex-wrap: wrap; gap: 4px;
            max-height: 68px; overflow-y: auto;
            padding: 5px; border: 1px solid #2a475e; border-radius: 6px;
            background: #1b2838;
        }
        .genre-tag {
            display: inline-block; padding: 2px 9px; font-size: 0.78em;
            border-radius: 12px; cursor: pointer; user-select: none;
            border: 1px solid #2a475e; color: #8f98a0;
            background: transparent; transition: all 0.15s;
            white-space: nowrap;
        }
        .genre-tag.active {
            background: #66c0f4; color: #1b2838;
            border-color: #66c0f4; font-weight: 600;
        }
        .genre-tag:hover { border-color: #66c0f4; }
        .genre-actions {
            display: inline-flex; gap: 6px; margin-left: 10px;
        }
        .genre-actions button {
            font-size: 0.75em; padding: 2px 10px; border-radius: 4px;
            border: 1px solid #2a475e; background: transparent;
            color: #66c0f4; cursor: pointer;
        }
        .genre-actions button:hover { background: #2a475e; }
        .genre-header {
            display: flex; align-items: center; margin-bottom: 5px;
        }

        /* Tier checkboxes */
        .tier-checks {
            display: flex; gap: 16px;
        }
        .tier-check {
            display: flex; align-items: center; gap: 5px;
            cursor: pointer; font-size: 0.9em; white-space: nowrap;
        }
        .tier-check input {
            accent-color: #66c0f4; cursor: pointer;
            width: 15px; height: 15px;
        }

        /* Year range — two number inputs */
        .year-range-row {
            display: flex; align-items: center; gap: 8px;
        }
        .year-input {
            width: 70px; padding: 4px 8px;
            background: #1b2838; border: 1px solid #2a475e;
            border-radius: 4px; color: #66c0f4;
            font-size: 0.95em; font-weight: 600;
            text-align: center;
        }
        .year-input:focus { outline: none; border-color: #66c0f4; }
        .year-sep { color: #8f98a0; font-size: 0.85em; }

        /* ── Dashboard layout ── */
        .dashboard {
            max-width: 1200px; margin: 0 auto; padding: 16px;
        }
        .chart-section {
            margin: 16px 0; padding: 16px;
            background: rgba(42, 71, 94, 0.25);
            border-radius: 10px; border: 1px solid #2a475e;
        }
        .chart-section h2 {
            font-size: 1.2em; color: #66c0f4;
            margin-bottom: 8px; padding-bottom: 5px;
            border-bottom: 1px solid #2a475e;
        }
        .chart-section .subtitle {
            font-size: 0.82em; color: #8f98a0; margin-top: -4px; margin-bottom: 8px;
        }

        /* KPI row — always 4 per row */
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
        }
        .kpi-card {
            background: #2a475e; border-radius: 8px;
            padding: 12px 10px; text-align: center;
        }
        .kpi-card .kpi-value {
            font-size: 1.5em; font-weight: 700; color: #66c0f4;
        }
        .kpi-card .kpi-label {
            font-size: 0.75em; color: #8f98a0; margin-top: 2px;
        }

        /* Two-column layout for side-by-side charts */
        .two-col {
            display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
        }
        .two-col > .chart-section { margin: 0; }
        @media (max-width: 900px) {
            .two-col { grid-template-columns: 1fr; }
            .kpi-row { grid-template-columns: repeat(2, 1fr); }
        }

        /* Findings table */
        .findings table {
            width: 100%; border-collapse: collapse; margin-top: 8px;
        }
        .findings th, .findings td {
            text-align: left; padding: 8px 12px;
            border-bottom: 1px solid #2a475e;
        }
        .findings th { color: #66c0f4; }

        footer {
            text-align: center; padding: 24px; font-size: 0.85em;
            opacity: 0.6; border-top: 1px solid #2a475e; margin-top: 30px;
        }
    </style>
</head>
<body>

<header>
    <h1>Steam Games Analysis</h1>
    <p>Interactive dashboard exploring <strong>150,000+ games</strong> — filter by
       genre, year, and publisher tier to find your own insights.</p>
</header>

<!-- ══ FILTER BAR ══ -->
<div class="filter-bar">
    <div class="filter-bar-inner">
        <!-- Row 1: Genre picker (full width) -->
        <div class="filter-row-genres">
            <div class="genre-header">
                <span class="filter-group-label" style="margin-bottom:0;">Genres</span>
                <div class="genre-actions">
                    <button onclick="selectAllGenres()">All</button>
                    <button onclick="selectNoneGenres()">None</button>
                    <button onclick="selectDefaultGenres()">Reset</button>
                </div>
            </div>
            <div class="genre-picker" id="genre-picker"></div>
        </div>
        <!-- Row 2: Year + Tier side by side -->
        <div class="filter-row-controls">
            <div>
                <span class="filter-group-label">Year Range</span>
                <div class="year-range-row">
                    <input type="number" id="year-min" class="year-input"
                           min="2005" max="2025" value="2005">
                    <span class="year-sep">to</span>
                    <input type="number" id="year-max" class="year-input"
                           min="2005" max="2025" value="2025">
                </div>
            </div>
            <div>
                <span class="filter-group-label">Publisher Tier</span>
                <div class="tier-checks">
                    <label class="tier-check">
                        <input type="checkbox" id="tier-indie" checked> Indie
                    </label>
                    <label class="tier-check">
                        <input type="checkbox" id="tier-midtier" checked> Mid-tier
                    </label>
                    <label class="tier-check">
                        <input type="checkbox" id="tier-aaa" checked> AAA
                    </label>
                </div>
            </div>
            <div style="margin-left:auto;">
                <span class="filter-group-label">Filtered</span>
                <span id="filter-summary" style="font-size:0.85em; color:#8f98a0;"></span>
            </div>
        </div>
    </div>
</div>

<!-- ══ DASHBOARD ══ -->
<div class="dashboard">

    <!-- KPI Cards -->
    <section class="chart-section">
        <h2>Key Metrics</h2>
        <p class="subtitle">Filtered summary of the selected data</p>
        <div class="kpi-row" id="kpi-row"></div>
    </section>

    <!-- Releases per year + Genre treemap -->
    <div class="two-col">
        <section class="chart-section">
            <h2>Games Released Per Year</h2>
            <div id="chart-yearly" style="height:380px;"></div>
        </section>
        <section class="chart-section">
            <h2>Genre Landscape</h2>
            <p class="subtitle">Top genres by game count (filtered)</p>
            <div id="chart-treemap" style="height:380px;"></div>
        </section>
    </div>

    <!-- Price distribution + Sweet spots -->
    <div class="two-col">
        <section class="chart-section">
            <h2>Price Distribution</h2>
            <div id="chart-price-dist" style="height:380px;"></div>
        </section>
        <section class="chart-section">
            <h2>Price Sweet Spots</h2>
            <p class="subtitle">Best price bracket per genre (by avg recommendations)</p>
            <div id="chart-sweet-spot" style="height:380px;"></div>
        </section>
    </div>

    <!-- Indie vs AAA -->
    <section class="chart-section">
        <h2>Indie vs AAA Comparison</h2>
        <p class="subtitle">Metrics compared across publisher tiers (filtered by selected genres)</p>
        <div id="chart-tier-compare" style="height:400px;"></div>
    </section>

    <!-- Genre Forecast -->
    <section class="chart-section">
        <h2>Genre Popularity Forecast</h2>
        <p class="subtitle">Polynomial regression on popularity score — select genres to compare trends</p>
        <div id="chart-forecast" style="height:550px;"></div>
    </section>

    <!-- Top Games -->
    <section class="chart-section">
        <h2>Most Recommended Games</h2>
        <p class="subtitle">Top 20 games matching your filters</p>
        <div id="chart-top-games" style="height:520px;"></div>
    </section>

    <!-- Sentiment (static) -->
    <section class="chart-section">
        <h2>Player Sentiment</h2>
        <p class="subtitle">NLP analysis of 53K English reviews (not affected by filters)</p>
        <div id="chart-sentiment" style="height:360px;"></div>
    </section>

    <!-- Key findings -->
    <section class="chart-section findings">
        <h2>Key Takeaways</h2>
        <table>
            <tr><th>Finding</th><th>Detail</th></tr>
            <tr><td><strong>Steam is booming</strong></td>
                <td>20K+ games released in the peak year, up from ~100 in 2005</td></tr>
            <tr><td><strong>Indie dominates</strong></td>
                <td>103K indie games vs 14K AAA — growing 2x faster</td></tr>
            <tr><td><strong>Price != success</strong></td>
                <td>Correlation between price and recommendations ~ 0.035</td></tr>
            <tr><td><strong>Quality is tier-agnostic</strong></td>
                <td>Metacritic scores nearly identical across indie / mid / AAA</td></tr>
            <tr><td><strong>Reviews skew positive</strong></td>
                <td>55.7% positive; sentiment stable since 2015</td></tr>
            <tr><td><strong>Sweet spots vary</strong></td>
                <td>RPGs do best at $50-100; Indie titles peak at $30-50</td></tr>
        </table>
    </section>
</div>

<!-- ══ DATA SOURCE ══ -->
<div class="dashboard-section" style="margin-top:24px;">
    <details style="cursor:pointer;">
        <summary style="font-size:1.1em; color:#66c0f4; font-weight:bold; padding:8px 0;">
            About This Data &amp; Methodology
        </summary>
        <div style="margin-top:12px; line-height:1.7; font-size:0.92em;">
            <p><strong>Source:</strong>
                <a href="https://github.com/vintagedon/steam-dataset-2025" target="_blank"
                   style="color:#66c0f4;">Steam Dataset 2025</a> by vintagedon
                &nbsp;|&nbsp; License: CC BY 4.0
                &nbsp;|&nbsp; DOI:
                <a href="https://doi.org/10.5281/zenodo.17286923" target="_blank"
                   style="color:#66c0f4;">10.5281/zenodo.17286923</a>
            </p>
            <p>A point-in-time snapshot (September 2025) of the Steam catalog collected
               exclusively from official Steam Web APIs. No scraping or third-party estimates.</p>

            <table style="width:100%; border-collapse:collapse; margin:12px 0;">
                <tr><th style="text-align:left; padding:6px 12px; color:#66c0f4; border-bottom:1px solid #2a475e;">Metric</th>
                    <th style="text-align:left; padding:6px 12px; color:#66c0f4; border-bottom:1px solid #2a475e;">Value</th></tr>
                <tr><td style="padding:6px 12px; border-bottom:1px solid #1b2838;">Applications</td>
                    <td style="padding:6px 12px; border-bottom:1px solid #1b2838;">239,664</td></tr>
                <tr><td style="padding:6px 12px; border-bottom:1px solid #1b2838;">Reviews</td>
                    <td style="padding:6px 12px; border-bottom:1px solid #1b2838;">1,048,148</td></tr>
                <tr><td style="padding:6px 12px; border-bottom:1px solid #1b2838;">Developers</td>
                    <td style="padding:6px 12px; border-bottom:1px solid #1b2838;">54,321</td></tr>
                <tr><td style="padding:6px 12px; border-bottom:1px solid #1b2838;">Publishers</td>
                    <td style="padding:6px 12px; border-bottom:1px solid #1b2838;">39,876</td></tr>
                <tr><td style="padding:6px 12px; border-bottom:1px solid #1b2838;">Time span</td>
                    <td style="padding:6px 12px; border-bottom:1px solid #1b2838;">1997 &ndash; 2025 (28 years)</td></tr>
                <tr><td style="padding:6px 12px;">Successful retrieval rate</td>
                    <td style="padding:6px 12px;">56% (134K of 240K apps)</td></tr>
            </table>

            <p style="color:#ff9800; margin-top:8px;"><strong>Known Limitations:</strong></p>
            <ul style="margin-top:4px; padding-left:20px;">
                <li><strong>Incomplete metadata</strong> — 44% of apps failed retrieval (delisted games, regional restrictions, invalid IDs)</li>
                <li><strong>US-centric</strong> — Pricing is USD-only; Asia-Pacific coverage ~60%, China ~20%</li>
                <li><strong>Non-English genres</strong> — ~32% of apps have non-English metadata; some major titles only have Arabic genre tags</li>
                <li><strong>Review counts are partial</strong> — The recommendations field is a snapshot, not lifetime totals</li>
                <li><strong>Junk release dates filtered</strong> — Games with placeholder years (e.g. 9998) are excluded from analysis</li>
            </ul>
        </div>
    </details>
</div>

<footer>
    Analysis based on the <a href="https://github.com/vintagedon/steam-dataset-2025"
    target="_blank" style="color:#66c0f4;">Steam Dataset 2025</a> (CC BY 4.0)
    — 240K applications, 150K games, 1M+ reviews.
</footer>

<script>
// ══════════════════════════════════════════════════════════════════════════════
// EMBEDDED DATA
// ══════════════════════════════════════════════════════════════════════════════
const DATA = """ + data_json + """;

const STEAM = DATA.steamColors;
const COLORS = DATA.chartColors;
const LAYOUT_BASE = {
    paper_bgcolor: '#1b2838',
    plot_bgcolor: '#2a475e',
    font: { color: '#c7d5e0', size: 12 },
    margin: { l: 60, r: 30, t: 40, b: 60 },
    xaxis: { gridcolor: 'rgba(198,213,224,0.15)', zerolinecolor: 'rgba(198,213,224,0.3)' },
    yaxis: { gridcolor: 'rgba(198,213,224,0.15)', zerolinecolor: 'rgba(198,213,224,0.3)' },
};

// ══════════════════════════════════════════════════════════════════════════════
// FILTER STATE
// ══════════════════════════════════════════════════════════════════════════════
let selectedGenres = new Set(DATA.defaultGenres);
let yearMin = 2005;
let yearMax = 2025;
let selectedTiers = new Set(['Indie', 'Mid-tier', 'AAA']);

// ── Genre picker setup ──
const picker = document.getElementById('genre-picker');
DATA.validGenres.forEach(genre => {
    const tag = document.createElement('span');
    tag.className = 'genre-tag' + (selectedGenres.has(genre) ? ' active' : '');
    tag.textContent = genre;
    tag.dataset.genre = genre;
    tag.addEventListener('click', () => {
        if (selectedGenres.has(genre)) {
            selectedGenres.delete(genre);
            tag.classList.remove('active');
        } else {
            selectedGenres.add(genre);
            tag.classList.add('active');
        }
        updateAll();
    });
    picker.appendChild(tag);
});

function selectAllGenres() {
    selectedGenres = new Set(DATA.validGenres);
    picker.querySelectorAll('.genre-tag').forEach(t => t.classList.add('active'));
    updateAll();
}
function selectNoneGenres() {
    selectedGenres.clear();
    picker.querySelectorAll('.genre-tag').forEach(t => t.classList.remove('active'));
    updateAll();
}
function selectDefaultGenres() {
    selectedGenres = new Set(DATA.defaultGenres);
    picker.querySelectorAll('.genre-tag').forEach(t => {
        t.classList.toggle('active', selectedGenres.has(t.dataset.genre));
    });
    updateAll();
}

// ── Year range number inputs ──
const yearMinInput = document.getElementById('year-min');
const yearMaxInput = document.getElementById('year-max');

yearMinInput.addEventListener('change', () => {
    yearMin = Math.max(2005, Math.min(2025, parseInt(yearMinInput.value) || 2005));
    if (yearMin > yearMax) { yearMax = yearMin; yearMaxInput.value = yearMax; }
    yearMinInput.value = yearMin;
    updateAll();
});
yearMaxInput.addEventListener('change', () => {
    yearMax = Math.max(2005, Math.min(2025, parseInt(yearMaxInput.value) || 2025));
    if (yearMax < yearMin) { yearMin = yearMax; yearMinInput.value = yearMin; }
    yearMaxInput.value = yearMax;
    updateAll();
});

// ── Tier checkboxes ──
['indie', 'midtier', 'aaa'].forEach(id => {
    const tierMap = { indie: 'Indie', midtier: 'Mid-tier', aaa: 'AAA' };
    document.getElementById('tier-' + id).addEventListener('change', (e) => {
        if (e.target.checked) selectedTiers.add(tierMap[id]);
        else selectedTiers.delete(tierMap[id]);
        updateAll();
    });
});


// ══════════════════════════════════════════════════════════════════════════════
// FILTER HELPERS
// ══════════════════════════════════════════════════════════════════════════════
function filterYearly() {
    return DATA.yearly.filter(r =>
        selectedGenres.has(r.genre_name) &&
        selectedTiers.has(r.tier) &&
        r.release_year >= yearMin && r.release_year <= yearMax
    );
}

function filterPriceBrackets() {
    return DATA.priceBrackets.filter(r =>
        selectedGenres.has(r.genre_name) && selectedTiers.has(r.tier)
    );
}

function filterTopGames() {
    return DATA.topGames.filter(g =>
        selectedTiers.has(g.tier) &&
        g.year >= yearMin && g.year <= yearMax &&
        (selectedGenres.size === 0 || g.genres.some(gn => selectedGenres.has(gn)))
    );
}


// ══════════════════════════════════════════════════════════════════════════════
// CHART RENDERERS
// ══════════════════════════════════════════════════════════════════════════════

function renderKPIs() {
    const rows = filterYearly();
    let totalGames = 0, freeGames = 0;
    let priceSum = 0, priceCount = 0;
    let metaSum = 0, metaCount = 0;
    const yearCounts = {};

    rows.forEach(r => {
        totalGames += r.game_count;
        freeGames += r.free_count;
        if (r.avg_price > 0) { priceSum += r.avg_price * r.game_count; priceCount += r.game_count; }
        if (r.avg_metacritic > 0) { metaSum += r.avg_metacritic * r.metacritic_count; metaCount += r.metacritic_count; }
        yearCounts[r.release_year] = (yearCounts[r.release_year] || 0) + r.game_count;
    });

    const paidGames = totalGames - freeGames;
    const avgPrice = priceCount > 0 ? (priceSum / priceCount) : 0;
    const avgMeta = metaCount > 0 ? (metaSum / metaCount) : 0;

    let peakYear = '-', peakCount = 0;
    for (const [yr, cnt] of Object.entries(yearCounts)) {
        if (cnt > peakCount) { peakYear = yr; peakCount = cnt; }
    }

    // Exactly 4 KPIs per row (2 rows of 4)
    const kpis = [
        { label: 'Total Games', value: totalGames.toLocaleString() },
        { label: 'Paid Games', value: paidGames.toLocaleString() },
        { label: 'Free Games', value: freeGames.toLocaleString() },
        { label: 'Avg Price', value: '$' + avgPrice.toFixed(2) },
        { label: 'Avg Metacritic', value: avgMeta > 0 ? avgMeta.toFixed(1) : 'N/A' },
        { label: 'Peak Year', value: peakYear },
        { label: 'Peak Year Games', value: peakCount.toLocaleString() },
        { label: 'Free-to-Play %', value: totalGames > 0 ? (freeGames / totalGames * 100).toFixed(1) + '%' : '0%' },
    ];

    const container = document.getElementById('kpi-row');
    container.innerHTML = kpis.map(k =>
        `<div class="kpi-card"><div class="kpi-value">${k.value}</div><div class="kpi-label">${k.label}</div></div>`
    ).join('');

    // Update the filter summary in the sticky bar
    const summaryEl = document.getElementById('filter-summary');
    if (summaryEl) {
        summaryEl.textContent = `${selectedGenres.size} genres · ${yearMin}–${yearMax} · ${selectedTiers.size} tiers`;
    }
}


function renderYearly() {
    const rows = filterYearly();
    // Sum game_count per year across genres + tiers
    const yearMap = {};
    rows.forEach(r => { yearMap[r.release_year] = (yearMap[r.release_year] || 0) + r.game_count; });
    const years = Object.keys(yearMap).sort();
    const counts = years.map(y => yearMap[y]);
    const maxCount = Math.max(...counts);
    const colors = counts.map(c => c === maxCount ? STEAM.accent_orange : STEAM.light_blue);

    Plotly.react('chart-yearly', [{
        type: 'bar', x: years, y: counts,
        marker: { color: colors },
        hovertemplate: '%{x}: %{y:,} games<extra></extra>',
    }], {
        ...LAYOUT_BASE,
        xaxis: { ...LAYOUT_BASE.xaxis, title: 'Year' },
        yaxis: { ...LAYOUT_BASE.yaxis, title: 'Games Released' },
        margin: { l: 60, r: 20, t: 10, b: 50 },
    }, { responsive: true });
}


function renderTreemap() {
    const rows = filterYearly();
    // Sum game_count per genre
    const genreMap = {};
    rows.forEach(r => { genreMap[r.genre_name] = (genreMap[r.genre_name] || 0) + r.game_count; });
    // Sort and take top 15
    const sorted = Object.entries(genreMap).sort((a, b) => b[1] - a[1]).slice(0, 15);
    const labels = sorted.map(s => s[0]);
    const values = sorted.map(s => s[1]);

    Plotly.react('chart-treemap', [{
        type: 'treemap',
        labels: labels, parents: labels.map(() => ''), values: values,
        textinfo: 'label+value',
        texttemplate: '<b>%{label}</b><br>%{value:,}',
        marker: {
            colors: values,
            colorscale: [[0, '#2a475e'], [1, '#66c0f4']],
            line: { width: 2, color: '#1b2838' },
        },
        hovertemplate: '<b>%{label}</b><br>%{value:,} games<extra></extra>',
    }], {
        paper_bgcolor: '#1b2838',
        margin: { l: 5, r: 5, t: 5, b: 5 },
    }, { responsive: true });
}


function renderPriceDist() {
    const rows = filterPriceBrackets();
    // Sum game_count per bracket across genres + tiers
    const bracketMap = {};
    DATA.bracketLabels.forEach(b => { bracketMap[b] = 0; });
    rows.forEach(r => { bracketMap[r.price_bracket] = (bracketMap[r.price_bracket] || 0) + r.game_count; });
    const brackets = DATA.bracketLabels;
    const counts = brackets.map(b => bracketMap[b]);

    Plotly.react('chart-price-dist', [{
        type: 'bar', x: brackets, y: counts,
        marker: { color: COLORS.slice(0, brackets.length) },
        text: counts.map(c => c.toLocaleString()),
        textposition: 'outside',
        textfont: { color: '#c7d5e0', size: 10 },
        hovertemplate: '%{x}: %{y:,} games<extra></extra>',
    }], {
        ...LAYOUT_BASE,
        xaxis: { ...LAYOUT_BASE.xaxis, title: 'Price Bracket' },
        yaxis: { ...LAYOUT_BASE.yaxis, title: 'Number of Games' },
        margin: { l: 60, r: 20, t: 10, b: 50 },
        showlegend: false,
    }, { responsive: true });
}


function renderSweetSpot() {
    const rows = filterPriceBrackets();
    // Group by genre + bracket, sum game_count and weighted avg_recs
    const key = (g, b) => g + '|||' + b;
    const agg = {};
    rows.forEach(r => {
        const k = key(r.genre_name, r.price_bracket);
        if (!agg[k]) agg[k] = { genre: r.genre_name, bracket: r.price_bracket, recsSum: 0, countSum: 0 };
        agg[k].recsSum += r.avg_recs * r.game_count;
        agg[k].countSum += r.game_count;
    });

    // Find best bracket per genre (min 10 games)
    const genreBest = {};
    Object.values(agg).forEach(a => {
        if (a.countSum < 10) return;
        const avgRecs = a.recsSum / a.countSum;
        if (!genreBest[a.genre] || avgRecs > genreBest[a.genre].avgRecs) {
            genreBest[a.genre] = { bracket: a.bracket, avgRecs: avgRecs, count: a.countSum };
        }
    });

    const sorted = Object.entries(genreBest)
        .sort((a, b) => a[1].avgRecs - b[1].avgRecs);
    const labels = sorted.map(s => s[0] + '  (' + s[1].bracket + ')');
    const values = sorted.map(s => s[1].avgRecs);

    if (labels.length === 0) {
        Plotly.react('chart-sweet-spot', [], {
            ...LAYOUT_BASE,
            margin: { l: 60, r: 30, t: 30, b: 50 },
            annotations: [{ text: 'No data for current filters', showarrow: false,
                x: 0.5, y: 0.5, xref: 'paper', yref: 'paper',
                font: { size: 16, color: '#8f98a0' } }],
        }, { responsive: true });
        return;
    }

    Plotly.react('chart-sweet-spot', [{
        type: 'bar', y: labels, x: values, orientation: 'h',
        marker: { color: STEAM.light_blue },
        text: values.map(v => Math.round(v).toLocaleString() + ' avg recs'),
        textposition: 'outside',
        textfont: { color: '#c7d5e0', size: 10 },
    }], {
        paper_bgcolor: '#1b2838',
        plot_bgcolor: '#2a475e',
        font: { color: '#c7d5e0', size: 12 },
        xaxis: { title: 'Avg Recommendations', gridcolor: 'rgba(198,213,224,0.15)',
                 zerolinecolor: 'rgba(198,213,224,0.3)' },
        yaxis: { type: 'category', automargin: true,
                 gridcolor: 'rgba(198,213,224,0.15)' },
        margin: { l: 10, r: 80, t: 10, b: 50 },
    }, { responsive: true });
}


function renderTierCompare() {
    const rows = filterYearly();
    const tiers = ['Indie', 'Mid-tier', 'AAA'];
    const tierColors = { 'Indie': '#66c0f4', 'Mid-tier': '#ff9800', 'AAA': '#4caf50' };

    // Aggregate per tier
    const tierStats = {};
    tiers.forEach(t => {
        tierStats[t] = { priceSum: 0, priceCount: 0, freeCount: 0, totalCount: 0,
                         metaSum: 0, metaCount: 0, recsSum: 0, recsCount: 0 };
    });
    rows.forEach(r => {
        const s = tierStats[r.tier];
        if (!s) return;
        s.totalCount += r.game_count;
        s.freeCount += r.free_count;
        if (r.median_price > 0) { s.priceSum += r.median_price * r.game_count; s.priceCount += r.game_count; }
        if (r.avg_metacritic > 0) { s.metaSum += r.avg_metacritic * r.metacritic_count; s.metaCount += r.metacritic_count; }
        s.recsSum += r.median_recs * r.game_count;
        s.recsCount += r.game_count;
    });

    const metrics = [
        { title: 'Median Price ($)', fn: s => s.priceCount > 0 ? s.priceSum / s.priceCount : 0, fmt: v => '$' + v.toFixed(2) },
        { title: 'F2P Rate (%)', fn: s => s.totalCount > 0 ? (s.freeCount / s.totalCount * 100) : 0, fmt: v => v.toFixed(1) + '%' },
        { title: 'Avg Metacritic', fn: s => s.metaCount > 0 ? s.metaSum / s.metaCount : 0, fmt: v => v.toFixed(1) },
        { title: 'Median Recs', fn: s => s.recsCount > 0 ? s.recsSum / s.recsCount : 0, fmt: v => Math.round(v).toLocaleString() },
    ];

    const traces = [];
    tiers.forEach(tier => {
        if (!selectedTiers.has(tier)) return;
        const s = tierStats[tier];
        traces.push({
            type: 'bar',
            x: metrics.map(m => m.title),
            y: metrics.map(m => m.fn(s)),
            name: tier,
            marker: { color: tierColors[tier] },
            text: metrics.map(m => m.fmt(m.fn(s))),
            textposition: 'outside',
            textfont: { color: '#c7d5e0', size: 11 },
        });
    });

    Plotly.react('chart-tier-compare', traces, {
        ...LAYOUT_BASE,
        barmode: 'group',
        legend: {
            bgcolor: 'rgba(42,71,94,0.8)', bordercolor: '#c7d5e0', borderwidth: 1,
            orientation: 'h', yanchor: 'bottom', y: 1.05, xanchor: 'center', x: 0.5,
        },
        margin: { l: 60, r: 30, t: 50, b: 60 },
    }, { responsive: true });
}


function renderForecast() {
    const traces = [];
    let colorIdx = 0;

    DATA.forecastGenres.forEach(genre => {
        if (!selectedGenres.has(genre)) return;
        const fd = DATA.forecast[genre];
        if (!fd) return;
        const color = COLORS[colorIdx % COLORS.length];
        colorIdx++;

        // Actual line
        traces.push({
            type: 'scatter', mode: 'lines',
            x: fd.actual_periods,
            y: fd.actual_scores,
            name: genre,
            line: { color: color, width: 2.5 },
            legendgroup: genre,
        });

        // Forecast (dashed), connecting from last actual point
        const lastActual = fd.actual_scores.filter(v => v !== null).pop() || fd.forecast_scores[0];
        traces.push({
            type: 'scatter', mode: 'lines',
            x: [fd.actual_periods[fd.actual_periods.length - 1], ...fd.forecast_periods],
            y: [lastActual, ...fd.forecast_scores],
            name: genre + ' (forecast)',
            line: { color: color, width: 2.5, dash: 'dash' },
            legendgroup: genre,
            showlegend: false,
        });
    });

    const fPeriods = DATA.forecast[DATA.forecastGenres[0]].forecast_periods;
    const shapes = [{
        type: 'rect',
        xref: 'x', yref: 'paper',
        x0: fPeriods[0], x1: fPeriods[fPeriods.length - 1],
        y0: 0, y1: 1,
        fillcolor: 'rgba(255,152,0,0.08)',
        line: { color: 'rgba(255,152,0,0.4)', width: 2, dash: 'dot' },
    }];

    const annotations = [{
        x: fPeriods[Math.floor(fPeriods.length / 2)],
        y: 1.02, yref: 'paper',
        text: 'FORECAST', showarrow: false,
        font: { size: 18, color: '#ff9800', family: 'Arial Black' },
        opacity: 0.8,
    }];

    Plotly.react('chart-forecast', traces, {
        ...LAYOUT_BASE,
        shapes: shapes,
        annotations: annotations,
        xaxis: { ...LAYOUT_BASE.xaxis, title: 'Month', dtick: 3, tickangle: 45 },
        yaxis: { ...LAYOUT_BASE.yaxis, title: 'Popularity Score' },
        legend: {
            bgcolor: 'rgba(42,71,94,0.9)', bordercolor: '#c7d5e0', borderwidth: 1,
            font: { size: 11 },
        },
        margin: { l: 60, r: 30, t: 40, b: 70 },
    }, { responsive: true });
}


function renderTopGames() {
    const filtered = filterTopGames().slice(0, 20);
    // Sort ascending for horizontal bar (bottom = highest)
    filtered.sort((a, b) => a.recs - b.recs);

    const names = filtered.map(g => g.name);
    const recs = filtered.map(g => g.recs);
    const colors = filtered.map(g => g.is_free ? STEAM.accent_green : STEAM.light_blue);

    if (names.length === 0) {
        Plotly.react('chart-top-games', [], {
            ...LAYOUT_BASE,
            annotations: [{ text: 'No games match current filters', showarrow: false,
                x: 0.5, y: 0.5, xref: 'paper', yref: 'paper',
                font: { size: 16, color: '#8f98a0' } }],
        }, { responsive: true });
        return;
    }

    Plotly.react('chart-top-games', [{
        type: 'bar', y: names, x: recs, orientation: 'h',
        marker: { color: colors },
        text: recs.map(r => r.toLocaleString()),
        textposition: 'outside',
        textfont: { color: '#c7d5e0', size: 11 },
        hovertemplate: '<b>%{y}</b><br>Recommendations: %{x:,}<extra></extra>',
    }], {
        paper_bgcolor: '#1b2838',
        plot_bgcolor: '#2a475e',
        font: { color: '#c7d5e0', size: 12 },
        xaxis: { title: 'Total Recommendations', gridcolor: 'rgba(198,213,224,0.15)',
                 zerolinecolor: 'rgba(198,213,224,0.3)' },
        yaxis: { type: 'category', automargin: true,
                 gridcolor: 'rgba(198,213,224,0.15)' },
        margin: { l: 220, r: 80, t: 10, b: 50 },
        annotations: [{
            x: 0.98, y: 0.02, xref: 'paper', yref: 'paper',
            text: '<span style="color:#4caf50">&#9632;</span> Free  <span style="color:#66c0f4">&#9632;</span> Paid',
            showarrow: false, font: { size: 13, color: '#c7d5e0' },
            bgcolor: 'rgba(42,71,94,0.8)', bordercolor: '#c7d5e0', borderwidth: 1, borderpad: 6,
        }],
    }, { responsive: true });
}


function renderSentiment() {
    const s = DATA.sentiment;
    const distTrace = {
        type: 'pie',
        labels: ['Positive (> 0.05)', 'Neutral (-0.05 to 0.05)', 'Negative (< -0.05)'],
        values: [s.distribution.positive, s.distribution.neutral, s.distribution.negative],
        marker: {
            colors: [STEAM.accent_green, STEAM.accent_orange, STEAM.accent_red],
            line: { color: '#1b2838', width: 2 },
        },
        textinfo: 'label+percent',
        textfont: { size: 11 },
        domain: { x: [0, 0.45], y: [0, 1] },
        hovertemplate: '%{label}<br>%{value:,} reviews<extra></extra>',
    };

    const thumbsTrace = {
        type: 'bar',
        x: ['Thumbs Up', 'Thumbs Down'],
        y: [s.thumbs.up, s.thumbs.down],
        marker: { color: [STEAM.accent_green, STEAM.accent_red] },
        text: [s.thumbs.up.toFixed(3), s.thumbs.down.toFixed(3)],
        textposition: 'outside',
        textfont: { color: '#c7d5e0', size: 14 },
        xaxis: 'x2', yaxis: 'y2',
        showlegend: false,
    };

    Plotly.react('chart-sentiment', [distTrace, thumbsTrace], {
        paper_bgcolor: '#1b2838', plot_bgcolor: '#2a475e',
        font: { color: '#c7d5e0', size: 12 },
        showlegend: false,
        margin: { l: 40, r: 30, t: 40, b: 50 },
        xaxis2: { domain: [0.55, 1], gridcolor: 'rgba(198,213,224,0.15)' },
        yaxis2: { anchor: 'x2', gridcolor: 'rgba(198,213,224,0.15)', title: 'Avg Polarity' },
        annotations: [
            { text: 'Sentiment Distribution', x: 0.22, y: 1.08, xref: 'paper', yref: 'paper',
              showarrow: false, font: { size: 14, color: '#c7d5e0' } },
            { text: 'TextBlob vs Thumbs Vote', x: 0.78, y: 1.08, xref: 'paper', yref: 'paper',
              showarrow: false, font: { size: 14, color: '#c7d5e0' } },
        ],
    }, { responsive: true });
}


// ══════════════════════════════════════════════════════════════════════════════
// MASTER UPDATE — called whenever any filter changes
// ══════════════════════════════════════════════════════════════════════════════
function updateAll() {
    renderKPIs();
    renderYearly();
    renderTreemap();
    renderPriceDist();
    renderSweetSpot();
    renderTierCompare();
    renderForecast();
    renderTopGames();
    // Sentiment is static, only render once
}

// Initial render
updateAll();
renderSentiment();

</script>
</body>
</html>"""

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"\nDashboard saved to: {OUTPUT_PATH}")
print(f"File size: {file_size_mb:.1f} MB")
print(f"Open in browser: file://{OUTPUT_PATH}")
