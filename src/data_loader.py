"""
data_loader.py - Centralized data loading and cleaning for Steam Games Analysis

This module handles all the data loading, cleaning, and merging so that
each notebook can just import the functions it needs without repeating
boilerplate code.
"""

import os
import pandas as pd
import numpy as np

# Path to the data directory (relative to the project root)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_applications():
    """
    Load and clean the main applications (games) dataset.
    Filters to only 'game' type entries and cleans up data types.
    """
    print("Loading applications data...")
    apps = pd.read_csv(os.path.join(DATA_DIR, "applications.csv"))

    # Filter to only games (not DLC, demos, etc.)
    games = apps[apps["type"] == "game"].copy()
    print(f"  Filtered to {len(games)} games out of {len(apps)} total applications")

    # Convert release_date to datetime
    games["release_date"] = pd.to_datetime(games["release_date"], errors="coerce")

    # Ensure is_free is boolean (handle both string and bool formats)
    if games["is_free"].dtype == object:
        games["is_free"] = games["is_free"].map({"True": True, "False": False})
    else:
        games["is_free"] = games["is_free"].astype(bool)

    # Convert prices from cents to dollars
    games["price_dollars"] = games["mat_final_price"].fillna(0) / 100
    games["original_price_dollars"] = games["mat_initial_price"].fillna(0) / 100

    # Extract year and month for time-based analysis
    games["release_year"] = games["release_date"].dt.year
    games["release_month"] = games["release_date"].dt.month
    games["release_year_month"] = games["release_date"].dt.to_period("M")

    print(f"  Date range: {games['release_date'].min()} to {games['release_date'].max()}")
    print("  Done!")

    return games


def load_genres():
    """Load the genres lookup table."""
    print("Loading genres...")
    genres = pd.read_csv(os.path.join(DATA_DIR, "genres.csv"))
    print(f"  Found {len(genres)} genres")
    return genres


def load_application_genres():
    """Load the junction table linking apps to genres."""
    print("Loading application-genre mappings...")
    app_genres = pd.read_csv(os.path.join(DATA_DIR, "application_genres.csv"))
    print(f"  Found {len(app_genres)} mappings")
    return app_genres


def load_categories():
    """Load the categories lookup table."""
    categories = pd.read_csv(os.path.join(DATA_DIR, "categories.csv"))
    return categories


def load_application_categories():
    """Load the junction table linking apps to categories."""
    app_cats = pd.read_csv(os.path.join(DATA_DIR, "application_categories.csv"))
    return app_cats


def load_developers():
    """Load the developers lookup table."""
    devs = pd.read_csv(os.path.join(DATA_DIR, "developers.csv"))
    return devs


def load_application_developers():
    """Load the junction table linking apps to developers."""
    app_devs = pd.read_csv(os.path.join(DATA_DIR, "application_developers.csv"))
    return app_devs


def load_publishers():
    """Load the publishers lookup table."""
    pubs = pd.read_csv(os.path.join(DATA_DIR, "publishers.csv"))
    return pubs


def load_application_publishers():
    """Load the junction table linking apps to publishers."""
    app_pubs = pd.read_csv(os.path.join(DATA_DIR, "application_publishers.csv"))
    return app_pubs


def load_platforms():
    """Load the platforms lookup table."""
    platforms = pd.read_csv(os.path.join(DATA_DIR, "platforms.csv"))
    return platforms


def load_application_platforms():
    """Load the junction table linking apps to platforms."""
    app_plats = pd.read_csv(os.path.join(DATA_DIR, "application_platforms.csv"))
    return app_plats


def load_reviews(sample_size=None):
    """
    Load the reviews dataset.

    Args:
        sample_size: If provided, randomly sample this many reviews to save memory.
                     The full dataset is ~4M rows which can be slow to process.
    """
    print("Loading reviews data...")

    if sample_size:
        # Read the full file but sample to keep memory manageable
        reviews = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))
        print(f"  Full dataset: {len(reviews)} reviews")
        reviews = reviews.sample(n=min(sample_size, len(reviews)), random_state=42)
        print(f"  Sampled down to: {len(reviews)} reviews")
    else:
        reviews = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))
        print(f"  Loaded {len(reviews)} reviews")

    # Convert timestamps from Unix epoch to datetime
    reviews["review_date"] = pd.to_datetime(
        reviews["timestamp_created"], unit="s", errors="coerce"
    )
    reviews["review_updated"] = pd.to_datetime(
        reviews["timestamp_updated"], unit="s", errors="coerce"
    )

    # Convert voted_up from string to boolean if needed
    if reviews["voted_up"].dtype == object:
        reviews["voted_up"] = reviews["voted_up"].map({"True": True, "False": False})

    print("  Done!")
    return reviews


def build_games_with_genres(games=None):
    """
    Merge games with their genre information.
    Returns a DataFrame where each row is a game-genre pair.

    A single game can appear multiple times if it has multiple genres.
    """
    if games is None:
        games = load_applications()

    genres = load_genres()
    app_genres = load_application_genres()

    # Merge: app_genres -> genres (to get genre names)
    genre_mapped = app_genres.merge(genres, left_on="genre_id", right_on="id", how="left")
    genre_mapped = genre_mapped.rename(columns={"name": "genre_name"})

    # Merge: games -> genre_mapped (to attach genre info to each game)
    games_genres = games.merge(genre_mapped, on="appid", how="left")

    print(f"Built games-with-genres dataset: {len(games_genres)} rows")
    return games_genres


def build_games_with_developers(games=None):
    """
    Merge games with their developer information.
    """
    if games is None:
        games = load_applications()

    devs = load_developers()
    app_devs = load_application_developers()

    dev_mapped = app_devs.merge(devs, left_on="developer_id", right_on="id", how="left")
    dev_mapped = dev_mapped.rename(columns={"name": "developer_name"})

    games_devs = games.merge(dev_mapped, on="appid", how="left")

    print(f"Built games-with-developers dataset: {len(games_devs)} rows")
    return games_devs


def build_games_with_publishers(games=None):
    """
    Merge games with their publisher information.
    """
    if games is None:
        games = load_applications()

    pubs = load_publishers()
    app_pubs = load_application_publishers()

    pub_mapped = app_pubs.merge(pubs, left_on="publisher_id", right_on="id", how="left")
    pub_mapped = pub_mapped.rename(columns={"name": "publisher_name"})

    games_pubs = games.merge(pub_mapped, on="appid", how="left")

    print(f"Built games-with-publishers dataset: {len(games_pubs)} rows")
    return games_pubs


# -- Shared chart styling constants --
# Steam-inspired color palette
STEAM_COLORS = {
    "dark_blue": "#1b2838",
    "medium_blue": "#2a475e",
    "light_blue": "#66c0f4",
    "off_white": "#c7d5e0",
    "accent_green": "#4caf50",
    "accent_orange": "#ff9800",
    "accent_red": "#f44336",
}

# Color list for charts with multiple series
CHART_COLORS = [
    "#66c0f4",  # Steam light blue
    "#ff9800",  # Orange
    "#4caf50",  # Green
    "#f44336",  # Red
    "#9c27b0",  # Purple
    "#00bcd4",  # Cyan
    "#ffeb3b",  # Yellow
    "#e91e63",  # Pink
    "#3f51b5",  # Indigo
    "#8bc34a",  # Light green
]
