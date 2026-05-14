# Steam Games Analysis - Predictive Genre Trends & Market Insights

A data science portfolio project that analyzes **240,000+ Steam games** and **4 million reviews** to predict gaming genre trends and uncover market insights for game development planning.

## Project Goal

As an aspiring game developer, I wanted to answer:
- **What genres will be popular in the next 12 months?**
- **What's the optimal price point for each genre?**
- **Do players actually enjoy the games they buy?** (Sentiment analysis)
- **Can indie developers compete with AAA studios?**

## Key Findings

| Insight | Details |
|---------|---------|
| Genre Trends | Predicted top trending genres for the next 12 months using polynomial regression |
| Price Sweet Spot | Found optimal price ranges that maximize player satisfaction per genre |
| Sentiment Gap | Discovered which genres have the biggest gap between hype and actual player satisfaction |
| Indie Opportunity | Identified genres where indie developers consistently outperform AAA studios |

## Screenshots

### Genre trend forecast
![Genre Trends](docs/screenshots/genre_trend_forecast.png)

### Price sweet spot analysis
![Price Analysis](docs/screenshots/price_sweetspot.png)

### Interactive dashboard
![Dashboard](docs/screenshots/dashboard_overview.png)

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 00 | [Executive Summary](notebooks/00_executive_summary.ipynb) | High-level KPIs, key charts, and takeaways from all analyses |
| 01 | [Data Exploration](notebooks/01_data_exploration.ipynb) | EDA on 240K games — distributions, trends, and dataset overview |
| 02 | [Genre Trend Prediction](notebooks/02_genre_trends.ipynb) | ML model predicting genre popularity for the next 12 months |
| 03 | [Price Sweet-Spot Analysis](notebooks/03_price_analysis.ipynb) | Finding the optimal price point per genre |
| 04 | [Sentiment Analysis](notebooks/04_sentiment_analysis.ipynb) | NLP on 100K reviews — what do players really think? |
| 05 | [Indie vs AAA](notebooks/05_indie_vs_aaa.ipynb) | Comparing indie and AAA games across success metrics |

An **interactive HTML dashboard** with genre/year/publisher filters is also available at `output/dashboard.html`.

## Tech Stack

- **Python** — pandas, numpy, scikit-learn, statsmodels
- **Visualization** — Plotly (interactive), Matplotlib, Seaborn
- **NLP** — TextBlob for sentiment analysis
- **ML** — Linear Regression with Polynomial Features for time series prediction

## Dataset

**Source:** [Steam Dataset 2025](https://github.com/vintagedon/steam-dataset-2025) by vintagedon
**License:** CC BY 4.0 | **DOI:** [10.5281/zenodo.17286923](https://doi.org/10.5281/zenodo.17286923)

A point-in-time snapshot (September 2025) of the Steam catalog collected exclusively from official Steam Web APIs. No scraping or third-party estimates.

| Table | Records | Description |
|-------|---------|-------------|
| applications | 239,664 | Full accessible Steam catalog (games, DLC, demos, software) |
| reviews | 1,048,148 | User reviews with metadata, playtime, and votes |
| developers | 54,321 | Developer entities |
| publishers | 39,876 | Publisher entities |
| genres | 154 | Genre labels (includes non-English names) |
| categories | 462 | Steam store categories (multiplayer, controller support, etc.) |

Junction tables link applications to genres, categories, platforms, developers, and publishers via many-to-many relationships.

### Known Data Limitations

These limitations are documented by the dataset author and are worth noting when interpreting results:

- **56% retrieval rate** — 134K of 240K apps returned full metadata. Failures are mostly delisted games (~45K), regional restrictions (~28K), and invalid IDs (~19K)
- **US-centric collection** — Pricing is USD-only. Coverage: North America ~95%, Western Europe ~90%, Asia-Pacific ~60%, China ~20%
- **Non-English genre names** — ~32% of retrieved applications have non-English metadata. Some popular games (e.g., Counter-Strike 2) only have Arabic genre tags in this dataset
- **Incomplete review counts** — The `recommendations_total` field reflects the API snapshot, not lifetime totals. Major titles like Dota 2 show far fewer recommendations than reality
- **Junk release dates** — Some games have placeholder dates (e.g., year 9998) set by developers. We filter to ≤2025 in our analysis
- **Prices in cents** — The `mat_final_price` field stores values in cents (999 = $9.99)

## How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/SteamGamesAnalysis.git
cd SteamGamesAnalysis

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place the Steam dataset CSVs in the data/ directory

# Launch Jupyter
jupyter notebook notebooks/
```

## Project Structure

```
SteamGamesAnalysis/
├── data/                    # Raw CSV data (not tracked in git)
├── notebooks/               # Jupyter notebooks (main deliverables)
│   ├── 01_data_exploration.ipynb
│   ├── 02_genre_trends.ipynb
│   ├── 03_price_analysis.ipynb
│   ├── 04_sentiment_analysis.ipynb
│   └── 05_indie_vs_aaa.ipynb
├── src/                     # Reusable Python modules
│   ├── data_loader.py       # Data loading and cleaning
│   └── utils.py             # Chart styling helpers
├── scripts/
│   └── generate_dashboard.py  # Generates the interactive HTML dashboard
├── output/                  # Generated dashboard and charts (not tracked in git)
├── requirements.txt
└── README.md
```

## Author

Built as a portfolio project to demonstrate data analysis, machine learning, and visualization skills applied to the gaming industry.
