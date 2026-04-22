# 📈 EGX Stock Screener & Portfolio Dashboard

A full-featured stock analysis dashboard for the **Egyptian Stock Exchange (EGX)** built with Python + Streamlit. No paid APIs required — uses web scraping with intelligent fallbacks.

---

## 🚀 Quick Start

### 1. Clone / Download the project

```bash
# If downloaded as zip, extract it, then:
cd egx_dashboard
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate:
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Open your browser to: **http://localhost:8501**

---

## 🗂️ Project Structure

```
egx_dashboard/
│
├── app.py                        # Main Streamlit application (all 7 pages)
├── requirements.txt              # Python dependencies
│
├── scrapers/
│   ├── __init__.py
│   └── data_scraper.py           # All web scraping logic
│       ├── scrape_yahoo_egx_stocks()   # Yahoo Finance scraper
│       ├── scrape_yahoo_history()      # Historical OHLCV scraper
│       ├── scrape_financial_news()     # Multi-source news scraper
│       ├── fetch_all_stocks()          # Master fetcher with fallback
│       └── _generate_demo_market_data()  # Fallback demo data
│
└── analytics/
    ├── __init__.py
    ├── technical.py              # SMA, EMA, RSI, MACD, Bollinger Bands, signals
    ├── monte_carlo.py            # GBM-based Monte Carlo simulation engine
    ├── sentiment.py              # AI + rule-based news sentiment analysis
    ├── recommendation.py         # Combined recommendation engine
    ├── portfolio.py              # Portfolio state management & P&L
    └── charts.py                 # All Plotly chart builders
```

---

## 📄 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Market Overview | All EGX stocks with filtering, sorting, sector breakdown |
| 📰 News & Sentiment | Scraped financial news with AI/NLP sentiment analysis |
| 💰 Trading & Transactions | Trade logger, P&L calculator, trade history |
| 📊 Technical Analysis | Full OHLCV chart + RSI, MACD, Bollinger Bands, signals |
| 🎯 Recommendation Engine | Combined AI recommendation with full explanation |
| 🧮 Monte Carlo Forecast | 1,000+ simulated price paths, distribution, risk metrics |
| 📂 Portfolio Tracker | Holdings tracker, unrealized P&L, risk exposure per holding |

---

## 🔑 Optional: AI-Powered Sentiment (Anthropic)

For **AI-enhanced** news sentiment analysis (vs. rule-based fallback):

1. Get an API key from [console.anthropic.com](https://console.anthropic.com)
2. Enter it in the sidebar under **"Anthropic API Key (optional)"**
3. Or set it as an environment variable:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

Without the key, the app uses a fast rule-based NLP sentiment analyzer — fully functional.

---

## 🏗️ Architecture & Future-Proofing

### Adding a new data source

The scraper is **fully modular**. To add a new data provider:

```python
# In scrapers/data_scraper.py

def scrape_my_new_source() -> pd.DataFrame:
    """Returns DataFrame with columns: symbol, name, price, change, change_pct, volume"""
    ...

# Then add to fetch_all_stocks():
df = scrape_my_new_source()
if not df.empty:
    return _clean_market_data(df)
```

### Adding an official API (future)

When an official EGX API becomes available:

```python
# Create: scrapers/egx_api.py
def fetch_from_egx_api(api_key: str) -> pd.DataFrame:
    """Official EGX API integration"""
    ...
```

Then import and call it first in `fetch_all_stocks()` before the scraper fallbacks.

---

## 📊 Technical Analysis Indicators

| Indicator | Parameters | Signal Logic |
|-----------|-----------|-------------|
| SMA | 20, 50 periods | Golden/Death Cross |
| EMA | 12, 26 periods | Trend direction |
| RSI | 14 periods | <30 oversold, >70 overbought |
| MACD | 12/26/9 | Crossover signals |
| Bollinger Bands | 20 periods, 2σ | Price at band edges |

---

## 🧮 Monte Carlo Simulation

Uses **Geometric Brownian Motion (GBM)**:

```
S(t) = S(0) · exp((μ - σ²/2)·t + σ·W(t))
```

Where:
- `μ` = estimated daily drift from historical returns
- `σ` = estimated daily volatility from historical returns  
- `W(t)` = Wiener process (standard normal random shocks)

Outputs: probability of gain/loss, 5th/25th/75th/95th percentile price bands, annualized volatility, 95% Value at Risk.

---

## 🛡️ Robustness Features

- **Rate limiting**: Random delays between requests (0.5–1.5s)
- **Exponential backoff**: 3 retries with increasing delay
- **Rotating user agents**: 4 browser signatures
- **Graceful fallback**: Auto-generates realistic demo data if scraping fails
- **Data validation**: All numeric values coerced and NaN-cleaned
- **Error isolation**: Each scraper fails independently without crashing the app

---

## 🔧 Troubleshooting

**"No data loaded"** → Click **🔄 Refresh All Data** in the sidebar

**Slow loading** → The app is scraping live data; first load takes 10–30 seconds

**Scraping blocked** → The app automatically falls back to realistic demo data based on actual EGX stock characteristics

**Chart not showing** → Select a stock and click the run/generate button

---

## ⚖️ Disclaimer

This dashboard is for **educational and informational purposes only**. It does not constitute financial advice. Always do your own research before making investment decisions.
