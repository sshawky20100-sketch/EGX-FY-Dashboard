"""
EGX Stock Screener & Portfolio Dashboard
==========================================
Multi-page Streamlit app for the Egyptian Stock Exchange.

Run with:  streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from scrapers.data_scraper import fetch_all_stocks, fetch_stock_history, scrape_financial_news
from analytics.technical   import compute_all_indicators, generate_signal
from analytics.monte_carlo import run_monte_carlo, monte_carlo_to_recommendation
from analytics.sentiment   import analyze_news_sentiment, aggregate_sentiment_by_stock
from analytics.recommendation import generate_recommendation
from analytics.portfolio   import (
    init_portfolio, add_holding, remove_holding,
    get_portfolio, add_trade, get_trades, compute_portfolio_summary,
)
from analytics.charts import (
    candlestick_chart, technical_chart,
    monte_carlo_paths_chart, monte_carlo_distribution_chart,
    portfolio_pie_chart, portfolio_performance_chart, sentiment_bar_chart,
    COLORS,
)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EGX Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0D1117;
    --surface: #161B22;
    --border: #30363D;
    --green: #00C896;
    --red: #FF4B6E;
    --gold: #FFD166;
    --blue: #3D8EFF;
    --text: #E6EDF3;
    --muted: #8B949E;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--muted);
    transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: var(--text); }

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; }
[data-testid="stMetricLabel"] { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace; }

/* Tables */
.stDataFrame { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }

/* Signal badges */
.badge-buy  { background:#00C896; color:#000; padding:3px 10px; border-radius:4px; font-weight:600; font-family:'IBM Plex Mono',monospace; }
.badge-sell { background:#FF4B6E; color:#fff; padding:3px 10px; border-radius:4px; font-weight:600; font-family:'IBM Plex Mono',monospace; }
.badge-hold { background:#FFD166; color:#000; padding:3px 10px; border-radius:4px; font-weight:600; font-family:'IBM Plex Mono',monospace; }

/* Reason boxes */
.reason-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--blue);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.83rem;
    margin: 4px 0;
    color: var(--text);
}
.reason-section {
    color: var(--gold);
    font-weight: 600;
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 12px 0 4px;
}

/* News cards */
.news-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    margin: 6px 0;
}
.sentiment-pos { color: var(--green); font-weight: 600; }
.sentiment-neg { color: var(--red);   font-weight: 600; }
.sentiment-neu { color: var(--muted); font-weight: 600; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1e3a5f, #2d5a8e) !important;
    color: white !important;
    border: 1px solid var(--blue) !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2d5a8e, #3D8EFF) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(61,142,255,0.3) !important;
}

/* Page title */
.page-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--gold);
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

/* Divider */
hr { border-color: var(--border); }

/* Input widgets */
.stSelectbox > div, .stTextInput > div, .stNumberInput > div {
    background: var(--surface) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "market_data": pd.DataFrame(),
        "news_data":   [],
        "sentiment_by_company": {},
        "data_loaded": False,
        "last_refresh": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    init_portfolio()

init_state()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.5rem'>
      <div style='font-family:"IBM Plex Mono",monospace; font-size:1.4rem; color:#FFD166; font-weight:700'>
        📈 EGX
      </div>
      <div style='font-size:0.72rem; color:#8B949E; letter-spacing:0.15em; text-transform:uppercase; margin-top:2px'>
        Stock Dashboard
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠  Market Overview",
            "📰  News & Sentiment",
            "💰  Trading & Transactions",
            "📊  Technical Analysis",
            "🎯  Recommendation Engine",
            "🧮  Monte Carlo Forecast",
            "📂  Portfolio Tracker",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # API Key (optional for AI sentiment)
    api_key = st.text_input(
        "Anthropic API Key (optional)",
        type="password",
        placeholder="sk-ant-...",
        help="Optional: enables AI-powered news sentiment analysis",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.markdown("---")

    if st.button("🔄  Refresh All Data", use_container_width=True):
        with st.spinner("Fetching market data..."):
            st.session_state["market_data"]  = fetch_all_stocks()
        with st.spinner("Scraping news..."):
            raw_news = scrape_financial_news(30)
            st.session_state["news_data"] = analyze_news_sentiment(raw_news, api_key or None)
            st.session_state["sentiment_by_company"] = aggregate_sentiment_by_stock(
                st.session_state["news_data"]
            )
        st.session_state["data_loaded"]   = True
        st.session_state["last_refresh"]  = datetime.now().strftime("%H:%M:%S")
        st.success("✅ Data refreshed!")

    if st.session_state["last_refresh"]:
        st.caption(f"Last refresh: {st.session_state['last_refresh']}")
    else:
        st.caption("Click Refresh to load live data")

    # Auto-load on first visit
    if not st.session_state["data_loaded"]:
        with st.spinner("Loading initial data..."):
            st.session_state["market_data"]  = fetch_all_stocks()
            raw_news = scrape_financial_news(20)
            st.session_state["news_data"] = analyze_news_sentiment(raw_news, api_key or None)
            st.session_state["sentiment_by_company"] = aggregate_sentiment_by_stock(
                st.session_state["news_data"]
            )
            st.session_state["data_loaded"]  = True
            st.session_state["last_refresh"] = datetime.now().strftime("%H:%M:%S")
        st.rerun()


# Shorthand
mdf = st.session_state["market_data"]
news = st.session_state["news_data"]
sent = st.session_state["sentiment_by_company"]

_PAGE = page.split("  ", 1)[-1].strip()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if _PAGE == "Market Overview":
    st.markdown('<div class="page-title">📈 EGX Market Overview</div>', unsafe_allow_html=True)

    if mdf.empty:
        st.warning("No data loaded. Press **Refresh All Data** in the sidebar.")
    else:
        # KPI row
        gainers  = (mdf["change_pct"] > 0).sum()
        losers   = (mdf["change_pct"] < 0).sum()
        flat     = len(mdf) - gainers - losers
        avg_chg  = mdf["change_pct"].mean()
        top_gain = mdf.loc[mdf["change_pct"].idxmax()]
        top_loss = mdf.loc[mdf["change_pct"].idxmin()]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Stocks",  len(mdf))
        c2.metric("Gainers",  f"🟢 {gainers}")
        c3.metric("Losers",   f"🔴 {losers}")
        c4.metric("Avg Change", f"{avg_chg:+.2f}%",
                  delta_color="normal" if avg_chg >= 0 else "inverse")
        c5.metric("Top Gainer", top_gain["symbol"],
                  f"+{top_gain['change_pct']:.2f}%")

        st.markdown("---")

        # Filters
        col_s, col_sec, col_sort = st.columns([2, 2, 2])
        search   = col_s.text_input("🔍 Search", placeholder="Symbol or name...")
        sectors  = ["All"] + sorted(mdf["sector"].unique().tolist()) if "sector" in mdf.columns else ["All"]
        sec_flt  = col_sec.selectbox("Sector", sectors)
        sort_by  = col_sort.selectbox("Sort by", ["change_pct", "price", "volume", "symbol"])

        filtered = mdf.copy()
        if search:
            mask = (
                filtered["symbol"].str.contains(search.upper(), na=False) |
                filtered["name"].str.contains(search, case=False, na=False)
            )
            filtered = filtered[mask]
        if sec_flt != "All" and "sector" in filtered.columns:
            filtered = filtered[filtered["sector"] == sec_flt]

        filtered = filtered.sort_values(sort_by, ascending=(sort_by == "symbol"))

        # Color helpers for display
        def color_change(val):
            if val > 0:  return f'color: {COLORS["green"]}'
            if val < 0:  return f'color: {COLORS["red"]}'
            return f'color: {COLORS["muted"]}'

        display_cols = ["symbol", "name", "price", "change", "change_pct", "volume"]
        if "sector" in filtered.columns:
            display_cols += ["sector"]

        display = filtered[display_cols].copy()
        display.columns = [c.replace("_pct", " %").replace("_", " ").title() for c in display_cols]
        display.rename(columns={"Change %": "Chg %", "Change Pct": "Chg %"}, inplace=True)

        st.dataframe(
            display.style.applymap(color_change, subset=["Change", "Chg %"])
                         .format({
                             "Price": "{:.2f}",
                             "Change": "{:+.2f}",
                             "Chg %": "{:+.2f}%",
                             "Volume": "{:,.0f}",
                         }),
            use_container_width=True,
            height=480,
        )

        st.markdown("---")

        # Bubble chart: price vs change
        col_l, col_r = st.columns(2)
        with col_l:
            fig = px.scatter(
                filtered.head(25), x="price", y="change_pct",
                size="volume", color="change_pct",
                color_continuous_scale=["#FF4B6E", "#8B949E", "#00C896"],
                hover_name="symbol", text="symbol",
                title="Price vs Daily Change (bubble = volume)",
            )
            fig.update_traces(textposition="top center", textfont_size=9)
            fig.update_layout(
                paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
                font=dict(color=COLORS["text"]), height=380,
                xaxis=dict(gridcolor=COLORS["border"]),
                yaxis=dict(gridcolor=COLORS["border"]),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            if "sector" in filtered.columns:
                sec_agg = filtered.groupby("sector")["change_pct"].mean().reset_index()
                sec_agg.columns = ["Sector", "Avg Chg %"]
                fig2 = go.Figure(go.Bar(
                    x=sec_agg["Sector"], y=sec_agg["Avg Chg %"],
                    marker_color=[COLORS["green"] if v >= 0 else COLORS["red"]
                                  for v in sec_agg["Avg Chg %"]],
                ))
                fig2.update_layout(
                    title="Average Change by Sector",
                    paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
                    font=dict(color=COLORS["text"]), height=380,
                    xaxis=dict(gridcolor=COLORS["border"], tickangle=-30),
                    yaxis=dict(gridcolor=COLORS["border"]),
                )
                st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — NEWS & SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
elif _PAGE == "News & Sentiment":
    st.markdown('<div class="page-title">📰 News & Sentiment Analysis</div>', unsafe_allow_html=True)

    if not news:
        st.info("No news loaded yet. Press **Refresh All Data** in the sidebar.")
    else:
        col_l, col_r = st.columns([3, 2])

        with col_r:
            st.subheader("Sentiment by Company")
            if sent:
                fig = sentiment_bar_chart(sent)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Company Sentiment Summary")
                for company, data in sent.items():
                    cls = "pos" if data["sentiment"] == "positive" else (
                          "neg" if data["sentiment"] == "negative" else "neu")
                    icon = "🟢" if cls == "pos" else ("🔴" if cls == "neg" else "⚪")
                    st.markdown(
                        f"**{icon} {company}** — score: `{data['avg_score']:+.2f}` "
                        f"({data['article_count']} article{'s' if data['article_count']>1 else ''})",
                    )
            else:
                st.info("No companies detected in current news batch.")

        with col_l:
            st.subheader(f"Latest News ({len(news)} articles)")

            # Filter by sentiment
            sent_flt = st.selectbox("Filter by sentiment", ["All", "positive", "negative", "neutral"])
            filtered_news = news if sent_flt == "All" else [n for n in news if n.get("sentiment") == sent_flt]

            for article in filtered_news[:20]:
                s   = article.get("sentiment", "neutral")
                cls = "pos" if s == "positive" else ("neg" if s == "negative" else "neu")
                icon = "🟢" if s == "positive" else ("🔴" if s == "negative" else "⚪")
                companies_str = ", ".join(article.get("companies", [])) or "—"
                score = article.get("sentiment_score", 0)
                insight = article.get("key_insight", "")

                with st.expander(f"{icon} {article.get('title', 'No title')[:90]}"):
                    st.markdown(f"""
                    <div style='font-size:0.82rem; color:{COLORS["muted"]}'>
                        Source: <b>{article.get('source','Unknown')}</b> &nbsp;|&nbsp;
                        {article.get('published','')}
                    </div>
                    """, unsafe_allow_html=True)
                    if article.get("summary"):
                        st.write(article["summary"][:300])
                    cols = st.columns(3)
                    cols[0].markdown(f"**Sentiment:** <span class='sentiment-{cls}'>{s.upper()}</span>", unsafe_allow_html=True)
                    cols[1].markdown(f"**Score:** `{score:+.2f}`")
                    cols[2].markdown(f"**Companies:** {companies_str}")
                    if insight:
                        st.info(f"💡 {insight}")
                    if article.get("url"):
                        st.markdown(f"[🔗 Read full article]({article['url']})")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — TRADING & TRANSACTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif _PAGE == "Trading & Transactions":
    st.markdown('<div class="page-title">💰 Trading & Transactions</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("📝 Log a Trade")
        with st.form("trade_form"):
            sym      = st.text_input("Symbol", placeholder="e.g. COMI")
            qty      = st.number_input("Quantity (shares)", min_value=0.0, step=1.0)
            buy_px   = st.number_input("Buy Price (EGP)", min_value=0.0, step=0.01, format="%.2f")
            sell_px  = st.number_input("Sell Price (EGP)", min_value=0.0, step=0.01, format="%.2f")
            comm     = st.number_input("Commission (EGP)", min_value=0.0, step=0.01, format="%.2f",
                                       help="Brokerage commission in EGP")
            submitted = st.form_submit_button("Calculate & Save Trade")

        if submitted and sym and qty > 0 and buy_px > 0:
            trade = add_trade(sym, buy_px, sell_px, qty, comm)
            gross = trade["gross_profit"]
            net   = trade["net_profit"]
            ret   = trade["return_pct"]

            # Result card
            color = COLORS["green"] if net >= 0 else COLORS["red"]
            direction = "PROFIT" if net >= 0 else "LOSS"
            st.markdown(f"""
            <div style='background:{COLORS["surface"]}; border:1px solid {color};
                        border-radius:10px; padding:1.25rem; margin-top:1rem'>
                <div style='font-size:0.78rem; color:{COLORS["muted"]}; text-transform:uppercase;
                            letter-spacing:.1em; margin-bottom:.5rem'>Trade Result</div>
                <div style='font-size:1.8rem; font-weight:700; color:{color};
                            font-family:"IBM Plex Mono",monospace'>
                    {"+" if net >= 0 else ""}{net:,.2f} EGP
                </div>
                <div style='color:{COLORS["muted"]}; font-size:.85rem; margin-top:.3rem'>
                    {direction} &nbsp;|&nbsp; Return: {ret:+.2f}% &nbsp;|&nbsp; Gross: {gross:+,.2f} &nbsp;|&nbsp; Comm: {comm:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.subheader("📊 Quick P&L Calculator")
        st.caption("Instant calculator — does not save to trade history")
        with st.form("calc_form"):
            c_sym   = st.text_input("Symbol", key="calc_sym")
            c_buy   = st.number_input("Buy Price", min_value=0.0, step=0.01, key="c_buy")
            c_sell  = st.number_input("Sell Price", min_value=0.0, step=0.01, key="c_sell")
            c_qty   = st.number_input("Quantity", min_value=0.0, step=1.0, key="c_qty")
            c_comm  = st.number_input("Commission %", min_value=0.0, max_value=5.0, step=0.01,
                                       key="c_comm", help="Commission as percentage of trade value")
            calc    = st.form_submit_button("Calculate")

        if calc and c_qty > 0 and c_buy > 0:
            gross = (c_sell - c_buy) * c_qty
            comm_amt = (c_buy + c_sell) * c_qty * c_comm / 100
            net  = gross - comm_amt
            ret  = (c_sell - c_buy) / c_buy * 100
            col_a, col_b = st.columns(2)
            col_a.metric("Gross Profit",  f"{gross:+,.2f} EGP")
            col_b.metric("Net Profit",    f"{net:+,.2f} EGP",
                         delta=f"{ret:+.2f}%")
            col_a.metric("Commission",    f"{comm_amt:.2f} EGP")
            col_b.metric("Break-even",    f"{c_buy + comm_amt/c_qty:.2f} EGP")

    # Trade history
    st.markdown("---")
    st.subheader("📋 Trade History")
    trades = get_trades()
    if trades:
        tdf = pd.DataFrame(trades)
        disp_cols = ["date", "symbol", "buy_price", "sell_price", "quantity",
                     "commission", "gross_profit", "net_profit", "return_pct"]
        tdf_display = tdf[disp_cols].copy()
        tdf_display.columns = ["Date", "Symbol", "Buy", "Sell", "Qty",
                                "Commission", "Gross P&L", "Net P&L", "Return %"]

        # Summary row
        total_net = tdf["net_profit"].sum()
        winners   = (tdf["net_profit"] > 0).sum()
        losers    = (tdf["net_profit"] <= 0).sum()
        win_rate  = winners / len(tdf) * 100 if tdf.shape[0] > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Net P&L", f"{total_net:+,.2f} EGP")
        c2.metric("Win Rate", f"{win_rate:.0f}%")
        c3.metric("Winning Trades", f"🟢 {winners}")
        c4.metric("Losing Trades",  f"🔴 {losers}")

        def color_pnl(val):
            if isinstance(val, (int, float)):
                return f"color:{COLORS['green']}" if val > 0 else (f"color:{COLORS['red']}" if val < 0 else "")
            return ""

        st.dataframe(
            tdf_display.style
                .applymap(color_pnl, subset=["Gross P&L", "Net P&L", "Return %"])
                .format({"Buy": "{:.2f}", "Sell": "{:.2f}",
                         "Gross P&L": "{:+,.2f}", "Net P&L": "{:+,.2f}",
                         "Commission": "{:.2f}", "Return %": "{:+.2f}%"}),
            use_container_width=True,
        )
    else:
        st.info("No trades logged yet. Use the form above to record trades.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif _PAGE == "Technical Analysis":
    st.markdown('<div class="page-title">📊 Technical Analysis</div>', unsafe_allow_html=True)

    symbols = mdf["symbol"].tolist() if not mdf.empty else ["COMI", "ETEL", "ESRS"]
    sym = st.selectbox("Select Stock", symbols)
    days = st.slider("History (days)", 30, 365, 180)

    if sym:
        with st.spinner(f"Loading {sym} data..."):
            hist = fetch_stock_history(sym, days)
            hist_ind = compute_all_indicators(hist)
            signal_result = generate_signal(hist_ind)

        if hist_ind.empty:
            st.error("No historical data available.")
        else:
            # Signal banner
            sig   = signal_result["signal"]
            conf  = signal_result["confidence"]
            color = COLORS["green"] if sig == "BUY" else (COLORS["red"] if sig == "SELL" else COLORS["gold"])
            st.markdown(f"""
            <div style='background:{COLORS["surface"]}; border:2px solid {color};
                        border-radius:10px; padding:1rem 1.5rem; margin-bottom:1rem;
                        display:flex; align-items:center; gap:1.5rem'>
                <div>
                    <div style='font-size:0.7rem; color:{COLORS["muted"]}; letter-spacing:.15em;
                                text-transform:uppercase'>Signal</div>
                    <div style='font-size:2.2rem; font-weight:800; color:{color};
                                font-family:"IBM Plex Mono",monospace'>{sig}</div>
                </div>
                <div>
                    <div style='font-size:0.7rem; color:{COLORS["muted"]}; letter-spacing:.15em;
                                text-transform:uppercase'>Confidence</div>
                    <div style='font-size:2.2rem; font-weight:800;
                                font-family:"IBM Plex Mono",monospace'>{conf}%</div>
                </div>
                <div style='margin-left:auto; text-align:right'>
                    <div style='font-size:.85rem; color:{COLORS["muted"]}'>Latest Close</div>
                    <div style='font-size:1.6rem; font-family:"IBM Plex Mono",monospace'>
                        {hist_ind["close"].iloc[-1]:.2f} EGP
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Main technical chart
            st.plotly_chart(technical_chart(hist_ind, sym), use_container_width=True)

            # Indicator reasons
            st.subheader("Signal Breakdown")
            for reason in signal_result.get("reasons", []):
                if reason.startswith("──"):
                    st.markdown(f'<div class="reason-section">{reason}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)

            # Latest indicator values
            st.markdown("---")
            st.subheader("Latest Indicator Values")
            last = hist_ind.iloc[-1]
            i1, i2, i3, i4, i5 = st.columns(5)
            i1.metric("RSI(14)",    f"{last.get('rsi', 0):.1f}")
            i2.metric("MACD",       f"{last.get('macd', 0):.3f}")
            i3.metric("SMA 20",     f"{last.get('sma_20', 0):.2f}")
            i4.metric("SMA 50",     f"{last.get('sma_50', 0):.2f}")
            i5.metric("BB Width",
                      f"{(last.get('bb_upper',0) - last.get('bb_lower',0)):.2f}")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
elif _PAGE == "Recommendation Engine":
    st.markdown('<div class="page-title">🎯 AI Recommendation Engine</div>', unsafe_allow_html=True)

    symbols = mdf["symbol"].tolist() if not mdf.empty else ["COMI", "ETEL", "ESRS"]
    col_a, col_b = st.columns([2, 1])
    sym  = col_a.selectbox("Select Stock for Analysis", symbols)
    days = col_b.slider("History (days)", 30, 365, 120, key="rec_days")

    if sym and st.button(f"🎯 Generate Recommendation for {sym}", use_container_width=True):
        with st.spinner("Running full analysis..."):
            hist     = fetch_stock_history(sym, days)
            hist_ind = compute_all_indicators(hist)
            mc_res   = run_monte_carlo(hist_ind["close"], n_simulations=1000, n_days=30)
            sym_sent = sent.get(sym, sent.get(sym.upper(), {}))
            rec      = generate_recommendation(sym, hist_ind, mc_res, sym_sent)

        sig   = rec["signal"]
        conf  = rec["confidence"]
        color = COLORS["green"] if sig == "BUY" else (COLORS["red"] if sig == "SELL" else COLORS["gold"])

        # Hero recommendation card
        st.markdown(f"""
        <div style='background:linear-gradient(135deg, {COLORS["surface"]}, {COLORS["bg"]});
                    border: 2px solid {color}; border-radius:16px; padding:2rem;
                    margin-bottom:1.5rem; text-align:center'>
            <div style='font-size:.8rem; color:{COLORS["muted"]}; letter-spacing:.2em;
                        text-transform:uppercase; margin-bottom:.5rem'>Final Recommendation</div>
            <div style='font-size:4rem; font-weight:900; color:{color};
                        font-family:"IBM Plex Mono",monospace; line-height:1'>{sig}</div>
            <div style='font-size:1rem; color:{COLORS["text"]}; margin:.75rem 0'>
                {sym} &nbsp;—&nbsp; Confidence: <b style="color:{color}">{conf}%</b>
            </div>
            <div style='font-size:.85rem; color:{COLORS["muted"]}; font-style:italic'>
                {rec.get("summary", "")}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence gauge
        col1, col2, col3 = st.columns(3)
        src = rec.get("source_scores", {})
        tech_s = src.get("technical", 0)
        mc_s   = src.get("monte_carlo", 0)
        sent_s = src.get("sentiment", 0)

        def score_label(s):
            return "🟢 Bullish" if s > 0.2 else ("🔴 Bearish" if s < -0.2 else "⚪ Neutral")

        col1.metric("Technical Signal", score_label(tech_s), f"weight: 45%")
        col2.metric("Monte Carlo Signal", score_label(mc_s), f"weight: 35%")
        col3.metric("Sentiment Signal", score_label(sent_s), f"weight: 20%")

        # Reasons breakdown
        st.markdown("---")
        st.subheader("📋 Why This Recommendation?")
        st.caption("Detailed reasoning from each analysis layer:")

        for reason in rec.get("all_reasons", []):
            if reason.startswith("──"):
                st.markdown(f'<div class="reason-section">{reason}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)

    else:
        st.info("👆 Select a stock and click **Generate Recommendation** to run the full analysis.")

        # Show batch overview if market data loaded
        if not mdf.empty:
            st.markdown("---")
            st.subheader("Quick Scan: All Stocks")
            st.caption("Pre-computed signals based on recent price momentum")

            quick_rows = []
            for _, row in mdf.head(15).iterrows():
                chg = row["change_pct"]
                sig = "BUY" if chg > 2 else ("SELL" if chg < -2 else "HOLD")
                quick_rows.append({
                    "Symbol": row["symbol"],
                    "Name": row.get("name", ""),
                    "Price": row["price"],
                    "Chg %": row["change_pct"],
                    "Quick Signal": sig,
                })
            qdf = pd.DataFrame(quick_rows)
            st.dataframe(qdf, use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
elif _PAGE == "Monte Carlo Forecast":
    st.markdown('<div class="page-title">🧮 Monte Carlo Simulation & Forecasting</div>', unsafe_allow_html=True)

    symbols = mdf["symbol"].tolist() if not mdf.empty else ["COMI", "ETEL", "ESRS"]

    col_a, col_b, col_c = st.columns(3)
    sym    = col_a.selectbox("Select Stock", symbols, key="mc_sym")
    n_sims = col_b.selectbox("Simulations", [500, 1000, 2000, 5000], index=1)
    n_days = col_c.slider("Forecast Horizon (days)", 10, 90, 30)

    if st.button(f"🎲 Run {n_sims:,} Simulations for {sym}", use_container_width=True):
        with st.spinner(f"Running {n_sims:,} Monte Carlo simulations..."):
            hist = fetch_stock_history(sym, 180)
            hist_ind = compute_all_indicators(hist)
            mc = run_monte_carlo(hist_ind["close"], n_simulations=n_sims, n_days=n_days)

        if not mc.get("stats"):
            st.error("Could not run simulation — insufficient data.")
        else:
            s = mc["stats"]
            current = mc["current_price"]
            forecast = mc["forecast"].upper()
            fc_color = COLORS["green"] if forecast == "BULLISH" else (
                       COLORS["red"] if forecast == "BEARISH" else COLORS["gold"])

            # KPI row
            st.markdown(f"""
            <div style='background:{COLORS["surface"]}; border:1px solid {fc_color};
                        border-radius:12px; padding:1.5rem; margin-bottom:1rem;
                        display:flex; gap:2rem; align-items:center; flex-wrap:wrap'>
                <div>
                    <div style='font-size:.72rem; color:{COLORS["muted"]}; letter-spacing:.15em;
                                text-transform:uppercase'>Forecast</div>
                    <div style='font-size:1.8rem; font-weight:800; color:{fc_color};
                                font-family:"IBM Plex Mono",monospace'>{forecast}</div>
                </div>
                <div>
                    <div style='font-size:.72rem; color:{COLORS["muted"]}; text-transform:uppercase'>Prob. Gain</div>
                    <div style='font-size:1.6rem; color:{COLORS["green"]}; font-family:"IBM Plex Mono",monospace'>
                        {mc["prob_gain"]*100:.1f}%
                    </div>
                </div>
                <div>
                    <div style='font-size:.72rem; color:{COLORS["muted"]}; text-transform:uppercase'>Expected Return</div>
                    <div style='font-size:1.6rem; color:{"#00C896" if mc["expected_return_pct"]>=0 else "#FF4B6E"};
                                font-family:"IBM Plex Mono",monospace'>
                        {mc["expected_return_pct"]:+.1f}%
                    </div>
                </div>
                <div>
                    <div style='font-size:.72rem; color:{COLORS["muted"]}; text-transform:uppercase'>Annualized Vol.</div>
                    <div style='font-size:1.6rem; color:{COLORS["gold"]}; font-family:"IBM Plex Mono",monospace'>
                        {mc["volatility"]:.1f}%
                    </div>
                </div>
                <div>
                    <div style='font-size:.72rem; color:{COLORS["muted"]}; text-transform:uppercase'>95% VaR</div>
                    <div style='font-size:1.6rem; color:{COLORS["red"]}; font-family:"IBM Plex Mono",monospace'>
                        -{mc["var_95"]:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Price range
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Current Price", f"{current:.2f}")
            c2.metric("Min (p5)",      f"{s['p5']:.2f}")
            c3.metric("Median",        f"{s['median']:.2f}")
            c4.metric("Mean",          f"{s['mean']:.2f}")
            c5.metric("Max (p95)",     f"{s['p95']:.2f}")

            # Charts
            st.plotly_chart(monte_carlo_paths_chart(mc, sym), use_container_width=True)
            st.plotly_chart(monte_carlo_distribution_chart(mc, sym), use_container_width=True)

            # Insights
            st.subheader("📊 Simulation Insights")
            mc_rec = monte_carlo_to_recommendation(mc)
            for reason in mc_rec.get("reasons", []):
                st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)

    else:
        st.info("👆 Configure parameters and click **Run Simulations** to generate the forecast.")
        st.markdown("""
        **How Monte Carlo works here:**
        - Historical price returns are fitted to a log-normal distribution (Geometric Brownian Motion)
        - Thousands of random future price paths are simulated
        - Statistical output shows probability ranges and risk metrics
        - Results feed directly into the Recommendation Engine
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 7 — PORTFOLIO TRACKER
# ══════════════════════════════════════════════════════════════════════════════
elif _PAGE == "Portfolio Tracker":
    st.markdown('<div class="page-title">📂 Portfolio Tracker</div>', unsafe_allow_html=True)

    col_form, col_holdings = st.columns([1, 2])

    with col_form:
        st.subheader("➕ Add Holding")
        symbols = mdf["symbol"].tolist() if not mdf.empty else []
        names   = dict(zip(mdf["symbol"], mdf.get("name", mdf["symbol"]))) if not mdf.empty else {}

        with st.form("portfolio_form"):
            p_sym = st.selectbox("Stock", [""] + symbols) if symbols else st.text_input("Symbol")
            p_qty = st.number_input("Quantity (shares)", min_value=0.0, step=1.0)
            p_px  = st.number_input("Purchase Price (EGP)", min_value=0.0, step=0.01)
            p_dt  = st.date_input("Purchase Date")
            add   = st.form_submit_button("Add to Portfolio")

        if add and p_sym and p_qty > 0 and p_px > 0:
            p_name = names.get(p_sym, p_sym)
            add_holding(p_sym, p_name, p_qty, p_px, str(p_dt))
            st.success(f"✅ Added {p_qty:.0f} × {p_sym} @ {p_px:.2f} EGP")

        # Remove holding
        holdings = get_portfolio()
        if holdings:
            st.markdown("---")
            st.subheader("🗑️ Remove Holding")
            remove_idx = st.selectbox(
                "Select holding to remove",
                range(len(holdings)),
                format_func=lambda i: f"{holdings[i]['symbol']} — {holdings[i]['quantity']:.0f} shares",
            )
            if st.button("Remove Selected"):
                remove_holding(remove_idx)
                st.success("Holding removed.")
                st.rerun()

    with col_holdings:
        port_df = compute_portfolio_summary(mdf)

        if port_df.empty:
            st.info("📋 Your portfolio is empty. Add holdings using the form on the left.")
        else:
            total_cost    = port_df["Cost Basis"].sum()
            total_val     = port_df["Current Value"].sum()
            total_pl      = port_df["Unrealized P&L"].sum()
            total_ret_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0

            # Summary KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Cost",    f"{total_cost:,.2f} EGP")
            k2.metric("Market Value",  f"{total_val:,.2f} EGP")
            k3.metric("Total P&L",     f"{total_pl:+,.2f} EGP",
                      delta=f"{total_ret_pct:+.2f}%")
            k4.metric("# Holdings",   len(port_df))

            # Color P&L
            def color_pnl(val):
                if isinstance(val, (int, float)):
                    if val > 0: return f"color:{COLORS['green']}"
                    if val < 0: return f"color:{COLORS['red']}"
                return ""

            st.dataframe(
                port_df.style
                    .applymap(color_pnl, subset=["Unrealized P&L", "Return %", "Today's Chg %"])
                    .format({
                        "Buy Price":      "{:.2f}",
                        "Current Price":  "{:.2f}",
                        "Today's Chg %":  "{:+.2f}%",
                        "Cost Basis":     "{:,.2f}",
                        "Current Value":  "{:,.2f}",
                        "Unrealized P&L": "{:+,.2f}",
                        "Return %":       "{:+.2f}%",
                    }),
                use_container_width=True,
                height=320,
            )

            # Charts
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.plotly_chart(portfolio_pie_chart(port_df), use_container_width=True)
            with chart_col2:
                st.plotly_chart(portfolio_performance_chart(port_df), use_container_width=True)

            # Risk section
            st.markdown("---")
            st.subheader("⚠️ Portfolio Risk Analysis")
            st.caption("Monte Carlo risk estimates for each holding")

            for _, row in port_df.iterrows():
                sym_h = row["Symbol"]
                with st.expander(f"{sym_h} — {row['Qty']:.0f} shares @ {row['Current Price']:.2f}"):
                    with st.spinner(f"Running risk simulation for {sym_h}..."):
                        h_hist = fetch_stock_history(sym_h, 90)
                        h_ind  = compute_all_indicators(h_hist)
                        mc_r   = run_monte_carlo(h_ind["close"], n_simulations=500, n_days=20)

                    if mc_r.get("stats"):
                        r1, r2, r3 = st.columns(3)
                        r1.metric("Prob. Gain (20d)", f"{mc_r['prob_gain']*100:.0f}%")
                        r2.metric("Volatility",       f"{mc_r['volatility']:.1f}%")
                        r3.metric("95% VaR",          f"-{mc_r['var_95']:.1f}%")
                        # Monetary VaR
                        pos_val = row["Current Value"]
                        var_egp = pos_val * mc_r["var_95"] / 100
                        st.caption(f"💰 Max expected loss (95% confidence): **{var_egp:,.2f} EGP** over 20 trading days")
