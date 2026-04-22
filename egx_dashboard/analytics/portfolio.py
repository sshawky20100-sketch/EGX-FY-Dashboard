"""
Portfolio State Manager
========================
Handles in-session portfolio storage and P&L calculations.
Uses Streamlit session_state as persistence layer.
"""

import json
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional


PORTFOLIO_KEY = "egx_portfolio"
TRADES_KEY    = "egx_trades"


def init_portfolio():
    """Initialize portfolio in session state if not present."""
    if PORTFOLIO_KEY not in st.session_state:
        st.session_state[PORTFOLIO_KEY] = []
    if TRADES_KEY not in st.session_state:
        st.session_state[TRADES_KEY] = []


def add_holding(symbol: str, name: str, qty: float, buy_price: float, buy_date: str = None):
    """Add a new holding to the portfolio."""
    init_portfolio()
    holding = {
        "id": len(st.session_state[PORTFOLIO_KEY]),
        "symbol": symbol.upper(),
        "name": name,
        "quantity": qty,
        "buy_price": buy_price,
        "buy_date": buy_date or datetime.now().strftime("%Y-%m-%d"),
        "cost_basis": round(qty * buy_price, 2),
    }
    st.session_state[PORTFOLIO_KEY].append(holding)


def remove_holding(idx: int):
    """Remove a holding by index."""
    init_portfolio()
    portfolio = st.session_state[PORTFOLIO_KEY]
    if 0 <= idx < len(portfolio):
        st.session_state[PORTFOLIO_KEY] = [h for i, h in enumerate(portfolio) if i != idx]


def get_portfolio() -> List[Dict]:
    init_portfolio()
    return st.session_state.get(PORTFOLIO_KEY, [])


def add_trade(symbol: str, buy_price: float, sell_price: float,
              qty: float, commission: float = 0.0):
    """Log a completed trade and compute P&L."""
    init_portfolio()
    gross_profit = (sell_price - buy_price) * qty
    net_profit   = gross_profit - commission
    return_pct   = ((sell_price - buy_price) / buy_price * 100) if buy_price > 0 else 0

    trade = {
        "id": len(st.session_state[TRADES_KEY]),
        "symbol": symbol.upper(),
        "buy_price": buy_price,
        "sell_price": sell_price,
        "quantity": qty,
        "commission": commission,
        "gross_profit": round(gross_profit, 2),
        "net_profit": round(net_profit, 2),
        "return_pct": round(return_pct, 2),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state[TRADES_KEY].append(trade)
    return trade


def get_trades() -> List[Dict]:
    init_portfolio()
    return st.session_state.get(TRADES_KEY, [])


def compute_portfolio_summary(market_data: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches portfolio holdings with current market prices and P&L.
    """
    holdings = get_portfolio()
    if not holdings:
        return pd.DataFrame()

    rows = []
    for h in holdings:
        sym = h["symbol"]
        # Look up current price from market data
        match = market_data[market_data["symbol"] == sym] if not market_data.empty else pd.DataFrame()
        current_price = float(match["price"].iloc[0]) if not match.empty else h["buy_price"]
        chg_pct = float(match["change_pct"].iloc[0]) if not match.empty else 0.0

        cost_basis    = h["buy_price"] * h["quantity"]
        current_value = current_price * h["quantity"]
        unrealized_pl = current_value - cost_basis
        unrealized_pct = (unrealized_pl / cost_basis * 100) if cost_basis > 0 else 0

        rows.append({
            "Symbol":         sym,
            "Name":           h.get("name", sym),
            "Qty":            h["quantity"],
            "Buy Price":      h["buy_price"],
            "Current Price":  round(current_price, 2),
            "Today's Chg %":  chg_pct,
            "Cost Basis":     round(cost_basis, 2),
            "Current Value":  round(current_value, 2),
            "Unrealized P&L": round(unrealized_pl, 2),
            "Return %":       round(unrealized_pct, 2),
            "Buy Date":       h.get("buy_date", ""),
        })

    return pd.DataFrame(rows)
