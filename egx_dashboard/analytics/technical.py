"""
Technical Analysis Engine
==========================
Computes SMA, EMA, RSI, MACD and generates Buy/Sell/Hold signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def compute_ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series,
                 fast: int = 12, slow: int = 26, signal: int = 9
                 ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    sma = compute_sma(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all technical indicators to a price DataFrame.
    Expects columns: date, open, high, low, close, volume
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df["close"]

    # Moving Averages
    df["sma_20"]  = compute_sma(close, 20)
    df["sma_50"]  = compute_sma(close, 50)
    df["ema_12"]  = compute_ema(close, 12)
    df["ema_26"]  = compute_ema(close, 26)

    # RSI
    df["rsi"] = compute_rsi(close, 14)

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(close)

    # Bollinger Bands
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = compute_bollinger_bands(close)

    # Volume MA
    df["vol_ma_20"] = compute_sma(df["volume"].astype(float), 20)

    # Daily Returns
    df["daily_return"] = close.pct_change()

    return df


def generate_signal(df: pd.DataFrame) -> Dict:
    """
    Generates a composite Buy/Sell/Hold signal from latest indicator values.
    Returns signal, confidence (0-100), and explanation list.
    """
    if df.empty or len(df) < 30:
        return {"signal": "HOLD", "confidence": 50, "reasons": ["Insufficient data"]}

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) >= 2 else latest

    score = 0       # positive = bullish, negative = bearish
    reasons = []
    max_score = 0

    # ── RSI ────────────────────────────────────────────────────────────────────
    rsi = latest.get("rsi", 50)
    max_score += 2
    if rsi < 30:
        score += 2
        reasons.append(f"🟢 RSI oversold ({rsi:.1f} < 30) → potential reversal")
    elif rsi > 70:
        score -= 2
        reasons.append(f"🔴 RSI overbought ({rsi:.1f} > 70) → potential pullback")
    elif 40 <= rsi <= 60:
        score += 0
        reasons.append(f"⚪ RSI neutral ({rsi:.1f})")
    else:
        score += 1 if rsi < 50 else -1
        reasons.append(f"{'🟡 RSI bearish zone' if rsi > 50 else '🟡 RSI bullish zone'} ({rsi:.1f})")

    # ── MACD ───────────────────────────────────────────────────────────────────
    macd     = latest.get("macd", 0)
    macd_sig = latest.get("macd_signal", 0)
    prev_macd     = prev.get("macd", 0)
    prev_macd_sig = prev.get("macd_signal", 0)
    max_score += 3

    bullish_cross = (prev_macd < prev_macd_sig) and (macd > macd_sig)
    bearish_cross = (prev_macd > prev_macd_sig) and (macd < macd_sig)

    if bullish_cross:
        score += 3
        reasons.append("🟢 MACD bullish crossover (MACD crossed above signal)")
    elif bearish_cross:
        score -= 3
        reasons.append("🔴 MACD bearish crossover (MACD crossed below signal)")
    elif macd > macd_sig:
        score += 1
        reasons.append(f"🟡 MACD above signal line (bullish momentum)")
    else:
        score -= 1
        reasons.append(f"🟡 MACD below signal line (bearish momentum)")

    # ── Moving Average Crossover ────────────────────────────────────────────
    sma20 = latest.get("sma_20", 0)
    sma50 = latest.get("sma_50", 0)
    prev_sma20 = prev.get("sma_20", 0)
    prev_sma50 = prev.get("sma_50", 0)
    close  = latest.get("close", 0)
    max_score += 3

    golden_cross = (prev_sma20 < prev_sma50) and (sma20 > sma50)
    death_cross  = (prev_sma20 > prev_sma50) and (sma20 < sma50)

    if golden_cross:
        score += 3
        reasons.append("🟢 Golden Cross: SMA20 crossed above SMA50 (strong bullish)")
    elif death_cross:
        score -= 3
        reasons.append("🔴 Death Cross: SMA20 crossed below SMA50 (strong bearish)")
    elif sma20 > sma50:
        score += 1
        reasons.append(f"🟡 SMA20 ({sma20:.2f}) above SMA50 ({sma50:.2f}) — uptrend")
    else:
        score -= 1
        reasons.append(f"🟡 SMA20 ({sma20:.2f}) below SMA50 ({sma50:.2f}) — downtrend")

    # Price vs SMA
    max_score += 2
    if close > sma20:
        score += 1
        reasons.append(f"🟢 Price ({close:.2f}) above SMA20 ({sma20:.2f})")
    else:
        score -= 1
        reasons.append(f"🔴 Price ({close:.2f}) below SMA20 ({sma20:.2f})")

    # ── Bollinger Bands ─────────────────────────────────────────────────────
    bb_upper = latest.get("bb_upper", close * 1.05)
    bb_lower = latest.get("bb_lower", close * 0.95)
    max_score += 2

    if close < bb_lower:
        score += 2
        reasons.append("🟢 Price below lower Bollinger Band → oversold bounce likely")
    elif close > bb_upper:
        score -= 2
        reasons.append("🔴 Price above upper Bollinger Band → overbought condition")
    else:
        reasons.append("⚪ Price within Bollinger Bands (normal range)")

    # ── Compute final signal ───────────────────────────────────────────────
    if max_score == 0:
        confidence = 50
    else:
        raw_confidence = (score / max_score + 1) / 2  # normalize to [0,1]
        confidence = int(raw_confidence * 100)

    if score >= 4:
        signal = "BUY"
    elif score <= -4:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "score": score,
        "max_score": max_score,
        "confidence": confidence,
        "reasons": reasons,
    }
