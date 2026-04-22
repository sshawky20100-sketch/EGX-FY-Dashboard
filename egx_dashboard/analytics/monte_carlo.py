"""
Monte Carlo Simulation Engine
==============================
Runs N simulations of future price paths using Geometric Brownian Motion.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def run_monte_carlo(
    close_prices: pd.Series,
    n_simulations: int = 1000,
    n_days: int = 30,
    seed: int = 42,
) -> Dict:
    """
    Runs Monte Carlo simulation on historical price data.

    Parameters
    ----------
    close_prices : pd.Series of closing prices
    n_simulations : number of simulated paths
    n_days : forecast horizon in trading days
    seed : random seed for reproducibility

    Returns
    -------
    dict with keys:
        paths         - (n_simulations × n_days) array of simulated prices
        final_prices  - array of final prices across all simulations
        current_price - last observed price
        stats         - dict with mean/median/std/min/max/percentiles
        prob_gain     - probability of ending above current price
        prob_loss     - probability of ending below current price
        forecast      - "bullish" | "bearish" | "neutral"
        volatility    - annualized historical volatility
        var_95        - 95% Value at Risk (% loss)
    """
    np.random.seed(seed)

    prices = close_prices.dropna().values
    if len(prices) < 10:
        return _empty_result()

    current_price = prices[-1]

    # ── Estimate parameters from historical data ──────────────────────────
    log_returns = np.diff(np.log(prices))
    mu    = log_returns.mean()          # daily drift
    sigma = log_returns.std()           # daily volatility

    # Annualized volatility
    annualized_vol = sigma * np.sqrt(252) * 100  # percent

    # ── Simulate GBM paths ──────────────────────────────────────────────────
    # S(t) = S(0) * exp((mu - sigma²/2)*t + sigma * W(t))
    dt = 1  # 1 day per step
    rand_shocks = np.random.normal(0, 1, (n_simulations, n_days))
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * rand_shocks

    # Cumulative sum over time axis
    cumulative = np.cumsum(drift + diffusion, axis=1)
    paths = current_price * np.exp(cumulative)

    final_prices = paths[:, -1]

    # ── Statistics ─────────────────────────────────────────────────────────
    prob_gain = float(np.mean(final_prices > current_price))
    prob_loss = float(np.mean(final_prices < current_price))

    stats = {
        "current": float(current_price),
        "mean": float(np.mean(final_prices)),
        "median": float(np.median(final_prices)),
        "std": float(np.std(final_prices)),
        "min": float(np.min(final_prices)),
        "max": float(np.max(final_prices)),
        "p5": float(np.percentile(final_prices, 5)),
        "p25": float(np.percentile(final_prices, 25)),
        "p75": float(np.percentile(final_prices, 75)),
        "p95": float(np.percentile(final_prices, 95)),
    }

    # 95% VaR
    var_95 = float((current_price - stats["p5"]) / current_price * 100)

    # Forecast direction
    expected_return = (stats["mean"] - current_price) / current_price
    if prob_gain > 0.60:
        forecast = "bullish"
    elif prob_loss > 0.60:
        forecast = "bearish"
    else:
        forecast = "neutral"

    return {
        "paths": paths,
        "final_prices": final_prices,
        "current_price": current_price,
        "stats": stats,
        "prob_gain": prob_gain,
        "prob_loss": prob_loss,
        "forecast": forecast,
        "volatility": float(annualized_vol),
        "var_95": var_95,
        "n_simulations": n_simulations,
        "n_days": n_days,
        "expected_return_pct": float(expected_return * 100),
    }


def _empty_result() -> Dict:
    return {
        "paths": np.array([]),
        "final_prices": np.array([]),
        "current_price": 0,
        "stats": {},
        "prob_gain": 0.5,
        "prob_loss": 0.5,
        "forecast": "neutral",
        "volatility": 0,
        "var_95": 0,
        "n_simulations": 0,
        "n_days": 0,
        "expected_return_pct": 0,
    }


def monte_carlo_to_recommendation(mc_result: Dict) -> Dict:
    """
    Translates Monte Carlo results into recommendation signal and score.
    Used by the Recommendation Engine.
    """
    if not mc_result or not mc_result.get("stats"):
        return {"signal": "HOLD", "confidence": 50, "reasons": ["No MC data"]}

    prob_gain = mc_result["prob_gain"]
    forecast  = mc_result["forecast"]
    vol       = mc_result["volatility"]
    var_95    = mc_result["var_95"]
    exp_ret   = mc_result["expected_return_pct"]

    reasons = []

    if forecast == "bullish":
        signal = "BUY"
        reasons.append(f"🟢 Monte Carlo: {prob_gain*100:.0f}% of simulations end higher")
        reasons.append(f"🟢 Expected return: +{exp_ret:.1f}% over {mc_result['n_days']} days")
    elif forecast == "bearish":
        signal = "SELL"
        reasons.append(f"🔴 Monte Carlo: only {prob_gain*100:.0f}% of simulations end higher")
        reasons.append(f"🔴 Expected return: {exp_ret:.1f}% over {mc_result['n_days']} days")
    else:
        signal = "HOLD"
        reasons.append(f"⚪ Monte Carlo: uncertain outcome ({prob_gain*100:.0f}% gain probability)")

    if vol > 40:
        reasons.append(f"⚠️ High volatility ({vol:.1f}% annualized) — elevated risk")
    elif vol < 15:
        reasons.append(f"🟢 Low volatility ({vol:.1f}% annualized) — stable stock")

    reasons.append(f"📊 95% VaR: max expected loss {var_95:.1f}%")

    confidence = int(abs(prob_gain - 0.5) * 200)  # 0-100 scale

    return {
        "signal": signal,
        "confidence": confidence,
        "reasons": reasons,
        "prob_gain": prob_gain,
        "forecast": forecast,
    }
