"""
Recommendation Engine
======================
Combines technical indicators + sentiment + Monte Carlo into a final recommendation.
"""

from typing import Dict, List, Optional
from analytics.technical import generate_signal
from analytics.monte_carlo import monte_carlo_to_recommendation


WEIGHT_TECHNICAL  = 0.45
WEIGHT_MC         = 0.35
WEIGHT_SENTIMENT  = 0.20


def generate_recommendation(
    symbol: str,
    tech_df,           # DataFrame with indicators
    mc_result: Dict,   # Monte Carlo result dict
    sentiment: Dict,   # per-company sentiment dict
) -> Dict:
    """
    Combines three signal sources into a unified recommendation.

    Returns dict with:
        signal      : "BUY" | "SELL" | "HOLD"
        confidence  : 0-100
        all_reasons : list of reason strings
        source_scores: breakdown by source
    """
    # ── Technical signal ─────────────────────────────────────────────────────
    tech = generate_signal(tech_df)
    tech_score = _signal_to_score(tech["signal"])

    # ── Monte Carlo signal ────────────────────────────────────────────────────
    mc = monte_carlo_to_recommendation(mc_result)
    mc_score = _signal_to_score(mc["signal"])

    # ── Sentiment signal ──────────────────────────────────────────────────────
    sent_score = 0.0
    sent_reasons = []
    if sentiment:
        avg = sentiment.get("avg_score", 0)
        sent_score = avg  # already in [-1, 1]
        if avg > 0.2:
            sent_reasons.append(f"🟢 News sentiment: positive (score {avg:.2f})")
        elif avg < -0.2:
            sent_reasons.append(f"🔴 News sentiment: negative (score {avg:.2f})")
        else:
            sent_reasons.append(f"⚪ News sentiment: neutral (score {avg:.2f})")
    else:
        sent_reasons.append("⚪ No recent news found for this stock")

    # ── Weighted composite ────────────────────────────────────────────────────
    composite = (
        tech_score  * WEIGHT_TECHNICAL +
        mc_score    * WEIGHT_MC +
        sent_score  * WEIGHT_SENTIMENT
    )

    # Map composite [-1, 1] → signal
    if composite >= 0.25:
        final_signal = "BUY"
    elif composite <= -0.25:
        final_signal = "SELL"
    else:
        final_signal = "HOLD"

    # Confidence = how far from 0
    confidence = int(min(abs(composite) * 100, 99))
    confidence = max(confidence, 30)  # floor

    all_reasons = (
        ["── Technical Analysis ──"]
        + tech.get("reasons", [])
        + ["── Monte Carlo Forecast ──"]
        + mc.get("reasons", [])
        + ["── News Sentiment ──"]
        + sent_reasons
    )

    # Summary line
    summary_parts = []
    if tech["signal"] != "HOLD":
        summary_parts.append(f"Technical: {tech['signal']}")
    if mc["signal"] != "HOLD":
        summary_parts.append(f"Monte Carlo: {mc['signal']}")
    if sentiment and abs(sentiment.get("avg_score", 0)) > 0.2:
        s = sentiment.get("sentiment", "neutral").upper()
        summary_parts.append(f"Sentiment: {s}")

    summary = " | ".join(summary_parts) if summary_parts else "All signals neutral"

    return {
        "signal": final_signal,
        "confidence": confidence,
        "summary": summary,
        "all_reasons": all_reasons,
        "source_scores": {
            "technical": tech_score,
            "monte_carlo": mc_score,
            "sentiment": sent_score,
        },
        "composite_score": composite,
    }


def _signal_to_score(signal: str) -> float:
    return {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}.get(signal, 0.0)
