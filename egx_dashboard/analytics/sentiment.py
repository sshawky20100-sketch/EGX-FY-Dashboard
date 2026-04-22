"""
AI Sentiment Analysis Engine
==============================
Uses Claude API to analyze news sentiment and extract company mentions.
"""

import os
import json
import logging
import anthropic
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Known EGX company names / tickers for entity extraction
EGX_ENTITIES = {
    "CIB": ["cib", "commercial international bank", "بنك التجاري الدولي"],
    "COMI": ["comi", "commercial international"],
    "Telecom Egypt": ["telecom egypt", "etel", "المصرية للاتصالات"],
    "Talaat Moustafa": ["talaat moustafa", "tmgh", "طلعت مصطفى"],
    "Eastern Company": ["eastern company", "east", "الشرقية للدخان"],
    "El Sewedy": ["el sewedy", "esrs", "السويدي"],
    "Orascom": ["orascom", "ocdi", "otmt", "أوراسكوم"],
    "Palm Hills": ["palm hills", "phdc"],
    "Heliopolis Housing": ["heliopolis", "hrho", "مصر الجديدة"],
    "ISPH": ["integrated diagnostics", "isph"],
    "Abu Kir": ["abu kir", "abuk", "أبو قير"],
    "EFG Hermes": ["efg", "hermes", "efg hermes", "هيرمس"],
    "Banque Misr": ["banque misr", "بنك مصر"],
    "NBE": ["nbe", "national bank", "البنك الأهلي"],
}


def analyze_news_sentiment(articles: List[Dict], api_key: Optional[str] = None) -> List[Dict]:
    """
    Uses Claude to analyze sentiment of news articles and extract company mentions.
    Falls back to rule-based analysis if API key is unavailable.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    analyzed = []
    for article in articles:
        title   = article.get("title", "")
        summary = article.get("summary", "")
        text    = f"{title}. {summary}".strip()

        if api_key and api_key.startswith("sk-ant"):
            result = _ai_analyze(text, api_key)
        else:
            result = _rule_based_analyze(text)

        analyzed.append({
            **article,
            "sentiment": result.get("sentiment", "neutral"),
            "sentiment_score": result.get("score", 0.0),
            "companies": result.get("companies", []),
            "key_insight": result.get("insight", ""),
        })

    return analyzed


def _ai_analyze(text: str, api_key: str) -> Dict:
    """AI-powered sentiment analysis via Claude."""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""Analyze this Egyptian financial news headline/summary and respond ONLY with valid JSON.

Text: "{text}"

Return JSON with exactly these keys:
{{
  "sentiment": "positive" | "negative" | "neutral",
  "score": float between -1.0 (very negative) and 1.0 (very positive),
  "companies": list of EGX-listed company names mentioned (empty list if none),
  "insight": one-sentence key insight for investors (max 100 chars)
}}"""

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown code fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"AI sentiment failed: {e}")
        return _rule_based_analyze(text)


def _rule_based_analyze(text: str) -> Dict:
    """Keyword-based sentiment scoring (fallback)."""
    text_lower = text.lower()

    bullish_words = [
        "profit", "growth", "increase", "surge", "rally", "beat", "record",
        "expansion", "investment", "positive", "gains", "strong", "rose",
        "dividend", "upgrade", "ارتفع", "نمو", "أرباح", "ارتفاع", "صعود",
    ]
    bearish_words = [
        "loss", "decline", "drop", "fall", "miss", "cut", "weak", "risk",
        "debt", "concern", "sell", "downgrade", "warning", "deficit",
        "انخفض", "خسارة", "هبوط", "تراجع", "ضعف",
    ]

    bull_count = sum(1 for w in bullish_words if w in text_lower)
    bear_count = sum(1 for w in bearish_words if w in text_lower)

    if bull_count > bear_count:
        sentiment = "positive"
        score = min(0.9, 0.3 + bull_count * 0.15)
    elif bear_count > bull_count:
        sentiment = "negative"
        score = max(-0.9, -0.3 - bear_count * 0.15)
    else:
        sentiment = "neutral"
        score = 0.0

    # Entity extraction
    companies = []
    for company, keywords in EGX_ENTITIES.items():
        if any(kw in text_lower for kw in keywords):
            companies.append(company)

    return {
        "sentiment": sentiment,
        "score": round(score, 2),
        "companies": companies,
        "insight": f"{'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'} market signal detected.",
    }


def aggregate_sentiment_by_stock(analyzed_articles: List[Dict]) -> Dict[str, Dict]:
    """Aggregates per-article sentiment into per-company summary."""
    company_data: Dict[str, Dict] = {}

    for article in analyzed_articles:
        for company in article.get("companies", []):
            if company not in company_data:
                company_data[company] = {"scores": [], "articles": [], "count": 0}
            company_data[company]["scores"].append(article.get("sentiment_score", 0))
            company_data[company]["articles"].append(article.get("title", ""))
            company_data[company]["count"] += 1

    result = {}
    for company, data in company_data.items():
        scores = data["scores"]
        avg_score = sum(scores) / len(scores) if scores else 0
        result[company] = {
            "avg_score": round(avg_score, 2),
            "sentiment": "positive" if avg_score > 0.1 else ("negative" if avg_score < -0.1 else "neutral"),
            "article_count": data["count"],
            "latest_article": data["articles"][0] if data["articles"] else "",
        }

    return result
