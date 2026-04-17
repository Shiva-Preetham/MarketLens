from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from statistics import mean

import yfinance as yf


POSITIVE_WORDS = {
    "beat",
    "bullish",
    "buy",
    "expands",
    "gain",
    "growth",
    "high",
    "jump",
    "outperform",
    "profit",
    "rally",
    "record",
    "rise",
    "strong",
    "surge",
    "upgrade",
}

NEGATIVE_WORDS = {
    "bearish",
    "crash",
    "cut",
    "decline",
    "downgrade",
    "drop",
    "fall",
    "fraud",
    "loss",
    "miss",
    "probe",
    "risk",
    "sell",
    "slump",
    "weak",
}


def _safe_datetime(raw_value) -> str | None:
    if raw_value in (None, ""):
        return None

    if isinstance(raw_value, str):
        try:
            return datetime.fromisoformat(raw_value.replace("Z", "+00:00")).isoformat()
        except Exception:
            return raw_value

    try:
        return datetime.fromtimestamp(int(raw_value), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _text_from(value, default: str = "unknown") -> str:
    if value in (None, ""):
        return default

    if isinstance(value, dict):
        for key in ("displayName", "name", "providerName", "publisher", "title", "url"):
            nested = value.get(key)
            if nested not in (None, ""):
                return _text_from(nested, default)
        return default

    if isinstance(value, list):
        cleaned = [_text_from(item, "") for item in value]
        cleaned = [item for item in cleaned if item]
        return ", ".join(cleaned) if cleaned else default

    return str(value)


def _url_from(value) -> str | None:
    if value in (None, ""):
        return None

    if isinstance(value, dict):
        for key in ("url", "href", "link"):
            nested = value.get(key)
            if nested not in (None, ""):
                return _url_from(nested)
        return None

    return str(value)


def _score_headline(headline: str) -> tuple[float, str]:
    if not headline:
        return 0.0, "neutral"

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        vader_score = SentimentIntensityAnalyzer().polarity_scores(headline)["compound"]
    except ImportError:
        words = set(headline.lower().split())
        pos = len(words & POSITIVE_WORDS)
        neg = len(words & NEGATIVE_WORDS)
        vader_score = (pos - neg) / (pos + neg) if (pos + neg) else 0.0

    label = "positive" if vader_score > 0.15 else "negative" if vader_score < -0.15 else "neutral"
    return float(vader_score), label


def _extract_headline(item: dict) -> dict:
    content = item.get("content") or {}
    title = _text_from(content.get("title") or item.get("title"), default="")
    publisher = _text_from(content.get("provider") or item.get("publisher"))
    link = _url_from(content.get("canonicalUrl") or item.get("link"))
    published_at = _safe_datetime(
        content.get("pubDate")
        or content.get("displayTime")
        or item.get("providerPublishTime")
        or item.get("published_at")
    )
    score, label = _score_headline(title)

    return {
        "title": title,
        "publisher": publisher,
        "link": link,
        "published_at": published_at,
        "sentiment_score": round(score, 4),
        "sentiment_label": label,
    }


def _aggregate_signal(scores: list[float]) -> tuple[float, str]:
    if not scores:
        return 0.0, "neutral"

    aggregate = float(mean(scores))
    label = "bullish" if aggregate > 0.12 else "bearish" if aggregate < -0.12 else "neutral"
    return round(aggregate, 4), label


def build_market_intelligence(symbol: str, limit: int = 8) -> dict:
    ticker = yf.Ticker(symbol)

    try:
        raw_news = ticker.news or []
    except Exception:
        raw_news = []

    headlines = [_extract_headline(item) for item in raw_news[:limit]]
    scored_headlines = [headline for headline in headlines if headline["title"]]
    scores = [headline["sentiment_score"] for headline in scored_headlines]
    aggregate_score, signal = _aggregate_signal(scores)
    counts = Counter(headline["sentiment_label"] for headline in scored_headlines)

    return {
        "symbol": symbol.upper(),
        "news_count": len(scored_headlines),
        "sentiment_score": aggregate_score,
        "signal": signal,
        "sentiment_breakdown": {
            "positive": counts.get("positive", 0),
            "neutral": counts.get("neutral", 0),
            "negative": counts.get("negative", 0),
        },
        "headlines": scored_headlines,
        "explanation": (
            "Phase 1 uses headline sentiment to produce a lightweight market-intelligence signal. "
            "Later phases can upgrade this to FinBERT and richer multi-source news ingestion."
        ),
    }
