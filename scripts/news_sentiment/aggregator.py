"""
Sentiment Aggregator — Convert raw articles to per-ticker sentiment features
=============================================================================
Pipeline:
  1. Map articles to tickers (symbol matching + entity extraction)
  2. Aggregate by (date, ticker) → score, count, std, trend
  3. Compute market-level sentiment
  4. Produce pandas DataFrame ready for feature merging

Output format matches what `features.py` expects:
  columns: date, symbol, sentiment_score, news_count, [trend, volatility, ...]
"""
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config_news import WATCH_KEYWORDS

logger = logging.getLogger("aggregator")

# ═══════════════════════════════════════════════════════════════════════
# Ticker Mapping
# ═══════════════════════════════════════════════════════════════════════

# Known symbol patterns in articles (e.g., $AAPL, $MSFT)
_SYMBOL_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')

# US common tickers that appear as bare words in text
_US_COMMON_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "TSLA", "META",
    "AVGO", "JPM", "V", "BAC", "WMT", "JNJ", "PG", "XOM", "CVX",
    "UNH", "HD", "DIS", "MA", "NFLX", "ADBE", "CRM", "INTC",
    "AMD", "PYPL", "BABA", "JD", "NIO", "TSM", "SPY", "QQQ",
    "DIA", "IWM", "VIX", "KWEB", "FXI", "EWJ", "EWY",
}

# Chinese stock symbol patterns (e.g., 600519, 000858)
_CN_SYMBOL_PATTERN = re.compile(r'\b(\d{6})\b')

# Chinese company name → ticker mapping (common ones)
_CN_NAME_TO_TICKER = {
    "贵州茅台": "600519.SS",
    "五粮液": "000858.SZ",
    "宁德时代": "300750.SZ",
    "中国平安": "601318.SS",
    "招商银行": "600036.SS",
    "美的集团": "000333.SZ",
    "格力电器": "000651.SZ",
    "恒瑞医药": "600276.SS",
    "药明康德": "603259.SS",
    "隆基绿能": "601012.SS",
    "比亚迪": "002594.SZ",
    "迈瑞医疗": "300760.SZ",
    "海康威视": "002415.SZ",
    "中芯国际": "688981.SS",
    "立讯精密": "002475.SZ",
    "东方财富": "300059.SZ",
    "中信证券": "600030.SS",
    "工商银行": "601398.SS",
    "农业银行": "601288.SS",
    "建设银行": "601939.SS",
}


def extract_tickers(text: str, market: str = "US") -> List[str]:
    """Extract stock tickers mentioned in article text.

    Supports:
      - $TICKER format ($AAPL, $MSFT)
      - Chinese 6-digit codes (600519)
      - Chinese company names → ticker mapping
      - Known US tickers appearing as bare words

    Args:
        text: The article/posts text
        market: Market code

    Returns:
        List of identified stock tickers
    """
    found: List[str] = []

    if market == "CN":
        # Chinese 6-digit codes
        codes = _CN_SYMBOL_PATTERN.findall(text)
        for code in codes:
            prefix = "60" if code.startswith("60") else "00" if code.startswith("00") else "30" if code.startswith("30") else "68"
            suffix = "SS" if prefix in ("60", "68") else "SZ"
            found.append(f"{code}.{suffix}")

        # Company name mapping
        for name, ticker in _CN_NAME_TO_TICKER.items():
            if name in text:
                found.append(ticker)
    else:
        # $TICKER format
        found.extend(match.group(1) for match in _SYMBOL_PATTERN.finditer(text))

        # Bare-word known tickers
        words = re.findall(r'\b[A-Z]{1,5}\b', text)
        for w in words:
            if w in _US_COMMON_TICKERS and w not in found:
                found.append(w)

    # Deduplicate while preserving order
    seen = set()
    return [t for t in found if not (t in seen or seen.add(t))]


def extract_tickers_from_av(item: Dict) -> List[str]:
    """Extract tickers from Alpha Vantage ticker_sentiments field."""
    ticker_map = item.get("ticker_sentiments", {})
    if ticker_map:
        return list(ticker_map.keys())
    return []


# ═══════════════════════════════════════════════════════════════════════
# Temporal Resolution
# ═══════════════════════════════════════════════════════════════════════


def _resolve_date(item: Dict) -> Optional[str]:
    """Extract a date string from an article in YYYY-MM-DD format.

    Falls back to collected_at if no published date available.
    """
    pub = item.get("published", "")
    if pub:
        # Try common formats
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S",
                     "%Y-%m-%d %H:%M:%S", "%a, %d %b %Y %H:%M:%S"):
            try:
                return datetime.strptime(pub[:19], fmt).strftime("%Y-%m-%d")
            except (ValueError, IndexError):
                continue
        # If nothing matched, take first 10 chars as YYYY-MM-DD
        date_part = pub[:10]
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_part):
            return date_part

    # Fallback to collected_at
    collected = item.get("collected_at", "")
    if collected:
        return collected[:10]

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ═══════════════════════════════════════════════════════════════════════
# Core Aggregation
# ═══════════════════════════════════════════════════════════════════════


def aggregate_by_ticker(
    articles: List[Dict],
    tickers: Optional[List[str]] = None,
    market: str = "US",
) -> Dict[str, Dict[str, float]]:
    """Aggregate sentiment per ticker across all articles.

    Args:
        articles: List of article dicts with sentiment and ticker info
        tickers: Optional list of target tickers (filter to these)
        market: Market code

    Returns:
        Dict of {ticker: {sentiment_score, news_count, ...}}
    """
    # Group sentiment by ticker
    ticker_scores: Dict[str, List[float]] = defaultdict(list)
    ticker_weights: Dict[str, List[float]] = defaultdict(list)

    for article in articles:
        sentiment = article.get("sentiment", 0.0)
        matched_tickers = article.get("_matched_tickers", [])

        # Try to match tickers if not pre-matched
        if not matched_tickers:
            matched_tickers = extract_tickers(article.get("text", ""), market=market)
            # Also check Alpha Vantage ticker sentiments
            av_tickers = extract_tickers_from_av(article)
            matched_tickers.extend(av_tickers)

        if not matched_tickers and tickers:
            # If no ticker found but we have target list, assign to all
            # (general market news affects all)
            pass

        if not matched_tickers:
            # General market news — contributes to market-level, skip ticker
            continue

        for ticker in matched_tickers:
            if tickers and ticker not in tickers:
                continue

            # Compute weight: source credibility + engagement
            weight = _compute_article_weight(article)
            ticker_scores[ticker].append(sentiment)
            ticker_weights[ticker].append(weight)

    # Aggregate per ticker
    result: Dict[str, Dict[str, float]] = {}
    for ticker, scores in ticker_scores.items():
        weights = ticker_weights[ticker]
        total_weight = sum(weights)
        if total_weight > 0:
            weighted_avg = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            weighted_avg = sum(scores) / max(len(scores), 1)

        result[ticker] = {
            "sentiment_score": round(weighted_avg, 4),
            "news_count": len(scores),
            "sentiment_std": round(float(np.std(scores)), 4) if len(scores) > 1 else 0.0,
            "positive_pct": round(
                sum(1 for s in scores if s > 0.15) / max(len(scores), 1), 4
            ),
            "negative_pct": round(
                sum(1 for s in scores if s < -0.15) / max(len(scores), 1), 4
            ),
            "neutral_pct": round(
                sum(1 for s in scores if -0.15 <= s <= 0.15) / max(len(scores), 1), 4
            ),
        }

    logger.debug("Ticker aggregation: %d tickers matched from %d articles",
                 len(result), len(articles))
    return result


def _compute_article_weight(article: Dict) -> float:
    """Compute article importance weight based on source and engagement.

    Factors:
      - Source type (RSS > API > Social)
      - Social engagement (followers, retweets, likes)
      - Relevance score
    """
    weight = 1.0

    # Source type base weight
    src_type = article.get("type", "rss")
    if src_type == "rss":
        weight = 1.0
    elif src_type == "api":
        weight = 1.2  # API-provided articles tend to be more curated
    elif src_type == "social":
        weight = 0.6  # Social posts are noisier

    # Engagement boost (for social media)
    followers = article.get("followers", 0)
    if followers > 100000:
        weight *= 1.5
    elif followers > 10000:
        weight *= 1.2

    retweets = article.get("retweets", 0)
    likes = article.get("likes", 0)
    comments = article.get("comments", 0)
    score = article.get("score", 0)

    engagement = retweets * 2 + likes + comments * 3 + score
    if engagement > 1000:
        weight *= min(1.5, 1.0 + engagement / 10000)

    return weight


# ═══════════════════════════════════════════════════════════════════════
# Temporal Aggregation (per ticker per day)
# ═══════════════════════════════════════════════════════════════════════


def aggregate_temporal(
    articles: List[Dict],
    tickers: Optional[List[str]] = None,
    market: str = "US",
) -> List[Dict]:
    """Aggregate sentiment per (date, ticker) pair.

    Produces daily-level records suitable for merging into feature DataFrame.

    Returns:
        List of dicts with: date, symbol, sentiment_score, news_count,
        sentiment_std, sentiment_trend
    """
    # Group by (date, ticker)
    daily: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for article in articles:
        date = _resolve_date(article)
        sentiment = article.get("sentiment", 0.0)

        matched_tickers = article.get("_matched_tickers", [])
        if not matched_tickers:
            matched_tickers = extract_tickers(article.get("text", ""), market=market)

        if not matched_tickers:
            continue

        for ticker in matched_tickers:
            if tickers and ticker not in tickers:
                continue
            daily[(date, ticker)].append(sentiment)

    # Aggregate each (date, ticker) group
    records: List[Dict] = []
    for (date, ticker), scores in daily.items():
        scores_arr = np.array(scores)
        records.append({
            "date": date,
            "symbol": ticker,
            "sentiment_score": round(float(np.mean(scores_arr)), 4),
            "news_count": len(scores),
            "sentiment_std": round(float(np.std(scores_arr)), 4) if len(scores) > 1 else 0.0,
            "max_sentiment": round(float(np.max(scores_arr)), 4),
            "min_sentiment": round(float(np.min(scores_arr)), 4),
        })

    # Sort by date, ticker
    records.sort(key=lambda r: (r["date"], r["symbol"]))
    logger.debug("Temporal aggregation: %d (date, ticker) records", len(records))
    return records


# ═══════════════════════════════════════════════════════════════════════
# Market-Level Sentiment
# ═══════════════════════════════════════════════════════════════════════


def compute_market_sentiment(articles: List[Dict]) -> float:
    """Compute aggregate market-level sentiment from all articles.

    Uses weighted average of all article sentiments.
    General market news (no specific ticker) is included in market score.

    Returns:
        Overall market sentiment in [-1.0, 1.0]
    """
    if not articles:
        return 0.0

    scores: List[float] = []
    weights: List[float] = []

    for article in articles:
        sentiment = article.get("sentiment", 0.0)
        weight = _compute_article_weight(article)
        scores.append(sentiment)
        weights.append(weight)

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    weighted_avg = sum(s * w for s, w in zip(scores, weights)) / total_weight
    return round(weighted_avg, 4)


# ═══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════


def aggregate_sentiment(
    articles: List[Dict],
    tickers: Optional[List[str]] = None,
    market: str = "US",
) -> Dict:
    """Full aggregation pipeline: ticker-level + temporal + market-level.

    Args:
        articles: List of article/post dicts with 'sentiment' key
        tickers: Optional list of target tickers for filtering
        market: Market code

    Returns:
        Dict with:
          - sentiment_df: List of daily (date, symbol, score) records
          - ticker_sentiment: Dict per ticker with aggregate stats
          - market_score: Aggregate market sentiment
    """
    # Pre-match tickers for efficiency
    for article in articles:
        if "_matched_tickers" not in article:
            article["_matched_tickers"] = extract_tickers(
                article.get("text", ""), market=market
            )
            # Also check Alpha Vantage structured sentiment
            av_tickers = extract_tickers_from_av(article)
            existing = set(article["_matched_tickers"])
            for t in av_tickers:
                if t not in existing:
                    article["_matched_tickers"].append(t)

    # Per-ticker aggregation
    ticker_sent = aggregate_by_ticker(articles, tickers, market)

    # Temporal aggregation
    temporal = aggregate_temporal(articles, tickers, market)

    # Build DataFrame-friendly records
    df_records = []
    for rec in temporal:
        df_records.append({
            "date": rec["date"],
            "symbol": rec["symbol"],
            "sentiment_score": rec["sentiment_score"],
            "news_count": rec["news_count"],
            "sentiment_std": rec["sentiment_std"],
        })

    return {
        "sentiment_df": df_records,
        "ticker_sentiment": ticker_sent,
        "market_score": compute_market_sentiment(articles),
    }
