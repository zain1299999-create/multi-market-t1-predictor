"""
News Collector — RSS + API Sources
===================================
Supports: financial RSS feeds, Alpha Vantage News, NewsAPI, Marketaux, Finviz.

Graceful degradation — each source failure is caught independently.
"""
import logging
import os
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from urllib.parse import urlencode

import requests
import feedparser
import socket
from bs4 import BeautifulSoup

from .config_news import FINANCIAL_RSS_FEEDS, API_NEWS_SOURCES, WATCH_KEYWORDS, SCRAPE_DELAYS

logger = logging.getLogger("news_collector")

# ═══════════════════════════════════════════════════════════════════════
# RSS Collection
# ═══════════════════════════════════════════════════════════════════════


def fetch_rss_feeds(market: str = "US", limit: int = 200) -> List[Dict]:
    """Fetch articles from all configured financial RSS feeds.

    Args:
        market: Market code for keyword relevance filtering
        limit: Max articles to return

    Returns:
        List of article dicts with metadata, sorted by sentiment magnitude
    """
    articles: List[Dict] = []
    keywords = WATCH_KEYWORDS.get(market, [])

    for source in FINANCIAL_RSS_FEEDS:
        if not source.enabled:
            continue

        # Language-based market filtering
        if market == "CN" and source.language not in ("zh",):
            continue
        if market not in ("CN", "JP") and source.language not in ("en", "kr", "jp"):
            continue

        for feed_url in source.feeds:
            try:
                # Use requests with timeout instead of feedparser's built-in HTTP
                resp = requests.get(feed_url, timeout=15,
                                    headers={'User-Agent': 'Mozilla/5.0'})
                feed = feedparser.parse(resp.text)
                if not feed.entries:
                    logger.debug("RSS %s: empty feed from %s", source.name, feed_url)
                    continue

                for entry in feed.entries[:source.max_items]:
                    title = entry.get("title", "") or ""
                    summary = entry.get("summary", "") or ""
                    text = f"{title} {summary}".strip()

                    if not text:
                        continue

                    # Compute quick keyword sentiment
                    sentiment_score = _compute_quick_sentiment(text, market)

                    # Relevance score: how many watch keywords match
                    relevance = _keyword_relevance(text, keywords)

                    articles.append({
                        "source": source.name,
                        "source_cn": getattr(source, "name_cn", source.name),
                        "type": "rss",
                        "title": title,
                        "summary": summary[:500],
                        "url": entry.get("link", ""),
                        "published": _parse_pub_date(entry),
                        "text": text[:2000],
                        "sentiment": sentiment_score,
                        "relevance": relevance,
                        "language": source.language,
                        "market": market,
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                    })

                logger.debug("RSS %s (%s): %d articles parsed",
                             source.name, feed_url, len(feed.entries))
            except Exception as exc:
                logger.debug("RSS feed error [%s] %s: %s", source.name, feed_url, exc)

    # Sort by absolute sentiment (most extreme first), then by relevance
    articles.sort(key=lambda x: (abs(x.get("sentiment", 0)), x.get("relevance", 0)), reverse=True)
    logger.info("RSS collected: %d unique articles for market=%s", len(articles), market)
    return articles[:limit]


def _compute_quick_sentiment(text: str, market: str) -> float:
    """Quick keyword-based sentiment heuristic for RSS items.

    Used before full NLP to provide an initial score.
    Scans for positive/negative financial keywords.
    """
    text_lower = text.lower()

    if market == "CN":
        positive = [
            "利好", "大涨", "牛市", "突破", "反弹", "涨停",
            "增长", "创新高", "超预期", "放量大涨", "飙升",
            "逆转", "回暖", "攀升", "强势",
        ]
        negative = [
            "利空", "大跌", "熊市", "暴跌", "跌停", "风险",
            "亏损", "减持", "利空出尽", "恐慌", "抛售",
            "破位", "下行", "低迷", "崩盘",
        ]
    else:
        positive = [
            "rally", "surge", "breakout", "bullish", "upgrade",
            "beat", "growth", "record high", "positive", "outperform",
            "boom", "recovery", "momentum", "soar", "jump",
        ]
        negative = [
            "crash", "plunge", "bearish", "downgrade", "miss",
            "loss", "decline", "recession", "negative", "slump",
            "tumble", "selloff", "downturn", "default", "bankruptcy",
        ]

    score = 0.0
    for word in positive:
        if word.lower() in text_lower:
            score += 0.15
    for word in negative:
        if word.lower() in text_lower:
            score -= 0.15

    # Clamp to [-1.0, 1.0]
    return max(-1.0, min(1.0, score))


def _keyword_relevance(text: str, keywords: List[str]) -> int:
    """Count how many watch keywords appear in text (relevance proxy)."""
    if not keywords:
        return 0
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def _parse_pub_date(entry) -> str:
    """Try to extract a consistent ISO date string from RSS entry."""
    for attr in ("published", "updated", "created"):
        val = entry.get(attr)
        if val:
            return val
    return ""


# ═══════════════════════════════════════════════════════════════════════
# API-based News Collection
# ═══════════════════════════════════════════════════════════════════════


def fetch_api_news(market: str = "US", limit: int = 200) -> List[Dict]:
    """Fetch news from all configured API providers that have keys set.

    Each provider is called independently; failure in one doesn't block others.
    """
    articles: List[Dict] = []

    # Build a list of (name, fetch_fn) pairs
    providers = []

    av_key = os.getenv("ALPHA_VANTAGE_KEY", "")
    if av_key:
        providers.append(("alpha_vantage", lambda: _fetch_alpha_vantage(av_key, market)))

    na_key = os.getenv("NEWSAPI_KEY", "")
    if na_key:
        providers.append(("newsapi", lambda: _fetch_newsapi(na_key, market)))

    ma_key = os.getenv("MARKETAUX_KEY", "")
    if ma_key:
        providers.append(("marketaux", lambda: _fetch_marketaux(ma_key, market)))

    fv_key = os.getenv("FINVIZ_KEY", "")
    if fv_key:
        providers.append(("finviz", lambda: _fetch_finviz(fv_key, market)))

    for name, fetch_fn in providers:
        try:
            result = fetch_fn()
            articles.extend(result)
            logger.debug("API %s: %d articles", name, len(result))
        except Exception as e:
            logger.debug("API %s error: %s", name, e)

    articles.sort(key=lambda x: abs(x.get("sentiment", 0)), reverse=True)
    logger.info("API news collected: %d articles for market=%s", len(articles), market)
    return articles[:limit]


def _fetch_alpha_vantage(api_key: str, market: str) -> List[Dict]:
    """Fetch news from Alpha Vantage NEWS_SENTIMENT endpoint.

    Supports ticker-level sentiment scoring natively.
    """
    import time

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": api_key,
        "limit": 50,
        "sort": "LATEST",
    }

    # Build topic string based on market
    topic_map = {
        "CN": "finance:china",
        "US": "finance:us",
        "KR": "finance:korea",
        "JP": "finance:japan",
    }
    topic = topic_map.get(market, "finance")
    params["topics"] = topic

    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
    except Exception as e:
        logger.debug("Alpha Vantage HTTP error: %s", e)
        return []

    articles = []
    for item in data.get("feed", []):
        title = item.get("title", "") or ""
        summary = item.get("summary", "") or ""
        pub_raw = item.get("time_published", "")
        pub_dt = _parse_av_timestamp(pub_raw)

        ticker_sentiments = item.get("ticker_sentiment", [])
        ticker_map = {}
        for ts in ticker_sentiments:
            sym = ts.get("ticker", "")
            ticker_map[sym] = {
                "score": float(ts.get("ticker_sentiment_score", 0)),
                "label": ts.get("ticker_sentiment_label", "Neutral"),
            }

        articles.append({
            "source": "alpha_vantage",
            "source_cn": "Alpha Vantage",
            "type": "api",
            "title": title,
            "summary": summary[:500],
            "url": item.get("url", ""),
            "published": pub_dt,
            "text": f"{title} {summary}"[:2000],
            "sentiment": float(item.get("overall_sentiment_score", 0)),
            "sentiment_label": item.get("overall_sentiment_label", "Neutral"),
            "relevance_score": float(item.get("relevance_score", 0)),
            "language": "en",
            "market": market,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "ticker_sentiments": ticker_map,
        })

    return articles


def _parse_av_timestamp(ts: str) -> str:
    """Convert Alpha Vantage timestamp YYYYMMDDTHHMMSS to ISO."""
    if not ts or len(ts) < 8:
        return ts
    try:
        return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
    except Exception:
        return ts


def _fetch_newsapi(api_key: str, market: str) -> List[Dict]:
    """Fetch financial news from NewsAPI /v2/everything."""
    query_map = {
        "CN": "stock market China",
        "US": "stock market US finance",
        "KR": "stock market Korea finance",
        "JP": "stock market Japan finance",
    }
    query = query_map.get(market, "stock market finance")

    params = {
        "q": query,
        "apiKey": api_key,
        "language": "en" if market not in ("CN",) else "zh",
        "sortBy": "publishedAt",
        "pageSize": 50,
    }

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params=params, timeout=15
        )
        data = resp.json()
    except Exception as e:
        logger.debug("NewsAPI HTTP error: %s", e)
        return []

    articles = []
    for item in data.get("articles", []):
        title = (item.get("title") or "").strip()
        description = (item.get("description") or "").strip()
        text = f"{title} {description}".strip()
        if not text:
            continue

        sentiment = _compute_quick_sentiment(text, market)
        keywords = WATCH_KEYWORDS.get(market, [])
        relevance = _keyword_relevance(text, keywords)

        articles.append({
            "source": "newsapi",
            "source_cn": "NewsAPI",
            "type": "api",
            "title": title,
            "summary": description[:500],
            "url": item.get("url", ""),
            "published": item.get("publishedAt", ""),
            "text": text[:2000],
            "sentiment": sentiment,
            "relevance": relevance,
            "language": "en" if market != "CN" else "zh",
            "market": market,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "source_name": item.get("source", {}).get("name", ""),
        })
    return articles


def _fetch_marketaux(api_key: str, market: str) -> List[Dict]:
    """Fetch news from Marketaux /v1/news/all.

    Marketaux returns per-article sentiment natively.
    """
    country_map = {"US": "us", "CN": "cn", "KR": "kr", "JP": "jp"}
    country = country_map.get(market, "us")

    params = {
        "api_token": api_key,
        "limit": 50,
        "published_after": "7d",
        "sort": "published_at",
        "countries": country,
        "language": "en" if market not in ("CN",) else "zh",
        "filter_entities": "true",
    }

    try:
        resp = requests.get(
            "https://api.marketaux.com/v1/news/all",
            params=params, timeout=15,
        )
        data = resp.json()
    except Exception as e:
        logger.debug("Marketaux HTTP error: %s", e)
        return []

    articles = []
    for item in data.get("data", []):
        title = (item.get("title") or "").strip()
        description = (item.get("description") or "").strip()
        text = f"{title} {description}".strip()
        if not text:
            continue

        raw_sent = item.get("sentiment")
        sentiment = float(raw_sent) if raw_sent is not None else 0.0

        articles.append({
            "source": "marketaux",
            "source_cn": "MarketAux",
            "type": "api",
            "title": title,
            "summary": description[:500],
            "url": item.get("url", ""),
            "published": item.get("published_at", ""),
            "text": text[:2000],
            "sentiment": sentiment,
            "language": "en",
            "market": market,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "entities": item.get("entities", []),
        })
    return articles


def _fetch_finviz(api_key: str, market: str) -> List[Dict]:
    """Fetch financial news via Finviz (scrape-based, requires key)."""
    # Finviz doesn't have a standard API; this is a RSS/scrape approach
    url = "https://finviz.com/news.ashx"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("table.news tr")
    except Exception as e:
        logger.debug("Finviz scrape error: %s", e)
        return []

    articles = []
    for row in rows[:50]:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        timestamp = cells[0].get_text(strip=True)
        headline = cells[2].get_text(strip=True)
        link_tag = cells[2].find("a")
        url_link = link_tag.get("href", "") if link_tag else ""
        if not headline:
            continue

        sentiment = _compute_quick_sentiment(headline, market)

        articles.append({
            "source": "finviz",
            "source_cn": "Finviz",
            "type": "scraper",
            "title": headline,
            "summary": "",
            "url": url_link,
            "published": timestamp,
            "text": headline[:2000],
            "sentiment": sentiment,
            "language": "en",
            "market": market,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        })

    return articles
