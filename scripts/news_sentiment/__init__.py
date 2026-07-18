"""
Multi-Source News & Social Media Sentiment Module
==================================================
Collects financial news and social media posts from multiple sources,
runs sentiment analysis (VADER + SnowNLP + financial lexicons),
caches results, and provides aggregated multi-language sentiment features.

Usage:
    from news_sentiment import collect_and_analyze
    sentiment_features = collect_and_analyze(market="US", tickers=["AAPL", "MSFT"])

Dependency groups (optional, graceful degradation):
    - feedparser / requests      → RSS & API news
    - snscrape                   → social media (Twitter, Weibo, Reddit)
    - youtube-search / youtube-comment-downloader → YouTube
    - vaderSentiment             → English VADER
    - snownlp                    → Chinese SnowNLP
"""
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

from .config_news import WATCH_KEYWORDS, SENTIMENT_MODELS, MAX_CACHE_AGE_HOURS
from .news_collector import fetch_rss_feeds, fetch_api_news, _compute_quick_sentiment
from .social_collector import fetch_social_sentiment, fetch_youtube_comments
from .sentiment_engine import analyze_all_items, enrich_with_sentiment
from .cache_manager import NewsCache
from .aggregator import aggregate_sentiment, compute_market_sentiment

logger = logging.getLogger(__name__)


def collect_and_analyze(
    market: str = "US",
    tickers: Optional[List[str]] = None,
    run_rss: bool = True,
    run_api: bool = True,
    run_social: bool = True,
    use_cache: bool = True,
) -> Dict[str, object]:
    """Main entry point: collect, analyze, cache, and aggregate sentiment.

    Args:
        market: Market code ("CN", "US", "KR", "JP")
        tickers: Stock tickers to focus on (optional, for keyword matching)
        run_rss: Whether to collect RSS feeds
        run_api: Whether to collect API-based news
        run_social: Whether to collect social media
        use_cache: Whether to use disk cache

    Returns:
        Dict with keys:
            - "sentiment_df": pd.DataFrame with daily sentiment per ticker
            - "market_score": float (-1..1) aggregate market sentiment
            - "total_articles": int
            - "articles_by_source": Dict[str, int]
            - "collected_at": ISO timestamp
    """
    start_ts = datetime.now(timezone.utc)
    cache = NewsCache() if use_cache else None

    # ── Check cache ────────────────────────────────────────────────────
    if cache and cache.is_fresh(market, MAX_CACHE_AGE_HOURS):
        cached = cache.load(market)
        if cached is not None:
            logger.info("Cache HIT for market=%s (%d articles)",
                        market, len(cached.get("articles", [])))
            return cached
    logger.info("Cache MISS for market=%s — starting collection", market)

    # ── Collect from all sources ───────────────────────────────────────
    all_articles: List[Dict] = []

    # RSS feeds
    if run_rss:
        try:
            rss = fetch_rss_feeds(market=market, limit=200)
            all_articles.extend(rss)
            logger.info("RSS: %d articles", len(rss))
        except Exception as e:
            logger.warning("RSS collection failed: %s", e)

    # API news
    if run_api:
        try:
            api = fetch_api_news(market=market, limit=200)
            all_articles.extend(api)
            logger.info("API news: %d articles", len(api))
        except Exception as e:
            logger.warning("API news failed: %s", e)

    # Social media
    if run_social:
        try:
            social = fetch_social_sentiment(market=market, limit=200)
            all_articles.extend(social)
            logger.info("Social media: %d posts", len(social))
        except Exception as e:
            logger.warning("Social media failed: %s", e)

    # YouTube
    if run_social:
        try:
            yt = fetch_youtube_comments(market=market, limit=50)
            all_articles.extend(yt)
            logger.info("YouTube: %d videos", len(yt))
        except Exception as e:
            logger.debug("YouTube skipped: %s", e)

    # ── Deduplicate by URL ─────────────────────────────────────────────
    seen_urls = set()
    deduped = []
    for art in all_articles:
        url = art.get("url", "")
        if url and url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(art)
    all_articles = deduped

    if not all_articles:
        logger.warning("No articles collected for %s", market)
        empty = _empty_result(market, start_ts)
        if cache:
            cache.save(market, empty)
        return empty

    logger.info("Total unique articles: %d", len(all_articles))

    # ── Run sentiment analysis ─────────────────────────────────────────
    try:
        all_articles = analyze_all_items(all_articles, market=market)
        logger.info("Sentiment analysis complete")
    except Exception as e:
        logger.warning("Sentiment analysis error: %s", e)

    # ── Aggregate ──────────────────────────────────────────────────────
    try:
        agg_result = aggregate_sentiment(
            all_articles, tickers=tickers, market=market
        )
    except Exception as e:
        logger.warning("Aggregation error: %s", e)
        agg_result = {
            "sentiment_df": None,
            "ticker_sentiment": {},
        }

    # ── Market-level score ─────────────────────────────────────────────
    try:
        market_score = compute_market_sentiment(all_articles)
    except Exception as e:
        logger.debug("Market score compute error: %s", e)
        market_score = 0.0

    # ── Article source breakdown ───────────────────────────────────────
    source_counts: Dict[str, int] = {}
    for art in all_articles:
        src = art.get("source_cn", art.get("source", "unknown"))
        source_counts[src] = source_counts.get(src, 0) + 1

    result = {
        "sentiment_df": agg_result.get("sentiment_df"),
        "ticker_sentiment": agg_result.get("ticker_sentiment", {}),
        "market_score": market_score,
        "total_articles": len(all_articles),
        "articles_by_source": source_counts,
        "articles": all_articles,
        "collected_at": start_ts.isoformat(),
    }

    # ── Save cache ─────────────────────────────────────────────────────
    if cache:
        try:
            cache.save(market, result)
        except Exception as e:
            logger.debug("Cache save error: %s", e)

    logger.info("Sentiment collection complete: %d articles, market_score=%.3f",
                len(all_articles), market_score)
    return result


def collect_and_analyze_multi(
    markets: Optional[List[str]] = None,
    tickers_by_market: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, object]]:
    """Collect sentiment for multiple markets in one call.

    Args:
        markets: List of markets (e.g. ["CN", "US", "KR", "JP"])
        tickers_by_market: Dict of {market: [tickers]}

    Returns:
        Dict of {market: result_dict}
    """
    if markets is None:
        from .config_news import (  # noqa: F811
            FINANCIAL_RSS_FEEDS, API_NEWS_SOURCES
        )
        # Detect markets from RSS language config
        markets = ["US", "CN", "KR", "JP"]
    if tickers_by_market is None:
        tickers_by_market = {}

    results = {}
    for market in markets:
        try:
            results[market] = collect_and_analyze(
                market=market,
                tickers=tickers_by_market.get(market),
            )
        except Exception as e:
            logger.warning("Market %s sentiment collection failed: %s", market, e)
            results[market] = _empty_result(market, datetime.now(timezone.utc))
    return results


def _empty_result(market: str, ts: datetime) -> Dict:
    """Return an empty result dict for a market."""
    return {
        "sentiment_df": None,
        "ticker_sentiment": {},
        "market_score": 0.0,
        "total_articles": 0,
        "articles_by_source": {},
        "articles": [],
        "collected_at": ts.isoformat(),
    }
