"""
News & Social Media Sources Configuration
==========================================
Central configuration for all news/social sources, language settings,
watch keywords per market, and sentiment model selection.
All sources are opt-in — missing API keys gracefully disable their sources.

Usage:
    from news_sentiment.config_news import FINANCIAL_RSS_FEEDS, WATCH_KEYWORDS
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NewsSource:
    """Configuration for a single news/social media source."""
    name: str
    type: str  # "rss" | "api" | "scraper"
    feeds: List[str] = field(default_factory=list)
    url: str = ""
    api_key_env: str = ""
    enabled: bool = True
    language: str = "en"  # "en" | "zh" | "kr" | "jp"
    max_items: int = 50
    name_cn: str = ""


# ═══════════════════════════════════════════════════════════════════════
# Financial News RSS (free, no auth required)
# ═══════════════════════════════════════════════════════════════════════
FINANCIAL_RSS_FEEDS: List[NewsSource] = [
    # ── 中文财经 ──
    NewsSource(
        name="cls", type="rss",
        feeds=["https://www.cls.cn/telegraph"],
        language="zh", max_items=50,
        name_cn="财联社",
    ),
    NewsSource(
        name="eastmoney", type="rss",
        feeds=["https://feed.eastmoney.com/rss/finance"],
        language="zh", max_items=50,
        name_cn="东方财富",
    ),
    NewsSource(
        name="sina_finance", type="rss",
        feeds=["https://feed.sina.com.cn/rss/finance.xml"],
        language="zh", max_items=50,
        name_cn="新浪财经",
    ),
    NewsSource(
        name="xueqiu", type="rss",
        feeds=["https://xueqiu.com/statuses/original/timeline.json"],
        language="zh", max_items=50,
        name_cn="雪球",
    ),
    # ── English financial ──
    NewsSource(
        name="reuters", type="rss",
        feeds=["https://www.reutersagency.com/feed/",
               "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best"],
        language="en", max_items=50,
        name_cn="路透社",
    ),
    NewsSource(
        name="bloomberg", type="rss",
        feeds=["https://feeds.bloomberg.com/markets/news.rss"],
        language="en", max_items=50,
        name_cn="彭博社",
    ),
    NewsSource(
        name="yahoo_finance", type="rss",
        feeds=["https://finance.yahoo.com/news/rssindex"],
        language="en", max_items=50,
        name_cn="雅虎财经",
    ),
    NewsSource(
        name="seeking_alpha", type="rss",
        feeds=["https://seekingalpha.com/feed.xml"],
        language="en", max_items=50,
        name_cn="Seeking Alpha",
    ),
    NewsSource(
        name="investing_com", type="rss",
        feeds=["https://www.investing.com/rss/news.rss"],
        language="en", max_items=50,
        name_cn="Investing.com",
    ),
    NewsSource(
        name="market_watch", type="rss",
        feeds=["https://feeds.marketwatch.com/marketwatch/topstories/"],
        language="en", max_items=50,
        name_cn="MarketWatch",
    ),
    NewsSource(
        name="cnbc", type="rss",
        feeds=["https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"],
        language="en", max_items=50,
        name_cn="CNBC",
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# API-based News (require API keys via env vars; silently disabled if missing)
# ═══════════════════════════════════════════════════════════════════════
API_NEWS_SOURCES: List[NewsSource] = [
    NewsSource(
        name="alpha_vantage", type="api",
        api_key_env="ALPHA_VANTAGE_KEY",
        max_items=200, name_cn="Alpha Vantage新闻",
    ),
    NewsSource(
        name="newsapi", type="api",
        api_key_env="NEWSAPI_KEY",
        max_items=100, name_cn="NewsAPI",
    ),
    NewsSource(
        name="marketaux", type="api",
        api_key_env="MARKETAUX_KEY",
        max_items=100, name_cn="MarketAux",
    ),
    NewsSource(
        name="finviz", type="api",
        api_key_env="FINVIZ_KEY",
        max_items=100, name_cn="Finviz",
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# Social Media (scraper-based, works without auth via snscrape)
# snscrape is optional — if not installed, these sources gracefully skip
# ═══════════════════════════════════════════════════════════════════════
SOCIAL_SOURCES: List[NewsSource] = [
    NewsSource(
        name="twitter", type="scraper",
        enabled=True, max_items=200,
        name_cn="Twitter/X",
    ),
    NewsSource(
        name="weibo", type="scraper",
        enabled=True, max_items=200, language="zh",
        name_cn="微博",
    ),
    NewsSource(
        name="reddit", type="scraper",
        enabled=True, max_items=100,
        name_cn="Reddit",
    ),
    NewsSource(
        name="telegram", type="scraper",
        enabled=True, max_items=100,
        name_cn="Telegram频道",
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# Advanced sources (require browser automation, disabled by default)
# ═══════════════════════════════════════════════════════════════════════
ADVANCED_SOURCES: List[NewsSource] = [
    NewsSource(
        name="xiaohongshu", type="scraper",
        enabled=False, max_items=50, language="zh",
        name_cn="小红书",
    ),
    NewsSource(
        name="douyin", type="scraper",
        enabled=False, max_items=50, language="zh",
        name_cn="抖音",
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# YouTube Sources
# ═══════════════════════════════════════════════════════════════════════
YOUTUBE_SOURCES: List[NewsSource] = [
    NewsSource(
        name="youtube", type="api",
        api_key_env="YOUTUBE_API_KEY",
        max_items=50, name_cn="YouTube",
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# Stock-specific watch keywords per market
# Used for sentiment boosting and relevance filtering
# ═══════════════════════════════════════════════════════════════════════
WATCH_KEYWORDS = {
    "CN": [
        # Market-wide
        "A股", "大盘", "涨停", "跌停", "利好", "利空", "牛市", "熊市",
        "反弹", "回调", "震荡", "突破", "放量", "缩量",
        # Sectors (high-frequency in China)
        "新能源", "半导体", "医药", "白酒", "银行", "地产", "光伏",
        "锂电池", "人工智能", "AI", "券商", "保险", "消费",
        "汽车", "芯片", "5G", "基建",
        # Policy
        "降准", "降息", "加息", "央行", "证监会", "北向资金",
        "量化", "印花税", "注册制",
    ],
    "US": [
        # Market-wide
        "stock market", "rally", "crash", "Fed", "rate hike", "rate cut",
        "earnings", "S&P 500", "Nasdaq", "Dow Jones", "recession",
        "inflation", "bull market", "bear market", "volatility",
        # Macro
        "CPI", "GDP", "unemployment", "jobs report", "treasury yield",
        "quantitative easing", "tightening", "soft landing",
        # Tech megacaps
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
        "AI", "cloud", "semiconductor", "cybersecurity",
    ],
    "KR": [
        "코스피", "코스닥", "긴축", "금리", "원화", "환율",
        "삼성전자", "SK하이닉스", "네이버", "카카오",
    ],
    "JP": [
        "日経", "TOPIX", "円高", "円安", "株価",
        "日銀", "金融政策", "ソニー", "トヨタ", "ファーストリテイリング",
    ],
}

# ═══════════════════════════════════════════════════════════════════════
# Sentiment Engine Configuration
# ═══════════════════════════════════════════════════════════════════════
SENTIMENT_MODELS = {
    "vader": True,        # English VADER (pip install vaderSentiment)
    "snownlp": True,      # Chinese SnowNLP (pip install snownlp)
    "finbert": False,     # Financial BERT (requires torch, optional heavier)
    "textblob": True,     # English TextBlob (lightweight alternative)
    "pattern": True,      # Pattern analyzer (included in TextBlob)
}

# ═══════════════════════════════════════════════════════════════════════
# Cache Settings
# ═══════════════════════════════════════════════════════════════════════
MAX_CACHE_AGE_HOURS = 6
CACHE_DIR = "data/news_cache"

# ═══════════════════════════════════════════════════════════════════════
# Advanced: scrape interval (seconds) per source type
# ═══════════════════════════════════════════════════════════════════════
SCRAPE_DELAYS = {
    "rss": 0.0,       # RSS feeds are fast, no delay needed
    "api": 0.5,       # APIs need rate limiting
    "scraper": 1.0,   # Scrapers should be polite
}
