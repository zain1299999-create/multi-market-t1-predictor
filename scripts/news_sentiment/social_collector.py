"""
Social Media Collector — snscrape-based
========================================
Sources: Twitter/X, Weibo, Reddit, Telegram (public content, no auth required).
YouTube via youtube-search + youtube-comment-downloader.

Design principle: Graceful degradation at every level.
  - If snscrape is not installed → return empty lists silently
  - If a specific scraper fails → log debug and continue
  - Each source is independently try/except wrapped
"""
import logging
import importlib.util
from datetime import datetime, timezone
from typing import List, Dict

from .config_news import SOCIAL_SOURCES, WATCH_KEYWORDS

logger = logging.getLogger("social_collector")

# Check if snscrape is installed at import time
_SNSCRAPE_AVAIL = importlib.util.find_spec("snscrape") is not None


# ═══════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════


def fetch_social_sentiment(market: str = "US", limit: int = 200) -> List[Dict]:
    """Aggregate social media posts for a given market.

    Falls back gracefully if snscrape is not installed or scraping fails.
    """
    articles: List[Dict] = []

    if not _SNSCRAPE_AVAIL:
        logger.info("snscrape not installed — skipping social media scraping "
                     "(install with: pip install snscrape)")
        return articles

    try:
        # Twitter — stock-related keywords for the market
        tw = _fetch_twitter(market, max(1, limit // 2))
        articles.extend(tw)
        logger.debug("Twitter: %d posts", len(tw))
    except Exception as e:
        logger.debug("Twitter scrape error: %s", e)

    try:
        # Weibo — Chinese stock discussion
        wb = _fetch_weibo(market, max(1, limit // 4))
        articles.extend(wb)
        logger.debug("Weibo: %d posts", len(wb))
    except Exception as e:
        logger.debug("Weibo scrape error: %s", e)

    try:
        # Reddit — stock subreddits
        rd = _fetch_reddit(max(1, limit // 4))
        articles.extend(rd)
        logger.debug("Reddit: %d posts", len(rd))
    except Exception as e:
        logger.debug("Reddit scrape error: %s", e)

    try:
        # Telegram — public financial channels
        tg = _fetch_telegram(market, max(1, limit // 4))
        articles.extend(tg)
        logger.debug("Telegram: %d posts", len(tg))
    except Exception as e:
        logger.debug("Telegram scrape error: %s", e)

    logger.info("Social media collected: %d posts for market=%s", len(articles), market)
    return articles[:limit]


# ═══════════════════════════════════════════════════════════════════════
# Twitter/X Scraper
# ═══════════════════════════════════════════════════════════════════════


def _fetch_twitter(market: str, limit: int) -> List[Dict]:
    """Fetch tweets related to stock market keywords.

    Uses snscrape.modules.twitter.TwitterSearchScraper.
    """
    query_map = {
        "CN": "($AAPL OR $BABA OR $NIO OR $KWEB OR China stock) lang:en",
        "US": "($SPY OR $QQQ OR stock market OR earnings OR Fed) lang:en since:1d",
        "KR": "($EWY OR KOSPI OR KOSDAQ) lang:en",
        "JP": "($EWJ OR Nikkei OR TOPIX) lang:en",
    }
    query = query_map.get(market, "stock market lang:en since:1d")

    from snscrape.modules import twitter as sntwitter

    articles = []
    try:
        scraper = sntwitter.TwitterSearchScraper(query)
        for i, tweet in enumerate(scraper.get_items()):
            if i >= limit:
                break
            if not tweet.content:
                continue

            user = tweet.user
            articles.append({
                "source": "twitter",
                "source_cn": "Twitter/X",
                "type": "social",
                "title": tweet.content[:200] if tweet.content else "",
                "text": tweet.content[:2000],
                "url": tweet.url or "",
                "published": tweet.date.isoformat() if tweet.date else "",
                "user": user.username if user else "",
                "display_name": user.displayname if user else "",
                "sentiment": 0.0,  # Will be refined by sentiment engine
                "language": "en",
                "market": market,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "followers": user.followersCount if user else 0,
                "following": user.friendsCount if user else 0,
                "retweets": tweet.retweetCount or 0,
                "likes": tweet.likeCount or 0,
                "replies": tweet.replyCount or 0,
                "quote_count": tweet.quoteCount or 0,
                "is_retweet": tweet.retweetedTweet is not None,
            })
    except Exception as e:
        logger.debug("Twitter scraper iteration error: %s", e)

    return articles


# ═══════════════════════════════════════════════════════════════════════
# Weibo Scraper
# ═══════════════════════════════════════════════════════════════════════


def _fetch_weibo(market: str, limit: int) -> List[Dict]:
    """Fetch Weibo posts about stock market (Chinese).

    Only runs for CN and JP markets.
    """
    if market not in ("CN", "JP"):
        return []

    query = "A股 股市" if market == "CN" else "日本 株"

    from snscrape.modules import weibo as snweibo

    articles = []
    try:
        scraper = snweibo.WeiboSearcher(query)
        for i, post in enumerate(scraper.get_items()):
            if i >= limit:
                break
            content = post.content or ""
            if not content:
                continue

            user = getattr(post, 'user', None)
            articles.append({
                "source": "weibo",
                "source_cn": "微博",
                "type": "social",
                "title": content[:200] if content else "",
                "text": content[:2000],
                "url": post.url or "",
                "published": post.date.isoformat() if hasattr(post, 'date') and post.date else "",
                "user": user.username if user else "",
                "display_name": user.displayname if user else "",
                "sentiment": 0.0,
                "language": "zh",
                "market": market,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "likes": getattr(post, 'likeCount', 0) or 0,
                "reposts": getattr(post, 'repostCount', 0) or 0,
                "comments": getattr(post, 'commentCount', 0) or 0,
            })
    except Exception as e:
        logger.debug("Weibo scraper error: %s", e)

    return articles


# ═══════════════════════════════════════════════════════════════════════
# Reddit Scraper
# ═══════════════════════════════════════════════════════════════════════


def _fetch_reddit(limit: int) -> List[Dict]:
    """Fetch Reddit posts from stock/investing subreddits.

    Targets: wallstreetbets, stocks, investing, stockmarket.
    """
    subreddits = ["wallstreetbets", "stocks", "investing", "stockmarket"]

    from snscrape.modules import reddit as snreddit

    articles = []
    per_sub = max(1, limit // len(subreddits))

    for sub in subreddits:
        query = f"subreddit:{sub}"
        try:
            scraper = snreddit.RedditSearchScraper(query)
            for i, post in enumerate(scraper.get_items()):
                if i >= per_sub:
                    break
                title = (post.title or "")
                selftext = (getattr(post, 'selftext', '') or '')
                text = f"{title} {selftext}".strip()
                if not text:
                    continue

                articles.append({
                    "source": "reddit",
                    "source_cn": f"r/{sub}",
                    "type": "social",
                    "title": title[:200],
                    "text": text[:2000],
                    "url": post.url or "",
                    "published": post.date.isoformat() if hasattr(post, 'date') and post.date else "",
                    "user": post.author if hasattr(post, 'author') else "",
                    "sentiment": 0.0,
                    "language": "en",
                    "market": "US",
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "score": getattr(post, 'score', 0) or 0,
                    "comments": getattr(post, 'numComments', 0) or 0,
                    "upvote_ratio": getattr(post, 'upvoteRatio', 0.0) or 0.0,
                    "subreddit": sub,
                })
        except Exception as e:
            logger.debug("Reddit sub=%s error: %s", sub, e)

    return articles


# ═══════════════════════════════════════════════════════════════════════
# Telegram Scraper
# ═══════════════════════════════════════════════════════════════════════


def _fetch_telegram(market: str, limit: int) -> List[Dict]:
    """Fetch posts from public Telegram financial channels.

    Uses snscrape.modules.telegram.TelegramChannelScraper.
    """
    # Public financial Telegram channels
    channels_map = {
        "CN": ["A股情报", "财经早报", "股票交流"],
        "US": ["stockmarketnews", "wallstreetbets", "cryptomarket"],
        "default": ["financialnews"],
    }
    channels = channels_map.get(market, channels_map["default"])

    from snscrape.modules import telegram as sntelegram

    articles = []
    per_channel = max(1, limit // len(channels))

    for channel in channels:
        try:
            scraper = sntelegram.TelegramChannelScraper(channel)
            for i, msg in enumerate(scraper.get_items()):
                if i >= per_channel:
                    break
                content = msg.content or ""
                if not content:
                    continue

                articles.append({
                    "source": "telegram",
                    "source_cn": f"Telegram/{channel}",
                    "type": "social",
                    "title": content[:200],
                    "text": content[:2000],
                    "url": msg.url or "",
                    "published": msg.date.isoformat() if hasattr(msg, 'date') and msg.date else "",
                    "user": "",
                    "sentiment": 0.0,
                    "language": "en" if market != "CN" else "zh",
                    "market": market,
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "views": getattr(msg, 'views', 0) or 0,
                    "channel": channel,
                })
        except Exception as e:
            logger.debug("Telegram channel=%s error: %s", channel, e)

    return articles


# ═══════════════════════════════════════════════════════════════════════
# YouTube Scraper
# ═══════════════════════════════════════════════════════════════════════


def fetch_youtube_comments(market: str = "US", limit: int = 50) -> List[Dict]:
    """Fetch YouTube financial videos and optionally their comments.

    Requires: youtube-search and youtube-comment-downloader.
    Graceful fallback if either is missing.
    """
    query_map = {
        "CN": "A股分析 股票推荐 财经新闻",
        "US": "stock market analysis today finance news",
        "KR": "주식 분석 증시",
        "JP": "株式 分析 日本株",
    }
    query = query_map.get(market, "stock market analysis")

    # Check for youtube-search
    has_yt_search = importlib.util.find_spec("youtube_search") is not None
    if not has_yt_search:
        logger.debug("youtube-search not installed — skipping YouTube scraping")
        return []

    from youtube_search import YoutubeSearch

    articles: List[Dict] = []
    try:
        results = YoutubeSearch(query, max_results=5).to_dict()
    except Exception as e:
        logger.debug("YouTube search error: %s", e)
        return []

    for video in results:
        vid_id = video.get("id", "")
        title = video.get("title", "")
        if not vid_id or not title:
            continue

        art = {
            "source": "youtube",
            "source_cn": "YouTube",
            "type": "social",
            "title": title[:200],
            "text": title[:2000],
            "url": f"https://youtube.com/watch?v={vid_id}",
            "published": video.get("publish_time", ""),
            "sentiment": 0.0,
            "language": "en" if market != "CN" else "zh",
            "market": market,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "views": video.get("views", "0"),
            "channel": video.get("channel", ""),
            "duration": video.get("duration", ""),
            "comments_text": "",
        }

        # Try to fetch comments (optional enhancement)
        try:
            has_comment_dl = importlib.util.find_spec(
                "youtube_comment_downloader"
            ) is not None
            if has_comment_dl:
                from youtube_comment_downloader import YoutubeCommentDownloader
                downloader = YoutubeCommentDownloader()
                comments = downloader.get_comments(vid_id)
                comment_texts = [
                    c.get("text", "") for _, c in zip(range(20), comments)
                ]
                if comment_texts:
                    art["comments_text"] = " ".join(comment_texts)
                    art["text"] = f"{title} {' '.join(comment_texts)}"[:2000]
        except Exception as e:
            logger.debug("YouTube comment fetch error: %s", e)

        articles.append(art)

    logger.info("YouTube collected: %d videos for market=%s", len(articles), market)
    return articles[:limit]
