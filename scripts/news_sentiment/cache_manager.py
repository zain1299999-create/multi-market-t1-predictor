"""
Disk-Based Cache Manager for Collected News/Sentiment
======================================================
Purposes:
  1. Avoid re-fetching API data every pipeline run (rate limit savings)
  2. Avoid re-running NLP on same articles
  3. Enable incremental updates

Cache structure:
  data/news_cache/
    {market}_articles.json    — raw article data
    {market}_aggregated.json  — aggregated sentiment features
    cache_meta.json           — timestamps per market
"""
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from .config_news import CACHE_DIR, MAX_CACHE_AGE_HOURS

logger = logging.getLogger("cache_manager")


class NewsCache:
    """Disk cache for news articles and aggregated sentiment results.

    Design:
      - Per-market JSON files for articles and aggregated results
      - Metadata file tracks freshness per market
      - Corrupted cache files are silently deleted and treated as miss
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.articles_dir = self.cache_dir / "articles"
        self.agg_dir = self.cache_dir / "aggregated"
        self.meta_path = self.cache_dir / "cache_meta.json"

        # Ensure directories exist
        self.articles_dir.mkdir(parents=True, exist_ok=True)
        self.agg_dir.mkdir(parents=True, exist_ok=True)

        self._meta: Dict[str, str] = self._load_meta()

    # ── Public API ─────────────────────────────────────────────────────

    def is_fresh(self, market: str, max_age_hours: float = MAX_CACHE_AGE_HOURS) -> bool:
        """Check if cached data for a market is still fresh.

        Args:
            market: Market code
            max_age_hours: Max age in hours before cache is stale

        Returns:
            True if cache exists and is fresh
        """
        last_updated = self._meta.get(market)
        if not last_updated:
            return False

        try:
            last_dt = datetime.fromisoformat(last_updated)
            age = datetime.now(timezone.utc) - last_dt.replace(tzinfo=timezone.utc)
            return age < timedelta(hours=max_age_hours)
        except (ValueError, TypeError):
            return False

    def load(self, market: str) -> Optional[Dict]:
        """Load cached sentiment result for a market.

        Returns:
            The cached result dict, or None if no valid cache exists
        """
        cache_file = self.articles_dir / f"{market}_articles.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Also load aggregated if available
            agg_file = self.agg_dir / f"{market}_aggregated.json"
            if agg_file.exists():
                with open(agg_file, "r", encoding="utf-8") as af:
                    agg_data = json.load(af)
                    data["sentiment_df"] = agg_data.get("sentiment_df")
                    data["ticker_sentiment"] = agg_data.get("ticker_sentiment", {})

            logger.debug("Cache LOADED for market=%s (%d articles)",
                         market, len(data.get("articles", [])))
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Cache load error for %s: %s — clearing", market, e)
            self._clear_market(market)
            return None

    def save(self, market: str, data: Dict) -> None:
        """Save sentiment result to cache.

        Separates articles from aggregated data for efficient partial loads.
        """
        try:
            # Save articles
            articles_path = self.articles_dir / f"{market}_articles.json"
            articles_data = {
                "market": market,
                "total_articles": data.get("total_articles", 0),
                "market_score": data.get("market_score", 0.0),
                "articles_by_source": data.get("articles_by_source", {}),
                "articles": data.get("articles", []),
                "collected_at": data.get("collected_at", datetime.now(timezone.utc).isoformat()),
            }

            with open(articles_path, "w", encoding="utf-8") as f:
                json.dump(articles_data, f, ensure_ascii=False, indent=2)

            # Save aggregated data separately
            agg_path = self.agg_dir / f"{market}_aggregated.json"
            agg_data = {
                "market": market,
                "sentiment_df": data.get("sentiment_df"),
                "ticker_sentiment": data.get("ticker_sentiment", {}),
                "collected_at": data.get("collected_at"),
            }
            with open(agg_path, "w", encoding="utf-8") as f:
                json.dump(agg_data, f, ensure_ascii=False, indent=2)

            # Update metadata
            self._meta[market] = datetime.now(timezone.utc).isoformat()
            self._save_meta()

            logger.debug("Cache SAVED for market=%s (%d articles)",
                         market, len(data.get("articles", [])))
        except OSError as e:
            logger.warning("Cache save error for %s: %s", market, e)

    def get_cache_age(self, market: str) -> Optional[float]:
        """Get cache age in hours for a market.

        Returns:
            Age in hours, or None if no cache exists
        """
        last_updated = self._meta.get(market)
        if not last_updated:
            return None
        try:
            last_dt = datetime.fromisoformat(last_updated)
            age = datetime.now(timezone.utc) - last_dt.replace(tzinfo=timezone.utc)
            return age.total_seconds() / 3600.0
        except (ValueError, TypeError):
            return None

    def clear(self, market: Optional[str] = None) -> None:
        """Clear cache for one or all markets.

        Args:
            market: If provided, clear only this market; else clear all
        """
        if market:
            self._clear_market(market)
            self._meta.pop(market, None)
            self._save_meta()
            logger.info("Cache cleared for market=%s", market)
        else:
            # Clear all market caches
            for f in self.articles_dir.glob("*.json"):
                f.unlink()
            for f in self.agg_dir.glob("*.json"):
                f.unlink()
            self._meta.clear()
            self._save_meta()
            logger.info("All caches cleared")

    def list_markets(self) -> List[str]:
        """List markets with cached data."""
        return [f.stem.replace("_articles", "") for f in self.articles_dir.glob("*_articles.json")]

    # ── Internal ───────────────────────────────────────────────────────

    def _load_meta(self) -> Dict[str, str]:
        """Load metadata from disk."""
        if not self.meta_path.exists():
            return {}
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_meta(self) -> None:
        """Save metadata to disk."""
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self._meta, f, indent=2)
        except OSError as e:
            logger.debug("Meta save error: %s", e)

    def _clear_market(self, market: str) -> None:
        """Delete cache files for a specific market."""
        articles_file = self.articles_dir / f"{market}_articles.json"
        agg_file = self.agg_dir / f"{market}_aggregated.json"
        for p in (articles_file, agg_file):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    # ── Stat ───────────────────────────────────────────────────────────

    def get_stat(self) -> Dict:
        """Get cache statistics."""
        stats = {
            "total_markets": len(self._meta),
            "markets": {},
            "cache_dir_size_bytes": 0,
        }

        for market, last_updated in self._meta.items():
            age = self.get_cache_age(market)
            articles_file = self.articles_dir / f"{market}_articles.json"
            agg_file = self.agg_dir / f"{market}_aggregated.json"

            stats["markets"][market] = {
                "last_updated": last_updated,
                "age_hours": round(age, 2) if age is not None else None,
                "articles_file_exists": articles_file.exists(),
                "articles_file_size": articles_file.stat().st_size if articles_file.exists() else 0,
                "aggregated_file_exists": agg_file.exists(),
            }

            # Accumulate size
            for f in [articles_file, agg_file]:
                if f.exists():
                    stats["cache_dir_size_bytes"] += f.stat().st_size

        stats["cache_dir_size_mb"] = round(stats["cache_dir_size_bytes"] / (1024 * 1024), 2)
        return stats
