#!/usr/bin/env python3
"""
News Sentiment Timeline — 7×24 Hourly Polling Engine
=====================================================
Maintains a rolling parquet file of per-hour sentiment snapshots.
Designed to be called via cron every hour.

Data flow:
  poll → sentiment modules → append to rolling parquet → predictor reads 24h curve

Output:
  data/news_timeline/sentiment_series.parquet  (per-hour snapshots, last 72h)
  data/news_timeline/sentiment_curve.json      (lightweight 24h snapshot for quick read)

Usage:
  # Run once (for cron):
  python scripts/news_timeline.py

  # Run with verbose logging:
  python scripts/news_timeline.py --verbose

  # Query the saved curve:
  python scripts/news_timeline.py --show

  # Set custom market(s):
  python scripts/news_timeline.py --markets CN,US
"""
import sys
import os
import json
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "news_timeline"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SERIES_PARQUET = DATA_DIR / "sentiment_series.parquet"
CURVE_JSON     = DATA_DIR / "sentiment_curve.json"
LOG_DIR        = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────
MAX_HISTORY_HOURS = 72   # rolling window
POLL_INTERVAL_H   = 1    # expected interval (for gap detection)

logger = logging.getLogger("news_timeline")


def setup_logging(verbose: bool = False):
    """Configure logging to file + stderr with optional verbose mode."""
    level = logging.DEBUG if verbose else logging.INFO
    log_path = LOG_DIR / f"news_timeline_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════
# Storage Layer
# ═══════════════════════════════════════════════════════════════════════

def _load_existing() -> pd.DataFrame:
    """Load existing sentiment series, or return empty DataFrame."""
    if SERIES_PARQUET.exists():
        try:
            df = pd.read_parquet(SERIES_PARQUET)
            logger.info("Loaded existing timeline: %d rows, date range [%s, %s]",
                        len(df),
                        df["timestamp"].min() if not df.empty else "N/A",
                        df["timestamp"].max() if not df.empty else "N/A")
            return df
        except Exception as e:
            logger.warning("Failed to load existing timeline: %s", e)
    return pd.DataFrame()


def _prune_old(df: pd.DataFrame, max_hours: int = MAX_HISTORY_HOURS) -> pd.DataFrame:
    """Remove entries older than max_hours from the reference time."""
    if df.empty:
        return df
    if "timestamp" not in df.columns:
        return df
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_hours)
    before = len(df)
    df = df[df["timestamp"] >= cutoff].copy()
    pruned = before - len(df)
    if pruned:
        logger.info("Pruned %d old entries (kept %d)", pruned, len(df))
    return df


def _detect_gaps(df: pd.DataFrame) -> list:
    """Detect polling gaps (missing hours) for logging."""
    if df.empty or "timestamp" not in df.columns:
        return []
    timestamps = sorted(pd.to_datetime(df["timestamp"]))
    gaps = []
    for i in range(1, len(timestamps)):
        gap = (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600
        if gap > 1.5 * POLL_INTERVAL_H:
            gaps.append({
                "from": timestamps[i - 1].isoformat(),
                "to": timestamps[i].isoformat(),
                "gap_hours": round(gap, 1),
            })
    return gaps


def save_timeline(df: pd.DataFrame):
    """Save the timeline to parquet + lightweight json snapshot."""
    # Parquet
    try:
        df.to_parquet(SERIES_PARQUET, index=False)
        logger.info("Timeline saved: %s (%d rows, %.1f KB)",
                    SERIES_PARQUET, len(df),
                    SERIES_PARQUET.stat().st_size / 1024)
    except Exception as e:
        logger.warning("Failed to save parquet: %s", e)

    # Lightweight JSON (last 24h for quick reads)
    if not df.empty:
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            recent = df[pd.to_datetime(df["timestamp"]) >= cutoff]
            snapshot = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "total_snapshots": len(recent),
                "markets": {},
            }
            if "market" in recent.columns:
                for market in recent["market"].unique():
                    mdf = recent[recent["market"] == market]
                    snapshot["markets"][market] = {
                        "current_score": float(mdf.sort_values("timestamp", ascending=False).iloc[0]["sentiment_score"]),
                        "score_min": float(mdf["sentiment_score"].min()),
                        "score_max": float(mdf["sentiment_score"].max()),
                        "score_mean": float(mdf["sentiment_score"].mean()),
                        "score_std": float(mdf["sentiment_score"].std()),
                        "trend": "rising" if len(mdf) > 1 and mdf["sentiment_score"].iloc[-1] > mdf["sentiment_score"].iloc[0] else "falling",
                        "snapshots": len(mdf),
                        "first_ts": mdf["timestamp"].min(),
                        "last_ts": mdf["timestamp"].max(),
                    }
            CURVE_JSON.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))
            logger.info("Lightweight snapshot saved: %s", CURVE_JSON)
        except Exception as e:
            logger.debug("JSON snapshot save failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════
# Polling Core
# ═══════════════════════════════════════════════════════════════════════

def collect_sentiment(markets: list = None) -> pd.DataFrame:
    """
    Poll sentiment from all configured sources and return a single-row
    DataFrame with the latest snapshot.

    Returns:
        DataFrame with columns: timestamp, market, sentiment_score,
        article_count, source_counts, top_tickers (JSON), is_trading_hour
    """
    if markets is None:
        from config import ACTIVE_MARKETS
        markets = ACTIVE_MARKETS

    from news_sentiment import collect_and_analyze

    snapshots = []
    now = datetime.now(timezone.utc)
    cst_hour = (now + timedelta(hours=8)).hour  # CST hour
    is_trading = 9 <= cst_hour <= 15 and now.weekday() < 5

    for market in markets:
        try:
            result = collect_and_analyze(
                market=market,
                tickers=None,
                run_rss=True,
                run_api=bool(os.getenv("ALPHA_VANTAGE_KEY")),
                run_social=True,
                use_cache=False,  # Force fresh fetch for timeline
            )

            sentiment_df = result.get("sentiment_df")
            if sentiment_df is not None and len(sentiment_df) > 0:
                # Aggregate to market-level score
                scores = [r.get("sentiment_score", 0) for r in sentiment_df]
                market_score = float(np.mean(scores)) if scores else 0.0
            else:
                market_score = result.get("market_score", 0.0)

            # Top tickers by absolute sentiment
            ticker_sent = result.get("ticker_sentiment", {})
            sorted_tickers = sorted(
                ticker_sent.items(),
                key=lambda x: abs(x[1].get("sentiment_score", 0)),
                reverse=True
            )[:10]

            snapshots.append({
                "timestamp": now,
                "market": market,
                "sentiment_score": round(market_score, 4),
                "article_count": result.get("total_articles", 0),
                "source_counts": json.dumps(result.get("articles_by_source", {}), ensure_ascii=False),
                "top_tickers": json.dumps([
                    {"ticker": t[0], "score": t[1].get("sentiment_score", 0),
                     "count": t[1].get("news_count", 0)}
                    for t in sorted_tickers
                ], ensure_ascii=False),
                "is_trading_hour": is_trading,
                "collection_time_s": round(
                    (datetime.now(timezone.utc) - now).total_seconds(), 1
                ),
            })

            logger.info("[%s] sentiment=%.4f articles=%d tickers=%d",
                        market, market_score, result.get("total_articles", 0),
                        len(sorted_tickers))

        except Exception as e:
            logger.error("Sentiment poll failed for %s: %s", market, e)
            snapshots.append({
                "timestamp": now,
                "market": market,
                "sentiment_score": 0.0,
                "article_count": 0,
                "source_counts": "{}",
                "top_tickers": "[]",
                "is_trading_hour": is_trading,
                "collection_time_s": 0.0,
                "error": str(e),
            })

    return pd.DataFrame(snapshots)


def run_poll(markets: list = None, verbose: bool = False):
    """Main polling function: fetch sentiment, merge with history, save."""
    setup_logging(verbose)
    logger.info("=== News Sentiment Timeline Poll ===")
    logger.info("Markets: %s", markets or ["CN", "US"])

    # ── Collect fresh sentiment ──
    new = collect_sentiment(markets)
    if new.empty:
        logger.warning("No data collected this poll")
        return

    logger.info("Collected %d market snapshots", len(new))

    # ── Merge with history ──
    existing = _load_existing()
    combined = pd.concat([existing, new], ignore_index=True) if not existing.empty else new

    # Deduplicate (if this hour was already polled, keep the latest)
    if "timestamp" in combined.columns:
        combined["_hour"] = pd.to_datetime(combined["timestamp"]).dt.strftime("%Y-%m-%d %H:00:00")
        combined = combined.sort_values("timestamp").drop_duplicates(
            subset=["_hour", "market"], keep="last"
        ).drop(columns=["_hour"])

    # Prune old data
    combined = _prune_old(combined)

    # Check for gaps
    gaps = _detect_gaps(combined)
    if gaps:
        logger.warning("Polling gaps detected: %d gaps", len(gaps))
        for g in gaps:
            logger.warning("  Gap: %s → %s (%.1f hours)", g["from"], g["to"], g["gap_hours"])

    # Save
    save_timeline(combined)

    # ── Summary ──
    logger.info("Timeline: %d total snapshots, %.1f hours window",
                len(combined), MAX_HISTORY_HOURS)
    logger.info("=== Poll Complete ===")


# ═══════════════════════════════════════════════════════════════════════
# Curve Query API (used by predictor)
# ═══════════════════════════════════════════════════════════════════════

def load_sentiment_curve(market: str = "CN", hours: int = 24) -> pd.DataFrame:
    """
    Load the sentiment curve for a market over the last N hours.

    Returns:
        DataFrame with columns: timestamp, sentiment_score, article_count,
        market_score (smoothed), sentiment_trend, volatility
    """
    if not SERIES_PARQUET.exists():
        logger.info("No sentiment timeline data yet")
        return pd.DataFrame()

    try:
        df = pd.read_parquet(SERIES_PARQUET)
    except Exception as e:
        logger.warning("Failed to load sentiment curve: %s", e)
        return pd.DataFrame()

    if df.empty:
        return df

    # Filter market
    if "market" in df.columns and market:
        df = df[df["market"] == market].copy()

    if df.empty:
        logger.info("No data for market=%s in timeline", market)
        return pd.DataFrame()

    # Filter time window
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    df = df[pd.to_datetime(df["timestamp"]) >= cutoff].copy()

    if df.empty:
        return df

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute derived features
    scores = df["sentiment_score"].values
    df["market_score_smoothed"] = pd.Series(
        np.convolve(scores, np.ones(3) / 3, mode="same")
        if len(scores) >= 3 else scores
    )
    df["sentiment_trend"] = df["sentiment_score"].diff().fillna(0).round(4)
    df["volatility"] = df["sentiment_score"].rolling(5, min_periods=2).std().fillna(0).round(4)

    logger.info("Loaded sentiment curve: %d points, score=%.4f trend=%.4f vol=%.4f",
                len(df),
                df["sentiment_score"].iloc[-1] if not df.empty else 0,
                df["sentiment_trend"].iloc[-1] if len(df) > 1 else 0,
                df["volatility"].iloc[-1] if len(df) > 1 else 0)
    return df


def get_sentiment_summary(market: str = "CN") -> dict:
    """Return a concise summary of the current sentiment state."""
    curve = load_sentiment_curve(market, hours=24)
    if curve.empty:
        return {"market": market, "available": False}

    last = curve.iloc[-1]
    first = curve.iloc[0]

    return {
        "market": market,
        "available": True,
        "current_score": float(last["sentiment_score"]),
        "trend_1h": float(last["sentiment_trend"]) if "sentiment_trend" in last else 0,
        "change_24h": float(last["sentiment_score"] - first["sentiment_score"]),
        "volatility": float(last["volatility"]) if "volatility" in last else 0,
        "article_count_24h": int(curve["article_count"].sum()),
        "snapshots_24h": len(curve),
        "smoothed_score": float(last.get("market_score_smoothed", last["sentiment_score"])),
        "title": "📈 情感积极" if last["sentiment_score"] > 0.15 else (
            "📉 情感消极" if last["sentiment_score"] < -0.15 else "➖ 情感中性"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def show_curve(market: str = "CN"):
    """Display the sentiment curve for a market."""
    curve = load_sentiment_curve(market, hours=48)
    if curve.empty:
        print(f"No sentiment data for {market}")
        return

    print(f"\n{'='*60}")
    print(f"📊 Sentiment Timeline — {market}")
    print(f"{'='*60}")
    print(f"  Snapshots: {len(curve)}")
    print(f"  Range:     {curve['timestamp'].min().strftime('%m/%d %H:%M')} → {curve['timestamp'].max().strftime('%m/%d %H:%M')}")
    print(f"  Score:     {curve['sentiment_score'].iloc[-1]:.4f} (range [{curve['sentiment_score'].min():.4f}, {curve['sentiment_score'].max():.4f}])")
    print(f"  Volatility:{curve['volatility'].iloc[-1]:.4f}")
    print(f"  Articles:  {int(curve['article_count'].sum())} total")

    # Recent trend (last 5 points)
    print(f"\n  Recent 5 snapshots:")
    for i in range(max(0, len(curve) - 5), len(curve)):
        r = curve.iloc[i]
        marker = "🟢" if r["sentiment_score"] > 0.05 else "🔴" if r["sentiment_score"] < -0.05 else "⚪"
        print(f"    {marker} {r['timestamp'].strftime('%H:%M')}  {r['sentiment_score']:+.4f}  ({r['article_count']} articles)")

    # Trend direction
    print(f"\n  → 24h trend: ", end="")
    if len(curve) >= 2:
        change = curve["sentiment_score"].iloc[-1] - curve["sentiment_score"].iloc[0]
        if change > 0.05:
            print(f"📈 上涨 {change:+.4f}")
        elif change < -0.05:
            print(f"📉 下跌 {change:+.4f}")
        else:
            print(f"➡️ 横盘 {change:+.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="News Sentiment Timeline — Hourly Polling Engine")
    parser.add_argument("--markets", default="CN,US",
                        help="Comma-separated market codes (default: CN,US)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging (DEBUG level)")
    parser.add_argument("--show", "-s", action="store_true",
                        help="Show current sentiment curve and exit")
    parser.add_argument("--summarize", action="store_true",
                        help="Quick summary for cron notification")
    parser.add_argument("--check-gaps", action="store_true",
                        help="Check for polling gaps without collecting")

    args = parser.parse_args()

    if args.show:
        for market in args.markets.split(","):
            show_curve(market.strip())
        return

    if args.summarize:
        for market in args.markets.split(","):
            summary = get_sentiment_summary(market.strip())
            if summary.get("available"):
                print(f"[{market}] {summary['title']} score={summary['current_score']:.4f} "
                      f"trend={summary['trend_1h']:+.4f} articles={summary['article_count_24h']}")
            else:
                print(f"[{market}] No data")
        return

    if args.check_gaps:
        existing = _load_existing()
        gaps = _detect_gaps(existing)
        if gaps:
            print(f"Found {len(gaps)} gaps:")
            for g in gaps:
                print(f"  {g['from']} → {g['to']} ({g['gap_hours']:.1f}h)")
        else:
            print("No gaps detected")
        return

    markets = [m.strip() for m in args.markets.split(",")]
    run_poll(markets, verbose=args.verbose)


if __name__ == "__main__":
    main()
