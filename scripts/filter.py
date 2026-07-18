"""
Multi-Market T+1 Predictor — Market-Specific Filters
Merged from:
  - a-share: ST/*ST/limit-up/IPO/low-liquidity/industry-diversify
  - multi-market: per-market price limits, T+ rules, min volume
"""
import logging
import pandas as pd
import numpy as np

from config import (MIN_TRADE_DAYS, MIN_VOLUME_RATIO,
                    INDUSTRY_MAX_SHARE, MARKET_RULES)

logger = logging.getLogger(__name__)


def filter_market(df: pd.DataFrame, market: str = "CN",
                  top_k: int = 10) -> pd.DataFrame:
    """Apply market-specific filters for a single market at latest date.

    Returns filtered DataFrame with only the latest date's rows.
    """
    if df.empty:
        return df

    latest_date = df["date"].max()
    today = df[df["date"] == latest_date].copy()
    if today.empty:
        return today

    before = len(today)
    rules = MARKET_RULES.get(market, {})

    # ── CN-specific: ST filter ──
    if rules.get("st_filter") and "name" in today.columns:
        st = today["name"].str.contains("ST|退|暂停", na=False)
        n_st = st.sum()
        today = today[~st]
        if n_st:
            logger.info("  [%s] Removed %d ST stocks", market, n_st)

    # ── Min trading days ──
    if "symbol" in today.columns and "symbol" in df.columns:
        trade_counts = df.groupby("symbol").size()
        today = today[today["symbol"].map(trade_counts).fillna(0) >= MIN_TRADE_DAYS]

    # ── Price limit filter ──
    price_limit = rules.get("price_limit")
    if price_limit and "pct_chg" in today.columns:
        hit_limit = today["pct_chg"].abs() > (price_limit * 0.995)
        n_limit = hit_limit.sum()
        today = today[~hit_limit]
        if n_limit:
            logger.info("  [%s] Removed %d limit-hit stocks", market, n_limit)

    # ── CN-specific limit-up/down ──
    if market == "CN" and rules.get("limit_up_filter") and "pct_chg" in today.columns:
        close_to_limit = today["pct_chg"].abs() > 9.95
        today = today[~close_to_limit]

    # ── Liquidity: volume ratio ──
    if "volume_ratio_5" in today.columns:
        thin = today["volume_ratio_5"] < MIN_VOLUME_RATIO
        n_thin = thin.sum()
        today = today[~thin]
        if n_thin:
            logger.info("  [%s] Removed %d low-liquidity stocks", market, n_thin)

    # ── Min amount/volume (market-specific) ──
    min_amount_keys = {"CN": "min_volume_cny", "US": "min_volume_usd",
                       "KR": "min_volume_krw", "JP": "min_volume_jpy"}
    key = min_amount_keys.get(market)
    if key and key in rules and "amount" in today.columns:
        min_val = rules[key]
        today = today[today["amount"] >= min_val]

    after = len(today)
    logger.info("  [%s] %d → %d after filters (%d removed)",
                market, before, after, before - after)
    return today


def diversify_picks(scored_df: pd.DataFrame,
                    top_k: int = 10) -> pd.DataFrame:
    """Diversify picks: limit stocks per industry/symbol group."""
    df = scored_df.copy()
    if "industry" not in df.columns:
        df["industry"] = "unknown"

    df = df.sort_values("pred_ret", ascending=False).reset_index(drop=True)

    selected = []
    remaining = df.copy()
    while len(selected) < top_k and not remaining.empty:
        available = remaining.groupby("industry").head(INDUSTRY_MAX_SHARE)
        next_pick = available.iloc[0]
        selected.append(next_pick)
        remaining = remaining[remaining["symbol"] != next_pick["symbol"]].reset_index(drop=True)

    result = pd.DataFrame(selected)
    logger.info("Diversified picks: %d from %d", len(result), len(df))
    return result


def compute_rank_metrics(df: pd.DataFrame, top_k: int = 10) -> dict:
    """Compute quality metrics on predictions."""
    if df.empty or "pred_ret" not in df.columns:
        return {}
    preds = df["pred_ret"].dropna()
    if len(preds) == 0:
        return {}
    return {
        "n_picks": min(len(df), top_k),
        "avg_pred_ret": float(preds.head(top_k).mean()),
        "max_pred_ret": float(preds.head(top_k).max()),
        "min_pred_ret": float(preds.head(top_k).min()),
        "pred_std":     float(preds.head(top_k).std()),
    }
