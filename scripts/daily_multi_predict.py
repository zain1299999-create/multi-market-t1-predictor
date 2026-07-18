#!/usr/bin/env python3
"""
Multi-Market T+1 Next-Day Return Predictor — Main Entry
==========================================================
Workflow:
  1. Detect trading days per market
  2. Fetch universe (index components or preset tickers)
  3. Fetch OHLCV + sentiment + macro for each active market
  4. Feature engineering (24+ technical + sentiment + cross-market)
  5. Train or load LightGBM model (time-series CV)
  6. Predict on latest cross-section per market
  7. Market-specific filters (ST/limit/IPO/liquidity/industry-diversify)
  8. Diversified TopK per market → consolidated CSV output

Usage:
    python scripts/daily_multi_predict.py

Output: outputs/multi_t1_pred_YYYYMMDD.csv
"""
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (ACTIVE_MARKETS, TOP_K, LOG_LEVEL, LOG_FORMAT,
                    TRAIN_YEARS, OUTPUTS, LOGS, DATA_CACHE, CONFIG_DIR,
                    MARKET_RULES)
from data_fetcher import (fetch_cn_index_components,
                          fetch_market_data, is_trading_day,
                          fetch_yf_ohlcv)
from features import build_all_features
from model import (prepare_training_data, train_model, predict,
                   load_model, should_retrain, get_model_age_days)
from filter import filter_market, diversify_picks, compute_rank_metrics

logger = logging.getLogger("multi_t1_predict")

log_filename = datetime.now().strftime("multi_daily_%Y%m%d.log")
log_path = LOGS / log_filename
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[logging.FileHandler(log_path, encoding="utf-8"),
              logging.StreamHandler()],
)
logger.info("=" * 60)
logger.info("Multi-Market T+1 Predictor — Starting")
logger.info("Active markets: %s | TopK: %d | Train years: %d",
            ACTIVE_MARKETS, TOP_K, TRAIN_YEARS)

# ── Market ticker presets ──────────────────────────────────────────────
MARKET_TICKERS = {
    "US": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AVGO", "JPM", "V"],
    "KR": ["005930.KS", "000660.KS", "035420.KQ", "091990.KQ"],
    "JP": ["7203.T", "9984.T", "6758.T", "9432.T"],
    "CN": None,  # fetched from index components
}


def get_universe(market: str) -> list:
    """Get ticker list for a given market."""
    if market == "CN":
        comp = fetch_cn_index_components("hs300")
        if not comp.empty:
            return comp["code"].tolist()[:500]
        # Fallback: common A-share
        return ["600519.SS", "000858.SZ", "300750.SZ", "601318.SS",
                "000333.SZ", "600036.SS", "002415.SZ", "688981.SS"]
    return MARKET_TICKERS.get(market, [])


def main():
    today_str = datetime.now().strftime("%Y%m%d")
    all_picks = []

    for market in ACTIVE_MARKETS:
        logger.info("\n─── Market: %s ───", market)

        if not is_trading_day(market):
            logger.info("  Not a trading day, skipping")
            continue

        # ── Step 1: Universe ──
        tickers = get_universe(market)
        if not tickers:
            logger.warning("  No tickers for %s, skipping", market)
            continue
        logger.info("  Universe: %d tickers", len(tickers))

        # ── Step 2: Data ──
        logger.info("  Fetching data...")
        period = f"{TRAIN_YEARS}y" if market != "CN" else "1y"
        data = fetch_market_data(market, tickers, period=period)
        ohlcv = data.get("ohlcv", pd.DataFrame())
        sentiment = data.get("sentiment", pd.DataFrame())
        macro = data.get("macro", pd.DataFrame())

        if ohlcv.empty:
            logger.warning("  No OHLCV data for %s", market)
            continue
        logger.info("  OHLCV: %d rows, %d symbols",
                    len(ohlcv), ohlcv["symbol"].nunique() if "symbol" in ohlcv.columns else 0)

        # ── Step 3: Features ──
        logger.info("  Building features...")
        features = build_all_features(
            ohlcv, market=market,
            sentiment_df=sentiment, macro_df=macro
        )
        if features.empty:
            logger.warning("  No features for %s", market)
            continue
        logger.info("  Features: %d rows x %d cols", len(features), len(features.columns))

        # ── Step 4: Model (cross-market = shared across all markets) ──
        # For simplicity, train shared model across all data
        logger.info("  Model stage...")
        model = None
        feature_cols = None

        model_age = get_model_age_days()
        needs_train = should_retrain(model_age)

        if needs_train:
            logger.info("  Training new model (age=%d days)...", model_age)
            X, y, idx, fcols = prepare_training_data(features)
            if X is not None:
                model, val_mae = train_model(X, y, fcols)
                feature_cols = fcols
        else:
            logger.info("  Loading existing model...")
            model, feature_cols = load_model()

        if model is None:
            # Train on this market's data as fallback
            X, y, idx, fcols = prepare_training_data(features)
            if X is not None:
                model, _ = train_model(X, y, fcols)
                feature_cols = fcols
            if model is None:
                logger.warning("  No model for %s, skipping prediction", market)
                continue

        # ── Step 5: Predict latest cross-section ──
        latest_date = features["date"].max()
        latest = features[features["date"] == latest_date].copy()
        if latest.empty:
            logger.warning("  No latest-date rows for %s", market)
            continue

        if feature_cols is None:
            exclude = {"date", "symbol", "label", "name", "market",
                       "high_20", "low_20"}
            feature_cols = [c for c in latest.columns if c not in exclude]

        available_cols = [c for c in feature_cols if c in latest.columns]
        if not available_cols:
            logger.warning("  No feature columns available for %s", market)
            continue

        X_latest = latest[available_cols].fillna(0).values.astype(np.float32)
        preds = predict(model, X_latest)
        latest["pred_ret"] = preds
        logger.info("  Predicted %d stocks", len(preds))

        # ── Step 6: Filter ──
        filtered = filter_market(latest, market=market, top_k=TOP_K)
        if filtered.empty:
            logger.warning("  All stocks filtered out for %s", market)
            continue

        # ── Step 7: Diversify ──
        diversified = diversify_picks(filtered, top_k=TOP_K)
        picks = diversified.head(TOP_K).copy()
        picks["rank"] = range(1, len(picks) + 1)
        picks["market"] = market

        # ── Step 8: Append ──
        all_picks.append(picks)

        metrics = compute_rank_metrics(picks, TOP_K)
        logger.info("  Metrics: %s", metrics)

    # ── Consolidate ──────────────────────────────────────────────────────
    if not all_picks:
        logger.warning("No predictions for any market!")
        return

    final = pd.concat(all_picks, ignore_index=True)

    out_cols = ["market", "rank", "symbol", "pred_ret"]
    for extra in ["close", "name", "pct_chg", "rsi_14", "bb_position",
                   "volume_ratio_5", "sentiment_score"]:
        if extra in final.columns:
            out_cols.append(extra)

    output_name = f"multi_t1_pred_{today_str}.csv"
    output_path = OUTPUTS / output_name
    final[out_cols].round(4).to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("\n✅ Predictions saved: %s", output_path)
    logger.info("=" * 60)

    print(f"\n✅ Multi-market TopK → {output_path}")
    print(final[out_cols].round(4).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        print(f"\n❌ Fatal error: {exc}")
        traceback.print_exc()
        sys.exit(1)
