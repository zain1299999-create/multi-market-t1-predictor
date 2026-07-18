#!/usr/bin/env python3
"""
Multi-Market T+1 Next-Day Return Predictor — Main Entry (Upgraded)
==================================================================
Workflow:
  1. Detect trading days per market
  2. Fetch universe (index components or preset tickers)
  3. Fetch OHLCV + sentiment + macro for each active market
  4. Feature engineering (Alpha158 + sentiment + cross-market, ~130+ factors)
  5. Signal preprocessing (winsorize → cross-sectional rank → fill)
  6. Train or load LightGBM model (time-series CV, optional ensemble)
  7. Compute factor IC & log top-10 features
  8. Predict on latest cross-section per market
  9. Market-specific filters (ST/limit/IPO/liquidity/industry-diversify)
  10. Diversified TopK per market → backtest → consolidated CSV

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
                    TRAIN_YEARS, OUTPUTS, LOGS, DATA_CACHE,
                    WFA_ENABLED, OPTUNA_ENABLED, ENSEMBLE_ENABLED,
                    BACKTEST_TOP_K, REPORTS)
from data_fetcher import (fetch_cn_index_components,
                          fetch_market_data, fetch_news_sentiment,
                          is_trading_day)
from features import build_all_features
from config import NEWS_SENTIMENT_ENABLED, NEWS_SENTIMENT_LIMIT
from signal_processor import preprocess, get_feature_cols
from model import (prepare_training_data, train_model, predict,
                   load_model, should_retrain, get_model_age_days,
                   run_walk_forward, compute_factor_ic)
from filter import filter_market, diversify_picks, compute_rank_metrics
from backtest import run_backtest, backtest_top_k

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
logger.info("Multi-Market T+1 Predictor v2 — Alpha158 Upgrade")
logger.info("Active: %s | TopK: %d | Train years: %d | Ensemble=%s | Optuna=%s",
            ACTIVE_MARKETS, TOP_K, TRAIN_YEARS, ENSEMBLE_ENABLED, OPTUNA_ENABLED)

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
        return ["600519.SS", "000858.SZ", "300750.SZ", "601318.SS",
                "000333.SZ", "600036.SS", "002415.SZ", "688981.SS"]
    return MARKET_TICKERS.get(market, [])


def main():
    today_str = datetime.now().strftime("%Y%m%d")
    all_picks = []
    all_predictions = []  # for backtest

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

        # ── Step 2b: News & Social Media Sentiment (new module) ──
        news_sentiment = pd.DataFrame()
        if NEWS_SENTIMENT_ENABLED:
            logger.info("  Fetching multi-source news & social media sentiment...")
            try:
                news_sentiment = fetch_news_sentiment(
                    market=market,
                    tickers=tickers if len(tickers) <= 50 else None,
                    use_cache=True,
                )
                if not news_sentiment.empty:
                    logger.info("  News sentiment: %d rows, %d tickers",
                                len(news_sentiment),
                                news_sentiment["symbol"].nunique() if "symbol" in news_sentiment.columns else 0)
                    # Merge with existing API sentiment (alpha_vantage/marketaux)
                    # Prefer news_sentiment for methods that also have API sentiment
                    if not sentiment.empty:
                        sentiment = pd.concat(
                            [sentiment, news_sentiment], ignore_index=True
                        )
                        sentiment = sentiment.groupby(["date", "symbol"]).agg(
                            sentiment_score=("sentiment_score", "mean"),
                            news_count=("news_count", "sum"),
                        ).reset_index()
                        logger.info("  Merged sentiment: %d rows", len(sentiment))
                    else:
                        # Rename columns to match the format features.py expects
                        sentiment = news_sentiment.rename(
                            columns={"sentiment_score": "sentiment_score"}
                        )
                else:
                    logger.info("  No news sentiment data collected")
            except Exception as e:
                logger.warning("  News sentiment fetch failed: %s", e)

        if ohlcv.empty:
            logger.warning("  No OHLCV data for %s", market)
            continue
        logger.info("  OHLCV: %d rows, %d symbols",
                    len(ohlcv), ohlcv["symbol"].nunique() if "symbol" in ohlcv.columns else 0)

        # ── Step 3: Features ──
        logger.info("  Building features (Alpha158 + sentiment + macro + news)...")
        features = build_all_features(
            ohlcv, market=market,
            sentiment_df=sentiment, macro_df=macro
        )
        if features.empty:
            logger.warning("  No features for %s", market)
            continue
        logger.info("  Raw features: %d rows x %d cols", len(features), len(features.columns))

        # ── Step 4: Signal processing ──
        logger.info("  Signal preprocessing...")
        features = preprocess(features, do_winsorize=True, do_rank=True, do_zscore=False, do_fill=True)
        if features.empty:
            logger.warning("  Preprocessing empty for %s", market)
            continue
        logger.info("  Preprocessed: %d rows x %d cols", len(features), len(features.columns))

        # ── Step 5: Model ──
        logger.info("  Model stage...")
        model = None
        feature_cols = None
        model_age = get_model_age_days()
        needs_train = should_retrain(model_age) or WFA_ENABLED

        if needs_train:
            logger.info("  Training new model (age=%d days)...", model_age)

            # Walk-Forward Analysis (when enabled)
            if WFA_ENABLED:
                logger.info("  Walk-Forward Analysis mode...")
                X, y, idx, group, fcols = prepare_training_data(features)
                if X is not None:
                    wfa_preds, wfa_model = run_walk_forward(features, fcols)
                    if not wfa_preds.empty:
                        all_predictions.append(wfa_preds)
                        logger.info("  WFA: %d predictions generated", len(wfa_preds))
                    if wfa_model is not None:
                        model = wfa_model
                        feature_cols = fcols
            else:
                # Standard training
                X, y, idx, group, fcols = prepare_training_data(features)
                if X is not None:
                    model, val_mae = train_model(X, y, idx, group, fcols)
                    feature_cols = fcols

        if model is None and not WFA_ENABLED:
            # Fallback: try loading or train on this market
            model, feature_cols = load_model()
            if model is None:
                X, y, idx, group, fcols = prepare_training_data(features)
                if X is not None:
                    model, _ = train_model(X, y, idx, group, fcols)
                    feature_cols = fcols
            if model is None:
                logger.warning("  No model for %s, skipping prediction", market)
                continue

        # ── Step 6: Predict latest cross-section ──
        latest_date = features["date"].max()
        latest = features[features["date"] == latest_date].copy()
        if latest.empty:
            logger.warning("  No latest-date rows for %s", market)
            continue

        if feature_cols is None:
            feature_cols = get_feature_cols(latest)

        available_cols = [c for c in feature_cols if c in latest.columns]
        if not available_cols:
            logger.warning("  No feature columns available for %s", market)
            continue

        X_latest = latest[available_cols].fillna(0).values.astype(np.float32)

        try:
            preds = predict(model, X_latest)
            latest["pred_ret"] = preds
            logger.info("  Predicted %d stocks", len(preds))

            # Factor importance from model
            if hasattr(model, 'feature_importance'):
                try:
                    imp = model.feature_importance()
                    if len(imp) == len(available_cols):
                        f_imp = sorted(zip(available_cols, imp), key=lambda x: x[1], reverse=True)
                        logger.info("  Top-10 factors by importance:")
                        for fname, sc in f_imp[:10]:
                            logger.info("    %s: %.1f", fname, sc)
                    else:
                        logger.warning("  Feature importance shape mismatch: %d vs %d",
                                       len(imp), len(available_cols))
                except Exception as e:
                    logger.debug("  Feature importance unavailable: %s", e)
        except Exception as exc:
            logger.warning("  Prediction failed for %s: %s", market, exc)
            continue

        # ── Step 7: Backtest (predictions) ──
        if needs_train and not WFA_ENABLED and "label" in latest.columns:
            # Use full feature df for backtest predictions
            try:
                X_all = features[available_cols].fillna(0).values.astype(np.float32)
                all_preds = predict(model, X_all)
                bt_df = features[["date", "symbol", "label"]].copy()
                bt_df["pred_ret"] = all_preds
                all_predictions.append(bt_df)
                logger.info("  Backtest predictions: %d rows", len(bt_df))
            except Exception as exc:
                logger.debug("  Backtest prediction skip: %s", exc)

        # ── Step 8: Filter ──
        filtered = filter_market(latest, market=market, top_k=TOP_K)
        if filtered.empty:
            logger.warning("  All stocks filtered out for %s", market)
            continue

        # ── Step 9: Diversify ──
        diversified = diversify_picks(filtered, top_k=TOP_K)
        picks = diversified.head(TOP_K).copy()
        picks["rank"] = range(1, len(picks) + 1)
        picks["market"] = market
        all_picks.append(picks)

        metrics = compute_rank_metrics(picks, TOP_K)
        logger.info("  Metrics: %s", metrics)

    # ── Run full backtest ─────────────────────────────────────────────────
    if all_predictions:
        bt_data = pd.concat(all_predictions, ignore_index=True)
        logger.info("\n─── Running Backtest ───")
        logger.info("  Predictions: %d rows, %d-%d",
                    len(bt_data),
                    bt_data["date"].min() if "date" in bt_data.columns else None,
                    bt_data["date"].max() if "date" in bt_data.columns else None)
        bt_results = run_backtest(bt_data, top_k_list=BACKTEST_TOP_K)

        # Save backtest summary
        bt_summary = backtest_top_k(bt_data, top_k_list=BACKTEST_TOP_K)
        if not bt_summary.empty:
            bt_path = REPORTS / f"backtest_summary_{today_str}.csv"
            bt_summary.to_csv(bt_path, index=False, encoding="utf-8-sig")
            logger.info("Backtest summary saved: %s", bt_path)

        # Save cumulative curves
        curves = bt_results.get("cumulative_curves", {})
        for k, cum in curves.items():
            if not cum.empty:
                cum_path = REPORTS / f"cumulative_top{k}_{today_str}.csv"
                cum.to_csv(cum_path, index=False, encoding="utf-8-sig")

    # ── Consolidate picks ─────────────────────────────────────────────────
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
    print(f"\n📊 Reports → {REPORTS}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        print(f"\n❌ Fatal error: {exc}")
        traceback.print_exc()
        sys.exit(1)
