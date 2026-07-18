"""
Multi-Market T+1 Predictor — Feature Engineering
Merged factors:
  - A-share project: 24 factors (MA/ROC/Volatility/Bollinger/MACD/RSI/CCI/ATR/Volume/Momentum)
  - Multi-market project: cross-market features (VIX, FX, global indices)
  - Multi-market project: sentiment features (sentiment_score, MA, change, news_volume)
  - Label: next-day return (T+1)
"""
import numpy as np
import pandas as pd
from typing import Optional

from config import (MA_WINDOWS, ROC_WINDOWS, STD_WINDOWS,
                    VOLUME_MA_WINDOWS, MOMENTUM_WINDOWS)


def compute_features_for_one_stock(group: pd.DataFrame) -> pd.DataFrame:
    """Compute all 24+ technical factors for a single stock.

    Input columns: date, open, high, low, close, volume, [amount, pct_chg, turnover]
    sorted by date ascending.
    """
    df = group.copy().sort_values("date").reset_index(drop=True)
    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    pct = df.get("pct_chg", pd.Series(0.0, index=df.index)).values.astype(np.float64)
    turnover = df.get("turnover", pd.Series(0.0, index=df.index)).values.astype(np.float64)
    amount = df.get("amount", pd.Series(0.0, index=df.index)).values.astype(np.float64)

    # ── 1. Moving Averages ──
    for w in MA_WINDOWS:
        df[f"ma_{w}"] = _roll_mean(close, w)
        df[f"close_ma_{w}_ratio"] = close / np.maximum(_roll_mean(close, w), 1e-10) - 1.0

    # ── 2. ROC ──
    for w in ROC_WINDOWS:
        df[f"roc_{w}"] = _roll_pct(close, w)

    # ── 3. Volatility ──
    ret = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-10)
    ret = np.where(np.isfinite(ret), ret, 0.0)
    for w in STD_WINDOWS:
        df[f"volatility_{w}"] = _roll_std(ret, w)

    # ── 4. Bollinger ──
    bb_mean = _roll_mean(close, 20)
    bb_std  = _roll_std(close, 20)
    df["bb_position"] = np.where(bb_std > 1e-10, (close - bb_mean) / (2 * bb_std), 0.0)
    df["bb_width"]    = bb_std / np.maximum(bb_mean, 1e-10)

    # ── 5. MACD ──
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    df["macd"]         = ema12 - ema26
    df["macd_signal"]  = _ema(df["macd"].values, 9)
    df["macd_hist"]    = df["macd"] - df["macd_signal"]

    # ── 6. RSI ──
    df["rsi_14"] = _rsi(close, 14)

    # ── 7. CCI ──
    tp = (high + low + close) / 3.0
    sma_tp = _roll_mean(tp, 20)
    mad_tp = _roll_mad(tp, 20)
    df["cci_20"] = np.where(mad_tp > 1e-10, (tp - sma_tp) / (0.015 * mad_tp), 0.0)

    # ── 8. ATR ──
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close),
                               np.abs(low - prev_close)))
    df["atr_14"]   = _roll_mean(tr, 14)
    df["atr_pct"]  = df["atr_14"] / np.maximum(close, 1e-10)

    # ── 9. Price Position ──
    df["high_20"] = _roll_max(high, 20)
    df["low_20"]  = _roll_min(low, 20)
    rng = df["high_20"] - df["low_20"]
    df["price_position_20"] = np.where(rng > 1e-10, (close - df["low_20"]) / rng, 0.5)

    # ── 10. Volume ──
    for w in VOLUME_MA_WINDOWS:
        df[f"volume_ma_{w}"] = _roll_mean(volume, w)
    df["volume_ratio_5"]  = volume / np.maximum(df["volume_ma_5"].values, 1e-10)
    df["volume_change_1"] = volume / np.maximum(np.roll(volume, 1), 1e-10) - 1.0
    df["volume_change_1"] = df["volume_change_1"].fillna(0.0)

    # ── 11. Turnover ──
    if turnover.sum() > 0:
        df["turnover_ma_20"] = _roll_mean(turnover, 20)
        df["turnover_ratio"] = turnover / np.maximum(df["turnover_ma_20"].values, 1e-10)

    # ── 12. Amount (liquidity) ──
    df["amount_ma_20"] = _roll_mean(amount, 20) if amount.sum() > 0 else 0.0
    df["amount_ratio"] = amount / np.maximum(df["amount_ma_20"].values, 1e-10)

    # ── 13. Momentum ──
    for d in MOMENTUM_WINDOWS:
        df[f"momentum_{d}d"] = close / np.maximum(np.roll(close, d), 1e-10) - 1.0

    # ── 14. Daily amplitude ──
    df["amplitude"] = (high - low) / np.maximum(close, 1e-10)

    # ── Label: next-day return (T+1) ──
    df["label"] = np.roll(close, -1) / np.maximum(close, 1e-10) - 1.0
    df["label"] = df["label"].fillna(0.0)

    # Clean inf/nan
    _feature_cols = [c for c in df.columns if c not in ("date", "symbol", "label")]
    df[_feature_cols] = df[_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def add_sentiment_features(df: pd.DataFrame,
                           sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Merge sentiment features into main feature DataFrame.

    sentiment_df columns: date, symbol, sentiment_score, news_count
    """
    if sentiment_df.empty:
        df["sentiment_score"] = 0.0
        df["sentiment_ma_5"]  = 0.0
        df["sentiment_change"] = 0.0
        df["news_count"]      = 0
        return df

    df = df.merge(
        sentiment_df[["date", "symbol", "sentiment_score", "news_count"]],
        on=["date", "symbol"], how="left"
    )
    df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
    df["news_count"]      = df["news_count"].fillna(0).astype(int)

    # Rolling sentiment features per symbol
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["sentiment_ma_5"] = df.groupby("symbol")["sentiment_score"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    df["sentiment_change"] = df.groupby("symbol")["sentiment_score"].transform(
        lambda s: s.diff().fillna(0.0)
    )
    return df


def add_cross_market_features(df: pd.DataFrame,
                               macro_df: pd.DataFrame) -> pd.DataFrame:
    """Attach macro/cross-market features aligned by date."""
    if macro_df.empty:
        for col in ["vix", "usd_cny", "vix_change", "sp500_ret", "hsi_ret"]:
            df[col] = 0.0
        return df

    # Flatten macro: daily → flat columns
    macro_flat = macro_df.copy()
    if isinstance(macro_flat.index, pd.DatetimeIndex):
        macro_flat = macro_flat.reset_index()
    macro_flat.columns = [str(c).lower() for c in macro_flat.columns]
    if "index" in macro_flat.columns:
        macro_flat.rename(columns={"index": "date"}, inplace=True)

    macro_flat["date"] = pd.to_datetime(macro_flat["date"])
    # Merge on date
    df = df.merge(macro_flat, on="date", how="left")

    # Calculate change features
    for col in macro_flat.columns:
        if col == "date":
            continue
        try:
            arr = df[col].values.astype(np.float64)
            df[f"{col}_change"] = arr / np.maximum(np.roll(arr, 1), 1e-10) - 1.0
            df[f"{col}_change"] = df[f"{col}_change"].fillna(0.0)
        except Exception:
            df[f"{col}_change"] = 0.0

    df = df.fillna(0.0)
    return df


def build_all_features(ohlcv_df: pd.DataFrame,
                       market: str = "CN",
                       sentiment_df: Optional[pd.DataFrame] = None,
                       macro_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Complete feature pipeline: technical + sentiment + cross-market."""
    if ohlcv_df.empty:
        return pd.DataFrame()

    # 1. Per-stock technical features
    parts = []
    for sym, grp in ohlcv_df.groupby("symbol"):
        try:
            feat = compute_features_for_one_stock(grp)
            parts.append(feat)
        except Exception as exc:
            logger.debug("Feature failed for %s: %s", sym, exc)
            continue

    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)

    # 2. Sentiment features
    if sentiment_df is not None:
        df = add_sentiment_features(df, sentiment_df)

    # 3. Cross-market features
    if macro_df is not None:
        df = add_cross_market_features(df, macro_df)

    # 4. Market tag
    df["market"] = market

    return df


# ─── Low-level helpers ─────────────────────────────────────────────────

def _roll_mean(arr, w):
    s = pd.Series(arr)
    return s.rolling(w, min_periods=w // 2).mean().values


def _roll_std(arr, w):
    s = pd.Series(arr)
    return s.rolling(w, min_periods=w // 2).std(ddof=0).fillna(0.0).values


def _roll_pct(arr, w):
    shifted = np.roll(arr, w)
    shifted[:w] = arr[:w]
    return (arr - shifted) / np.maximum(shifted, 1e-10)


def _roll_max(arr, w):
    s = pd.Series(arr)
    return s.rolling(w, min_periods=1).max().values


def _roll_min(arr, w):
    s = pd.Series(arr)
    return s.rolling(w, min_periods=1).min().values


def _roll_mad(arr, w):
    s = pd.Series(arr)
    mean = s.rolling(w, min_periods=w // 2).mean()
    mad = (s - mean).abs().rolling(w, min_periods=w // 2).mean()
    return mad.fillna(0.0).values


def _ema(arr, span):
    s = pd.Series(arr)
    return s.ewm(span=span, adjust=False).mean().values


def _rsi(arr, period=14):
    s = pd.Series(arr)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    return (100 - 100 / (1 + rs)).fillna(50.0).values


import logging
logger = logging.getLogger(__name__)
