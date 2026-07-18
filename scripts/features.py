"""
Multi-Market T+1 Predictor — Feature Engineering
================================================
Merged factor system:
  Layer 1 — 24 technical factors (MA/ROC/Vol/Bollinger/MACD/RSI/CCI/ATR/Volume/Momentum)
  Layer 2 — Qlib Alpha158 (KBar 9 + BETA/RSQR/RESI/QTLU/QTLD/RANK/RSV + IMAX/IMIN/IMXD +
             CORR/CORD + CNTP/CNTN/CNTD/SUMP/SUMN/SUMD + VMA/VSTD/WVMA/VSUMP/VSUMN/VSUMD)
  Layer 3 — Sentiment features (score, rolling MA, change, news count)
  Layer 4 — Cross-market features (global indices, VIX, FX)
  Layer 5 — Label: next-day return (T+1)
"""
import numpy as np
import pandas as pd
from typing import Optional

from config import (MA_WINDOWS, ROC_WINDOWS, STD_WINDOWS,
                    VOLUME_MA_WINDOWS, MOMENTUM_WINDOWS, ALPHA158_WINDOWS)


def compute_features_for_one_stock(group: pd.DataFrame) -> pd.DataFrame:
    """Compute ALL factors for a single stock.

    Input columns: date, open, high, low, close, volume, [amount, pct_chg, turnover]
    sorted by date ascending.

    Returns df with all factor columns + label.
    """
    df = group.copy().sort_values("date").reset_index(drop=True)
    close  = df["close"].values.astype(np.float64)
    open_  = df["open"].values.astype(np.float64)
    high   = df["high"].values.astype(np.float64)
    low    = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    n      = len(close)
    eps    = 1e-12

    turnover = df.get("turnover", pd.Series(0.0, index=df.index)).values.astype(np.float64)
    amount   = df.get("amount", pd.Series(0.0, index=df.index)).values.astype(np.float64)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LAYER 1 — Original 24 technical factors (keep unchanged)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 1a. Moving Averages
    for w in MA_WINDOWS:
        df[f"ma_{w}"] = _roll_mean(close, w)
        df[f"close_ma_{w}_ratio"] = close / np.maximum(_roll_mean(close, w), eps) - 1.0

    # 1b. ROC
    for w in ROC_WINDOWS:
        df[f"roc_{w}"] = _roll_pct(close, w)

    # 1c. Volatility of returns
    ret = np.diff(close, prepend=close[0]) / np.maximum(close, eps)
    ret = np.where(np.isfinite(ret), ret, 0.0)
    for w in STD_WINDOWS:
        df[f"volatility_{w}"] = _roll_std(ret, w)

    # 1d. Bollinger
    bb_mean = _roll_mean(close, 20)
    bb_std  = _roll_std(close, 20)
    df["bb_position"] = np.where(bb_std > eps, (close - bb_mean) / (2 * bb_std), 0.0)
    df["bb_width"]    = bb_std / np.maximum(bb_mean, eps)

    # 1e. MACD
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    df["macd"]         = ema12 - ema26
    df["macd_signal"]  = _ema(df["macd"].values, 9)
    df["macd_hist"]    = df["macd"] - df["macd_signal"]

    # 1f. RSI
    df["rsi_14"] = _rsi(close, 14)

    # 1g. CCI
    tp = (high + low + close) / 3.0
    sma_tp = _roll_mean(tp, 20)
    mad_tp = _roll_mad(tp, 20)
    df["cci_20"] = np.where(mad_tp > eps, (tp - sma_tp) / (0.015 * mad_tp), 0.0)

    # 1h. ATR
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close),
                               np.abs(low - prev_close)))
    df["atr_14"]  = _roll_mean(tr, 14)
    df["atr_pct"] = df["atr_14"] / np.maximum(close, eps)

    # 1i. Price Position (original)
    df["high_20"] = _roll_max(high, 20)
    df["low_20"]  = _roll_min(low, 20)
    rng = df["high_20"] - df["low_20"]
    df["price_position_20"] = np.where(rng > eps, (close - df["low_20"]) / rng, 0.5)

    # 1j. Volume
    for w in VOLUME_MA_WINDOWS:
        df[f"volume_ma_{w}"] = _roll_mean(volume, w)
    df["volume_ratio_5"]  = volume / np.maximum(df["volume_ma_5"].values, eps)
    df["volume_change_1"] = volume / np.maximum(np.roll(volume, 1), eps) - 1.0
    df["volume_change_1"] = df["volume_change_1"].fillna(0.0)

    # 1k. Turnover & Amount
    if turnover.sum() > 0:
        df["turnover_ma_20"] = _roll_mean(turnover, 20)
        df["turnover_ratio"] = turnover / np.maximum(df["turnover_ma_20"].values, eps)
    if amount.sum() > 0:
        df["amount_ma_20"] = _roll_mean(amount, 20)
        df["amount_ratio"] = amount / np.maximum(df["amount_ma_20"].values, eps)

    # 1l. Momentum
    for d in MOMENTUM_WINDOWS:
        s = np.roll(close, d)
        s[:d] = close[:d]
        df[f"momentum_{d}d"] = close / np.maximum(s, eps) - 1.0

    # 1m. Daily amplitude
    df["amplitude"] = (high - low) / np.maximum(close, eps)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LAYER 2 — Alpha158 factor groups
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── 2a. KBar morphology (9 factors) ──
    df["KMID"]   = (close - open_) / np.maximum(open_, eps)
    df["KLEN"]   = (high - low) / np.maximum(open_, eps)
    df["KMID2"]  = (close - open_) / (high - low + eps)
    upper_wick   = high - np.maximum(open_, close)
    df["KUP"]    = upper_wick / np.maximum(open_, eps)
    df["KUP2"]   = upper_wick / (high - low + eps)
    lower_wick   = np.minimum(open_, close) - low
    df["KLOW"]   = lower_wick / np.maximum(open_, eps)
    df["KLOW2"]  = lower_wick / (high - low + eps)
    df["KSFT"]   = (2.0 * close - high - low) / np.maximum(open_, eps)
    df["KSFT2"]  = (2.0 * close - high - low) / (high - low + eps)

    # ── 2b. Rolling regression trend — BETA / RSQR / RESI ──
    for w in ALPHA158_WINDOWS:
        beta, rsqr, resi = _linreg_stats(close, w)
        df[f"BETA_{w}"]  = beta / np.maximum(close, eps)
        df[f"RSQR_{w}"]  = rsqr
        df[f"RESI_{w}"]  = resi / np.maximum(close, eps)

    # ── 2c. Price quantile / rank / RSV ──
    for w in ALPHA158_WINDOWS:
        df[f"QTLU_{w}"] = _roll_quantile(close, w, 0.8) / np.maximum(close, eps)
        df[f"QTLD_{w}"] = _roll_quantile(close, w, 0.2) / np.maximum(close, eps)
        df[f"RANK_{w}"] = _roll_percentile(close, w)

        hh = _roll_max(high, w)
        ll = _roll_min(low, w)
        df[f"RSV_{w}"]  = (close - ll) / (hh - ll + eps)

    # ── 2d. Aroon indicators: IMAX / IMIN / IMXD ──
    for w in ALPHA158_WINDOWS:
        df[f"IMAX_{w}"] = _idx_max(high, w) / w
        df[f"IMIN_{w}"] = _idx_min(low, w) / w
        df[f"IMXD_{w}"] = (df[f"IMAX_{w}"] - df[f"IMIN_{w}"]).values

    # ── 2e. Price-volume correlation: CORR / CORD ──
    log_vol = np.log(volume + 1.0)
    close_chg = close / np.maximum(np.roll(close, 1), eps)
    vol_chg   = volume / np.maximum(np.roll(volume, 1), eps)
    log_vol_chg = np.log(np.maximum(vol_chg, eps))

    s_close = pd.Series(close)
    s_log_vol = pd.Series(log_vol)
    s_close_chg = pd.Series(close_chg)
    s_log_vol_chg = pd.Series(log_vol_chg)

    for w in ALPHA158_WINDOWS:
        df[f"CORR_{w}"] = s_close.rolling(w, min_periods=w // 2).corr(s_log_vol).fillna(0.0).values
        df[f"CORD_{w}"] = s_close_chg.rolling(w, min_periods=w // 2).corr(s_log_vol_chg).fillna(0.0).values

    # ── 2f. Up/down count and gain/loss ──
    up   = np.maximum(close - np.roll(close, 1), 0.0)
    down = np.maximum(np.roll(close, 1) - close, 0.0)
    abs_chg = np.abs(close - np.roll(close, 1))
    # First element: no prior day
    up[0] = 0.0; down[0] = 0.0; abs_chg[0] = 0.0

    s_up     = pd.Series(up)
    s_down   = pd.Series(down)
    s_abs    = pd.Series(abs_chg)
    close_gt = pd.Series(close > np.roll(close, 1)).fillna(False).astype(float)
    close_lt = pd.Series(close < np.roll(close, 1)).fillna(False).astype(float)

    for w in ALPHA158_WINDOWS:
        # CNTP / CNTN / CNTD
        cntp = close_gt.rolling(w, min_periods=1).mean().fillna(0.5).values
        cntn = close_lt.rolling(w, min_periods=1).mean().fillna(0.5).values
        df[f"CNTP_{w}"] = cntp
        df[f"CNTN_{w}"] = cntn
        df[f"CNTD_{w}"] = cntp - cntn

        # SUMP / SUMN / SUMD (RSI-style gain/loss)
        sum_abs = s_abs.rolling(w, min_periods=w // 2).sum().fillna(eps).values
        sum_up  = s_up.rolling(w, min_periods=w // 2).sum().fillna(0.0).values
        sum_dn  = s_down.rolling(w, min_periods=w // 2).sum().fillna(0.0).values
        df[f"SUMP_{w}"] = sum_up / (sum_abs + eps)
        df[f"SUMN_{w}"] = sum_dn / (sum_abs + eps)
        df[f"SUMD_{w}"] = (sum_up - sum_dn) / (sum_abs + eps)

    # ── 2g. Volume statistics: VMA / VSTD / WVMA / VSUMP / VSUMN / VSUMD ──
    s_vol = pd.Series(volume)
    s_price_chg = pd.Series(np.abs(close / np.maximum(np.roll(close, 1), eps) - 1.0))
    vol_up   = np.maximum(volume - np.roll(volume, 1), 0.0)
    vol_dn   = np.maximum(np.roll(volume, 1) - volume, 0.0)
    vol_abs  = np.abs(volume - np.roll(volume, 1))
    vol_up[0] = 0.0; vol_dn[0] = 0.0; vol_abs[0] = 0.0
    s_vol_up  = pd.Series(vol_up)
    s_vol_dn  = pd.Series(vol_dn)
    s_vol_abs = pd.Series(vol_abs)
    vol_gt = pd.Series(volume > np.roll(volume, 1)).fillna(False).astype(float)
    vol_lt = pd.Series(volume < np.roll(volume, 1)).fillna(False).astype(float)

    for w in ALPHA158_WINDOWS:
        vma = s_vol.rolling(w, min_periods=w // 2).mean().fillna(eps).values
        df[f"VMA_{w}"]   = volume / (vma + eps)
        df[f"VSTD_{w}"]  = s_vol.rolling(w, min_periods=w // 2).std(ddof=0).fillna(0.0).values / (vma + eps)

        # WVMA: volatility of volume-weighted price change
        wvma_series = (s_price_chg * s_vol).rolling(w, min_periods=w // 2)
        wvma_mean   = wvma_series.mean().fillna(eps).values
        wvma_std    = wvma_series.std(ddof=0).fillna(0.0).values
        df[f"WVMA_{w}"] = wvma_std / (wvma_mean + eps)

        # Volume up/down percentages
        vsum_abs = s_vol_abs.rolling(w, min_periods=w // 2).sum().fillna(eps).values
        vsum_up  = s_vol_up.rolling(w, min_periods=w // 2).sum().fillna(0.0).values
        vsum_dn  = s_vol_dn.rolling(w, min_periods=w // 2).sum().fillna(0.0).values
        df[f"VSUMP_{w}"] = vsum_up / (vsum_abs + eps)
        df[f"VSUMN_{w}"] = vsum_dn / (vsum_abs + eps)
        df[f"VSUMD_{w}"] = (vsum_up - vsum_dn) / (vsum_abs + eps)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LABEL: next-day return (T+1)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    df["label"] = np.roll(close, -1) / np.maximum(close, eps) - 1.0
    df["label"] = df["label"].fillna(0.0)

    # ── Clean inf/nan on all feature columns ──
    _feature_cols = [c for c in df.columns if c not in ("date", "symbol", "label")]
    df[_feature_cols] = df[_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def add_sentiment_features(df: pd.DataFrame,
                           sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Merge sentiment features into main feature DataFrame.

    Layer 3 sentiment features include:
      - sentiment_score:  current weighted sentiment
      - sentiment_ma_5:   5-day rolling average
      - sentiment_change: day-over-day sentiment change
      - sentiment_std_5:  5-day rolling std (disagreement measure)
      - sentiment_trend:  trend direction over 3 days
      - news_count:       article count (volume proxy)
      - market_sentiment: aggregate market-level sentiment (from all sources)
    """
    if sentiment_df is not None and not sentiment_df.empty:
        # Merge news sentiment data
        merge_cols = ["date", "symbol", "sentiment_score", "news_count"]
        available = [c for c in merge_cols if c in sentiment_df.columns]

        # Also try "sentiment_std" if available
        if "sentiment_std" in sentiment_df.columns:
            available.append("sentiment_std")

        df = df.merge(
            sentiment_df[available],
            on=["date", "symbol"], how="left"
        )
        df["sentiment_score"] = df.get("sentiment_score", pd.Series(0.0)).fillna(0.0)
        df["news_count"]      = df.get("news_count", pd.Series(0.0)).fillna(0).astype(int)
        if "sentiment_std" in df.columns:
            df["sentiment_std"] = df["sentiment_std"].fillna(0.0)
    else:
        # No sentiment data — fill with zeros
        df["sentiment_score"] = 0.0
        df["news_count"]      = 0

    df["news_count"] = df["news_count"].clip(upper=1000)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Rolling sentiment features
    group_sent = df.groupby("symbol")["sentiment_score"]
    df["sentiment_ma_3"]   = group_sent.transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    ).fillna(0.0).values
    df["sentiment_ma_5"]   = group_sent.transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    ).fillna(0.0).values
    df["sentiment_ma_10"]  = group_sent.transform(
        lambda s: s.rolling(10, min_periods=1).mean()
    ).fillna(0.0).values
    df["sentiment_std_5"]  = group_sent.transform(
        lambda s: s.rolling(5, min_periods=1).std()
    ).fillna(0.0).values
    df["sentiment_change"] = group_sent.transform(
        lambda s: s.diff().fillna(0.0)
    ).values

    # Sentiment trend: slope over last 3 days (positive = improving)
    def _sentiment_trend(series):
        if len(series) < 3:
            return 0.0
        return (series.iloc[-1] - series.iloc[-3]) / 2.0

    df["sentiment_trend"] = df.groupby("symbol")["sentiment_score"] \
        .transform(lambda s: s.rolling(3, min_periods=1)
                   .apply(_sentiment_trend, raw=False)).fillna(0.0).values

    # Interaction: sentiment * volume ratio (sentiment-confirmed volume)
    if "volume_ratio_5" in df.columns:
        df["sentiment_volume_interaction"] = df["sentiment_score"] * df["volume_ratio_5"]
    else:
        df["sentiment_volume_interaction"] = df["sentiment_score"]

    return df



def add_cross_market_features(df: pd.DataFrame,
                               macro_df: pd.DataFrame) -> pd.DataFrame:
    """Attach macro/cross-market features aligned by date."""
    if macro_df.empty:
        for col in ["vix", "gspc", "n225", "ks11", "hsi", "dx_y_nyb",
                     "vix_change", "gspc_ret", "hsi_ret"]:
            df[col] = 0.0
        return df

    macro_flat = macro_df.copy()
    if isinstance(macro_flat.index, pd.DatetimeIndex):
        macro_flat = macro_flat.reset_index()
    macro_flat.columns = [str(c).lower() for c in macro_flat.columns]
    if "index" in macro_flat.columns:
        macro_flat.rename(columns={"index": "date"}, inplace=True)
    if "date" not in macro_flat.columns:
        macro_flat["date"] = pd.Timestamp.now().normalize()

    macro_flat["date"] = pd.to_datetime(macro_flat["date"])
    df = df.merge(macro_flat, on="date", how="left")

    for col in macro_flat.columns:
        if col == "date":
            continue
        try:
            arr = df[col].values.astype(np.float64)
            df[f"{col}_change"] = arr / np.maximum(np.roll(arr, 1), 1e-12) - 1.0
            df[f"{col}_change"] = df[f"{col}_change"].fillna(0.0)
        except Exception:
            df[f"{col}_change"] = 0.0

    df = df.fillna(0.0)
    return df


def build_all_features(ohlcv_df: pd.DataFrame,
                       market: str = "CN",
                       sentiment_df: Optional[pd.DataFrame] = None,
                       macro_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Complete feature pipeline: technical + Alpha158 + sentiment + cross-market."""
    if ohlcv_df.empty:
        return pd.DataFrame()

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

    if sentiment_df is not None:
        df = add_sentiment_features(df, sentiment_df)
    if macro_df is not None:
        df = add_cross_market_features(df, macro_df)

    df["market"] = market

    # ── Market-relative label neutralization ──
    # Subtract cross-sectional median T+1 return per-date
    # This removes the market beta component → model learns alpha
    if "label" in df.columns and "date" in df.columns:
        daily_median = df.groupby("date")["label"].transform("median")
        df["label"] = df["label"] - daily_median

    return df


# ═══════════════════════════════════════════════════════════════════════
# Low-level helpers
# ═══════════════════════════════════════════════════════════════════════

def _roll_mean(arr, w):
    return pd.Series(arr).rolling(w, min_periods=w // 2).mean().values

def _roll_std(arr, w):
    return pd.Series(arr).rolling(w, min_periods=w // 2).std(ddof=0).fillna(0.0).values

def _roll_pct(arr, w):
    s = np.roll(arr, w)
    s[:w] = arr[:w]
    return (arr - s) / np.maximum(s, 1e-12)

def _roll_max(arr, w):
    return pd.Series(arr).rolling(w, min_periods=1).max().values

def _roll_min(arr, w):
    return pd.Series(arr).rolling(w, min_periods=1).min().values

def _roll_mad(arr, w):
    s = pd.Series(arr)
    m = s.rolling(w, min_periods=w // 2).mean().values
    return (s - pd.Series(m)).abs().rolling(w, min_periods=w // 2).mean().fillna(0.0).values

def _ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values

def _rsi(arr, period=14):
    s = pd.Series(arr)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    return (100 - 100 / (1 + rs)).fillna(50.0).values

def _roll_quantile(arr, w, q):
    """Rolling quantile using expanding+rolling."""
    s = pd.Series(arr)
    return s.rolling(w, min_periods=w // 2).quantile(q).fillna(arr[0]).values

def _roll_percentile(arr, w):
    """Percentile rank of current value in rolling window."""
    s = pd.Series(arr)
    def _rank_pct(win):
        if len(win) < 2:
            return 0.5
        return (win.values <= win.iloc[-1]).sum() / len(win)
    return s.rolling(w, min_periods=w // 2).apply(_rank_pct, raw=False).fillna(0.5).values

def _idx_max(arr, w):
    """Days since highest high in window. Returns 0 at current if high is max."""
    s = pd.Series(arr)
    def _imax(win):
        if len(win) < 2:
            return 0.0
        return float(np.argmax(win.values))
    return s.rolling(w, min_periods=w // 2).apply(_imax, raw=False).fillna(0.0).values

def _idx_min(arr, w):
    """Days since lowest low in window."""
    s = pd.Series(arr)
    def _imin(win):
        if len(win) < 2:
            return 0.0
        return float(np.argmin(win.values))
    return s.rolling(w, min_periods=w // 2).apply(_imin, raw=False).fillna(0.0).values

def _linreg_stats(arr, w):
    """Rolling OLS: slope (beta), R², residuals of last point, all zero-padded."""
    n = len(arr)
    beta_arr  = np.zeros(n, dtype=np.float64)
    rsqr_arr  = np.zeros(n, dtype=np.float64)
    resi_arr  = np.zeros(n, dtype=np.float64)

    x = np.arange(w, dtype=np.float64)
    x_mean = x.mean()
    x_centered = x - x_mean
    ss_xx = np.sum(x_centered ** 2)

    for i in range(w - 1, n):
        y = arr[i - w + 1 : i + 1]
        y_mean = y.mean()
        y_centered = y - y_mean
        ss_xy = np.sum(x_centered * y_centered)
        ss_yy = np.sum(y_centered ** 2)

        if ss_xx < 1e-12 or ss_yy < 1e-12:
            continue

        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        resid = y - (slope * x + intercept)
        ss_res = np.sum(resid ** 2)

        beta_arr[i] = slope
        rsqr_arr[i] = 1.0 - ss_res / ss_yy if ss_yy > 1e-12 else 0.0
        resi_arr[i] = resid[-1]  # residual of latest point

    return beta_arr, rsqr_arr, resi_arr


import logging
logger = logging.getLogger(__name__)
