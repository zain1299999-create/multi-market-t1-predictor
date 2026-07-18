"""
Multi-Market T+1 Predictor — Signal Pre-Processor
=================================================
Pipeline:
  1. Winsorize (clip extreme values at configurable percentiles)
  2. Cross-sectional rank normalization (per date, per feature → 0~1 percentile)
  3. Time-series Z-Score normalization (per symbol rolling window)
  4. NaN/Inf handling (forward-fill then zero-fill)
"""
import numpy as np
import pandas as pd
import logging

from config import (ZSCORE_WINDOW, ZSCORE_MIN_PERIODS,
                    WINSORIZE_LOWER, WINSORIZE_UPPER)

logger = logging.getLogger(__name__)

# ── Columns excluded from signal processing ────────────────────────────
EXCLUDE_COLS = {"date", "symbol", "label", "market", "name", "industry",
                "weight", "pct_chg", "close"}


def winsorize(df: pd.DataFrame, feature_cols: list,
              lower: float = WINSORIZE_LOWER,
              upper: float = WINSORIZE_UPPER) -> pd.DataFrame:
    """Clip per-column extreme values at given percentile thresholds.

    Each column is clipped at its own empirical quantiles.
    Edge case: if lower == 0.0 and upper == 1.0, no clipping.
    """
    if lower <= 0.0 and upper >= 1.0:
        return df

    result = df.copy()
    for col in feature_cols:
        series = result[col]
        lo = series.quantile(lower) if lower > 0.0 else series.min()
        hi = series.quantile(upper) if upper < 1.0 else series.max()
        result[col] = series.clip(lower=lo, upper=hi)
    return result


def cross_sectional_rank(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Convert each feature to daily cross-sectional percentile ranks.

    For each (date, feature): rank stocks → map to [0, 1].
    This removes market-wide bias and outliers.
    """
    result = df.copy()
    for col in feature_cols:
        result[col] = result.groupby("date")[col].rank(pct=True).fillna(0.5)
    return result


def time_series_zscore(df: pd.DataFrame, feature_cols: list,
                       symbol_col: str = "symbol",
                       window: int = ZSCORE_WINDOW,
                       min_periods: int = ZSCORE_MIN_PERIODS) -> pd.DataFrame:
    """Per-symbol rolling z-score normalization.

    For each (symbol, feature): z = (x - rolling_mean) / (rolling_std + eps).
    Zeros out when std < 1e-10 (flat series).
    """
    result = df.copy()
    eps = 1e-10

    for col in feature_cols:
        grouped = result.groupby(symbol_col)[col]
        mean = grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
        std  = grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).std(ddof=0))

        z = (result[col] - mean) / (std + eps)
        result[col] = z.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    return result


def fill_missing(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Forward-fill then zero-fill all feature columns."""
    result = df.copy()
    for col in feature_cols:
        result[col] = result.groupby("symbol")[col].transform(
            lambda s: s.ffill().fillna(0.0)
        )
    return result


def get_feature_cols(df: pd.DataFrame) -> list:
    """Auto-detect feature columns (everything not in EXCLUDE_COLS)."""
    return [c for c in df.columns
            if c not in EXCLUDE_COLS
            and not c.startswith(("^", "dx-"))
            and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)]


def preprocess(df: pd.DataFrame,
               do_winsorize: bool = True,
               do_rank: bool = True,
               do_zscore: bool = False,
               do_fill: bool = True) -> pd.DataFrame:
    """Full signal preprocessing pipeline.

    Typical flow for training:
        preprocess(df, do_winsorize=True, do_rank=True, do_zscore=False, do_fill=True)
        → Winsorize → Rank → Fill (z-score optional, often done inside model)

    Typical flow for inference:
        preprocess(df, do_winsorize=True, do_rank=True, do_zscore=False, do_fill=True)

    Args:
        df: Feature DataFrame with date, symbol columns.
        do_winsorize: Clip extreme values.
        do_rank: Cross-sectional rank normalize.
        do_zscore: Time-series z-score normalize (slower).
        do_fill: Forward-fill + zero-fill NaN.

    Returns:
        Preprocessed DataFrame (same shape, same index).
    """
    if df.empty:
        return df

    feature_cols = get_feature_cols(df)
    if not feature_cols:
        logger.warning("No feature columns found for preprocessing")
        return df

    logger.info("Signal preprocessing: %d features × %d rows",
                len(feature_cols), len(df))

    result = df.copy()

    if do_winsorize:
        result = winsorize(result, feature_cols)
        logger.info("  Winsorize done [%.1f%%, %.1f%%]",
                    WINSORIZE_LOWER * 100, WINSORIZE_UPPER * 100)

    if do_rank:
        result = cross_sectional_rank(result, feature_cols)
        logger.info("  Cross-sectional rank done")

    if do_zscore:
        result = time_series_zscore(result, feature_cols)
        logger.info("  Time-series z-score done (window=%d)", ZSCORE_WINDOW)

    if do_fill:
        result = fill_missing(result, feature_cols)
        logger.info("  Missing values filled")

    return result
