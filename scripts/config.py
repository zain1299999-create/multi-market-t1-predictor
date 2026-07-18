"""
Multi-Market T+1 Predictor — Unified Configuration
Merged from: a-share-t1-predictor + multi_market_t1_predictor + Alpha158 upgrade
"""
import os
from pathlib import Path
from typing import Optional, List

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_CACHE    = PROJECT_ROOT / "data" / "cache"
OUTPUTS       = PROJECT_ROOT / "outputs"
LOGS          = PROJECT_ROOT / "logs"
REPORTS       = PROJECT_ROOT / "reports"
CONFIG_DIR    = PROJECT_ROOT / "config"

for p in (DATA_CACHE, OUTPUTS, LOGS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

# ── Market Universe ────────────────────────────────────────────────────
UNIVERSE      = "multi"            # "hs300" | "zz500" | "all_a" | "multi"
ACTIVE_MARKETS: List[str] = ["CN", "US"]  # CN, US, KR, JP

# ── Prediction ─────────────────────────────────────────────────────────
TOP_K            = 10
TRAIN_YEARS      = 2
RETRAIN_INTERVAL = 20

# ── Filter Rules ───────────────────────────────────────────────────────
MIN_TRADE_DAYS     = 60
MIN_VOLUME_RATIO   = 0.3
INDUSTRY_MAX_SHARE = 2

# ── Data sources ───────────────────────────────────────────────────────
DATA_START_DATE = "20200101"
CACHE_VERSION   = "v1"
ALPHA_VANTAGE_KEY: Optional[str] = os.getenv("ALPHA_VANTAGE_KEY")
MARKETAUX_KEY: Optional[str]     = os.getenv("MARKETAUX_KEY")

# ── Cross-market symbols ───────────────────────────────────────────────
MACRO_SYMBOLS: List[str] = [
    "^VIX", "^GSPC", "^N225", "^KS11", "^HSI", "DX-Y.NYB",
]

# ── Sentiment config ───────────────────────────────────────────────────
SENTIMENT_LOOKBACK_DAYS = 30
SENTIMENT_MAX_NEWS      = 200

# ── Feature engineering: existing ──────────────────────────────────────
MA_WINDOWS: List[int]       = [3, 5, 10, 20, 30, 60]
ROC_WINDOWS: List[int]      = [5, 10, 20, 60]
STD_WINDOWS: List[int]      = [5, 10, 20]
VOLUME_MA_WINDOWS: List[int] = [5, 20]
MOMENTUM_WINDOWS: List[int]  = [1, 5, 10, 20]

# ── Alpha158 rolling windows (for all new factor groups) ───────────────
ALPHA158_WINDOWS: List[int] = [5, 10, 20, 30, 60]

# ── Signal processing ──────────────────────────────────────────────────
ZSCORE_WINDOW       = 60      # rolling window for z-score normalization
ZSCORE_MIN_PERIODS  = 30
WINSORIZE_LOWER     = 0.01    # clip lower 1%
WINSORIZE_UPPER     = 0.99    # clip upper 1%

# ── Walk-Forward Analysis ──────────────────────────────────────────────
WFA_TRAIN_DAYS = 504           # 2 years ≈ 504 trading days
WFA_STEP_DAYS  = 20            # retrain every 20 days
WFA_ENABLED    = False         # set True for WFA (takes longer)

# ── Ensemble (Stacking) ────────────────────────────────────────────────
ENSEMBLE_ENABLED = True        # True = LGB + XGB + Ridge meta
META_MODEL       = "ridge"     # "ridge" | "linear" | "none"

# ── Model ──────────────────────────────────────────────────────────────
LGB_PARAMS = {
    "objective":        "regression",
    "metric":           "mae",
    "boosting_type":    "gbdt",
    "num_leaves":       31,
    "learning_rate":    0.05,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.80,
    "bagging_freq":     5,
    "verbose":          -1,
    "num_threads":      2,
    "seed":             42,
}
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30
OPTUNA_N_TRIALS = 30
OPTUNA_ENABLED  = False

# ── XGBoost params (used when ENSEMBLE_ENABLED) ────────────────────────
XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "mae",
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.80,
    "colsample_bytree": 0.85,
    "verbosity":        0,
    "nthread":          2,
    "seed":             42,
}

# ── Backtest ───────────────────────────────────────────────────────────
BACKTEST_TOP_K: List[int] = [5, 10, 20, 30]  # multiple K to compare
BACKTEST_GROUPS         = 5                   # Q1-Q5 quintile groups

# ── Market-specific rules ──────────────────────────────────────────────
MARKET_RULES = {
    "CN": {
        "price_limit": 0.10,
        "t_plus": 1,
        "min_volume_cny": 1e7,
        "st_filter": True,
        "limit_up_filter": True,
    },
    "US": {
        "price_limit": None,
        "t_plus": 1,
        "min_volume_usd": 1e6,
        "st_filter": False,
    },
    "KR": {
        "price_limit": 0.30,
        "t_plus": 2,
        "min_volume_krw": 5e8,
        "st_filter": False,
    },
    "JP": {
        "price_limit": 0.20,
        "t_plus": 2,
        "min_volume_jpy": 1e7,
        "st_filter": False,
    },
}

# ── Logging ────────────────────────────────────────────────────────────
# ── Alpha158 rolling windows ───────────────────────────────────────────
ALPHA158_WINDOWS = [5, 10, 20, 30, 60]

# ── Signal processing ───────────────────────────────────────────────────
ZSCORE_WINDOW = 60
WINSORIZE_LIMITS = (0.01, 0.99)

# ── Walk-Forward Analysis ───────────────────────────────────────────────
WFA_TRAIN_DAYS = 504       # 2 trading years ≈ 504 days
WFA_STEP_DAYS  = 20         # retrain every 20 days

# ── Ensemble / Stacking ─────────────────────────────────────────────────
ENSEMBLE_ENABLED = True     # stacking / single
META_MODEL       = "ridge"  # ridge | linear | none

# ── Backtest ────────────────────────────────────────────────────────────
TOP_K_BACKTEST = [5, 10, 20, 30]  # multiple top-K to compare

# ── News / Social Sentiment ──────────────────────────────────────────────
# Enables the multi-source news & social media sentiment module (news_sentiment)
NEWS_SENTIMENT_ENABLED = True

# Which collection tiers to enable
NEWS_RSS_ENABLED     = True   # RSS feeds (always free)
NEWS_API_ENABLED     = True   # API-based (needs keys; skips if missing)
NEWS_SOCIAL_ENABLED  = True   # Social media via snscrape (needs pip install)

# Cache control
NEWS_CACHE_ENABLED     = True
NEWS_CACHE_MAX_AGE_H   = 6      # hours before re-fetch
NEWS_SENTIMENT_LIMIT   = 200    # max articles per market

# Source priority for sentiment merging (lower = higher priority)
NEWS_SOURCE_WEIGHTS = {
    "alpha_vantage": 1.5,   # Has ticker-level sentiment built-in
    "marketaux":     1.3,
    "newsapi":       1.0,
    "reuters":       1.0,
    "bloomberg":     1.0,
    "cls":           1.0,   # 财联社
    "eastmoney":     1.0,
    "sina_finance":  1.0,
    "xueqiu":        1.0,
    "twitter":       0.6,
    "weibo":         0.5,
    "reddit":        0.5,
    "youtube":       0.4,
}

# ── Logging ────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
