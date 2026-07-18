"""
Multi-Market T+1 Predictor — Unified Configuration
Merged from: a-share-t1-predictor + multi_market_t1_predictor
"""
import os
from pathlib import Path
from typing import Optional

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
# "hs300" | "zz500" | "all_a" | "multi"
UNIVERSE = "multi"

# When UNIVERSE = "multi", which markets to include
ACTIVE_MARKETS = ["CN", "US"]  # CN, US, KR, JP

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

# API Keys (from env or config)
ALPHA_VANTAGE_KEY: Optional[str] = os.getenv("ALPHA_VANTAGE_KEY")
MARKETAUX_KEY: Optional[str]     = os.getenv("MARKETAUX_KEY")

# ── Cross-market symbols ───────────────────────────────────────────────
MACRO_SYMBOLS = [
    "^VIX",           # US VIX
    "USD/KRW=X",      # USD/KRW
    "USD/JPY=X",      # USD/JPY
    "USDCNY=X",       # USD/CNY
    "^GSPC",          # S&P 500
    "^N225",          # Nikkei 225
    "^KS11",          # KOSPI
    "^HSI",           # Hang Seng
    "DX-Y.NYB",       # US Dollar Index
]

# ── Sentiment config ───────────────────────────────────────────────────
SENTIMENT_LOOKBACK_DAYS = 30
SENTIMENT_MAX_NEWS     = 200

# ── Feature engineering ────────────────────────────────────────────────
MA_WINDOWS       = [3, 5, 10, 20, 30, 60]
ROC_WINDOWS      = [5, 10, 20, 60]
STD_WINDOWS      = [5, 10, 20]
VOLUME_MA_WINDOWS = [5, 20]
MOMENTUM_WINDOWS  = [1, 5, 10, 20]

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

# Optuna tuning
OPTUNA_N_TRIALS = 30
OPTUNA_ENABLED  = False  # set True for tuning, False for fast training

# ── Market-specific rules ──────────────────────────────────────────────
MARKET_RULES = {
    "CN": {
        "price_limit": 0.10,     # 10% daily limit
        "t_plus": 1,
        "min_volume_cny": 1e7,   # 10M CNY min turnover
        "st_filter": True,
        "limit_up_filter": True,
    },
    "US": {
        "price_limit": None,     # no daily limit
        "t_plus": 1,             # T+1 settlement
        "min_volume_usd": 1e6,   # 1M USD
        "st_filter": False,
    },
    "KR": {
        "price_limit": 0.30,     # 30% KOSPI/KOSDAQ
        "t_plus": 2,             # T+2 settlement
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
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
