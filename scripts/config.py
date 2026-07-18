"""
Multi-Market T+1 Predictor — Unified Configuration
Merged from: a-share-t1-predictor + multi_market_t1_predictor + Alpha158 upgrade
Optimized: LambdaRank, Optuna auto-tuning, WFA default-on, faster retrain
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
UNIVERSE       = "multi"            # "hs300" | "zz500" | "all_a" | "multi"
ACTIVE_MARKETS: List[str] = ["CN", "US"]  # CN, US, KR, JP

# ── Prediction ─────────────────────────────────────────────────────────
TOP_K            = 10
TRAIN_YEARS      = 2
RETRAIN_INTERVAL = 5                # Retrain every 5 trading days (was 20)

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

# ── Feature windows ───────────────────────────────────────────────────
MA_WINDOWS: List[int]         = [3, 5, 10, 20, 30, 60]
ROC_WINDOWS: List[int]        = [5, 10, 20, 60]
STD_WINDOWS: List[int]        = [5, 10, 20]
VOLUME_MA_WINDOWS: List[int]  = [5, 20]
MOMENTUM_WINDOWS: List[int]   = [1, 5, 10, 20]
ALPHA158_WINDOWS: List[int]   = [5, 10, 20, 30, 60]

# ── Signal processing ──────────────────────────────────────────────────
ZSCORE_WINDOW       = 60
ZSCORE_MIN_PERIODS  = 30
WINSORIZE_LOWER     = 0.01
WINSORIZE_UPPER     = 0.99

# ── Walk-Forward Analysis ──────────────────────────────────────────────
WFA_TRAIN_DAYS = 504           # 2 years ≈ 504 trading days
WFA_STEP_DAYS  = 5             # retrain every 5 days in WFA (was 20)
WFA_ENABLED    = True          # ✅ Enabled by default

# ── Ensemble (Stacking) ────────────────────────────────────────────────
ENSEMBLE_ENABLED = True
META_MODEL       = "ridge"

# ── Model: LightGBM (LambdaRank for ranking optimization) ─────────────
LGB_PARAMS = {
    "objective":        "lambdarank",         # ✅ Rank-optimized (was "regression")
    "metric":           "ndcg",               # ✅ NDCG@K (was "mae")
    "ndcg_eval_at":     [5, 10, 20],          # Evaluate ranking at top-K
    "boosting_type":    "gbdt",
    "num_leaves":       31,
    "learning_rate":    0.05,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.80,
    "bagging_freq":     5,
    "min_sum_hessian_in_leaf": 1e-3,
    "lambda_l1":        0.0,
    "lambda_l2":        0.0,
    "verbose":          -1,
    "num_threads":      2,
    "seed":             42,
}
NUM_BOOST_ROUND         = 1000    # ↑ More rounds with early stopping
EARLY_STOPPING_ROUNDS   = 50      # ↑ Longer patience for NDCG convergence

# ── Optuna hyperparameter search ──────────────────────────────────────
OPTUNA_N_TRIALS = 15              # ✅ Reduced for daily runs
OPTUNA_ENABLED  = True            # ✅ Enabled

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

# ── IC-based feature pruning ───────────────────────────────────────────
IC_PRUNING_THRESHOLD = 0.01       # ✅ Prune features with |IC| < 1%
IC_PRUNING_MIN_FEATURES = 30     # Keep at least this many

# ── Time series CV ─────────────────────────────────────────────────────
TS_CV_SPLITS     = 3             # 3-fold time-series CV
TS_EMBARGO_DAYS  = 5             # ✅ Embargo gap to prevent leakage

# ── Backtest ───────────────────────────────────────────────────────────
BACKTEST_TOP_K: List[int] = [5, 10, 20, 30]
BACKTEST_GROUPS         = 5

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

# ── News / Social Sentiment ──────────────────────────────────────────────
NEWS_SENTIMENT_ENABLED = True
NEWS_RSS_ENABLED       = True
NEWS_API_ENABLED       = True
NEWS_SOCIAL_ENABLED    = True
NEWS_CACHE_ENABLED     = True
NEWS_CACHE_MAX_AGE_H   = 6
NEWS_SENTIMENT_LIMIT   = 200

NEWS_SOURCE_WEIGHTS = {
    "alpha_vantage": 1.5,
    "marketaux":     1.3,
    "newsapi":       1.0,
    "reuters":       1.0,
    "bloomberg":     1.0,
    "cls":           1.0,
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
