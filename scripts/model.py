"""
Multi-Market T+1 Predictor — Model module
LightGBM cross-sectional regression with time-series CV
Merged from:
  - a-share project: proven cross-sectional approach, model persistence
  - multi-market project: Optuna hyperparameter tuning, classification/regression support
"""
import logging
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from config import (LGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS,
                    RETRAIN_INTERVAL, DATA_CACHE, OPTUNA_N_TRIALS, OPTUNA_ENABLED)

logger = logging.getLogger(__name__)

MODEL_PATH       = DATA_CACHE / "lgb_model.pkl"
FEATURE_COLS_PATH = DATA_CACHE / "feature_cols.pkl"


def prepare_training_data(feature_df: pd.DataFrame):
    """Split feature DataFrame into X, y, feature list."""
    exclude = {"date", "symbol", "label", "name", "industry", "weight",
               "pct_chg", "market", "high_20", "low_20", "sentiment_score",
               "sentiment_ma_5", "sentiment_change", "news_count"}
    feature_cols = [c for c in feature_df.columns
                    if c not in exclude and not c.startswith(("^", "dx-"))]

    df = feature_df.dropna(subset=["label"]).copy()
    # Filter rows where rolling features have settled
    if "ma_20" in df.columns:
        df = df[df["ma_20"].notna()].copy()

    if df.empty:
        return None, None, None, feature_cols

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    idx = df[["symbol", "date"]].reset_index(drop=True)

    logger.info("Training data: X %s, y mean=%.6f, std=%.6f",
                X.shape, float(y.mean()), float(y.std()))
    return X, y, idx, feature_cols


def _optuna_objective(trial, X, y, cv_splits=3):
    """Optuna hyperparameter search objective."""
    params = {
        "objective":        "regression",
        "metric":           "mae",
        "boosting_type":    "gbdt",
        "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq":     trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1":        trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2":        trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "verbose":          -1,
        "random_state":     42,
    }

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        tr_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=tr_data)

        model = lgb.train(
            params, tr_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        scores.append(-model.best_score["valid_0"]["l1"])  # negative MAE
    return float(np.mean(scores))


def train_model(X, y, feature_cols=None):
    """Train a LightGBM regression model with TimeSeriesSplit CV.

    If OPTUNA_ENABLED, runs Optuna before final training.
    """
    if X is None or X.shape[0] < 500:
        logger.warning("Training data too small (%d rows), skipping",
                       X.shape[0] if X is not None else 0)
        return None, None

    # ── Optuna tuning (optional) ──
    final_params = LGB_PARAMS.copy()
    if OPTUNA_ENABLED:
        try:
            import optuna
            logger.info("Running Optuna hyperparameter search (%d trials)...",
                        OPTUNA_N_TRIALS)
            study = optuna.create_study(direction="maximize",
                                        study_name="t1_predictor")
            study.optimize(lambda t: _optuna_objective(t, X, y),
                           n_trials=OPTUNA_N_TRIALS)
            final_params.update({
                k: v for k, v in study.best_params.items()
                if k in final_params or k not in LGB_PARAMS
            })
            logger.info("Best params: %s", study.best_params)
        except ImportError:
            logger.warning("optuna not installed, using default params")

    # ── Time-series CV training ──
    tscv = TimeSeriesSplit(n_splits=3)
    val_scores = []
    best_models = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        tr_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=tr_data)

        model = lgb.train(
            final_params, tr_data,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[tr_data, val_data],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS),
                       lgb.log_evaluation(0)],
        )
        val_mae = model.best_score["valid_1"]["l1"]
        val_scores.append(val_mae)
        best_models.append(model)
        logger.info("  Fold %d: val MAE = %.6f", fold + 1, val_mae)

    # Best fold
    best_idx = int(np.argmin(val_scores))
    model = best_models[best_idx]
    logger.info("Best model (fold %d): val MAE = %.6f",
                best_idx + 1, val_scores[best_idx])

    if feature_cols:
        _save_artifacts(model, feature_cols, final_params)
    return model, val_scores[best_idx]


def predict(model, X):
    """Predict returns for feature matrix."""
    return model.predict(X, num_iteration=model.best_iteration)


def should_retrain(num_days_since_last: int) -> bool:
    if not MODEL_PATH.exists():
        return True
    return num_days_since_last >= RETRAIN_INTERVAL


def get_model_age_days() -> int:
    if not MODEL_PATH.exists():
        return 9999
    mtime = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime)
    return (datetime.now() - mtime).days


def _save_artifacts(model, feature_cols, params=None):
    try:
        artifacts = {"model": model, "feature_cols": feature_cols, "params": params}
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(artifacts, f)
        logger.info("Model saved: %s", MODEL_PATH)
    except Exception as exc:
        logger.warning("Model save failed: %s", exc)


def load_model():
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                artifacts = pickle.load(f)
            if isinstance(artifacts, dict):
                return artifacts["model"], artifacts["feature_cols"]
            # Legacy: plain model
            return artifacts, None
        except Exception as exc:
            logger.warning("Model load failed: %s", exc)
    return None, None
