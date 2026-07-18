"""
Multi-Market T+1 Predictor — Model module
==========================================
Capabilities:
  - LightGBM cross-sectional regression with time-series CV (baseline)
  - Optuna hyperparameter search (optional)
  - Factor IC (Information Coefficient) analysis
  - Walk-Forward Analysis (WFA) rolling train
  - Ensemble Stacking (LGB + XGB + Ridge meta-learner)
  - Model persistence with pickle
"""
import logging
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr

from config import (LGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS,
                    RETRAIN_INTERVAL, DATA_CACHE, OPTUNA_N_TRIALS, OPTUNA_ENABLED,
                    WFA_TRAIN_DAYS, WFA_STEP_DAYS, WFA_ENABLED,
                    ENSEMBLE_ENABLED, META_MODEL, XGB_PARAMS)

logger = logging.getLogger(__name__)

MODEL_PATH        = DATA_CACHE / "lgb_model.pkl"
FEATURE_COLS_PATH = DATA_CACHE / "feature_cols.pkl"
IC_HISTORY_PATH   = DATA_CACHE / "ic_history.csv"
BACKTEST_OUTPUT   = DATA_CACHE / "wfa_predictions.csv"

# ── Training data preparation ─────────────────────────────────────────

def prepare_training_data(feature_df: pd.DataFrame):
    """Split feature DataFrame into X, y, feature list."""
    exclude = {"date", "symbol", "label", "name", "industry", "weight",
               "pct_chg", "market", "sentiment_score",
               "sentiment_ma_5", "sentiment_change", "news_count", "sentiment_std_5"}
    feature_cols = [c for c in feature_df.columns
                    if c not in exclude and not c.startswith(("^", "dx-"))]

    df = feature_df.dropna(subset=["label"]).copy()
    if df.empty:
        return None, None, None, feature_cols

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    idx = df[["symbol", "date"]].reset_index(drop=True)

    logger.info("Training data: X %s, y mean=%.6f, std=%.6f",
                X.shape, float(y.mean()), float(y.std()))
    return X, y, idx, feature_cols


# ── Factor IC analysis ─────────────────────────────────────────────────

def compute_factor_ic(predictions: np.ndarray, actuals: np.ndarray,
                      feature_names: list = None,
                      X: np.ndarray = None,
                      top_n: int = 10) -> dict:
    """Compute per-factor IC (Spearman rank correlation with future returns).

    Returns dict with:
      - rank_ic: overall Rank IC
      - ic_per_feature: list of (feature_name, ic_value) sorted by abs(IC)
      - top_features: top-N most important features by |IC|
    """
    if len(predictions) < 10 or len(actuals) < 10:
        return {"rank_ic": 0.0, "ic_per_feature": [], "top_features": []}

    # Overall Rank IC
    rank_ic, p_value = spearmanr(predictions, actuals)
    rank_ic = float(rank_ic) if not np.isnan(rank_ic) else 0.0

    result = {"rank_ic": rank_ic, "p_value": float(p_value)}

    # Per-feature IC (pearson corr between feature value and next-day return)
    ic_list = []
    if X is not None and feature_names is not None:
        for i, fname in enumerate(feature_names):
            if i >= X.shape[1]:
                break
            try:
                ic, _ = spearmanr(X[:, i], actuals)
                ic_val = float(ic) if not np.isnan(ic) else 0.0
                ic_list.append((fname, ic_val))
            except Exception:
                continue

        ic_list.sort(key=lambda x: abs(x[1]), reverse=True)
        result["ic_per_feature"] = ic_list
        result["top_features"] = ic_list[:top_n]
        result["bottom_features"] = ic_list[-top_n:] if len(ic_list) > top_n else []

    logger.info("  Rank IC = %.4f (p=%.4f)", rank_ic, result.get("p_value", 1.0))
    if ic_list:
        logger.info("  Top-5 factors by |IC|: %s",
                    [(n, round(v, 4)) for n, v in ic_list[:5]])

    return result


# ── LGBM cross-section model ───────────────────────────────────────────

def _train_lgb(X_train, y_train, X_val, y_val, params=None, num_round=None, es_round=None):
    """Train a single LightGBM model with early stopping."""
    p = (params or LGB_PARAMS).copy()
    nr = num_round or NUM_BOOST_ROUND
    es = es_round or EARLY_STOPPING_ROUNDS

    tr_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=tr_data)

    model = lgb.train(
        p, tr_data,
        num_boost_round=nr,
        valid_sets=[tr_data, val_data],
        callbacks=[lgb.early_stopping(es), lgb.log_evaluation(0)],
    )
    return model


def _train_xgb(X_train, y_train, X_val, y_val, params=None):
    """Train a single XGBoost model for ensemble."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("XGBoost not installed, skipping XGBoost ensemble")
        return None

    p = (params or XGB_PARAMS).copy()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    model = xgb.train(
        p, dtrain,
        num_boost_round=300,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    return model


# ── Ensemble (Stacking) ────────────────────────────────────────────────

def _build_ensemble(lgb_model, xgb_model, meta_model_type="ridge"):
    """Build a stacking ensemble: LGB + XGB predictions → Ridge meta."""
    if not ENSEMBLE_ENABLED or META_MODEL == "none":
        return None

    class StackingEnsemble:
        """Simple stacking: aggregate LGB + XGB predictions + optionally linear."""

        def __init__(self, lgb_m, xgb_m, meta_type):
            self.lgb_model = lgb_m
            self.xgb_model = xgb_m
            self.meta_type = meta_type
            self.meta_coef = None

        def predict(self, X):
            preds = []
            if self.lgb_model is not None:
                p = self.lgb_model.predict(X, num_iteration=self.lgb_model.best_iteration)
                preds.append(p.reshape(-1, 1))
            if self.xgb_model is not None:
                import xgboost as xgb
                dx = xgb.DMatrix(X)
                p = self.xgb_model.predict(dx)
                preds.append(p.reshape(-1, 1))

            stacked = np.hstack(preds)  # (n, n_models)

            if self.meta_type == "ridge" and stacked.shape[1] > 1:
                from sklearn.linear_model import Ridge
                return (stacked @ self.meta_coef.reshape(-1, 1)).flatten() if self.meta_coef is not None else stacked.mean(axis=1)
            return stacked.mean(axis=1)

        def fit_meta(self, X_meta, y_meta):
            if self.meta_type == "ridge" or self.meta_type == "linear":
                preds = []
                if self.lgb_model is not None:
                    p = self.lgb_model.predict(X_meta, num_iteration=self.lgb_model.best_iteration)
                    preds.append(p.reshape(-1, 1))
                if self.xgb_model is not None:
                    import xgboost as xgb
                    dx = xgb.DMatrix(X_meta)
                    p = self.xgb_model.predict(dx)
                    preds.append(p.reshape(-1, 1))

                stacked = np.hstack(preds)
                if stacked.shape[1] < 2:
                    return

                from sklearn.linear_model import Ridge, LinearRegression
                MetaCls = Ridge if self.meta_type == "ridge" else LinearRegression
                meta = MetaCls()
                meta.fit(stacked, y_meta)
                self.meta_coef = meta.coef_
                logger.info("  Meta-learner coefficients: %s", self.meta_coef)

    return StackingEnsemble(lgb_model, xgb_model, meta_model_type)


# ── Training ───────────────────────────────────────────────────────────

def train_model(X, y, feature_cols=None):
    """Train with time-series CV. Returns (model_or_ensemble, val_mae)."""
    if X is None or X.shape[0] < 500:
        logger.warning("Training data too small (%d rows), skipping", X.shape[0] if X is not None else 0)
        return None, None

    # ── Optuna (optional) ──
    final_lgb_params = LGB_PARAMS.copy()
    if OPTUNA_ENABLED:
        try:
            import optuna
            logger.info("Running Optuna hyperparam search (%d trials)...", OPTUNA_N_TRIALS)
            study = optuna.create_study(direction="maximize", study_name="t1_predictor")
            study.optimize(lambda t: _optuna_objective(t, X, y), n_trials=OPTUNA_N_TRIALS)
            final_lgb_params.update(study.best_params)
            logger.info("Best params: %s", study.best_params)
        except ImportError:
            logger.warning("optuna not installed, using defaults")

    # ── Time-series CV training ──
    tscv = TimeSeriesSplit(n_splits=3)
    models = []   # (model, val_score)
    xgb_models = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # LGB
        lgb_m = _train_lgb(X_tr, y_tr, X_val, y_val, final_lgb_params)
        val_mae = lgb_m.best_score["valid_1"]["l1"]
        models.append((lgb_m, val_mae))
        logger.info("  Fold %d LGB: val MAE = %.6f", fold + 1, val_mae)

        # XGB (if ensemble enabled)
        if ENSEMBLE_ENABLED:
            try:
                xgb_m = _train_xgb(X_tr, y_tr, X_val, y_val)
                if xgb_m is not None:
                    xgb_models.append(xgb_m)
            except Exception as exc:
                logger.debug("XGBoost fold %d failed: %s", fold + 1, exc)

    # Best LGB
    best_idx = int(np.argmin([s for _, s in models]))
    best_lgb = models[best_idx][0]
    val_score = models[best_idx][1]

    # Build ensemble
    final_model = best_lgb
    if ENSEMBLE_ENABLED and xgb_models:
        ensemble = _build_ensemble(best_lgb, xgb_models[best_idx] if len(xgb_models) > best_idx else None, META_MODEL)
        if ensemble is not None:
            # Fit meta on full training as approximation
            ensemble.fit_meta(X, y)
            final_model = ensemble
            logger.info("  Ensemble model built (meta=%s)", META_MODEL)

    # Compute IC
    if hasattr(final_model, 'predict'):
        preds = predict(final_model, X)
    else:
        preds = final_model.predict(X, num_iteration=final_model.best_iteration)

    ic_results = compute_factor_ic(preds, y, feature_cols, X)
    logger.info("  Overall Rank IC = %.4f", ic_results.get("rank_ic", 0.0))

    _save_artifacts(final_model, feature_cols, final_lgb_params, ic_results)
    return final_model, val_score


def _optuna_objective(trial, X, y, cv_splits=3):
    """Optuna hyperparameter search objective."""
    params = {
        "objective":        "regression",
        "metric":           "mae",
        "boosting_type":    "gbdt",
        "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
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
            params, tr_data, num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        scores.append(-model.best_score["valid_0"]["l1"])
    return float(np.mean(scores))


def predict(model, X):
    """Predict returns. Works for LGB model or StackingEnsemble."""
    if hasattr(model, 'predict'):
        return model.predict(X)
    return model.predict(X, num_iteration=model.best_iteration)


# ── Walk-Forward Analysis ─────────────────────────────────────────────

def run_walk_forward(feature_df: pd.DataFrame,
                     feature_cols: list = None) -> tuple:
    """Run walk-forward analysis: rolling train-predict window.

    Args:
        feature_df: Full feature DataFrame with date, label columns.
        feature_cols: Feature column names.

    Returns:
        (predictions_df, final_model)
        predictions_df has columns: date, symbol, pred_ret, label
        final_model is trained on the latest window.
    """
    if feature_cols is None:
        exclude = {"date", "symbol", "label", "name", "industry", "market"}
        feature_cols = [c for c in feature_df.columns if c not in exclude
                        and not c.startswith(("^", "dx-"))]

    df = feature_df.dropna(subset=["label"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    dates = sorted(df["date"].unique())
    if len(dates) < WFA_TRAIN_DAYS + WFA_STEP_DAYS:
        logger.warning("Not enough dates for WFA (%d < %d)", len(dates), WFA_TRAIN_DAYS + WFA_STEP_DAYS)
        return pd.DataFrame(), None

    logger.info("WFA: %d unique dates, train=%dd, step=%dd",
                len(dates), WFA_TRAIN_DAYS, WFA_STEP_DAYS)

    all_preds = []
    final_model = None

    for start_idx in range(0, len(dates) - WFA_TRAIN_DAYS, WFA_STEP_DAYS):
        train_end   = start_idx + WFA_TRAIN_DAYS
        train_dates = dates[start_idx:train_end]
        pred_dates  = dates[train_end:train_end + WFA_STEP_DAYS]
        if not pred_dates:
            break

        train_df = df[df["date"].isin(train_dates)].copy()
        pred_df  = df[df["date"].isin(pred_dates)].copy()

        X_train = train_df[feature_cols].fillna(0).values.astype(np.float32)
        y_train = train_df["label"].values.astype(np.float32)
        X_pred  = pred_df[feature_cols].fillna(0).values.astype(np.float32)

        if len(X_train) < 500:
            continue

        # Train LGB model
        tr_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            LGB_PARAMS, tr_data,
            num_boost_round=NUM_BOOST_ROUND,
            callbacks=[lgb.log_evaluation(0)],
        )
        final_model = model

        # Predict
        preds = model.predict(X_pred, num_iteration=model.best_iteration)
        pred_df = pred_df.copy()
        pred_df["pred_ret"] = preds
        all_preds.append(pred_df[["date", "symbol", "pred_ret", "label"]])

        logger.info("  WFA window [%s .. %s]: %d train → %d pred rows",
                    train_dates[0].strftime("%Y%m%d"),
                    pred_dates[-1].strftime("%Y%m%d"),
                    len(X_train), len(preds))

    if not all_preds:
        return pd.DataFrame(), None

    predictions = pd.concat(all_preds, ignore_index=True)
    logger.info("WFA done: %d total predictions", len(predictions))
    return predictions, final_model


# ── Persistence ───────────────────────────────────────────────────────

def should_retrain(num_days_since_last: int) -> bool:
    return (not MODEL_PATH.exists()) or (num_days_since_last >= RETRAIN_INTERVAL)


def get_model_age_days() -> int:
    if not MODEL_PATH.exists():
        return 9999
    mtime = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime)
    return (datetime.now() - mtime).days


def _save_artifacts(model, feature_cols, params=None, ic_results=None):
    try:
        artifacts = {
            "model": model,
            "feature_cols": feature_cols,
            "params": params,
            "ic_results": ic_results,
            "timestamp": datetime.now().isoformat(),
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(artifacts, f)
        logger.info("Model saved: %s", MODEL_PATH)

        # Save IC history
        if ic_results and ic_results.get("top_features"):
            df = pd.DataFrame(ic_results["top_features"], columns=["feature", "ic"])
            df.to_csv(IC_HISTORY_PATH, index=False)
            logger.info("IC history saved: %s", IC_HISTORY_PATH)

    except Exception as exc:
        logger.warning("Model save failed: %s", exc)


def load_model():
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                artifacts = pickle.load(f)
            if isinstance(artifacts, dict):
                return artifacts["model"], artifacts["feature_cols"]
            return artifacts, None
        except Exception as exc:
            logger.warning("Model load failed: %s", exc)
    return None, None
