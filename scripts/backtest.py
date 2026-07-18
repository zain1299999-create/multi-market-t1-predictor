"""
Multi-Market T+1 Predictor — Backtest Module
=============================================
Lightweight alpha validation without transaction costs.

Functions:
  - quintile_analysis():    Group by prediction score → Q1-Q5 → compute avg returns
  - long_short_spread():    Q1 (long) vs Q5 (short) return difference
  - compute_sharpe():       Annualized Sharpe ratio
  - cumulative_returns():   Cumulative return curve CSV
  - run_backtest():         Full backtest: quintiles + Sharpe + cumulative curve
  - backtest_top_k():       Compare multiple top-K thresholds
"""
import numpy as np
import pandas as pd
import logging

from config import BACKTEST_GROUPS, BACKTEST_TOP_K

logger = logging.getLogger(__name__)


def quintile_analysis(predictions: pd.DataFrame,
                      n_groups: int = BACKTEST_GROUPS) -> pd.DataFrame:
    """Group predictions into N quantiles and compute next-day returns.

    Args:
        predictions: DataFrame with columns [date, symbol, pred_ret, label]
                     label = actual next-day return
        n_groups: Number of quantile groups (default 5 = Q1-Q5)

    Returns:
        DataFrame with columns: group, avg_pred, avg_ret, count, hit_rate
        hit_rate = fraction of positive returns in the group
    """
    if predictions.empty or len(predictions) < n_groups * 2:
        logger.warning("Not enough predictions for quintile analysis (%d)", len(predictions))
        return pd.DataFrame()

    df = predictions.copy()
    df["group"] = pd.qcut(df["pred_ret"].rank(method="first"),
                          q=n_groups,
                          labels=[f"Q{i+1}" for i in range(n_groups)])

    result = df.groupby("group", observed=True).agg(
        avg_pred=("pred_ret", "mean"),
        avg_ret=("label", "mean"),
        std_ret=("label", "std"),
        count=("label", "count"),
        hit_rate=("label", lambda x: (x > 0).mean()),
    ).reset_index()

    result = result.sort_values("group")
    logger.info("Quintile analysis:")
    for _, row in result.iterrows():
        logger.info("  %s: pred=%.4f  ret=%.4f  hit=%.2f%%  n=%d",
                    row["group"], row["avg_pred"], row["avg_ret"],
                    row["hit_rate"] * 100, row["count"])

    return result


def long_short_spread(quintiles: pd.DataFrame) -> dict:
    """Compute long-short spread (Q1 - Q5) and t-stat.

    Args:
        quintiles: Output from quintile_analysis()

    Returns:
        dict with: spread, long_ret, short_ret, sharpe (if std available)
    """
    if quintiles.empty or len(quintiles) < 2:
        return {"spread": 0.0, "long_ret": 0.0, "short_ret": 0.0}

    q1 = quintiles[quintiles["group"] == "Q1"]
    q5 = quintiles[quintiles["group"] == f"Q{BACKTEST_GROUPS}"]

    if q1.empty or q5.empty:
        return {"spread": 0.0, "long_ret": 0.0, "short_ret": 0.0}

    long_ret  = float(q1["avg_ret"].iloc[0])
    short_ret = float(q5["avg_ret"].iloc[0])
    spread    = long_ret - short_ret

    result = {
        "spread": round(spread, 6),
        "long_ret": round(long_ret, 6),
        "short_ret": round(short_ret, 6),
    }

    if "std_ret" in quintiles.columns:
        # Approximate pooled std
        s1 = float(q1["std_ret"].iloc[0])
        s5 = float(q5["std_ret"].iloc[0])
        n1 = int(q1["count"].iloc[0])
        n5 = int(q5["count"].iloc[0])
        pooled_std = np.sqrt((s1**2 / n1) + (s5**2 / n5))
        if pooled_std > 1e-10:
            result["t_stat"] = round(spread / pooled_std, 4)

    logger.info("Long-Short spread: %.4f (Q1=%.4f, Q%d=%.4f)",
                spread, long_ret, BACKTEST_GROUPS, short_ret)
    return result


def compute_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio (assuming 252 trading days)."""
    if len(returns) < 5:
        return 0.0
    excess = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess / std)


def cumulative_returns(predictions: pd.DataFrame,
                       top_k: int = 10) -> pd.DataFrame:
    """Simulate daily equal-weighted top-K portfolio cumulative returns.

    Each day: pick top-K stocks by pred_ret, hold for 1 day, equal weight.
    Output: daily portfolio value (starting at 1.0).

    Args:
        predictions: DataFrame with [date, symbol, pred_ret, label]
        top_k: Number of stocks to pick each day

    Returns:
        DataFrame with columns: date, ret (daily), cum_ret (cumulative)
    """
    if predictions.empty:
        return pd.DataFrame()

    df = predictions.copy().sort_values(["date", "pred_ret"], ascending=[True, False])
    daily = df.groupby("date").head(top_k).groupby("date").agg(
        ret=("label", "mean"),
        n=("label", "count"),
    ).reset_index()
    daily = daily[daily["n"] >= top_k // 2].sort_values("date")

    daily["cum_ret"] = (1.0 + daily["ret"]).cumprod()
    daily["cum_ret"] = daily["cum_ret"].fillna(1.0)

    sharpe = compute_sharpe(daily["ret"].values)
    logger.info("Top-%d backtest: Sharpe=%.2f, final_cum=%.4f",
                top_k, sharpe, float(daily["cum_ret"].iloc[-1]) if not daily.empty else 1.0)

    return daily


def run_backtest(predictions: pd.DataFrame,
                 top_k_list: list = None) -> dict:
    """Full backtest suite.

    Args:
        predictions: DataFrame with [date, symbol, pred_ret, label] columns
        top_k_list: List of K values to test

    Returns:
        dict with keys: quintiles, long_short, top_k_results, cumulative_curves
    """
    if predictions.empty or "label" not in predictions.columns:
        logger.warning("No valid predictions for backtest")
        return {}

    top_k_list = top_k_list or BACKTEST_TOP_K
    results = {}

    # Quintile analysis
    quintiles = quintile_analysis(predictions)
    results["quintiles"] = quintiles

    # Long-short
    results["long_short"] = long_short_spread(quintiles)

    # Top-K backtests
    tk_results = {}
    curves = {}
    for k in top_k_list:
        cum = cumulative_returns(predictions, top_k=k)
        tk_results[k] = {
            "sharpe": compute_sharpe(cum["ret"].values) if not cum.empty else 0.0,
            "final_cum": float(cum["cum_ret"].iloc[-1]) if not cum.empty else 1.0,
            "n_days": len(cum),
            "avg_daily_ret": float(cum["ret"].mean()) if not cum.empty else 0.0,
        }
        curves[k] = cum

    results["top_k_results"] = tk_results
    results["cumulative_curves"] = curves

    logger.info("Backtest summary:")
    for k, v in tk_results.items():
        logger.info("  Top-%d: Sharpe=%.2f  Final=%.4f  Days=%d  AvgDaily=%.4f%%",
                    k, v["sharpe"], v["final_cum"], v["n_days"], v["avg_daily_ret"] * 100)

    return results


def backtest_top_k(predictions: pd.DataFrame,
                   top_k_list: list = None) -> pd.DataFrame:
    """Compare multiple top-K thresholds in a single table.

    Returns DataFrame: top_k, sharpe, final_cum, n_days, avg_daily_ret
    """
    results = run_backtest(predictions, top_k_list)
    tk = results.get("top_k_results", {})
    rows = []
    for k, v in tk.items():
        rows.append({
            "top_k": k,
            "sharpe": v["sharpe"],
            "final_cum": v["final_cum"],
            "days": v["n_days"],
            "avg_daily_ret_pct": v["avg_daily_ret"] * 100,
        })
    return pd.DataFrame(rows).sort_values("top_k") if rows else pd.DataFrame()
