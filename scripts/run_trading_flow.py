#!/usr/bin/env python3
"""
Trading Flow Orchestrator — 时间窗口调度
=========================================
统一管理 A+B 方案的全部定时任务：
  - ✅ 方案A — 7×24 情感轮询（调用 news_timeline.py）
  - ✅ 方案B — 开盘前/后额外计算

时间窗口（CST / 交易日）:
  时间（CST） | 动作
  ------------|------------------------------------------------
  07:00       | 美股收盘数据 + 隔夜新闻情感 → 预判A股开盘情绪
  09:25       | A股集合竞价数据 + 盘前综合情感 → 当日修正
  13:30       | 完整T+1预测 + 24h情感曲线 → 14:00买入信号
  09:55       | 隔夜新闻情感检查 → 10:00卖出决策

使用:
  # 自动检测时间窗口运行:
  python scripts/run_trading_flow.py

  # 手动指定窗口:
  python scripts/run_trading_flow.py --slot 1330

  # 查看最近状态:
  python scripts/run_trading_flow.py --status
"""
import sys
import os
import json
import logging
import argparse
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
OUTPUTS      = PROJECT_ROOT / "outputs"
LOGS         = PROJECT_ROOT / "logs"
DATA_DIR     = PROJECT_ROOT / "data"

# Ensure scripts dir is on path for module imports
sys.path.insert(0, str(SCRIPTS_DIR))

OUTPUTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

TRADING_JOURNAL = OUTPUTS / "trading_journal.json"
SELL_SIGNAL_FILE = OUTPUTS / "sell_signal.json"

logger = logging.getLogger("trading_flow")


def setup_logging():
    """Configure logging """
    log_path = LOGS / f"trading_flow_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════
# Time Utilities (CST = UTC+8)
# ═══════════════════════════════════════════════════════════════════════

def now_cst() -> datetime:
    """Current time in China Standard Time."""
    return datetime.now(timezone.utc) + timedelta(hours=8)


def detect_slot() -> str:
    """
    Detect which time slot we're in (CST).
    Returns: "0700" | "0925" | "1330" | "0955" | "idle"
    """
    now = now_cst()
    cst_hour = now.hour
    cst_min = now.minute

    # Convert to minutes since midnight for range checking
    cst_total_min = cst_hour * 60 + cst_min

    # Slot boundaries (±15 min tolerance)
    slots = {
        "0700": (7 * 60, 7 * 60 + 15),      # 07:00-07:15
        "0925": (9 * 60 + 25, 9 * 60 + 40),  # 09:25-09:40
        "1330": (13 * 60 + 30, 13 * 60 + 45),  # 13:30-13:45
        "0955": (9 * 60 + 55, 10 * 60 + 5),   # 09:55-10:05
    }

    for slot_id, (start, end) in slots.items():
        if start <= cst_total_min <= end:
            return slot_id

    return "idle"


def is_trading_day() -> bool:
    """Check if today is a trading day for CN market."""
    try:
        from data_fetcher import is_trading_day
        return is_trading_day("CN")
    except Exception:
        # Fallback: weekday check
        return now_cst().weekday() < 5


# ═══════════════════════════════════════════════════════════════════════
# Sentiment Timeline Access
# ═══════════════════════════════════════════════════════════════════════

def load_sentiment_curve(hours: int = 24) -> pd.DataFrame:
    """Load sentiment timeline from parquet."""
    timeline_file = DATA_DIR / "news_timeline" / "sentiment_series.parquet"
    if not timeline_file.exists():
        logger.info("No sentiment timeline data yet")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(timeline_file)
        if df.empty:
            return df
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        df = df[pd.to_datetime(df["timestamp"]) >= cutoff].sort_values("timestamp")
        return df
    except Exception as e:
        logger.warning("Failed to load sentiment timeline: %s", e)
        return pd.DataFrame()


def compute_sentiment_signals(market: str = "CN", hours: int = 24) -> dict:
    """Compute actionable sentiment signals for a market."""
    curve = load_sentiment_curve(hours)
    if curve.empty:
        return {"signal": "no_data", "confidence": 0}

    mdf = curve[curve["market"] == market].copy() if "market" in curve.columns else curve
    if mdf.empty:
        return {"signal": "no_data", "market": market, "confidence": 0}

    scores = mdf["sentiment_score"].values
    if len(scores) < 3:
        return {"signal": "insufficient_data", "snapshots": len(scores)}

    current = float(scores[-1])
    recent_avg = float(np.mean(scores[-3:]))
    trend_latest = scores[-1] - scores[-2] if len(scores) >= 2 else 0
    trend_24h = scores[-1] - scores[0]
    volatility = float(np.std(scores))

    # Sentiment signal logic
    if current > 0.15 and trend_latest >= 0:
        signal = "bullish"
        confidence = min(1.0, 0.5 + abs(current) + abs(trend_24h) / 0.3)
    elif current < -0.15 and trend_latest <= 0:
        signal = "bearish"
        confidence = min(1.0, 0.5 + abs(current) + abs(trend_24h) / 0.3)
    elif trend_24h > 0.1 and current > -0.05:
        signal = "improving"
        confidence = 0.5 + abs(trend_24h) / 0.4
    elif trend_24h < -0.1 and current < 0.05:
        signal = "deteriorating"
        confidence = 0.5 + abs(trend_24h) / 0.4
    else:
        signal = "neutral"
        confidence = 0.3

    confidence = min(1.0, max(0.0, confidence))

    return {
        "market": market,
        "signal": signal,
        "confidence": round(confidence, 4),
        "current_score": round(current, 4),
        "recent_avg": round(recent_avg, 4),
        "trend_latest": round(float(trend_latest), 4),
        "trend_24h": round(float(trend_24h), 4),
        "volatility": round(volatility, 4),
        "snapshots": len(mdf),
        "period_hours": hours,
    }


# ═══════════════════════════════════════════════════════════════════════
# Slot: 07:00 — 美股收盘 + 隔夜情感
# ═══════════════════════════════════════════════════════════════════════

def slot_0700_preopen():
    """07:00 CST: 美股收盘数据 + 隔夜新闻情感 → 预判A股开盘情绪."""
    logger.info("=== Slot 07:00 — 隔夜情感分析 ===")

    # 1. Load CN sentiment curve from last 12 hours (overnight)
    cn_signal = compute_sentiment_signals("CN", hours=12)
    us_signal = compute_sentiment_signals("US", hours=12)

    # 2. Combined judgment
    combined_score = (
        cn_signal.get("current_score", 0) * 0.6 +
        us_signal.get("current_score", 0) * 0.4
    )

    overnight_verdict = {
        "slot": "0700",
        "timestamp": now_cst().isoformat(),
        "cn_sentiment": cn_signal,
        "us_overnight": us_signal,
        "combined_score": round(combined_score, 4),
        "preopen_assessment": (
            "📈 积极偏多" if combined_score > 0.1 else
            "📉 消极偏空" if combined_score < -0.1 else
            "➖ 中性开市"
        ),
        "risk_level": "low" if abs(combined_score) < 0.1 else
                      "medium" if abs(combined_score) < 0.3 else "high",
    }

    # Save to journal
    _append_to_journal(overnight_verdict)
    logger.info("隔夜情绪综合: %.4f → %s", combined_score, overnight_verdict["preopen_assessment"])
    logger.info("  CN: %s (%.4f) | US: %s (%.4f)",
                cn_signal.get("signal", "?"), cn_signal.get("current_score", 0),
                us_signal.get("signal", "?"), us_signal.get("current_score", 0))
    return overnight_verdict


# ═══════════════════════════════════════════════════════════════════════
# Slot: 09:25 — 集合竞价 + 盘前修正
# ═══════════════════════════════════════════════════════════════════════

def slot_0925_morning_correction():
    """09:25 CST: 盘前综合情感 → 当日修正."""
    logger.info("=== Slot 09:25 — 盘前情感修正 ===")

    cn_signal = compute_sentiment_signals("CN", hours=24)

    # Weight recent hours more heavily
    recent_4h = compute_sentiment_signals("CN", hours=4)

    # Blended signal
    blended = (
        cn_signal.get("recent_avg", 0) * 0.4 +
        recent_4h.get("recent_avg", 0) * 0.3 +
        cn_signal.get("trend_24h", 0) * 0.3
    )

    verdict = {
        "slot": "0925",
        "timestamp": now_cst().isoformat(),
        "signal_24h": cn_signal,
        "signal_4h": recent_4h,
        "blended_score": round(blended, 4),
        "morning_verdict": (
            "🟢 积极开盘 — 关注追涨风险" if blended > 0.15 else
            "🔴 消极开盘 — 警惕低开" if blended < -0.15 else
            "🟡 中性开盘 — 等待趋势确认"
        ),
        "action": "proceed" if blended > -0.2 else "caution",
    }

    _append_to_journal(verdict)
    logger.info("盘前综合: %.4f → %s", blended, verdict["morning_verdict"])
    return verdict


# ═══════════════════════════════════════════════════════════════════════
# Slot: 13:30 — 完整T+1预测 → 14:00买入
# ═══════════════════════════════════════════════════════════════════════

def slot_1330_full_predict():
    """13:30 CST: 完整预测流水线 + 情感特征注入 → 生成买入信号."""
    logger.info("=== Slot 13:30 — 完整T+1预测 ===")

    # 1. Call the existing predictor
    predict_script = SCRIPTS_DIR / "daily_multi_predict.py"
    if not predict_script.exists():
        logger.error("Predictor script not found: %s", predict_script)
        return None

    logger.info("Starting full prediction pipeline...")
    result = subprocess.run(
        [sys.executable, str(predict_script)],
        capture_output=True, text=True, timeout=600,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    if result.returncode != 0:
        logger.error("Prediction failed (rc=%d):\n%s", result.returncode, result.stderr)
        # Try fallback
        logger.info("Attempting fallback: running on CN only")
        result = subprocess.run(
            [sys.executable, str(predict_script)],
            capture_output=True, text=True, timeout=600,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if result.returncode != 0:
            logger.error("Fallback also failed")
            return None

    # 2. Read the latest predictions
    today = now_cst().strftime("%Y%m%d")
    pred_files = list(OUTPUTS.glob(f"multi_t1_pred_{today}.csv"))
    pred_files.sort(reverse=True)

    if not pred_files:
        logger.warning("No prediction file found for today")
        return None

    latest_pred = pred_files[0]
    logger.info("Predictions file: %s", latest_pred)

    try:
        pred_df = pd.read_csv(latest_pred)
    except Exception as e:
        logger.error("Failed to read predictions: %s", e)
        return None

    # 3. Load sentiment signals as override/check
    cn_signal = compute_sentiment_signals("CN", hours=24)
    sentiment_risk_level = cn_signal.get("signal", "neutral")

    # 4. Build buy recommendation
    top_picks = pred_df.head(10).to_dict(orient="records")

    # Sentiment-adjusted picks
    adjusted_picks = []
    for pick in top_picks:
        entry = {
            "symbol": pick.get("symbol", ""),
            "rank": pick.get("rank", 0),
            "pred_ret": pick.get("pred_ret", 0),
            "sentiment_signal": sentiment_risk_level,
        }
        adjusted_picks.append(entry)

    buy_signal = {
        "slot": "1330",
        "timestamp": now_cst().isoformat(),
        "prediction_file": str(latest_pred),
        "total_picks": len(top_picks),
        "top_picks": adjusted_picks[:5],  # Top 5
        "market_sentiment": cn_signal,
        "sentiment_verdict": (
            "✅ 情感配合，按计划买入" if cn_signal.get("current_score", 0) > -0.2 else
            "⚠️ 情感偏空，减仓或观望"
        ),
        "action": "buy" if cn_signal.get("current_score", 0) > -0.2 else "buy_reduced",
    }

    _append_to_journal(buy_signal)
    logger.info("T+1预测完成: %d picks, top=%.4f, sentiment=%s",
                len(top_picks),
                top_picks[0].get("pred_ret", 0) if top_picks else 0,
                sentiment_risk_level)

    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 14:00 买入信号")
    print(f"{'='*60}")
    print(f"  预测文件: {latest_pred}")
    print(f"  市场情绪: {cn_signal.get('signal', '?')} ({cn_signal.get('current_score', 0):.4f})")
    print(f"  操作建议: {buy_signal['sentiment_verdict']}")
    print(f"\n  Top {min(5, len(adjusted_picks))} 推荐:")
    for pick in adjusted_picks[:5]:
        print(f"    #{pick['rank']} {pick['symbol']:12s}  pred_ret={pick['pred_ret']:+.4f}")
    print(f"{'='*60}\n")

    return buy_signal


# ═══════════════════════════════════════════════════════════════════════
# Slot: 09:55 — 卖出决策
# ═══════════════════════════════════════════════════════════════════════

def slot_0955_sell_decision():
    """09:55 CST: 检查隔夜新闻 → 决定10:00是否卖出."""
    logger.info("=== Slot 09:55 — 卖出决策 ===")

    cn_signal = compute_sentiment_signals("CN", hours=12)  # overnight
    us_signal = compute_sentiment_signals("US", hours=12)

    # Load yesterday's buys from journal
    yesterday_buys = _load_yesterday_buys()

    # Decision logic
    sentiment_ok = cn_signal.get("current_score", 0) > -0.2
    us_stable = us_signal.get("signal", "neutral") not in ("bearish", "deteriorating")
    volatility_ok = cn_signal.get("volatility", 0) < 0.3

    should_sell = sentiment_ok and us_stable and volatility_ok

    sell_decision = {
        "slot": "0955",
        "timestamp": now_cst().isoformat(),
        "cn_overnight": cn_signal,
        "us_overnight": us_signal,
        "yesterday_holdings": yesterday_buys,
        "checks": {
            "sentiment_ok": sentiment_ok,
            "us_stable": us_stable,
            "volatility_ok": volatility_ok,
        },
        "decision": (
            "SELL" if should_sell else "HOLD_CAUTION"
        ),
        "message": (
            "✅ 按计划10:00卖出" if should_sell else
            "⚠️ 隔夜有异动，建议谨慎/分批卖出"
        ),
    }

    # Write a dedicated sell signal file
    try:
        SELL_SIGNAL_FILE.write_text(json.dumps(sell_decision, indent=2, ensure_ascii=False))
        logger.info("卖出信号写入: %s", SELL_SIGNAL_FILE)
    except Exception as e:
        logger.warning("Failed to write sell signal: %s", e)

    _append_to_journal(sell_decision)

    print(f"\n{'='*60}")
    print(f"📊 10:00 卖出决策")
    print(f"{'='*60}")
    print(f"  CN情感: {cn_signal.get('signal', '?')} ({cn_signal.get('current_score', 0):.4f})")
    print(f"  US隔夜: {us_signal.get('signal', '?')} ({us_signal.get('current_score', 0):.4f})")
    print(f"  波动率: {cn_signal.get('volatility', 0):.4f} {'✅' if volatility_ok else '⚠️'}")
    print(f"  操作建议: {sell_decision['message']}")
    print(f"{'='*60}\n")

    return sell_decision


# ═══════════════════════════════════════════════════════════════════════
# Journal
# ═══════════════════════════════════════════════════════════════════════

def _append_to_journal(entry: dict):
    """Append an entry to the trading journal JSON."""
    journal = {}
    if TRADING_JOURNAL.exists():
        try:
            journal = json.loads(TRADING_JOURNAL.read_text())
        except (json.JSONDecodeError, Exception):
            journal = {}

    today = now_cst().strftime("%Y-%m-%d")
    if today not in journal:
        journal[today] = []

    journal[today].append(entry)

    try:
        TRADING_JOURNAL.write_text(json.dumps(journal, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.warning("Failed to save journal: %s", e)


def _load_journal() -> dict:
    """Load the trading journal."""
    if TRADING_JOURNAL.exists():
        try:
            return json.loads(TRADING_JOURNAL.read_text())
        except Exception as e:
            logger.warning("Failed to load journal: %s", e)
    return {}


def _load_yesterday_buys() -> list:
    """Load yesterday's buy decisions from the journal."""
    journal = _load_journal()
    yesterday = (now_cst() - timedelta(days=1)).strftime("%Y-%m-%d")
    day_before = (now_cst() - timedelta(days=2)).strftime("%Y-%m-%d")

    for day in [yesterday, day_before]:
        if day in journal:
            entries = journal[day]
            for entry in entries:
                if entry.get("slot") == "1330" and "top_picks" in entry:
                    return entry["top_picks"]
    return []


def show_status():
    """Display current flow status."""
    journal = _load_journal()
    today = now_cst().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"📊 交易流程状态 — {today}")
    print(f"{'='*60}")

    # Today's slots
    today_entries = journal.get(today, [])
    completed_slots = {e["slot"] for e in today_entries}
    expected_slots = {"0700", "0925", "1330", "0955"}

    for slot in ["0700", "0925", "1330", "0955"]:
        slot_names = {
            "0700": "🌙 07:00 隔夜情感",
            "0925": "🌅 09:25 盘前修正",
            "1330": "📈 13:30 T+1预测",
            "0955": "💸 09:55 卖出决策",
        }
        status = (
            f"✅ {slot_names.get(slot, slot)}" if slot in completed_slots else
            "⬜ 未执行" if slot not in expected_slots or now_cst().hour >= int(slot) // 100 else
            "⏳ 等待窗口"
        )
        print(f"  {status}")

    # Latest sentiment
    print(f"\n  📡 情感状态:")
    for market in ["CN", "US"]:
        sig = compute_sentiment_signals(market, hours=6)
        print(f"    {market}: {sig.get('signal', '?')} ({sig.get('current_score', 0):+.4f}) "
              f"conf={sig.get('confidence', 0):.2f}")

    # Sell signal checkpoint
    sell_signal = {}
    if SELL_SIGNAL_FILE.exists():
        try:
            sell_signal = json.loads(SELL_SIGNAL_FILE.read_text())
        except Exception:
            pass
    if sell_signal:
        print(f"\n  💰 最新卖出信号: {sell_signal.get('message', 'N/A')}")

    print(f"\n  昨日持仓: {_load_yesterday_buys()}")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════
# Main Dispatcher
# ═══════════════════════════════════════════════════════════════════════

def run_flow(slot: Optional[str] = None):
    """Run the trading flow for the specified slot or auto-detect."""
    setup_logging()

    if not is_trading_day():
        logger.info("Not a trading day — skipping")
        print("⏸️ 今日非交易日，跳过")
        return

    if slot is None:
        slot = detect_slot()
        if slot == "idle":
            # Also run emotion poll if it's time
            logger.info("No active trading slot at this time")
            print(f"⏰ 当前时间不在交易窗口内 (CST {now_cst().strftime('%H:%M')})")
            print("  预期窗口: 07:00-07:15, 09:25-09:40, 09:55-10:05, 13:30-13:45")
            return

    slot_map = {
        "0700": ("🌙 隔夜情感分析", slot_0700_preopen),
        "0925": ("🌅 盘前情感修正", slot_0925_morning_correction),
        "1330": ("📈 T+1预测 (→14:00买入)", slot_1330_full_predict),
        "0955": ("💸 卖出决策 (→10:00卖出)", slot_0955_sell_decision),
    }

    if slot not in slot_map:
        logger.warning("Unknown slot: %s", slot)
        print(f"⚠️ 未知时间窗口: {slot}")
        return

    name, func = slot_map[slot]
    logger.info("Running slot %s: %s", slot, name)

    print(f"\n{'='*60}")
    print(f"🎯 {slot} — {name}")
    print(f"{'='*60}")

    result = func()

    if result:
        logger.info("Slot %s complete: %s", slot, result.get("action", "done"))
    else:
        logger.warning("Slot %s returned no result", slot)


def main():
    parser = argparse.ArgumentParser(description="A+B Trading Flow Supervisor")
    parser.add_argument("--slot", "-s", choices=["0700", "0925", "1330", "0955"],
                        help="Force a specific time slot")
    parser.add_argument("--status", "-st", action="store_true",
                        help="Show current flow status")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Skip non-trading-day check")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.force:
        # Override trading day check
        setup_logging()
        slot_map = {
            "0700": ("🌙 隔夜情感分析", slot_0700_preopen),
            "0925": ("🌅 盘前情感修正", slot_0925_morning_correction),
            "1330": ("📈 T+1预测 (→14:00买入)", slot_1330_full_predict),
            "0955": ("💸 卖出决策 (→10:00卖出)", slot_0955_sell_decision),
        }
        slot = args.slot or "1330"
        if slot not in slot_map:
            print(f"⚠️ 未知窗口: {slot}")
            return
        name, func = slot_map[slot]
        print(f"\n{'='*60}")
        print(f"🎯 [强制] {slot} — {name}")
        print(f"{'='*60}")
        result = func()
        print(f"\n✅ 完成")
        return

    run_flow(args.slot)


if __name__ == "__main__":
    main()
