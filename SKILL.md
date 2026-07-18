---
name: multi-market-t1-predictor
description: A+T 隔日策略（14:00买入→10:00卖出）+ 7×24新闻情感轮询 + OpenClaw/GitHub Actions 双轨部署
---

# Multi-Market T+1 Predictor v4 — A+B Trading Flow + 7×24 News Sentiment

A 股 T+1 隔日策略，多市场预测 + Alpha158 因子引擎 + **7×24 实时新闻情感轮询** + OpenClaw / GitHub Actions 双轨调度。

## 核心结构

| 模块 | 能力 |
|------|------|
| `scripts/config.py` | 统一配置：市场规则、WFA、Ensemble、Backtest、新闻情感参数 |
| `scripts/data_fetcher.py` | CN:AKShare / US/KR/JP:yfinance / 新闻情感采集 / 宏观数据 |
| `scripts/features.py` | Alpha158 ~130+ 技术因子 + 情感因子 + 跨市场因子 = **~160+ 因子** |
| `scripts/signal_processor.py` | Winsorize → 截面Rank化 → 时序Z-Score → 填充 |
| `scripts/model.py` | LGB + XGBoost Ensemble + Ridge meta + Optuna + WFA + Factor IC |
| `scripts/backtest.py` | Quintile分组 / 多空Spread / Sharpe / 多TopK对比 |
| `scripts/filter.py` | 各市场规则过滤（涨跌停/ST/流动性/行业分散） |
| **`scripts/news_timeline.py`** | **🆕 v4**: 7×24 情感轮询引擎（增量 parquet, 72h 滚动窗口） |
| **`scripts/run_trading_flow.py`** | **🆕 v4**: 交易流程调度器（4个时间窗口 + 状态展示） |
| `scripts/news_sentiment/` | 📰 多源新闻&社交情感采集分析包 |
| `scripts/daily_multi_predict.py` | 主调度入口 → 每日TopK预测 |

## 📰 多源新闻情感模块

| 源类型 | 具体来源 | 认证需求 |
|--------|----------|---------|
| RSS免费（7源） | 36氪/Bloomberg/CNBC/Yahoo Finance/Seeking Alpha/Investing.com | 无 |
| API深度（3源） | Alpha Vantage(ticker级情感)/NewsAPI/MarketAux | API key(可选) |

情感分析: VADER(EN) + TextBlob(EN) + SnowNLP(ZH) + 金融词典(4市场增强)

## 交易时间窗口

| 时间（CST） | 动作 | 平台 |
|-------------|------|------|
| 每小时 | 📡 新闻情感采集 | OpenClaw 本地 |
| 工作日 07:00 | 🌙 隔夜情感分析 | OpenClaw 本地 |
| 工作日 09:25 | 🌅 盘前情感修正 | OpenClaw 本地 |
| 工作日 09:55 | 💸 卖出决策（→10:00卖出） | OpenClaw 本地 |
| 工作日 13:30 | 📈 T+1预测（→14:00买入） | GitHub Actions |

## 使用

```bash
pip install -r requirements.txt

# 如需API key
export ALPHA_VANTAGE_KEY=your_key
export MARKETAUX_KEY=your_key

# T+1 预测
python scripts/daily_multi_predict.py

# 情感轮询
python scripts/news_timeline.py

# 交易状态
python scripts/run_trading_flow.py --status
```

## 版本历史

| 版本 | 内容 |
|------|------|
| v1 | 24 因子 + 单市场 LightGBM |
| v1-merge | 四市场 + 宏观 + Optuna |
| v2 | Alpha158 130+因子 + Ensemble Stacking + WFA + Backtest |
| v3 | 多源新闻&社交情感模块 |
| **v4 (当前)** | **A+B 交易流程 + 7×24 情感轮询 + OpenClaw/GitHub Actions 双轨部署** |
