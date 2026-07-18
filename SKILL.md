---
name: multi-market-t1-predictor
description: 四市场（A股/美股/韩股/日股）T+1 涨幅预测，Qlib Alpha158 130+因子引擎 + LightGBM/XGBoost Ensemble + 信号预处理 + IC监控 + 回测
---

# Multi-Market T+1 Predictor v2 — Skill Overview

AKShare + yfinance + LightGBM/XGBoost + Qlib Alpha158 四市场 T+1 预测系统。

## 核心结构

| 模块 | 能力 |
|------|------|
| `scripts/config.py` | 统一配置：市场规则、WFA、Ensemble、Backtest 参数 |
| `scripts/data_fetcher.py` | CN:AKShare / US/KR/JP:yfinance / 情感:AlphaVantage / 宏观 |
| `scripts/features.py` | Alpha158 因子引擎：~130+ 因子（KBar/回归/分位/Aroon/价量相关/涨跌/成交量）+ 情感 + 跨市场 |
| `scripts/signal_processor.py` | 🆕 Winsorize → 截面Rank化 → 时序Z-Score → 填充 |
| `scripts/model.py` | LGB + XGBoost Ensemble + Ridge meta + Optuna + WFA + Factor IC |
| `scripts/backtest.py` | 🆕 Quintile分组 / 多空Spread / Sharpe / 多TopK对比 |
| `scripts/filter.py` | 各市场规则过滤（涨跌停/ST/流动性/行业分散） |
| `scripts/daily_multi_predict.py` | 主调度入口 |

## 使用

```bash
export ALPHA_VANTAGE_KEY=your_key   # 可选，情感增强
pip install -r requirements.txt
python scripts/daily_multi_predict.py
```

## 自动化

`.github/workflows/daily_multi_predict.yml` — 工作日北京 16:30 自动运行，输出预测 CSV + 回测报告。

## 升级记录

| 版本 | 内容 |
|------|------|
| v1 (原) | 24 因子 + 单市场 LightGBM |
| v1-merge | 四市场 + 情感 + 宏观 + Optuna |
| **v2 (当前)** | **Alpha158 130+因子 + signal preprocessing + Ensemble Stacking + Factor IC + WFA + Backtest** |
