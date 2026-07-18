---
name: multi-market-t1-predictor
description: 多市场（A股/美股/韩股/日股）T+1 次日涨幅预测系统。AKShare + yfinance + LightGBM 跨截面回归 + 新闻情感 + 跨市场宏观特征。触发词：T+1预测、次日涨幅、多市场、A股T+1、美股预测、量化选股、多市场因子、LightGBM多市场。
---

# Multi-Market T+1 Predictor — Skill Overview

AKShare + yfinance + LightGBM 四市场 T+1 预测系统。

## 核心结构

| 模块 | 能力 |
|------|------|
| `scripts/config.py` | 统一配置：市场规则、API keys、路径 |
| `scripts/data_fetcher.py` | CN:AKShare / US/KR/JP:yfinance / 情感:Alpha Vantage+Marketaux / 宏观 |
| `scripts/features.py` | 30+ 因子：24技术 + 情感score + 跨市场(VIX/汇率/全球指数) |
| `scripts/model.py` | 跨截面 LightGBM + TS-CV + 可选 Optuna 调参 |
| `scripts/filter.py` | 各市场规则过滤（涨跌停/ST/流动性/行业分散） |
| `scripts/daily_multi_predict.py` | 主调度入口 |

## 使用

```bash
export ALPHA_VANTAGE_KEY=your_key   # 可选，情感增强
pip install -r requirements.txt
python scripts/daily_multi_predict.py
```

## 自动化

`.github/workflows/daily_multi_predict.yml` — 工作日北京 16:30 自动运行。

## 合并记录

- a-share-t1-predictor（24因子+过滤+缓存）→ 底层实装
- multi_market_t1_predictor（四市场+情感+宏观+Optuna）→ 上层能力
- 全部 `pass` 替换为真实代码，无占位
