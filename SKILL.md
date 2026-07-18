---
name: multi-market-t1-predictor
description: 四市场（A股/美股/韩股/日股）T+1 预测，Alpha158 160+因子 + Ensemble + 多源新闻情感 + 信号预处理 + 回测
---

# Multi-Market T+1 Predictor v3 — News Sentiment + Alpha158

AKShare + yfinance + LightGBM/XGBoost + Qlib Alpha158 + **多源新闻/社交情感** 四市场 T+1 预测系统。

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
| **`scripts/news_sentiment/`** | **📰 v3 新增**: 多源新闻&社交情感采集分析包 |
| `scripts/daily_multi_predict.py` | 主调度入口 → 每日TopK预测 |

## 📰 多源新闻情感模块（v3）

| 源类型 | 具体来源 | 认证需求 |
|--------|----------|---------|
| RSS免费（11源） | 财联社/东方财富/新浪财经/Reuters/Bloomberg/Yahoo Finance/Seeking Alpha等 | 无 |
| API深度（3源） | Alpha Vantage(ticker级情感)/NewsAPI/MarketAux | API key(可选) |
| 社交爬虫（4源） | Twitter/微博/Reddit(r/wallstreetbets)/Telegram | 无(snscrape) |
| YouTube | 金融视频搜索+评论 | 无 |

情感分析: VADER(EN) + TextBlob(EN) + SnowNLP(ZH) + 金融词典(4市场增强)

输出特征: sentiment_score + MA3/5/10 + std5 + trend + news_count

## 使用

```bash
# 安装全部依赖（含情感模块）
pip install -r requirements.txt

# 可选：API key 开启深度新闻采集
export ALPHA_VANTAGE_KEY=your_key

# 可选：社交爬虫增强
pip install snscrape

# 运行
python scripts/daily_multi_predict.py
```

## 升级记录

| 版本 | 内容 |
|------|------|
| v1 | 24 因子 + 单市场 LightGBM |
| v1-merge | 四市场 + 宏观 + Optuna |
| v2 | Alpha158 130+因子 + Ensemble Stacking + WFA + Backtest |
| **v3 (当前)** | **多源新闻&社交情感模块 (RSS/API/Twitter/微博/Reddit/YouTube) + 情感特征因子 ~160+** |
