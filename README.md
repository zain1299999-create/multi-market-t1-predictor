# Multi-Market T+1 Predictor v3 — Multi-Source News Sentiment + Alpha158

**A 股 / 美股 / 韩股 / 日股** 四市场 T+1 次日涨幅预测系统  
AKShare + yfinance + LightGBM / XGBoost + Qlib Alpha158 因子引擎 + **多源新闻 & 社交情感** — **~160+ 因子 + 信号预处理 + 集成学习 + IC 监控 + 回测**

---

## 合并来源

| 项目 | 贡献 | 代码状态 |
|------|------|---------|
| **a-share-t1-predictor** | 24 技术因子、A 股硬过滤、parquet 缓存、跨截面 LightGBM、时间序列 CV | ✅ 全实装 |
| **multi_market_t1_predictor** | 四市场 yfinance 统一接入、跨市场宏观特征、Optuna 调参 | ✅ 全实装 |
| **Qlib Alpha158** | 130+ 因子表达式引擎（KBar/回归趋势/价格分位/Aroon/价量相关/涨跌统计/成交量统计） | ✅ 全实装 |
| **FinnewsHunter / StockAgent** | 多智能体金融情报架构（启发） | 🔍 借鉴设计 |

所有 `pass` / 伪代码全部替换为真实实现。

---

## 项目结构

```
multi_market_t1_predictor/
├── .github/workflows/daily_multi_predict.yml  # GitHub Actions 自动化（工作日 16:30 CST）
├── scripts/
│   ├── config.py              # 统一配置（市场/参数/API keys/WFA/Ensemble/Backtest/新闻情感）
│   ├── data_fetcher.py        # 数据获取（CN:AKShare / 其他:yfinance / 新闻情感 / 宏观）
│   ├── features.py            # Alpha158 因子引擎（~130+ 技术因子 + 情感 + 跨市场 + T+1 标签）
│   ├── signal_processor.py    # 信号预处理（Winsorize → 截面Rank化 → 时序Z-Score → 填充）
│   ├── model.py               # LightGBM + XGBoost Ensemble + Optuna + TimeSeriesCV + WFA + Factor IC
│   ├── backtest.py            # 回测模块（Quintile分组 / 多空Spread / Sharpe / 多TopK对比）
│   ├── filter.py              # 市场特定过滤（ST/涨跌停/流动性/行业分散）
│   ├── news_sentiment/        # 📰 多源新闻 & 社交情感采集分析包
│   │   ├── config_news.py     #   50+ 数据源配置（RSS/API/Social）
│   │   ├── news_collector.py  #   新闻采集（RSS + Alpha Vantage/NewsAPI/Marketaux）
│   │   ├── social_collector.py#   社交采集（Twitter/微博/Reddit/YouTube）
│   │   ├── sentiment_engine.py#   情感分析（VADER/TextBlob/SnowNLP + 金融词典）
│   │   ├── cache_manager.py   #   磁盘缓存（6h TTL）
│   │   ├── aggregator.py      #   股票代码聚合与时序评分
│   │   └── __init__.py        #   统一入口 collect_and_analyze()
│   └── daily_multi_predict.py # 🎯 主入口，编排完整流水线
├── config/
│   └── markets.yaml           # 多市场配置（备用扩展）
├── requirements.txt
├── outputs/                   # 预测结果（自动生成）
├── logs/                      # 日志（自动生成）
├── data/cache/                # Parquet 缓存（自动生成）
└── reports/                   # 回测报告（自动生成）
```

---

## 因子体系（~160+ 因子）

| 因子群 | 数量 | 来源 |
|--------|------|------|
| KBar 形态（KMID/KLEN/KUP/KLOW/KSFT 等） | 9 | Qlib Alpha158 |
| 回归趋势（BETA/RSQR/RESI） | 3×5=15 | Qlib Alpha158 |
| 价格分位（QTLU/QTLD/RANK/RSV） | 4×5=20 | Qlib Alpha158 |
| Aroon 指标（IMAX/IMIN/IMXD） | 3×5=15 | Qlib Alpha158 |
| 价量相关性（CORR/CORD） | 2×5=10 | Qlib Alpha158 |
| 涨跌统计（CNTP/CNTN/CNTD/SUMP/SUMN/SUMD） | 6×5=30 | Qlib Alpha158 |
| 成交量统计（VMA/VSTD/WVMA/VSUMP/VSUMN/VSUMD） | 6×5=30 | Qlib Alpha158 |
| 基础技术因子（MA/ROC/MACD/RSI/CCI/Bollinger/ATR） | 24 | 原项目保留 |
| 📰 **多源新闻情感因子**（score, MA3/5/10, std5, trend, news_count） | **7** | **多源新闻情感模块** |
| 跨市场因子（VIX/全球指数/汇率） | 6+ | yfinance |

**合计 ~160+ 原始特征 → 信号预处理（Winsorize + 截面Rank化）后进入模型**

---

## 📰 多源新闻 & 社交情感采集模块

> **新增 v3** — 整合 X/Twitter、微博、Reddit、YouTube、财联社、Reuters、Bloomberg 等多平台实时新闻情感

### 数据流

```
collect_and_analyze(market, tickers)
│
├─ 🆓 RSS 免费采集 ────── 财联社 / 东方财富 / 新浪财经 / Reuters / Bloomberg
│  (无 API key 也可)       Yahoo Finance / Seeking Alpha / MarketWatch 等 11 源
│
├─ 🔑 API 深度采集 ────── Alpha Vantage (内置ticker级情感) / NewsAPI / MarketAux
│
├─ 🐦 社交爬虫 ────────── Twitter/X / 微博 / Reddit(r/wallstreetbets, r/stocks) / Telegram
│   (通过 snscrape, 无需认证)
│
└─ 🎬 YouTube ────────── 搜索金融视频 + 评论情感
       │
       ▼
 Sentiment Engine             ← 多模型并行
 ├─ VADER       (英文基础情感)
 ├─ TextBlob    (英文备用)
 ├─ SnowNLP     (中文情感)
 └─ 金融词典     (CN/US/KR/JP 四市场专用词汇增强)
       │
       ▼
 Aggregator  →  ticker 级情感向量
 ├─ sentiment_score         当前加权情绪 (-1 ~ +1)
 ├─ sentiment_ma_3/5/10     短/中/长期滚动均线
 ├─ sentiment_std_5         观点分歧度（标准差）
 ├─ sentiment_trend         趋势方向 (1=正在变好, -1=变差)
 └─ news_count              新闻数量（信息热度）
```

### 来源权重

| 来源 | 权重 | 说明 |
|------|------|------|
| Alpha Vantage | 1.5× | 内置 ticker 级情感评分 |
| MarketAux | 1.3× | 包含新闻情感分数 |
| Reuters / Bloomberg / 财联社 | 1.0× | 权威财经媒体 |
| Twitter / Reddit | 0.5× | 社交噪音较高 |
| 微博 / Telegram | 0.5× | 中文社交 |

> 每个来源独立失败不影响系统。无 API key 时自动降级为 RSS + 社交爬虫。

---

## 模型架构

```
┌─────────────┐     ┌─────────────┐
│  LightGBM   │     │  XGBoost    │
│  (主模型)    │     │  (辅助模型)  │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └────────┬──────────┘
                ▼
       ┌────────────────┐
       │  Ridge / Linear│  ← Meta Learner
       │  Stacking      │
       └───────┬────────┘
               ▼
         Stage 3: Factor IC Analysis
         Output: Top-10 / Bottom-10 factors by |IC|

Stage 1: 时间序列 CV（3-fold TimeSeriesSplit, forward-chaining）
Stage 2: 可选 Optuna 超参搜索（30 trials）
Stage 4: 可选 Walk-Forward Analysis（504天窗口, 20天步进）
```

---

## 回测能力

每天按预测分值排序，等分 5 组（Q1-Q5），计算每组次日平均收益：

| 指标 | 说明 |
|------|------|
| **Quintile 分组收益** | Q1（最看好）vs Q5（最看空）差异 |
| **多空 Spread** | Q1 - Q5 收益率差 + t-stat |
| **年化 Sharpe** | TopK 组合的年化夏普比率 |
| **累计收益曲线** | CSV 输出，可回看历史表现 |

> 不含交易成本 / 滑点 — 纯 alpha 验证

---

## 快速开始

```bash
git clone https://github.com/zain1299999-create/multi-market-t1-predictor.git
cd multi-market-t1-predictor

pip install -r requirements.txt

# 可选：情感增强（环境变量）
export ALPHA_VANTAGE_KEY=your_key_here

# 运行完整预测管线
python scripts/daily_multi_predict.py
```

输出在 `outputs/multi_t1_pred_YYYYMMDD.csv`，回测报告在 `reports/`。

---

## GitHub Actions 自动化

每天北京 16:30（开盘日，周一至周五）自动运行：

- 判断各市场交易日状态
- 获取最新数据 → 计算 Alpha158 因子 → 信号预处理
- 训练/更新模型（每 20 天自动重训）
- 输出 TopK 预测 + 回测报告
- 结果以 Artifact 形式保存

---

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `ACTIVE_MARKETS` | `["CN", "US"]` | 可启用 KR/JP |
| `TOP_K` | `10` | 每市场推荐数 |
| `ENSEMBLE_ENABLED` | `True` | LGB+XGB+Ridge 集成 |
| `OPTUNA_ENABLED` | `False` | 是否超参搜索 |
| `WFA_ENABLED` | `False` | Walk-Forward 分析 |
| `BACKTEST_TOP_K` | `[5,10,20,30]` | 回测多 K 对比 |
| `NEWS_RSS_ENABLED` | `True` | RSS 新闻采集（免费） |
| `NEWS_API_ENABLED` | `True` | API 新闻（有 key 则用） |
| `NEWS_SOCIAL_ENABLED` | `True` | 社交爬虫（需 snscrape） |

详细配置见 `scripts/config.py` 和 `scripts/news_sentiment/config_news.py`
