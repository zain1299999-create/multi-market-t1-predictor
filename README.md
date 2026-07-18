# Multi-Market T+1 Predictor v4 — A+B Trading Flow + 7×24 News Sentiment

**A 股 T+1 隔日策略（14:00买入 → 次日10:00卖出）**  
多市场预测 + Alpha158 因子引擎 + **7×24 实时新闻情感轮询** + OpenClaw / GitHub Actions 双轨调度

---

## 交易策略：方案 A + B

| 方案 | 时间 | 动作 | 数据来源 |
|------|------|------|---------|
| **B — 模型预测** | 工作日 **13:30** → **14:00** 买入 | T+1 全量预测 → 买入信号 | Alpha158 + 行情 + 新闻情感 |
| **A — 情感轮询** | **7×24 每小时** | 新闻情感采集 → parquet 时间序列 | 11 RSS + 4 API + 社交 |
| **A + B 综合辅助** | 工作日 **07:00** | 隔夜情感分析 → 开盘情绪预判 | 情感曲线 + 隔夜新闻 |
| **A + B 综合辅助** | 工作日 **09:25** | 盘前情感修正 | 24h + 4h 情感加权 |
| **A + B 综合辅助** | 工作日 **09:55** | 卖出决策（→ **10:00** 卖出） | 隔夜情感 + 波动率 |

> **核心逻辑**：模型选股（B）+ 情感择时（A），双向验证 → 更稳健的买入/卖出决策

---

## 部署架构

| 部分 | 运行平台 | 频率 | 依赖 |
|------|---------|------|------|
| 📡 情感轮询引擎 | **OpenClaw**（本地） | 每小时 | 本地 parquet 时间序列 |
| 🌙 07:00 隔夜分析 | **OpenClaw**（本地） | 交易日 07:00 | 情感 parquet |
| 🌅 09:25 盘前修正 | **OpenClaw**（本地） | 交易日 09:25 | 情感 parquet |
| 💸 09:55 卖出决策 | **OpenClaw**（本地） | 交易日 09:55 | 情感 parquet |
| 📈 13:30 T+1 预测 | **GitHub Actions** | 交易日 13:30 (UTC 05:30) | API Key (Secrets) |

> 情感轮询依赖本地磁盘缓存（parquet），不适合纯 CI 环境，因此放在 OpenClaw 本地跑。
> T+1 模型预测完全自包含，在 GitHub Actions 上定时触发。

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
├── .github/workflows/
│   └── t1_predict_1330.yml         # GH Actions 自动化（工作日 13:30 CST）
├── scripts/
│   ├── config.py                   # 统一配置（市场/参数/API keys/WFA/Ensemble/Backtest/新闻情感）
│   ├── data_fetcher.py             # 数据获取（CN:AKShare / 其他:yfinance / 新闻情感 / 宏观）
│   ├── features.py                 # Alpha158 因子引擎（~130+ 技术因子 + 情感 + 跨市场 + T+1 标签）
│   ├── signal_processor.py         # 信号预处理（Winsorize → 截面Rank化 → 时序Z-Score → 填充）
│   ├── model.py                    # LightGBM + XGBoost Ensemble + Optuna + TimeSeriesCV + WFA + Factor IC
│   ├── backtest.py                 # 回测模块（Quintile分组 / 多空Spread / Sharpe / 多TopK对比）
│   ├── filter.py                   # 市场特定过滤（ST/涨跌停/流动性/行业分散）
│   ├── news_timeline.py            # 🆕 7×24 情感轮询引擎（增量 parquet, 72h 滚动窗口）
│   ├── run_trading_flow.py         # 🆕 交易流程调度器（4个时间窗口 + 状态展示）
│   ├── daily_multi_predict.py      # 🎯 主入口，编排 T+1 完整流水线
│   └── news_sentiment/             # 📰 多源新闻 & 社交情感采集分析包
│       ├── config_news.py          #   50+ 数据源配置（RSS/API/Social）
│       ├── news_collector.py       #   新闻采集（RSS + Alpha Vantage/NewsAPI/Marketaux）
│       ├── social_collector.py     #   社交采集（Twitter/微博/Reddit/YouTube）
│       ├── sentiment_engine.py     #   情感分析（VADER/TextBlob/SnowNLP + 金融词典）
│       ├── cache_manager.py        #   磁盘缓存（6h TTL）
│       ├── aggregator.py           #   股票代码聚合与时序评分
│       └── __init__.py             #   统一入口 collect_and_analyze()
├── config/
│   └── markets.yaml                # 多市场配置（备用扩展）
├── requirements.txt
├── outputs/                        # 预测结果 + 交易日志（自动生成）
├── logs/                           # 日志（自动生成）
├── data/
│   ├── cache/                      # Parquet 缓存（自动生成）
│   └── news_timeline/             # 🆕 情感时间序列 parquet（自动生成）
└── reports/                        # 回测报告（自动生成）
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

### 7×24 情感轮询

`scripts/news_timeline.py` 实现了一个独立的情感时间序列引擎：

```
每小时触发
│
├─ collect_and_analyze("CN")  → 中文RSS + API
├─ collect_and_analyze("US")  → 英文RSS + API
│
├─ 情感评分标准化 → 行情 ticker 聚合
│
├─ 写入 sentiment_series.parquet（增量追加）
└─ 72h 滚动窗口（自动清理过期数据）
```

情感曲线 JSON 格式：
```json
{
  "current_score": 0.784,
  "snapshots": 40,
  "trend_latest": 0.015,
  "trend_24h": 0.123,
  "volatility": 0.08,
  "signal": "bullish",
  "confidence": 0.72
}
```

### 数据流

```
collect_and_analyze(market, tickers)
│
├─ 🆓 RSS 免费采集 ────── 36氪 / Bloomberg / CNBC / Yahoo Finance 等活源
│  (无 API key 也可)
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
| Bloomberg / CNBC / Yahoo Finance | 1.0× | 权威财经媒体 |
| 36氪 | 1.0× | 中文科技/财经 |
| Twitter / Reddit | 0.5× | 社交噪音较高 |
| 微博 / Telegram | 0.5× | 中文社交 |

> 每个来源独立失败不影响系统。无 API key 时自动降级为 RSS 采集。

---

## 交易流程调度器

`scripts/run_trading_flow.py` 管理全部 4 个时间窗口：

| 时间窗口（CST） | 功能 | 产出 |
|----------------|------|------|
| `--slot 0700` | 隔夜情感分析 | 隔夜情绪综合评分 + 开盘预判 |
| `--slot 0925` | 盘前情感修正 | 24h+4h 混合信号 + 当日修正 |
| `--slot 0955` | 卖出决策检查 | sell_signal.json（是否10:00卖出） |
| `--slot 1330` | T+1 全量预测 | 调用 daily_multi_predict.py → 买入信号 |

辅助命令：
```bash
# 查看今日已完成的窗口和情感状态
python scripts/run_trading_flow.py --status

# 强制运行某个窗口
python scripts/run_trading_flow.py --slot 1330 --force
```

输出文件：
- `outputs/trading_journal.json` — 每日交易记录
- `outputs/sell_signal.json` — 10:00 卖出信号决策

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

# 可选：API key 开启深度新闻采集
export ALPHA_VANTAGE_KEY=your_key_here
export MARKETAUX_KEY=your_key_here

# 运行 T+1 预测
python scripts/daily_multi_predict.py

# 运行情感轮询（单次）
python scripts/news_timeline.py

# 查看交易日程状态
python scripts/run_trading_flow.py --status

# 强制运行卖出决策
python scripts/run_trading_flow.py --slot 0955 --force
```

输出路径：
- `outputs/multi_t1_pred_YYYYMMDD.csv` — T+1 TopK 预测
- `outputs/trading_journal.json` — 交易决策日志
- `data/news_timeline/sentiment_series.parquet` — 情感时间序列
- `reports/backtest_summary_*.csv` — 回测报告

---

## GitHub Actions 自动化

工作日 **13:30 CST（UTC 05:30）** 自动运行：

- 判断各市场交易日状态
- AKShare / yfinance 拉取最新行情
- 计算 Alpha158 因子 + 情感特征 → 信号预处理
- 训练/加载 LightGBM + XGBoost 集成模型
- 输出 TopK 预测 + 模型缓存（跨运行复用）
- 结果以 Artifact 形式保存 7 天

### 配置 Secrets

| Secret Name | 用途 | 获取方式 |
|-------------|------|---------|
| `ALPHA_VANTAGE_KEY` | 行情数据 + 新闻情感 | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| `MARKETAUX_KEY` | 市场新闻情感 | [marketaux.com](https://marketaux.com/) |

---

## OpenClaw 本地定时（情感轮询 + 辅助分析）

| Cron 任务 | 时间 | 执行脚本 |
|-----------|------|---------|
| 📡 新闻情感轮询 | 每小时 | `news_timeline.py --verbose` |
| 🌙 隔夜情感分析 | 交易日 07:00 | `run_trading_flow.py --slot 0700` |
| 🌅 盘前情感修正 | 交易日 09:25 | `run_trading_flow.py --slot 0925` |
| 💸 卖出决策 | 交易日 09:55 | `run_trading_flow.py --slot 0955` |

> 如需推送通知（微信/短信），在 OpenClaw cron 的 delivery 中配置对应渠道。

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

详细配置见 `scripts/config.py` 和 `scripts/news_sentiment/config_news.py`

---

## 版本历史

| 版本 | 内容 |
|------|------|
| v1 | 24 因子 + 单市场 LightGBM |
| v1-merge | 四市场 + 宏观 + Optuna |
| v2 | Alpha158 130+因子 + Ensemble Stacking + WFA + Backtest |
| v3 | 多源新闻&社交情感模块 (RSS/API/Twitter/微博/Reddit/YouTube) |
| **v4 (当前)** | **A+B 交易流程 + 7×24 情感轮询 + OpenClaw/GitHub Actions 双轨部署** |
