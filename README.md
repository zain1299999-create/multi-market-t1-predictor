# Multi-Market T+1 Predictor v2 — Alpha158 Upgrade

**A 股 / 美股 / 韩股 / 日股** 四市场 T+1 次日涨幅预测系统  
AKShare + yfinance + LightGBM / XGBoost + Qlib Alpha158 因子引擎 — **~130+ 技术因子 + 信号预处理 + 集成学习 + IC 监控 + 回测**

---

## 合并来源

| 项目 | 贡献 | 代码状态 |
|------|------|---------|
| **a-share-t1-predictor** | 24 技术因子、A 股硬过滤、parquet 缓存、跨截面 LightGBM、时间序列 CV | ✅ 全实装 |
| **multi_market_t1_predictor** | 四市场 yfinance 统一接入、Alpha Vantage 情感、跨市场宏观特征、Optuna 调参 | ✅ 全实装 |
| **Qlib Alpha158** | 130+ 因子表达式引擎（KBar/回归趋势/价格分位/Aroon/价量相关/涨跌统计/成交量统计） | ✅ 全实装 |

所有 `pass` / 伪代码全部替换为真实实现。

---

## 项目结构

```
multi_market_t1_predictor/
├── .github/workflows/daily_multi_predict.yml  # GitHub Actions 自动化（工作日 16:30 CST）
├── scripts/
│   ├── config.py              # 统一配置（市场/参数/API keys/WFA/Ensemble/Backtest）
│   ├── data_fetcher.py        # 三层数据获取（CN:AKShare / 其他:yfinance / 情感:Alpha Vantage / 宏观:yfinance）
│   ├── features.py            # Alpha158 因子引擎（~130+ 技术因子 + 情感 + 跨市场 + T+1 标签）
│   ├── signal_processor.py    # 🆕 信号预处理（Winsorize → 截面Rank化 → 时序Z-Score → 填充）
│   ├── model.py               # LightGBM + XGBoost Ensemble + Optuna + TimeSeriesCV + WFA + Factor IC
│   ├── backtest.py            # 🆕 回测模块（Quintile分组 / 多空Spread / Sharpe / 多TopK对比）
│   ├── filter.py              # 市场特定过滤（ST/涨跌停/流动性/行业分散）
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

## 因子体系（~130+ 因子）

| 因子群 | 数量 | 来源 |
|--------|------|------|
| KBar 形态（KMID/KLEN/KUP/KLOW/KSFT 等） | 9 | Qlib Alpha158 |
| 回归趋势（BETA/RSQR/RESI） | 3×5=15 | Qlib Alpha158 |
| 价格分位（QTLU/QTLD/RANK/RSV） | 4×5=20 | Qlib Alpha158 |
| Aroon 指标（IMAX/IMIN/IMXD） | 3×5=15 | Qlib Alpha158 |
| 价量相关性（CORR/CORD） | 2×5=10 | Qlib Alpha158 |
| 涨跌统计（CNTP/CNTN/CNTD/SUMP/SUMN/SUMD） | 6×5=30 | Qlib Alpha158 |
| 成交量统计（VMA/VSTD/WVMA/VSUMP/VSUMN/VSUMD） | 6×5=30 | Qlib Alpha158 |
| 基础技术因子（MA/ROC/MACD/RSI/CCI/Bollinger/ATR 等） | 24 | 原项目保留 |
| 情感因子（sentiment score/MA/change/News Count） | 5 | Alpha Vantage |
| 跨市场因子（VIX/全球指数/汇率） | 6+ | yfinance |

**合计 ~155 个原始特征 → 信号预处理（Winsorize + 截面Rank化）后进入模型**

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

详细配置见 `scripts/config.py`
