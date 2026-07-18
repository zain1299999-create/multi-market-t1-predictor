# Multi-Market T+1 Predictor (Merged & Optimized)

**A 股 / 美股 / 韩股 / 日股** 四市场 T+1 次日涨幅预测系统  
AKShare + yfinance + LightGBM + Cross-Market Sentiment — **全实装，无占位码**

---

## 合并来源

| 项目 | 贡献 | 代码状态 |
|------|------|---------|
| **a-share-t1-predictor** | 24 技术因子、A 股硬过滤、parquet 缓存、跨截面 LightGBM、时间序列 CV | ✅ 全实装 |
| **multi_market_t1_predictor** | 四市场 yfinance 统一接入、Alpha Vantage 情感、跨市场宏观特征、Optuna 调参 | ✅ 全实装 |

所有 `pass` / 伪代码全部替换为真实实现。

---

## 项目结构

```
multi_market_t1_predictor/
├── .github/workflows/daily_multi_predict.yml  # GitHub Actions 自动化
├── scripts/
│   ├── config.py              # 统一配置（市场/参数/API keys）
│   ├── data_fetcher.py        # 三层数据获取（CN:AKShare / 其他:yfinance / 情感:Alpha Vantage + Marketaux / 宏观:yfinance）
│   ├── features.py            # 因子工程（24 技术因子 + 情感特征 + 跨市场宏观特征 + T+1 标签）
│   ├── model.py               # LightGBM 跨截面单模型 + 时间序列 CV + Optuna 支持
│   ├── filter.py              # 市场特定过滤（ST/涨跌停/流动性/行业分散）/ 排重分散
│   └── daily_multi_predict.py # 🎯 主入口，编排完整多市场流水线
├── config/
│   └── markets.yaml           # 多市场配置（备用扩展）
├── requirements.txt
├── outputs/                   # 预测结果（自动生成）
├── logs/                      # 日志（自动生成）
├── data/cache/                # Parquet 缓存（自动生成）
└── reports/                   # 报告（自动生成）
```

---

## 支持市场

| 市场 | 数据源 | T+ 规则 | 涨跌停 | 过滤 |
|------|--------|---------|--------|------|
| 🇨🇳 **A股 (CN)** | AKShare → Baostock → yfinance | T+1 | 10% | ST/涨停/流动性/次新 |
| 🇺🇸 **美股 (US)** | yfinance → Alpha Vantage | T+1 | 无 | 流动性/成交量 |
| 🇰🇷 **韩股 (KR)** | yfinance (.KS/.KQ) | T+2 | 30% | 流动性 |
| 🇯🇵 **日股 (JP)** | yfinance (.T) | T+2 | 20% | 流动性 |

---

## 特征清单（30+ 因子）

| 类别 | 因子 |
|------|------|
| **均线** | MA(3/5/10/20/30/60), close/MA 偏离 |
| **动量** | ROC(5/10/20/60), Momentum(1d/5d/10d/20d) |
| **波动率** | STD(5/10/20), 振幅 |
| **布林带** | 位置、宽度 |
| **MACD** | DIF, Signal, Histogram |
| **RSI** | 14 日 |
| **CCI** | 20 日 |
| **ATR** | ATR(14), ATR% |
| **成交量** | 量比(5d)、成交量变化、换手率 |
| **成交额** | 成交额均值比（流动性过滤） |
| **价格位置** | 相对 20 日高低点 |
| **📰 情感因子** | sentiment_score, sentiment_MA(5d), sentiment_change, news_count |
| **🌎 跨市场** | VIX, USD/CNY, S&P500 收益, 相关性变化 |

---

## 快速开始

```bash
# 安装
pip install -r requirements.txt

# 运行（默认 CN + US 市场）
python scripts/daily_multi_predict.py

# 可选：开启情感数据（需要 API key）
export ALPHA_VANTAGE_KEY=your_key
python scripts/daily_multi_predict.py
```

首次运行：下载历史数据 + 训练模型（CN 约 3-5 分钟，US 约 1 分钟）  
后续运行：Parquet 缓存 + 已训练模型（30 秒 - 1 分钟）

**输出：** `outputs/multi_t1_pred_YYYYMMDD.csv`

---

## 自定义

编辑 `scripts/config.py`：

| 参数 | 说明 | 默认 |
|------|------|------|
| `ACTIVE_MARKETS` | 活跃市场 | `["CN", "US"]` |
| `TOP_K` | 每市场推荐数量 | `10` |
| `TRAIN_YEARS` | 训练用历史 | `2` |
| `OPTUNA_ENABLED` | 启用 Optuna 调参 | `False` |
| `UNIVERSE` | 股票池 | `"multi"` |

---

## 情感数据（可选增强）

支持的新闻 provider：
- **Alpha Vantage** `NEWS_SENTIMENT`（免费 tier，每日 5 次/分钟）
- **Marketaux**（免费 100 req/天）
- 设置环境变量 `ALPHA_VANTAGE_KEY` / `MARKETAUX_KEY` 即可启用

情感特征作为额外输入，可：
1. 在模型训练时作为因子（提升突发事件捕捉）
2. 在预测结果中作为辅助参考（输出列中查看）

---

## GitHub Actions 自动化

推送到 GitHub 即可自动：
- 工作日北京时间 ~16:30 运行
- 支持手动触发（workflow_dispatch）
- Artifacts 下载预测结果
- 自动 commit 输出到仓库

如需情感数据，在仓库 Secrets 添加 `ALPHA_VANTAGE_KEY`。

---

## 注意事项

- **仅供研究学习，不构成投资建议**
- 实盘前请充分回测 + 风控
- AKShare 接口可能因源站变更失效 → `pip install akshare --upgrade`
- 美股数据通过 yfinance（约 15min 延迟）

---

## License

MIT
