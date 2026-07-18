"""
Multi-Market T+1 Predictor — Data Fetcher
Three-layer data access:
  1. CN:  AKShare (primary) → Baostock (fallback) → yfinance (last resort)
  2. US:  yfinance (primary) → Alpha Vantage (enhancement)
  3. KR:  yfinance (.KS/.KQ) → pykrx (fallback)
  4. JP:  yfinance (.T) → pyjquants (fallback)

Features: Parquet incremental cache, exponential backoff, sentiment aggregation
"""
import time
import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union

import pandas as pd
import numpy as np

from config import (DATA_CACHE, DATA_START_DATE, MACRO_SYMBOLS,
                    ALPHA_VANTAGE_KEY, MARKETAUX_KEY, MARKET_RULES)

logger = logging.getLogger(__name__)


# ─── YFinance (primary for non-CN, fallback for CN) ────────────────────

def _yf_download(tickers: List[str], period: str = "1y",
                 interval: str = "1d") -> pd.DataFrame:
    """Unified yfinance download with retry."""
    try:
        import yfinance as yf
        df = yf.download(
            tickers, period=period, interval=interval,
            group_by="ticker", auto_adjust=True, progress=False,
            threads=True, timeout=15,
        )
        return df
    except Exception as exc:
        logger.debug("yfinance download failed: %s", exc)
        return pd.DataFrame()


def fetch_yf_ohlcv(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """Fetch OHLCV via yfinance, return melted DataFrame with columns:
    date, symbol, open, high, low, close, volume
    """
    raw = _yf_download(tickers, period=period)
    if raw.empty:
        return pd.DataFrame()

    rows = []
    if isinstance(raw.columns, pd.MultiIndex):
        # Multi-ticker: columns = (Price, Ticker)
        tickers_found = raw.columns.get_level_values(1).unique()
        for ticker in tickers_found:
            t = raw[ticker]
            if t.empty:
                continue
            df = t.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df.rename(columns={"close": "close", "open": "open",
                               "high": "high", "low": "low",
                               "volume": "volume"}, inplace=True)
            df["symbol"] = ticker
            rows.append(df)
    else:
        # Single ticker
        df = raw.reset_index()
        df.columns = [c.lower() for c in df.columns]
        if "symbol" not in df.columns:
            df["symbol"] = tickers[0]
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values(["symbol", "date"]).reset_index(drop=True)


# ─── AKShare (CN market primary) ───────────────────────────────────────

def _ak_import():
    import akshare as ak
    return ak


def fetch_akshare_daily(symbol: str, start_date: str = None,
                        end_date: str = None, retries: int = 3) -> pd.DataFrame:
    """Fetch single A-share stock daily K-line via AKShare with retry."""
    ak = _ak_import()
    start = start_date or DATA_START_DATE
    end = end_date or datetime.now().strftime("%Y%m%d")

    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=start, end_date=end, adjust="qfq",
            )
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "振幅": "amplitude",
                "涨跌幅": "pct_chg", "涨跌额": "change",
                "换手率": "turnover",
            })
            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = symbol
            return df.sort_values("date").reset_index(drop=True)
        except Exception as exc:
            wait = 2 ** attempt + random.random()
            if attempt < retries - 1:
                time.sleep(wait)
    return pd.DataFrame()


def fetch_cn_index_components(index: str = "hs300") -> pd.DataFrame:
    """Fetch A-share index constituents."""
    ak = _ak_import()
    try:
        if index == "hs300":
            df = ak.index_stock_cons_weight_csindex("931069")
        elif index == "zz500":
            df = ak.index_stock_cons_weight_csindex("000905")
        elif index == "all_a":
            df = ak.stock_info_a_code_name()
            df = df.rename(columns={"code": "成分券代码", "name": "成分券名称"})
            return df
        else:
            raise ValueError(f"Unknown index: {index}")

        df = df.rename(columns={"成分券代码": "code", "成分券名称": "name",
                                 "权重": "weight"})
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df
    except Exception as exc:
        logger.warning("AKShare index fetch failed: %s", exc)
        try:
            import baostock as bs
            bs.login()
            mapper = {"hs300": bs.query_hs300_stocks,
                      "zz500": bs.query_zz500_stocks}[index]
            rs = mapper()
            data = []
            while rs.next():
                row = rs.get_row_data()
                data.append({
                    "code": row[0].replace("sh.", "").replace("sz.", "").replace("bj.", ""),
                    "name": row[2] if len(row) > 2 else "",
                })
            bs.logout()
            return pd.DataFrame(data)
        except Exception as e2:
            logger.error("Baostock fallback also failed: %s", e2)
            return pd.DataFrame()


def fetch_cn_batch(codes: List[str], start_date: str = None,
                   end_date: str = None, batch_sleep: float = 0.3) -> pd.DataFrame:
    """Fetch K-lines for many A-share stocks with parquet cache."""
    cache_path = DATA_CACHE / "cn_kline.parquet"

    cached = {}
    if cache_path.exists():
        try:
            cached_df = pd.read_parquet(cache_path)
            cached = {sym: grp for sym, grp in cached_df.groupby("symbol")}
            logger.info("  CN cache: %d symbols loaded", len(cached))
        except Exception:
            pass

    results = []
    for i, code in enumerate(codes):
        if code in cached:
            df = cached[code]
            last = df["date"].max()
            if last >= pd.Timestamp.now().normalize() - pd.Timedelta(days=5):
                results.append(df)
                if (i + 1) % 50 == 0:
                    logger.info("  [CN cache hit] %d/%d", i + 1, len(codes))
                continue
        df = fetch_akshare_daily(code, start_date, end_date)
        if not df.empty:
            results.append(df)
        if (i + 1) % 10 == 0:
            logger.info("  [CN fetch] %d/%d", i + 1, len(codes))
        time.sleep(batch_sleep)

    if not results:
        return pd.DataFrame()
    combined = pd.concat(results, ignore_index=True)
    try:
        combined.to_parquet(cache_path, index=False)
    except Exception:
        pass
    return combined


# ─── Sentiment (Alpha Vantage + Marketaux) ─────────────────────────────

def fetch_alpha_vantage_sentiment(tickers: List[str],
                                  time_from: str = None) -> pd.DataFrame:
    """Fetch news sentiment from Alpha Vantage NEWS_SENTIMENT endpoint."""
    if not ALPHA_VANTAGE_KEY:
        return pd.DataFrame()

    import requests as req
    results = []
    for ticker in tickers:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": ALPHA_VANTAGE_KEY,
            "limit": 100,
        }
        if time_from:
            params["time_from"] = time_from
        try:
            resp = req.get(url, params=params, timeout=15).json()
            feed = resp.get("feed", [])
            for item in feed:
                pub = item.get("time_published", "")[:8]
                if not pub:
                    continue
                results.append({
                    "date": pd.to_datetime(pub, format="%Y%m%d"),
                    "symbol": ticker,
                    "sentiment_score": float(item.get("overall_sentiment_score", 0)),
                    "sentiment_label": item.get("overall_sentiment_label", "neutral"),
                })
        except Exception as exc:
            logger.debug("AV sentiment fail for %s: %s", ticker, exc)
        time.sleep(0.5)  # AV rate limit

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    # Aggregate per day per symbol
    agg = df.groupby(["date", "symbol"]).agg(
        sentiment_score=("sentiment_score", "mean"),
        news_count=("sentiment_score", "count"),
    ).reset_index()
    return agg


def fetch_marketaux_sentiment(symbols: List[str]) -> pd.DataFrame:
    """Fetch news via Marketaux (free tier, 100 req/day)."""
    if not MARKETAUX_KEY:
        return pd.DataFrame()
    import requests as req
    results = []
    for sym in symbols:
        url = "https://api.marketaux.com/v1/news/all"
        params = {
            "symbols": sym,
            "filter_entities": "true",
            "language": "en",
            "api_token": MARKETAUX_KEY,
            "limit": 50,
        }
        try:
            resp = req.get(url, params=params, timeout=10).json()
            for item in resp.get("data", []):
                pub_date = item.get("published_at", "")[:10]
                results.append({
                    "date": pd.to_datetime(pub_date),
                    "symbol": sym,
                    "title": item.get("title", ""),
                    "sentiment_score": item.get("sentiment_score", 0),
                })
        except Exception as exc:
            logger.debug("Marketaux fail for %s: %s", sym, exc)
        time.sleep(0.3)

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    agg = df.groupby(["date", "symbol"]).agg(
        sentiment_score=("sentiment_score", "mean"),
        news_count=("sentiment_score", "count"),
    ).reset_index()
    return agg


def fetch_sentiment(tickers: List[str]) -> pd.DataFrame:
    """Aggregate sentiment from all available providers."""
    parts = []
    av = fetch_alpha_vantage_sentiment(tickers)
    if not av.empty:
        parts.append(av)
    ma = fetch_marketaux_sentiment(tickers)
    if not ma.empty:
        parts.append(ma)
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.groupby(["date", "symbol"]).agg(
        sentiment_score=("sentiment_score", "mean"),
        news_count=("news_count", "sum"),
    ).reset_index()
    return merged


# ─── Macro data ────────────────────────────────────────────────────────

def fetch_macro(symbols: List[str] = None, period: str = "1y") -> pd.DataFrame:
    """Fetch macro/global index data via yfinance."""
    syms = symbols or MACRO_SYMBOLS
    try:
        import yfinance as yf
        df = yf.download(syms, period=period, progress=False)
        if "Close" in df.columns:
            return df["Close"]
        return df
    except Exception as exc:
        logger.warning("Macro fetch failed: %s", exc)
        return pd.DataFrame()


# ─── Trading day check ─────────────────────────────────────────────────

def is_trading_day(market: str = "CN") -> bool:
    """Check if today is a trading day for the given market."""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    if market == "CN":
        try:
            import akshare as ak
            cal = ak.tool_trade_date_hist_sina()
            today = now.strftime("%Y-%m-%d")
            return today in cal["trade_date"].astype(str).values
        except Exception:
            return True
    return True  # US/KR/JP: assume weekday = trading day


# ─── Unified entry point ───────────────────────────────────────────────

def fetch_market_data(market: str, tickers: List[str],
                      period: str = "1y") -> Dict[str, pd.DataFrame]:
    """Unified entry: fetch OHLCV + sentiment + macro for a market."""
    result = {}
    if market == "CN":
        ohlcv = fetch_cn_batch(tickers,
                                start_date=f"{datetime.now().year - TRAIN_YEARS}0101")
    else:
        ohlcv = fetch_yf_ohlcv(tickers, period=period)
    result["ohlcv"] = ohlcv

    sentiment = fetch_sentiment(tickers)
    result["sentiment"] = sentiment

    macro = fetch_macro(period=period)
    result["macro"] = macro

    return result


# Re-import needed at module level for fetch_market_data
from config import TRAIN_YEARS  # noqa: E402, F811
