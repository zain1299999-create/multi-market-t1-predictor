"""
Multi-Language Sentiment Analysis Engine
=========================================
Supports English (VADER + TextBlob) and Chinese (SnowNLP + financial lexicon).

Design:
  - Each analyzer is independently wrapped in try/except
  - Missing dependencies result in graceful fallback (0.0 score)
  - Market-specific financial lexicons boost/penalize domain terms
  - Results are normalized to [-1.0, 1.0] range
"""
import logging
import importlib.util
import re
from typing import Dict, List, Optional

from .config_news import SENTIMENT_MODELS, WATCH_KEYWORDS

logger = logging.getLogger("sentiment_engine")

# ═══════════════════════════════════════════════════════════════════════
# English Sentiment Analyzers
# ═══════════════════════════════════════════════════════════════════════

_VADER_AVAIL = importlib.util.find_spec("vaderSentiment") is not None
_TEXTBLOB_AVAIL = importlib.util.find_spec("textblob") is not None


def _analyze_vader(text: str) -> float:
    """English VADER sentiment analysis. Returns compound score in [-1, 1]."""
    if not _VADER_AVAIL:
        return 0.0
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)["compound"]
    except Exception as e:
        logger.debug("VADER error: %s", e)
        return 0.0


def _analyze_textblob(text: str) -> float:
    """English TextBlob sentiment analysis. Returns polarity in [-1, 1]."""
    if not _TEXTBLOB_AVAIL:
        return 0.0
    try:
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity
    except Exception as e:
        logger.debug("TextBlob error: %s", e)
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
# Chinese Sentiment Analyzers
# ═══════════════════════════════════════════════════════════════════════

_SNOWNLP_AVAIL = importlib.util.find_spec("snownlp") is not None


def _analyze_snownlp(text: str) -> float:
    """Chinese SnowNLP sentiment. Returns score in [0, 1], mapped to [-1, 1]."""
    if not _SNOWNLP_AVAIL:
        return 0.0
    try:
        from snownlp import SnowNLP
        s = SnowNLP(text)
        # SnowNLP returns 0–1; map to -1 to 1
        return s.sentiments * 2.0 - 1.0
    except Exception as e:
        logger.debug("SnowNLP error: %s", e)
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
# Financial Lexicon Boost
# ═══════════════════════════════════════════════════════════════════════

# Chinese financial sentiment lexicon
_CN_FINANCIAL_POSITIVE = [
    "涨停", "大涨", "暴涨", "飙升", "突破", "利好", "牛市", "反弹",
    "反转", "回暖", "攀升", "强势", "超预期", "创新高", "放量上涨",
    "触底反弹", "企稳回升", "业绩大增", "扭亏为盈", "分红", "回购",
    "评级上调", "买入", "增持", "优质", "龙头",
]
_CN_FINANCIAL_NEGATIVE = [
    "跌停", "大跌", "暴跌", "跳水", "崩盘", "利空", "熊市", "亏损",
    "减持", "卖出", "评级下调", "ST", "退市", "违约", "爆仓",
    "恐慌", "抛售", "破位", "下行", "低迷", "利空出尽",
    "业绩预亏", "商誉减值", "立案调查", "监管处罚",
]

# English financial sentiment lexicon
_EN_FINANCIAL_POSITIVE = [
    "beat", "earnings beat", "outperform", "upgrade", "buy",
    "bullish", "positive guidance", "record high", "rally",
    "surge", "breakout", "recovery", "momentum", "growth",
    "dividend increase", "buyback", "strong quarter",
    "raised guidance", "overweight", "market leader",
]
_EN_FINANCIAL_NEGATIVE = [
    "miss", "earnings miss", "downgrade", "sell", "bearish",
    "negative guidance", "crash", "plunge", "loss", "decline",
    "recession", "default", "bankruptcy", "selloff",
    "underperform", "underweight", "restructuring",
    "layoff", "investigation", "regulatory action",
]


def _apply_financial_lexicon(text: str, market: str, base_score: float) -> float:
    """Apply market-specific financial lexicon to boost/penalize base score.

    Modifies the score by +/-0.1 per matching financial term,
    capped at +/-0.5 total adjustment.
    """
    text_lower = text.lower()
    adjustment = 0.0

    if market == "CN":
        for word in _CN_FINANCIAL_POSITIVE:
            if word in text:
                adjustment += 0.08
        for word in _CN_FINANCIAL_NEGATIVE:
            if word in text:
                adjustment -= 0.08
    else:
        for word in _EN_FINANCIAL_POSITIVE:
            if word.lower() in text_lower:
                adjustment += 0.08
        for word in _EN_FINANCIAL_NEGATIVE:
            if word.lower() in text_lower:
                adjustment -= 0.08

    adjustment = max(-0.5, min(0.5, adjustment))
    return max(-1.0, min(1.0, base_score + adjustment))


# ═══════════════════════════════════════════════════════════════════════
# Main Sentiment Analysis Functions
# ═══════════════════════════════════════════════════════════════════════


def analyze_text(text: str, market: str = "US") -> float:
    """Complete sentiment analysis pipeline for a single text.

    1. Run language-appropriate analyzer
    2. Apply financial lexicon boost
    3. Normalize to [-1.0, 1.0]

    Args:
        text: The text to analyze
        market: Market code for language/lexicon selection

    Returns:
        Sentiment score in [-1.0, 1.0]
    """
    if not text or not text.strip():
        return 0.0

    text = text.strip()
    base_score = 0.0

    if market == "CN":
        # Chinese: SnowNLP, fallback to keyword-based
        if SENTIMENT_MODELS.get("snownlp", True):
            base_score = _analyze_snownlp(text)
    else:
        # English: VADER + TextBlob ensemble
        vader_score = 0.0
        if SENTIMENT_MODELS.get("vader", True):
            vader_score = _analyze_vader(text)

        tb_score = 0.0
        if SENTIMENT_MODELS.get("textblob", True):
            tb_score = _analyze_textblob(text)

        base_score = (vader_score + tb_score) / 2.0

    # Apply financial lexicon boost
    final_score = _apply_financial_lexicon(text, market, base_score)

    return final_score


def analyze_all_items(items: List[Dict], market: str = "US") -> List[Dict]:
    """Run sentiment analysis on a batch of articles/posts.

    Mutates each item dict by adding/updating 'sentiment' key.
    Handles text fields based on item type.

    Args:
        items: List of article/post dicts with 'text' key
        market: Market code for language selection

    Returns:
        Same list with enriched 'sentiment' values
    """
    for item in items:
        # Determine which text to analyze
        text = item.get("text", "")
        comments_text = item.get("comments_text", "")

        if comments_text:
            combined_text = f"{text} {comments_text}".strip()
        else:
            combined_text = text

        if not combined_text:
            item["sentiment"] = 0.0
            continue

        # Run analysis
        score = analyze_text(combined_text, market=market)

        # Weight by source authority if available
        followers = item.get("followers", 0)
        if followers > 10000:
            # Weight adjustment for high-follower accounts (less extreme)
            score = score * 0.8 + (score * min(followers / 1000000, 0.2))

        item["sentiment"] = max(-1.0, min(1.0, score))

        # Add sentiment label
        if score >= 0.35:
            item["sentiment_label"] = "positive"
        elif score <= -0.35:
            item["sentiment_label"] = "negative"
        else:
            item["sentiment_label"] = "neutral"

    return items


# ═══════════════════════════════════════════════════════════════════════
# Financial BERT (optional, heavier model)
# ═══════════════════════════════════════════════════════════════════════

def analyze_finbert(text: str) -> float:
    """Financial BERT sentiment (requires torch + transformers + finbert).

    Returns sentiment in [-1, 1].
    """
    if not SENTIMENT_MODELS.get("finbert", False):
        return 0.0

    try:
        from transformers import pipeline
        nlp = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )
        result = nlp(text[:512])[0]  # BERT 512 token limit
        label = result["label"]
        score = result["score"]
        if label == "positive":
            return score
        elif label == "negative":
            return -score
        else:
            return 0.0
    except Exception as e:
        logger.debug("FinBERT error: %s", e)
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
# Batch analysis with FinBERT (for when it's enabled)
# ═══════════════════════════════════════════════════════════════════════


def batch_finbert_analysis(items: List[Dict]) -> List[Dict]:
    """Batch sentiment analysis using FinBERT for items with English text.

    Only runs if FinBERT is enabled in config.
    """
    if not SENTIMENT_MODELS.get("finbert", False):
        return items

    try:
        from transformers import pipeline
        nlp = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            max_length=512,
            truncation=True,
        )

        for item in items:
            if item.get("language") != "en":
                continue
            text = item.get("text", "")[:512]
            if not text:
                continue
            try:
                result = nlp(text)[0]
                label = result["label"]
                score = result["score"]
                if label == "positive":
                    item["sentiment_finbert"] = score
                elif label == "negative":
                    item["sentiment_finbert"] = -score
                else:
                    item["sentiment_finbert"] = 0.0
            except Exception:
                continue
    except Exception as e:
        logger.debug("FinBERT batch error: %s", e)

    return items
