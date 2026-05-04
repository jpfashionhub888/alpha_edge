# models/sentiment_model.py

"""
Financial sentiment analysis.
Tries to use FinBERT (transformers) for best accuracy.
Falls back to a keyword-based scorer if torch/transformers
are unavailable (e.g. Windows DLL issue).
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try loading transformers — it depends on torch internally
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers loaded — FinBERT sentiment enabled")
except Exception as e:
    logger.warning(
        f"Transformers/torch not available ({type(e).__name__}). "
        f"Using keyword-based sentiment fallback instead."
    )


# ---------------------------------------------------------------------------
# Keyword-based fallback scorer (no torch needed)
# ---------------------------------------------------------------------------
_POSITIVE_WORDS = {
    'beat', 'beats', 'surpass', 'record', 'growth', 'profit',
    'gain', 'gains', 'rally', 'surge', 'soar', 'rise', 'rises',
    'upgrade', 'upgraded', 'buy', 'outperform', 'strong', 'bullish',
    'positive', 'revenue', 'earnings', 'exceeded', 'higher', 'boost',
    'partnership', 'deal', 'win', 'launch', 'innovative', 'expand',
}

_NEGATIVE_WORDS = {
    'miss', 'misses', 'missed', 'loss', 'losses', 'decline',
    'fall', 'falls', 'drop', 'drops', 'crash', 'plunge', 'sink',
    'downgrade', 'downgraded', 'sell', 'underperform', 'weak', 'bearish',
    'negative', 'lawsuit', 'investigation', 'fraud', 'recall', 'lower',
    'cut', 'layoff', 'layoffs', 'bankruptcy', 'debt', 'risk', 'warning',
    'disappointing', 'concern', 'concerns', 'trouble', 'fail', 'fails',
}


def _keyword_sentiment(text):
    """
    Simple keyword-based sentiment scorer.
    Returns float from -1.0 (very negative) to +1.0 (very positive).
    """
    if not text:
        return 0.0

    words = text.lower().split()
    pos = sum(1 for w in words if w.strip('.,!?;:') in _POSITIVE_WORDS)
    neg = sum(1 for w in words if w.strip('.,!?;:') in _NEGATIVE_WORDS)
    total = pos + neg

    if total == 0:
        return 0.0

    return (pos - neg) / total


class SentimentAnalyzer:
    """
    Financial sentiment analysis.
    Uses FinBERT when torch is available,
    keyword scoring otherwise.
    """

    def __init__(self):
        self.model = None
        self.loaded = False
        self.using_fallback = not TRANSFORMERS_AVAILABLE

    def load_model(self):
        """Load the FinBERT sentiment model."""

        if self.loaded:
            return

        if not TRANSFORMERS_AVAILABLE:
            self.using_fallback = True
            self.loaded = True
            print("   ⚠️  Using keyword-based sentiment (torch unavailable)")
            return

        print("\n   Loading sentiment model...")
        print("   (First run will download ~250MB model)")

        try:
            self.model = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                top_k=None
            )
            self.loaded = True
            self.using_fallback = False
            print("   ✅ FinBERT sentiment model loaded")
        except Exception as e:
            logger.warning(f"FinBERT load failed: {e}. Using keyword fallback.")
            self.using_fallback = True
            self.loaded = True

    def analyze_text(self, text):
        """
        Analyze a single text and return sentiment score.
        Returns float from -1 (very negative) to +1 (very positive).
        """

        if not self.loaded:
            self.load_model()

        if not text or len(text.strip()) == 0:
            return 0.0

        # Use keyword fallback if torch isn't available
        if self.using_fallback:
            return _keyword_sentiment(text)

        try:
            text = text[:512]
            result = self.model(text)[0]

            score = 0.0
            for item in result:
                label = item['label'].lower()
                conf = item['score']

                if label == 'positive':
                    score += conf
                elif label == 'negative':
                    score -= conf

            return score

        except Exception as e:
            logger.warning(f"Sentiment error: {e}")
            return _keyword_sentiment(text)

    def analyze_articles(self, articles):
        """
        Analyze a list of news articles.
        Returns aggregate sentiment score.
        """

        if not self.loaded:
            self.load_model()

        if not articles or len(articles) == 0:
            return {
                'sentiment_score': 0.0,
                'num_articles': 0,
                'positive_pct': 0.0,
                'negative_pct': 0.0,
                'avg_magnitude': 0.0,
            }

        scores = []
        for article in articles:
            title   = article.get('title', '')
            summary = article.get('summary', '')
            text    = f"{title}. {summary}"
            score   = self.analyze_text(text)
            scores.append(score)

        scores = np.array(scores)
        n = len(scores)

        positive = np.sum(scores > 0.1) / n
        negative = np.sum(scores < -0.1) / n
        avg_mag  = np.mean(np.abs(scores))

        return {
            'sentiment_score': float(np.mean(scores)),
            'num_articles': n,
            'positive_pct': float(positive),
            'negative_pct': float(negative),
            'avg_magnitude': float(avg_mag),
        }

    def get_sentiment_for_stocks(self, news_dict):
        """
        Get sentiment for all stocks.
        Returns dict of symbol -> sentiment data.
        """

        if not self.loaded:
            self.load_model()

        mode = "keyword" if self.using_fallback else "FinBERT"
        print(f"\n🧠 Analyzing sentiment ({mode})...")
        results = {}

        for symbol, articles in news_dict.items():
            print(f"   {symbol}...", end=" ")
            sentiment = self.analyze_articles(articles)
            results[symbol] = sentiment

            score = sentiment['sentiment_score']
            n     = sentiment['num_articles']

            if score > 0.1:
                emoji = "🟢"
            elif score < -0.1:
                emoji = "🔴"
            else:
                emoji = "⚪"

            print(
                f"{emoji} score: {score:+.3f}"
                f" ({n} articles)"
            )

        return results
