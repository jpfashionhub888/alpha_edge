# models/sentiment_model.py

import pandas as pd
import numpy as np
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Financial sentiment analysis using free
    pretrained FinBERT model. Runs locally.
    """

    def __init__(self):
        self.model = None
        self.loaded = False

    def load_model(self):
        """Load the sentiment model. First run downloads it."""

        if self.loaded:
            return

        print("\n   Loading sentiment model...")
        print("   (First run will download ~250MB model)")

        self.model = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            top_k=None
        )

        self.loaded = True
        print("   ✅ Sentiment model loaded")

    def analyze_text(self, text):
        """
        Analyze a single text and return sentiment score.
        Returns float from -1 (very negative) to +1 (very positive).
        """

        if not self.loaded:
            self.load_model()

        if not text or len(text.strip()) == 0:
            return 0.0

        try:
            # Truncate to 512 chars for model limit
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
            return 0.0

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
            title = article.get('title', '')
            summary = article.get('summary', '')
            text = f"{title}. {summary}"

            score = self.analyze_text(text)
            scores.append(score)

        scores = np.array(scores)
        n = len(scores)

        positive = np.sum(scores > 0.1) / n
        negative = np.sum(scores < -0.1) / n
        avg_mag = np.mean(np.abs(scores))

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

        print("\n🧠 Analyzing sentiment...")
        results = {}

        for symbol, articles in news_dict.items():
            print(f"   {symbol}...", end=" ")
            sentiment = self.analyze_articles(articles)
            results[symbol] = sentiment

            score = sentiment['sentiment_score']
            n = sentiment['num_articles']

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