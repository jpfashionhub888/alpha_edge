# data/news_data.py

import requests
import feedparser
import pandas as pd
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)


class NewsFetcher:
    """
    Fetches financial news from free RSS feeds.
    No API key required.
    """

    def __init__(self):
        self.feeds = {
            'yahoo': (
                'https://finance.yahoo.com/rss/headline'
                '?s={symbol}'
            ),
            'google': (
                'https://news.google.com/rss/search'
                '?q={symbol}+stock&hl=en-US&gl=US'
                '&ceid=US:en'
            ),
        }
        self.cache = {}

    def fetch_news(self, symbol, days_back=7):
        """Fetch recent news for a symbol."""

        all_articles = []
        clean_symbol = symbol.replace('/', '')

        for source, url_template in self.feeds.items():
            try:
                url = url_template.format(symbol=clean_symbol)
                feed = feedparser.parse(url)

                for entry in feed.entries:
                    published = None
                    if hasattr(entry, 'published_parsed'):
                        if entry.published_parsed:
                            published = datetime(
                                *entry.published_parsed[:6]
                            )

                    article = {
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'source': source,
                        'symbol': symbol,
                        'published': published,
                    }
                    all_articles.append(article)

                time.sleep(0.5)

            except Exception as e:
                logger.warning(
                    f"Error fetching {source} for {symbol}: {e}"
                )

        # Filter to recent articles
        cutoff = datetime.now() - timedelta(days=days_back)
        recent = []
        for article in all_articles:
            if article['published'] is None:
                recent.append(article)
            elif article['published'] > cutoff:
                recent.append(article)

        self.cache[symbol] = recent

        return recent

    def fetch_all(self, symbols, days_back=7):
        """Fetch news for all symbols."""

        print(f"\n📰 Fetching news for {len(symbols)} symbols...")
        all_news = {}

        for i, symbol in enumerate(symbols):
            print(
                f"   [{i+1}/{len(symbols)}] {symbol}...",
                end=" "
            )
            articles = self.fetch_news(symbol, days_back)
            all_news[symbol] = articles
            print(f"✅ {len(articles)} articles")
            time.sleep(1)

        return all_news