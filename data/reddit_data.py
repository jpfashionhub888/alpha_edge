# data/reddit_data.py

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)


class RedditFetcher:
    """
    Fetches stock discussion from Reddit.
    Uses free public JSON API. No API key needed.
    """

    def __init__(self):
        self.headers = {
            'User-Agent': 'AlphaEdge/1.0'
        }
        self.subreddits = [
            'wallstreetbets',
            'stocks',
            'investing',
            'stockmarket',
        ]

    def fetch_mentions(self, symbol, limit=50):
        """Search Reddit for stock mentions."""

        all_posts = []

        for sub in self.subreddits:
            try:
                url = (
                    f"https://www.reddit.com/r/{sub}"
                    f"/search.json"
                    f"?q={symbol}"
                    f"&sort=new"
                    f"&limit={limit}"
                    f"&t=week"
                    f"&restrict_sr=on"
                )

                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=10
                )

                if response.status_code != 200:
                    continue

                data = response.json()
                posts = data.get('data', {}).get('children', [])

                for post in posts:
                    post_data = post.get('data', {})

                    all_posts.append({
                        'title': post_data.get('title', ''),
                        'summary': post_data.get(
                            'selftext', ''
                        )[:500],
                        'source': f"reddit/{sub}",
                        'symbol': symbol,
                        'score': post_data.get('score', 0),
                        'comments': post_data.get(
                            'num_comments', 0
                        ),
                        'published': datetime.fromtimestamp(
                            post_data.get('created_utc', 0)
                        ),
                    })

                time.sleep(2)

            except Exception as e:
                logger.warning(
                    f"Reddit error for {sub}/{symbol}: {e}"
                )

        # Sort by score (most upvoted first)
        all_posts.sort(
            key=lambda x: x.get('score', 0),
            reverse=True
        )

        return all_posts

    def fetch_all(self, symbols, limit=25):
        """Fetch Reddit data for all symbols."""

        print(
            f"\n📱 Fetching Reddit data"
            f" for {len(symbols)} symbols..."
        )
        all_data = {}

        for i, symbol in enumerate(symbols):
            n = i + 1
            total = len(symbols)
            print(
                f"   [{n}/{total}] r/wallstreetbets"
                f" + others for {symbol}...",
                end=" "
            )

            posts = self.fetch_mentions(symbol, limit)
            all_data[symbol] = posts
            print(f"✅ {len(posts)} posts")

            time.sleep(2)

        return all_data

    def get_buzz_score(self, posts):
        """
        Calculate how much buzz a stock has on Reddit.
        High buzz + positive sentiment = strong signal.
        """

        if not posts or len(posts) == 0:
            return {
                'buzz_score': 0.0,
                'total_posts': 0,
                'total_upvotes': 0,
                'total_comments': 0,
                'avg_score': 0.0,
            }

        total_score = sum(p.get('score', 0) for p in posts)
        total_comments = sum(
            p.get('comments', 0) for p in posts
        )
        n = len(posts)

        buzz = (total_score + total_comments * 2) / max(n, 1)

        return {
            'buzz_score': buzz,
            'total_posts': n,
            'total_upvotes': total_score,
            'total_comments': total_comments,
            'avg_score': total_score / n,
        }