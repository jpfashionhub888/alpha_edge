# insider_tracker.py
# ALPHAEDGE - Insider Trading Tracker
# Uses SEC EDGAR official API (always free)
# https://efts.sec.gov/LATEST/search-index

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import json

logger = logging.getLogger(__name__)

# SEC EDGAR - Official Free API
SEC_COMPANY_URL = "https://data.sec.gov/submissions/CIK{}.json"
SEC_SEARCH_URL  = "https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22&dateRange=custom&startdt={start}&enddt={end}&forms=4"
TICKER_URL      = "https://www.sec.gov/files/company_tickers.json"

HEADERS = {
    'User-Agent': 'AlphaEdge Trading contact@alphaedge.com',
    'Accept-Encoding': 'gzip, deflate',
    'Host': 'data.sec.gov'
}


class InsiderTracker:
    """
    Tracks insider trading via SEC EDGAR.
    Official government data - always free.
    """

    def __init__(self):
        self.ticker_to_cik = {}
        self._load_ticker_map()

    def _load_ticker_map(self):
        """Load ticker to CIK mapping from SEC."""
        try:
            response = requests.get(
                TICKER_URL,
                headers={
                    'User-Agent': 'AlphaEdge contact@alphaedge.com'
                },
                timeout=15
            )
            if response.status_code == 200:
                data = response.json()
                for key, val in data.items():
                    ticker = val.get('ticker', '').upper()
                    cik    = str(val.get('cik_str', '')).zfill(10)
                    if ticker:
                        self.ticker_to_cik[ticker] = cik
                print(f"   Loaded {len(self.ticker_to_cik)} tickers from SEC ✅")
        except Exception as e:
            logger.warning(f"Ticker map load failed: {e}")

    def get_cik(self, symbol):
        """Get CIK for a ticker symbol."""
        return self.ticker_to_cik.get(symbol.upper(), None)

    def get_insider_trades(self, symbol, days_back=30):
        """
        Get Form 4 insider trades from SEC EDGAR.
        Returns list of recent transactions.
        """
        try:
            cik = self.get_cik(symbol)
            if not cik:
                return []

            # Get company submissions
            url      = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = requests.get(
                url,
                headers={
                    'User-Agent': 'AlphaEdge contact@alphaedge.com'
                },
                timeout=15
            )

            if response.status_code != 200:
                return []

            data     = response.json()
            filings  = data.get('filings', {}).get('recent', {})

            forms       = filings.get('form', [])
            dates       = filings.get('filingDate', [])
            accessions  = filings.get('accessionNumber', [])

            cutoff_date = datetime.now() - timedelta(days=days_back)
            trades      = []

            for i, form in enumerate(forms):
                if form == '4':
                    try:
                        filing_date = datetime.strptime(
                            dates[i], '%Y-%m-%d'
                        )
                        if filing_date >= cutoff_date:
                            trades.append({
                                'date'     : dates[i],
                                'form'     : form,
                                'accession': accessions[i],
                                'symbol'   : symbol,
                            })
                    except Exception:
                        continue

            return trades

        except Exception as e:
            logger.warning(f"SEC fetch failed for {symbol}: {e}")
            return []

    def get_insider_score(self, symbol, days_back=30):
        """
        Get insider trading score.
        Based on number of Form 4 filings recently.
        More filings = more insider activity.
        """
        try:
            trades = self.get_insider_trades(symbol, days_back)
            count  = len(trades)

            if count == 0:
                return 0.0
            elif count >= 5:
                return 0.15  # High insider activity
            elif count >= 3:
                return 0.10  # Moderate activity
            elif count >= 1:
                return 0.05  # Some activity
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Score failed for {symbol}: {e}")
            return 0.0

    def get_bulk_scores(self, symbols, days_back=30):
        """Get insider scores for multiple symbols."""
        scores = {}
        print(f"\n   Checking insider activity for {len(symbols)} stocks...")

        for symbol in symbols:
            try:
                score = self.get_insider_score(symbol, days_back)
                scores[symbol] = score

                if score >= 0.10:
                    print(f"   {symbol}: +{score:.2f} HIGH insider activity 🟢")
                elif score >= 0.05:
                    print(f"   {symbol}: +{score:.2f} Some insider activity 🟡")

                time.sleep(0.1)  # Be respectful to SEC servers

            except Exception as e:
                scores[symbol] = 0.0

        active = sum(1 for s in scores.values() if s > 0)
        print(f"   Insider activity found: {active}/{len(symbols)} stocks")
        return scores

    def get_recent_big_buys(self, watchlist, days_back=7):
        """
        Find stocks in watchlist with heavy insider activity.
        Returns list of high-activity symbols.
        """
        print(f"\n   Scanning for insider activity in last {days_back} days...")
        hot_stocks = []

        for symbol in watchlist:
            trades = self.get_insider_trades(symbol, days_back)
            if len(trades) >= 2:
                hot_stocks.append({
                    'symbol': symbol,
                    'count' : len(trades),
                })
            time.sleep(0.1)

        hot_stocks.sort(key=lambda x: x['count'], reverse=True)

        if hot_stocks:
            print(f"\n   HOT INSIDER STOCKS:")
            for stock in hot_stocks[:5]:
                print(
                    f"   {stock['symbol']}: "
                    f"{stock['count']} Form 4 filings"
                )

        return [s['symbol'] for s in hot_stocks]


if __name__ == '__main__':
    print("\nTesting SEC EDGAR Insider Tracker...")
    tracker = InsiderTracker()

    test_symbols = ['AAPL', 'MSFT', 'JPM', 'XOM', 'NVDA']

    print("\n--- Individual Scores ---")
    for symbol in test_symbols:
        score  = tracker.get_insider_score(symbol, days_back=30)
        trades = tracker.get_insider_trades(symbol, days_back=30)
        print(f"{symbol}: Score={score:+.2f} | Form4 Filings={len(trades)}")

    print("\n--- Hot Insider Stocks ---")
    hot = tracker.get_recent_big_buys(test_symbols, days_back=14)
    print(f"Active: {hot}")