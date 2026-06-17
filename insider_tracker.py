# insider_tracker.py
# ALPHAEDGE - Insider Trading Tracker
# Uses SEC EDGAR official API (always free)
#
# FIX P0-4: get_insider_score() now always returns 0.0.
#   Reason: Form 4 filing count cannot distinguish buys from sells.
#   A CEO dumping 5M shares produces the same positive boost as a CEO
#   buying 5M shares — a directionally wrong signal ~50% of the time.
#   The score has been removed from combined signal calculation in main.py.
#
#   Insider activity is still fetched and surfaced for DISPLAY only via
#   get_insider_activity() — the dashboard can show the raw filing count
#   so a human can judge direction. It just no longer moves the BUY signal.

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
    'User-Agent'     : 'AlphaEdge Trading contact@alphaedge.com',
    'Accept-Encoding': 'gzip, deflate',
    'Host'           : 'data.sec.gov'
}


class InsiderTracker:
    """
    Tracks insider trading via SEC EDGAR.
    Official government data - always free.

    NOTE: Filing counts (Form 4) do not distinguish buys from sells.
    Activity data is surfaced for human review only — it does NOT
    modify signal scores (see P0-4 fix notes above).
    """

    def __init__(self):
        self.ticker_to_cik = {}
        self._load_ticker_map()

    def _load_ticker_map(self):
        """Load ticker to CIK mapping from SEC."""
        try:
            response = requests.get(
                TICKER_URL,
                headers={'User-Agent': 'AlphaEdge contact@alphaedge.com'},
                timeout=15
            )
            if response.status_code == 200:
                data = response.json()
                for key, val in data.items():
                    ticker = val.get('ticker', '').upper()
                    cik    = str(val.get('cik_str', '')).zfill(10)
                    if ticker:
                        self.ticker_to_cik[ticker] = cik
                print(f"  Loaded {len(self.ticker_to_cik)} tickers from SEC ✅")
        except Exception as e:
            logger.warning(f"Ticker map load failed: {e}")

    def get_cik(self, symbol):
        """Get CIK for a ticker symbol."""
        return self.ticker_to_cik.get(symbol.upper(), None)

    def get_insider_trades(self, symbol, days_back=30):
        """
        Get Form 4 insider trades from SEC EDGAR.
        Returns list of recent filings (direction unknown — buys and sells mixed).
        """
        try:
            cik = self.get_cik(symbol)
            if not cik:
                return []

            url      = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = requests.get(
                url,
                headers={'User-Agent': 'AlphaEdge contact@alphaedge.com'},
                timeout=15
            )

            if response.status_code != 200:
                return []

            data      = response.json()
            filings   = data.get('filings', {}).get('recent', {})
            forms      = filings.get('form', [])
            dates      = filings.get('filingDate', [])
            accessions = filings.get('accessionNumber', [])

            cutoff_date = datetime.now() - timedelta(days=days_back)
            trades      = []

            for i, form in enumerate(forms):
                if form == '4':
                    try:
                        filing_date = datetime.strptime(dates[i], '%Y-%m-%d')
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
        DEPRECATED FOR SIGNAL USE — always returns 0.0.

        P0-4 FIX: Form 4 count cannot distinguish buys from sells.
        Returning a positive boost for any filing is directionally wrong
        ~50% of the time and was removed from signal generation.

        Use get_insider_activity() to get raw filing count for display.
        """
        return 0.0  # P0-4: removed from combined score

    def get_insider_activity(self, symbol, days_back=30):
        """
        Returns raw Form 4 filing count for DISPLAY purposes only.
        Does not imply direction (could be buys or sells).
        """
        try:
            trades = self.get_insider_trades(symbol, days_back)
            return len(trades)
        except Exception:
            return 0

    def get_bulk_scores(self, symbols, days_back=30):
        """
        Get insider scores for multiple symbols.

        P0-4 FIX: All scores returned are 0.0 — the boost has been removed.
        The dict is still returned so main.py call sites don't break.
        Activity counts are logged for awareness but don't affect signals.
        """
        scores = {}
        print(f"\n  Checking insider activity for {len(symbols)} stocks...")

        for symbol in symbols:
            scores[symbol] = 0.0  # P0-4: always 0.0 — directional unknown

            # Still fetch for display/logging awareness
            try:
                count = self.get_insider_activity(symbol, days_back)
                if count >= 3:
                    print(f"  {symbol}: {count} Form 4 filings (display only — direction unknown) 📋")
                time.sleep(0.1)
            except Exception as e:
                logger.debug(f"Activity check failed for {symbol}: {e}")

        active = sum(
            1 for sym in symbols
            if self.get_insider_activity(sym, days_back) > 0
        )
        print(f"  Insider Form 4 activity found: {active}/{len(symbols)} stocks (informational only)")
        return scores

    def get_recent_big_buys(self, watchlist, days_back=7):
        """
        Find stocks with heavy Form 4 filing activity.
        Returns list of high-activity symbols for DISPLAY only.
        Note: 'buys' in the function name is a misnomer — direction is unknown.
        """
        print(f"\n  Scanning for insider Form 4 activity in last {days_back} days...")

        hot_stocks = []
        for symbol in watchlist:
            trades = self.get_insider_trades(symbol, days_back)
            if len(trades) >= 2:
                hot_stocks.append({'symbol': symbol, 'count': len(trades)})
            time.sleep(0.1)

        hot_stocks.sort(key=lambda x: x['count'], reverse=True)

        if hot_stocks:
            print(f"\n  HIGH FORM 4 ACTIVITY (direction unknown):")
            for stock in hot_stocks[:5]:
                print(f"  {stock['symbol']}: {stock['count']} filings")

        return [s['symbol'] for s in hot_stocks]


if __name__ == '__main__':
    print("\nTesting SEC EDGAR Insider Tracker...")
    tracker      = InsiderTracker()
    test_symbols = ['AAPL', 'MSFT', 'JPM', 'XOM', 'NVDA']

    print("\n--- Activity Counts (display only, direction unknown) ---")
    for symbol in test_symbols:
        count = tracker.get_insider_activity(symbol, days_back=30)
        score = tracker.get_insider_score(symbol, days_back=30)
        print(f"{symbol}: Score={score:+.2f} (always 0) | Form4 Filings={count}")

    print("\n--- High-Activity Stocks ---")
    hot = tracker.get_recent_big_buys(test_symbols, days_back=14)
    print(f"Active: {hot}")
