# scripts/fetch_test_fixtures.py
"""
Fetch and save real historical OHLCV data from Yahoo Finance for use in tests.

Run this script once to populate tests/fixtures/:
    python scripts/fetch_test_fixtures.py

The resulting CSV files are committed to the repo so tests never need
internet access at runtime. Re-run this script to refresh the data.

Tickers chosen:
  AAPL  — large-cap tech, deep history, high liquidity
  NVDA  — high-volatility momentum stock (tests regime changes)
  SPY   — S&P 500 ETF (benchmark / market regime)
  BTC-USD — crypto proxy via Yahoo (tests crypto path in feature engine)

Period: 2020-01-01 → 2025-12-31 (6 years, covers bull/bear/COVID/rate-hike regimes)
"""

import os
import sys
import json
from datetime import datetime

import yfinance as yf
import pandas as pd

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')

TICKERS = {
    'AAPL'   : {'start': '2020-01-01', 'end': '2025-12-31', 'desc': 'Apple Inc — large-cap tech'},
    'NVDA'   : {'start': '2020-01-01', 'end': '2025-12-31', 'desc': 'NVIDIA — high-vol momentum'},
    'SPY'    : {'start': '2020-01-01', 'end': '2025-12-31', 'desc': 'S&P 500 ETF — benchmark'},
    'BTC-USD': {'start': '2020-01-01', 'end': '2025-12-31', 'desc': 'Bitcoin USD — crypto proxy'},
}


def fetch_and_save(ticker: str, config: dict) -> pd.DataFrame:
    print(f"  Downloading {ticker} ({config['desc']})...")
    df = yf.download(
        ticker,
        start=config['start'],
        end=config['end'],
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        print(f"  [WARN] No data returned for {ticker}")
        return df

    # Flatten multi-level columns if present (yfinance >= 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df.index.name = 'date'

    # Drop any all-NaN rows
    df = df.dropna(how='all')

    # Save as CSV
    filename = f"{ticker.replace('-', '_').lower()}_daily.csv"
    filepath = os.path.join(FIXTURES_DIR, filename)
    df.to_csv(filepath)

    print(f"  OK  {ticker}: {len(df)} rows -> {filepath}")
    return df



def save_manifest(results: dict):
    """Write a manifest.json so tests can programmatically discover fixtures."""
    manifest = {
        'generated_at': datetime.now().isoformat(),
        'source'      : 'Yahoo Finance via yfinance',
        'fixtures'    : {}
    }
    for ticker, df in results.items():
        if df is not None and not df.empty:
            manifest['fixtures'][ticker] = {
                'file'      : f"{ticker.replace('-', '_').lower()}_daily.csv",
                'rows'      : len(df),
                'start_date': str(df.index[0].date()),
                'end_date'  : str(df.index[-1].date()),
                'columns'   : list(df.columns),
            }

    manifest_path = os.path.join(FIXTURES_DIR, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✅ Manifest saved: {manifest_path}")


def main():
    print("=" * 60)
    print("  AlphaEdge — Fetching Real Historical Test Fixtures")
    print("=" * 60)
    print(f"  Output: {os.path.abspath(FIXTURES_DIR)}\n")

    os.makedirs(FIXTURES_DIR, exist_ok=True)

    results = {}
    for ticker, config in TICKERS.items():
        try:
            df = fetch_and_save(ticker, config)
            results[ticker] = df
        except Exception as e:
            print(f"  ❌ {ticker} failed: {e}")
            results[ticker] = None

    save_manifest(results)

    print("\n" + "=" * 60)
    total = sum(len(df) for df in results.values() if df is not None and not df.empty)
    print(f"  Done! {total:,} total rows across {len(results)} tickers")
    print("  Commit tests/fixtures/ to version control.")
    print("=" * 60)


if __name__ == '__main__':
    main()
