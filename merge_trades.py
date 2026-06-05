# merge_trades.py
"""
Merges stock (Alpaca) and crypto (Gate.io) paper trades
into a single logs/paper_trades.json that the dashboard reads.

Run automatically via cron every 5 minutes:
  */5 * * * * cd /root/alpha_edge && /root/alpha_edge/venv/bin/python merge_trades.py

Or run manually:
  python merge_trades.py
"""

import json
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── File paths ────────────────────────────────────────────────────────
STOCK_TRADES_FILE  = 'logs/paper_trades.json'
CRYPTO_TRADES_FILE = 'logs/gateio_paper_trades.json'
MERGED_FILE        = 'logs/paper_trades_merged.json'
DASHBOARD_FILE     = 'logs/paper_trades.json'   # what dashboard reads


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f'Failed to load {path}: {e}')
        return default


def normalize_trade(trade: dict, source: str) -> dict:
    """Ensure all trades have consistent fields regardless of source."""
    return {
        'action'      : trade.get('action', ''),
        'symbol'      : trade.get('symbol', ''),
        'shares'      : trade.get('shares', 0),
        'entry_price' : trade.get('entry_price', 0),
        'exit_price'  : trade.get('exit_price', trade.get('entry_price', 0)),
        'pnl'         : trade.get('pnl', 0),
        'pnl_pct'     : trade.get('pnl_pct', 0),
        'timestamp'   : trade.get('timestamp', ''),
        'reason'      : trade.get('reason', ''),
        'source'      : source,   # 'stocks' or 'crypto'
    }


def merge_trades():
    """Merge stock and crypto paper trades into unified file."""

    # Load both trade files
    stock_data  = load_json(STOCK_TRADES_FILE, {})
    crypto_data = load_json(CRYPTO_TRADES_FILE, {})

    # Extract trade histories
    stock_history  = stock_data.get('trade_history', [])
    crypto_history = crypto_data.get('trade_history', [])

    # Normalize and tag by source
    stock_trades  = [normalize_trade(t, 'stocks') for t in stock_history]
    crypto_trades = [normalize_trade(t, 'crypto') for t in crypto_history]

    # Combine
    all_trades = stock_trades + crypto_trades

    # Sort by timestamp
    try:
        all_trades.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    except Exception:
        pass

    # Combine positions
    stock_positions  = stock_data.get('positions', {})
    crypto_positions = crypto_data.get('positions', {})
    all_positions    = {**stock_positions, **crypto_positions}

    # Combine capital (sum both)
    stock_capital  = stock_data.get('capital', 10000)
    crypto_capital = crypto_data.get('capital', 10000)

    # Build merged structure matching what dashboard expects
    merged = {
        'capital'        : stock_capital + crypto_capital,
        'starting_capital': (
            stock_data.get('starting_capital', 10000) +
            crypto_data.get('starting_capital', 10000)
        ),
        'positions'      : all_positions,
        'trade_history'  : all_trades,
        'last_updated'   : datetime.now().isoformat(),
        'sources'        : {
            'stocks' : {
                'capital'  : stock_capital,
                'trades'   : len(stock_trades),
                'positions': len(stock_positions),
            },
            'crypto' : {
                'capital'  : crypto_capital,
                'trades'   : len(crypto_trades),
                'positions': len(crypto_positions),
            },
        },
    }

    # Save merged file
    os.makedirs('logs', exist_ok=True)
    with open(MERGED_FILE, 'w') as f:
        json.dump(merged, f, indent=2)

    # Also update the main dashboard file
    # (keeps original stock-only as backup)
    if os.path.exists(STOCK_TRADES_FILE):
        # Backup stock-only file
        with open('logs/paper_trades_stocks_only.json', 'w') as f:
            json.dump(stock_data, f, indent=2)

    with open(DASHBOARD_FILE, 'w') as f:
        json.dump(merged, f, indent=2)

    n_stock  = len(stock_trades)
    n_crypto = len(crypto_trades)
    logger.info(
        f'Merged: {n_stock} stock trades + {n_crypto} crypto trades'
        f' | stock capital=${stock_capital:,.2f}'
        f' | crypto capital=${crypto_capital:,.2f}'
        f' | total=${stock_capital + crypto_capital:,.2f}'
    )

    return merged


if __name__ == '__main__':
    merge_trades()
