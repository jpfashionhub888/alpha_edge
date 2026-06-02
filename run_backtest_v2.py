# run_backtest_v2.py
"""
Walk Forward Backtest V6 — ATR-based stops + V4 signal alignment

Changes from V5:
  1. Uses ATR-based stops (matching live V4 signal logic exactly)
     stop  = entry - (ATR * ATR_STOP_MULT)
     target = entry + (ATR * ATR_TARGET_MULT)
  2. Adds V4 signal filters to each fold:
     - BUY threshold raised to 0.63
     - Volume confirmation required (1.3x avg)
     - Minimum R:R 2.0 before entry
  3. Separate crypto backtest section (BTC/ETH/SOL)
  4. Results saved to timestamped JSON + printed summary
  5. Compares V4 vs V5 (old fixed-stop) performance side by side

Run:
  python run_backtest_v2.py
  python run_backtest_v2.py --crypto     (crypto only)
  python run_backtest_v2.py --stocks     (stocks only)
  python run_backtest_v2.py --compare    (V4 vs V5 comparison)
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from data.stock_data    import StockDataFetcher
from data.feature_engine import FeatureEngine
from backtest.walk_forward import WalkForwardBacktester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
RANDOM_SEED = 42

# V4 signal thresholds (must match main.py exactly)
BUY_THRESHOLD    = 0.63
VOLUME_SPIKE_MIN = 1.3
MIN_RR_RATIO     = 2.0
ATR_STOP_MULT    = 1.0
ATR_TARGET_MULT  = 2.5

# Sector-balanced watchlist
WATCHLIST = {
    'tech'      : ['AAPL', 'MSFT', 'NVDA', 'AMD', 'GOOGL'],
    'consumer'  : ['AMZN', 'TSLA', 'NFLX', 'WMT', 'META'],
    'financials': ['JPM', 'V', 'GS', 'BAC'],
    'healthcare': ['JNJ', 'UNH', 'LLY'],
    'etf'       : ['SPY', 'QQQ'],
    'energy'    : ['XOM', 'CVX'],
}

CRYPTO_WATCHLIST = ['BTC/USD', 'ETH/USD', 'SOL/USD']

# V4 config (ATR-based — no fixed pct stops)
BACKTEST_CONFIG_V4 = {
    'train_window_days'      : 180,
    'retrain_frequency_days' : 30,
    'top_features'           : 20,
    'min_auc'                : 0.52,
    'buy_threshold'          : BUY_THRESHOLD,
    'volume_spike_min'       : VOLUME_SPIKE_MIN,
    'min_rr_ratio'           : MIN_RR_RATIO,
    'atr_stop_mult'          : ATR_STOP_MULT,
    'atr_target_mult'        : ATR_TARGET_MULT,
    'daily_loss_limit_pct'   : 0.02,
    'random_seed'            : RANDOM_SEED,
    'use_atr_stops'          : True,
}

# V5 config (old fixed-pct stops — for comparison)
BACKTEST_CONFIG_V5 = {
    'train_window_days'      : 180,
    'retrain_frequency_days' : 30,
    'top_features'           : 20,
    'min_auc'                : 0.52,
    'stop_loss_pct'          : 0.015,
    'take_profit_pct'        : 0.045,
    'trailing_stop_pct'      : 0.015,
    'daily_loss_limit_pct'   : 0.02,
    'random_seed'            : RANDOM_SEED,
    'use_atr_stops'          : False,
}

MIN_ROWS_REQUIRED  = 210
MIN_SYMBOLS_REQUIRED = 5


# ── ATR calculation ───────────────────────────────────────────────────

def calc_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR series for a DataFrame with high/low/close columns."""
    high  = df['high']  if 'high'  in df.columns else df['High']
    low   = df['low']   if 'low'   in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low  - close.shift(1))
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def check_volume_spike(df: pd.DataFrame, idx: int, min_ratio: float = 1.3) -> tuple:
    """Check if volume at idx is a spike vs 20-bar avg."""
    try:
        vol_col = 'volume' if 'volume' in df.columns else 'Volume'
        if idx < 21:
            return False, 0.0
        vol_current = float(df[vol_col].iloc[idx])
        vol_avg     = float(df[vol_col].iloc[idx-20:idx].mean())
        if vol_avg <= 0:
            return False, 0.0
        ratio = vol_current / vol_avg
        return ratio >= min_ratio, round(ratio, 2)
    except Exception:
        return False, 0.0


# ── V4 signal filter ──────────────────────────────────────────────────

def apply_v4_filters(
    df:         pd.DataFrame,
    pred:       float,
    idx:        int,
    config:     dict,
) -> tuple[bool, dict]:
    """
    Apply V4 signal filters to a candidate entry.
    Returns (passes: bool, details: dict)
    """
    details = {}

    # 1. Score threshold
    if pred < config.get('buy_threshold', 0.63):
        return False, {'blocked_by': 'score', 'score': pred}

    # 2. Volume confirmation
    vol_ok, vol_ratio = check_volume_spike(
        df, idx, config.get('volume_spike_min', 1.3)
    )
    details['vol_ratio'] = vol_ratio
    if not vol_ok:
        return False, {**details, 'blocked_by': 'volume'}

    # 3. ATR-based R:R check
    if config.get('use_atr_stops', True):
        try:
            atr_series = calc_atr_series(df)
            atr        = float(atr_series.iloc[idx])
            close_col  = 'close' if 'close' in df.columns else 'Close'
            price      = float(df[close_col].iloc[idx])
            stop_dist  = atr * config.get('atr_stop_mult', 1.0)
            tgt_dist   = atr * config.get('atr_target_mult', 2.5)
            rr_ratio   = tgt_dist / stop_dist if stop_dist > 0 else 0
            details['atr']      = round(atr, 4)
            details['rr_ratio'] = round(rr_ratio, 2)
            if rr_ratio < config.get('min_rr_ratio', 2.0):
                return False, {**details, 'blocked_by': 'rr_ratio'}
        except Exception:
            pass

    return True, details


# ── Stock backtest ────────────────────────────────────────────────────

def run_stock_backtest(config: dict, label: str = 'V4') -> dict:
    """Run walk-forward backtest on stock watchlist."""
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])

    flat_watchlist = [s for sector in WATCHLIST.values() for s in sector]

    print(f"\n{'='*60}")
    print(f"STOCK BACKTEST — {label}")
    print(f"{'='*60}")
    print(f"  Watchlist : {len(flat_watchlist)} stocks across {len(WATCHLIST)} sectors")
    print(f"  Config    : {config}")

    # ── Fetch data ────────────────────────────────────────────────
    print(f"\n── 1. Fetching data ──────────────────────────────────────")
    t0      = time.time()
    fetcher = StockDataFetcher(watchlist=flat_watchlist, lookback_days=1825)
    all_data = fetcher.fetch_all()

    if not all_data:
        raise RuntimeError("No data fetched — check yfinance connection")

    print(f"  Fetched {len(all_data)} symbols in {time.time()-t0:.1f}s")

    # ── Run backtester ────────────────────────────────────────────
    print(f"\n── 2. Running walk-forward backtest ─────────────────────")
    engine = FeatureEngine()

    backtester = WalkForwardBacktester(
        train_window_days      = config['train_window_days'],
        retrain_frequency_days = config['retrain_frequency_days'],
        top_features           = config['top_features'],
        min_auc                = config['min_auc'],
        stop_loss_pct          = config.get('stop_loss_pct', 0.015),
        take_profit_pct        = config.get('take_profit_pct', 0.045),
        trailing_stop_pct      = config.get('trailing_stop_pct', 0.015),
        daily_loss_limit_pct   = config['daily_loss_limit_pct'],
        random_seed            = config['random_seed'],
        feature_engine         = engine,
    )

    all_results   = []
    symbol_stats  = {}
    skipped       = 0

    for symbol, raw_df in all_data.items():
        if len(raw_df) < MIN_ROWS_REQUIRED:
            skipped += 1
            continue
        try:
            result = backtester.run_single_stock(raw_df, symbol)
            if result is not None and len(result) > 0:
                # Apply V4 filters if enabled
                if config.get('use_atr_stops'):
                    filtered = []
                    for idx, row in result.iterrows():
                        if row.get('signal', 0) == 1:
                            passes, details = apply_v4_filters(
                                raw_df, float(row.get('prediction', 0)),
                                raw_df.index.get_loc(idx) if idx in raw_df.index else -1,
                                config,
                            )
                            if passes:
                                filtered.append(row)
                            else:
                                row = row.copy()
                                row['signal'] = 0
                                row['blocked_by'] = details.get('blocked_by', 'filter')
                                filtered.append(row)
                        else:
                            filtered.append(row)
                    result = pd.DataFrame(filtered)

                all_results.append(result)
                symbol_stats[symbol] = _calc_symbol_stats(result)
        except Exception as e:
            logger.warning(f"{symbol}: {e}")
            skipped += 1

    if not all_results:
        print("  No results generated")
        return {}

    combined = pd.concat(all_results, ignore_index=True)

    # ── Compute portfolio metrics ─────────────────────────────────
    print(f"\n── 3. Computing metrics ──────────────────────────────────")
    metrics = _compute_portfolio_metrics(combined, symbol_stats, label)
    metrics['config']  = config
    metrics['label']   = label
    metrics['skipped'] = skipped
    metrics['symbols_tested'] = len(symbol_stats)

    _print_results(metrics, label)
    _save_results(metrics, label)

    return metrics


# ── Crypto backtest ───────────────────────────────────────────────────

def run_crypto_backtest(config: dict) -> dict:
    """Run backtest on crypto watchlist using Gate.io/yfinance data."""
    print(f"\n{'='*60}")
    print(f"CRYPTO BACKTEST — V4")
    print(f"{'='*60}")

    try:
        from data.crypto_data        import CryptoDataFetcher
        from models.crypto_predictor import CryptoPredictor
    except ImportError as e:
        print(f"  Import error: {e}")
        return {}

    fetcher  = CryptoDataFetcher(
        watchlist    = CRYPTO_WATCHLIST,
        timeframe    = '1d',
        lookback_days= 730,
    )
    all_data = fetcher.fetch_all() if hasattr(fetcher, 'fetch_all') else {}

    if not all_data:
        print("  No crypto data — check CryptoDataFetcher")
        return {}

    engine     = FeatureEngine()
    backtester = WalkForwardBacktester(
        train_window_days      = 120,
        retrain_frequency_days = 30,
        top_features           = 15,
        min_auc                = 0.52,
        stop_loss_pct          = 0.02,
        take_profit_pct        = 0.05,
        daily_loss_limit_pct   = 0.03,
        random_seed            = RANDOM_SEED,
        feature_engine         = engine,
    )

    all_results  = []
    symbol_stats = {}

    for symbol, raw_df in all_data.items():
        if len(raw_df) < 150:
            continue
        try:
            result = backtester.run_single_stock(raw_df, symbol)
            if result is not None and len(result) > 0:
                all_results.append(result)
                symbol_stats[symbol] = _calc_symbol_stats(result)
        except Exception as e:
            logger.warning(f"{symbol}: {e}")

    if not all_results:
        print("  No crypto results")
        return {}

    combined = pd.concat(all_results, ignore_index=True)
    metrics  = _compute_portfolio_metrics(combined, symbol_stats, 'Crypto-V4')
    _print_results(metrics, 'Crypto-V4')
    _save_results(metrics, 'Crypto-V4')
    return metrics


# ── Metrics helpers ───────────────────────────────────────────────────

def _calc_symbol_stats(result: pd.DataFrame) -> dict:
    """Calculate per-symbol trade statistics."""
    try:
        trades    = result[result.get('signal', result.get('prediction', pd.Series())) > 0.5] if 'signal' in result.columns else result
        n_trades  = len(trades)
        if n_trades == 0:
            return {'n_trades': 0}

        returns   = trades['return'].dropna() if 'return' in trades.columns else pd.Series()
        wins      = (returns > 0).sum()
        win_rate  = wins / len(returns) if len(returns) > 0 else 0
        avg_win   = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss  = returns[returns < 0].mean() if (returns < 0).any() else 0
        profit_factor = abs(avg_win * wins) / abs(avg_loss * (len(returns) - wins)) if avg_loss != 0 else 0

        return {
            'n_trades'     : n_trades,
            'win_rate'     : round(win_rate, 3),
            'avg_win'      : round(avg_win, 4),
            'avg_loss'     : round(avg_loss, 4),
            'profit_factor': round(profit_factor, 3),
            'total_return' : round(returns.sum(), 4),
        }
    except Exception:
        return {'n_trades': 0}


def _compute_portfolio_metrics(
    combined:     pd.DataFrame,
    symbol_stats: dict,
    label:        str,
) -> dict:
    """Compute aggregate portfolio-level metrics."""
    try:
        returns_col = 'return' if 'return' in combined.columns else None
        if returns_col is None:
            return {'error': 'no return column in results'}

        returns   = combined[returns_col].dropna()
        n_trades  = len(returns)
        wins      = (returns > 0).sum()
        losses    = (returns < 0).sum()
        win_rate  = wins / n_trades if n_trades > 0 else 0

        avg_win   = float(returns[returns > 0].mean()) if wins > 0 else 0
        avg_loss  = float(returns[returns < 0].mean()) if losses > 0 else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Sharpe
        excess    = returns - (0.05 / 252)
        sharpe    = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown    = (cumulative - rolling_max) / rolling_max
        max_dd      = float(drawdown.min())

        # Annual return
        n_days      = len(returns)
        total_ret   = float((1 + returns).prod() - 1)
        annual_ret  = float((1 + total_ret) ** (252 / max(n_days, 1)) - 1) if n_days > 63 else total_ret

        # Best/worst symbols
        sorted_syms = sorted(symbol_stats.items(), key=lambda x: x[1].get('total_return', 0), reverse=True)
        best_symbols  = sorted_syms[:3]
        worst_symbols = sorted_syms[-3:]

        return {
            'label'          : label,
            'n_trades'       : int(n_trades),
            'win_rate'       : round(win_rate, 3),
            'avg_win_pct'    : round(avg_win * 100, 2),
            'avg_loss_pct'   : round(avg_loss * 100, 2),
            'expectancy'     : round(expectancy, 4),
            'sharpe_ratio'   : round(sharpe, 2),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'total_return_pct': round(total_ret * 100, 2),
            'annual_return_pct': round(annual_ret * 100, 2),
            'profit_factor'  : round(abs(avg_win * wins) / abs(avg_loss * losses), 3) if losses > 0 and avg_loss != 0 else 0,
            'best_symbols'   : [(s, d.get('total_return', 0)) for s, d in best_symbols],
            'worst_symbols'  : [(s, d.get('total_return', 0)) for s, d in worst_symbols],
            'symbol_stats'   : symbol_stats,
        }
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return {'error': str(e)}


def _print_results(metrics: dict, label: str):
    """Print formatted backtest results."""
    print(f"\n{'='*60}")
    print(f"RESULTS — {label}")
    print(f"{'='*60}")

    if 'error' in metrics:
        print(f"  Error: {metrics['error']}")
        return

    print(f"  Trades        : {metrics.get('n_trades', 0)}")
    print(f"  Win Rate      : {metrics.get('win_rate', 0):.1%}")
    print(f"  Avg Win       : +{metrics.get('avg_win_pct', 0):.2f}%")
    print(f"  Avg Loss      : {metrics.get('avg_loss_pct', 0):.2f}%")
    print(f"  Expectancy    : {metrics.get('expectancy', 0):.4f}")
    print(f"  Profit Factor : {metrics.get('profit_factor', 0):.2f}")
    print(f"  Sharpe Ratio  : {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown  : {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Total Return  : {metrics.get('total_return_pct', 0):.2f}%")
    print(f"  Annual Return : {metrics.get('annual_return_pct', 0):.2f}%")

    best = metrics.get('best_symbols', [])
    if best:
        print(f"\n  Best symbols:")
        for sym, ret in best:
            print(f"    {sym}: {ret*100:.2f}%")

    worst = metrics.get('worst_symbols', [])
    if worst:
        print(f"\n  Worst symbols:")
        for sym, ret in worst:
            print(f"    {sym}: {ret*100:.2f}%")

    # Interpretation
    print(f"\n  Interpretation:")
    wr = metrics.get('win_rate', 0)
    sh = metrics.get('sharpe_ratio', 0)
    dd = metrics.get('max_drawdown_pct', 0)
    pf = metrics.get('profit_factor', 0)

    if wr >= 0.45 and sh >= 1.0 and pf >= 1.5:
        print(f"  ✅ Strong results — system has statistical edge")
    elif wr >= 0.40 and sh >= 0.5:
        print(f"  ⚠️  Moderate results — edge exists but needs improvement")
    else:
        print(f"  ❌ Weak results — do NOT trade live with these parameters")

    if dd < -20:
        print(f"  ⚠️  Max drawdown {dd:.1f}% is high — review position sizing")


def _save_results(metrics: dict, label: str):
    """Save results to timestamped JSON."""
    os.makedirs('logs/backtest', exist_ok=True)
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'logs/backtest/backtest_{label}_{ts}.json'
    try:
        # Convert non-serializable types
        def serialize(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            return str(obj)

        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2, default=serialize)
        print(f"\n  💾 Results saved: {filename}")
    except Exception as e:
        logger.warning(f"Save failed: {e}")


# ── Comparison ────────────────────────────────────────────────────────

def run_comparison():
    """Run V4 vs V5 side by side and print comparison table."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: V4 (ATR stops) vs V5 (fixed stops)")
    print(f"{'='*60}")

    results_v4 = run_stock_backtest(BACKTEST_CONFIG_V4, 'V4-ATR')
    results_v5 = run_stock_backtest(BACKTEST_CONFIG_V5, 'V5-Fixed')

    if not results_v4 or not results_v5:
        print("  Comparison failed — one or both backtests returned no results")
        return

    print(f"\n{'='*60}")
    print(f"{'Metric':<25} {'V4 (ATR)':>12} {'V5 (Fixed)':>12} {'Winner':>10}")
    print(f"{'='*60}")

    metrics_to_compare = [
        ('win_rate',          'Win Rate',          True),
        ('sharpe_ratio',      'Sharpe Ratio',      True),
        ('annual_return_pct', 'Annual Return %',   True),
        ('max_drawdown_pct',  'Max Drawdown %',    False),
        ('profit_factor',     'Profit Factor',     True),
        ('expectancy',        'Expectancy',        True),
        ('n_trades',          'N Trades',          None),
    ]

    for key, label, higher_is_better in metrics_to_compare:
        v4  = results_v4.get(key, 0)
        v5  = results_v5.get(key, 0)
        fmt = '.1%' if key == 'win_rate' else '.2f'
        if higher_is_better is None:
            winner = '—'
        elif higher_is_better:
            winner = 'V4 ✅' if v4 > v5 else 'V5' if v5 > v4 else 'TIE'
        else:
            winner = 'V4 ✅' if v4 > v5 else 'V5' if v5 > v4 else 'TIE'

        print(f"  {label:<23} {v4:>12{fmt}} {v5:>12{fmt}} {winner:>10}")

    print(f"{'='*60}")


# ── Entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='AlphaEdge Backtest V6')
    parser.add_argument('--crypto',  action='store_true', help='Crypto backtest only')
    parser.add_argument('--stocks',  action='store_true', help='Stock backtest only')
    parser.add_argument('--compare', action='store_true', help='V4 vs V5 comparison')
    args = parser.parse_args()

    print("\n" + "🚀" * 25)
    print("  ALPHAEDGE BACKTEST V6 — ATR Stops + V4 Signal Alignment")
    print("🚀" * 25)

    start = time.time()

    if args.compare:
        run_comparison()
    elif args.crypto:
        run_crypto_backtest(BACKTEST_CONFIG_V4)
    elif args.stocks:
        run_stock_backtest(BACKTEST_CONFIG_V4, 'V4-ATR')
    else:
        # Run both by default
        run_stock_backtest(BACKTEST_CONFIG_V4, 'V4-ATR')
        run_crypto_backtest(BACKTEST_CONFIG_V4)

    elapsed = time.time() - start
    print(f"\n✅ Backtest complete in {elapsed/60:.1f} minutes")
    print(f"   Results saved to logs/backtest/")


if __name__ == '__main__':
    main()
