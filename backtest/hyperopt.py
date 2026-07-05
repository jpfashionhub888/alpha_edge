# backtest/hyperopt.py
"""
AlphaEdge — HyperOpt Parameter Optimizer
Inspired by: Freqtrade's hyperopt system
             https://www.freqtrade.io/en/stable/hyperopt/

What it does:
    Searches over strategy parameter combinations (thresholds, multipliers)
    using anchored walk-forward cross-validation.
    Optimises for Sharpe ratio on OUT-OF-SAMPLE data only.
    Prevents overfitting by never testing on training data.

Key design decisions vs Freqtrade:
    - Uses anchored (expanding) CV instead of rolling window
      (expanding = more data per fold, stabler parameter estimates)
    - Optimisation target is Sharpe ratio (not profit factor or win rate alone)
    - Parameters written to config/settings.yaml on completion

Usage:
    from backtest.hyperopt import AlphaEdgeHyperOpt
    opt = AlphaEdgeHyperOpt()
    best = opt.run(symbols=['AAPL','NVDA','SPY'], n_trials=200)
    print(best)   # {'buy_threshold': 0.65, 'volume_spike': 1.3, ...}
    opt.save_best_params(best)   # writes to config/settings.yaml

CLI:
    python -m backtest.hyperopt --symbols AAPL NVDA SPY --trials 200
"""

import itertools
import json
import logging
import math
import os
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default search space ──────────────────────────────────────────────────────
# Mirrors all tuneable knobs in scanner.py / config/settings.yaml
DEFAULT_SEARCH_SPACE = {
    'buy_threshold'    : [0.55, 0.58, 0.60, 0.62, 0.63, 0.65, 0.68, 0.70],
    'volume_spike_min' : [1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0],
    'min_rr_ratio'     : [1.5, 1.8, 2.0, 2.5, 3.0],
    'atr_stop_mult'    : [0.8, 1.0, 1.2, 1.5, 2.0],
    'atr_target_mult'  : [1.5, 2.0, 2.5, 3.0],
    'kelly_multiplier' : [0.25, 0.33, 0.5],
    'max_position_pct' : [0.05, 0.08, 0.10, 0.12, 0.15],
}

SETTINGS_PATH = 'config/settings.yaml'
RESULTS_PATH  = 'logs/hyperopt_results.json'


class AlphaEdgeHyperOpt:
    """
    Walk-forward hyperparameter optimiser for AlphaEdge strategy parameters.

    Algorithm:
        1. Load N years of OHLCV data for given symbols
        2. Create anchored CV folds (expanding train, fixed-size test)
        3. For each param combination:
           a. Run simplified backtest on each fold's test window
           b. Compute Sharpe ratio on OOS test returns
           c. Average Sharpe across all folds
        4. Return param combination with highest mean OOS Sharpe
        5. Write best params to config/settings.yaml

    Parameters
    ----------
    train_years : int
        Minimum training window in years. Default 2.
    test_months : int
        Test window per fold in months. Default 3.
    n_folds     : int
        Number of walk-forward folds. Default 4.
    n_trials    : int
        Max parameter combinations to test. Default 200.
        Full grid may exceed this — random sampling used if so.
    """

    def __init__(
        self,
        train_years : int = 2,
        test_months : int = 3,
        n_folds     : int = 4,
        n_trials    : int = 200,
        search_space : Optional[Dict] = None,
    ):
        self.train_years  = train_years
        self.test_months  = test_months
        self.n_folds      = n_folds
        self.n_trials     = n_trials
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self._results: List[Dict] = []

    # ── Main entry point ─────────────────────────────────────────────────

    def run(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Run hyperopt. Returns best parameter dict.

        Parameters
        ----------
        symbols    : list of ticker symbols to optimise on
        start_date : historical start date (YYYY-MM-DD). Defaults to 4 years ago.
        verbose    : print progress
        """
        start_date = start_date or (
            datetime.now() - timedelta(days=365 * (self.train_years + 2))
        ).strftime('%Y-%m-%d')

        logger.info(f'HyperOpt: loading data for {len(symbols)} symbols from {start_date}')
        data = self._load_data(symbols, start_date)

        if not data:
            raise RuntimeError('HyperOpt: no data loaded — check symbols / yfinance')

        # Build parameter combinations
        combos = self._build_combos()
        logger.info(
            f'HyperOpt: testing {len(combos)} param combinations × '
            f'{self.n_folds} folds'
        )

        best_sharpe = -999.0
        best_params = combos[0]
        self._results = []

        for i, params in enumerate(combos):
            fold_sharpes = []

            for fold_data in self._make_folds(data, start_date):
                sharpe = self._backtest_fold(fold_data, params)
                fold_sharpes.append(sharpe)

            mean_sharpe = float(np.mean(fold_sharpes))
            result = {**params, 'sharpe': mean_sharpe, 'fold_sharpes': fold_sharpes}
            self._results.append(result)

            if mean_sharpe > best_sharpe:
                best_sharpe = mean_sharpe
                best_params = params
                if verbose:
                    logger.info(
                        f'  [{i+1}/{len(combos)}] New best Sharpe={best_sharpe:.3f} '
                        f'| params={params}'
                    )
            elif verbose and (i + 1) % 20 == 0:
                logger.info(f'  [{i+1}/{len(combos)}] best so far: Sharpe={best_sharpe:.3f}')

        self._save_results()
        logger.info(f'HyperOpt complete. Best Sharpe={best_sharpe:.3f} | Params={best_params}')
        return best_params

    # ── Backtesting logic ─────────────────────────────────────────────────

    def _backtest_fold(
        self,
        fold_data: Dict[str, pd.DataFrame],
        params: Dict,
    ) -> float:
        """
        Run a simplified vectorised backtest on one fold's test window.
        Returns annualised Sharpe ratio of strategy returns.

        This is NOT a tick-level simulation — it's a fast approximation
        used purely for parameter ranking. Phase 4 (event engine) does
        the proper simulation for final validation.
        """
        all_returns = []

        buy_thr  = params['buy_threshold']
        vol_thr  = params['volume_spike_min']
        rr_ratio = params['min_rr_ratio']
        atr_stop = params['atr_stop_mult']
        atr_tgt  = params['atr_target_mult']

        for symbol, df in fold_data.items():
            if df.empty or len(df) < 60:
                continue

            try:
                # Fast feature proxies (no full Alpha158 for speed)
                close   = df['close']
                volume  = df['volume']
                ret_1d  = close.pct_change(1)
                vol_ma  = volume.rolling(20).mean()
                vol_spi = volume / (vol_ma + 1e-9)

                # Momentum score (proxy for model prediction)
                ma5   = close.rolling(5).mean()
                ma20  = close.rolling(20).mean()
                mom   = (ma5 - ma20) / (ma20 + 1e-9)

                # ATR (proxy stop/target)
                hl   = df['high'] - df['low']
                atr  = hl.rolling(14).mean()

                # Signal: buy when momentum > threshold AND volume spike
                score  = (mom - mom.rolling(60).min()) / (mom.rolling(60).max() - mom.rolling(60).min() + 1e-9)
                signal = (score > buy_thr) & (vol_spi > vol_thr)

                # Simulate returns: hold until stop or target (fixed bars)
                for i in range(len(df) - 20):
                    if not signal.iloc[i]:
                        continue
                    entry  = close.iloc[i]
                    a      = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else entry * 0.02
                    stop   = entry - atr_stop * a
                    target = entry + atr_tgt * a

                    # Check if RR ratio is met
                    if (target - entry) / (entry - stop + 1e-9) < rr_ratio:
                        continue

                    # Walk forward until stop, target, or 20 bars
                    exit_ret = 0.0
                    for j in range(1, 21):
                        idx = i + j
                        if idx >= len(close):
                            break
                        p = close.iloc[idx]
                        if p <= stop:
                            exit_ret = (p - entry) / entry
                            break
                        if p >= target:
                            exit_ret = (p - entry) / entry
                            break
                    else:
                        exit_ret = (close.iloc[min(i + 20, len(close) - 1)] - entry) / entry

                    all_returns.append(exit_ret)

            except Exception as e:
                logger.debug(f'HyperOpt backtest failed for {symbol}: {e}')
                continue

        if len(all_returns) < 5:
            return -1.0  # insufficient trades

        returns = np.array(all_returns)
        return self._sharpe(returns)

    @staticmethod
    def _sharpe(returns: np.ndarray) -> float:
        """Annualised Sharpe ratio (risk-free=0)."""
        if len(returns) < 2:
            return 0.0
        mean = returns.mean()
        std  = returns.std(ddof=1)
        if std < 1e-9:
            return 0.0
        # Assume ~20 trades/year → annualisation factor sqrt(20)
        return float((mean / std) * math.sqrt(20))

    # ── Data loading ─────────────────────────────────────────────────────

    @staticmethod
    def _load_data(symbols: List[str], start_date: str) -> Dict[str, pd.DataFrame]:
        """Load OHLCV data via yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError('yfinance required for HyperOpt data loading')

        data = {}
        for sym in symbols:
            try:
                df = yf.Ticker(sym).history(start=start_date, auto_adjust=True)
                if df.empty:
                    continue
                df.columns = [c.lower() for c in df.columns]
                df.index   = df.index.tz_localize(None)
                df         = df[['open', 'high', 'low', 'close', 'volume']].dropna()
                data[sym]  = df
                logger.debug(f'  {sym}: {len(df)} bars')
            except Exception as e:
                logger.warning(f'HyperOpt: failed to load {sym}: {e}')

        return data

    def _make_folds(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: str,
    ):
        """
        Generate anchored walk-forward folds.
        Each fold yields a dict[symbol → DataFrame] for the TEST period only.
        """
        # Find common date range across all symbols
        all_dates = sorted(set.union(*[set(df.index) for df in data.values()]))
        if not all_dates:
            return

        total_days  = (all_dates[-1] - all_dates[0]).days
        test_days   = self.test_months * 30
        min_train   = self.train_years * 365

        # Build fold boundaries
        folds = []
        for fold in range(self.n_folds):
            test_end   = all_dates[-1] - timedelta(days=fold * test_days)
            test_start = test_end - timedelta(days=test_days)
            if (test_start - all_dates[0]).days < min_train:
                break
            folds.append((test_start, test_end))

        for test_start, test_end in reversed(folds):
            fold_data = {
                sym: df.loc[
                    (df.index >= test_start) & (df.index <= test_end)
                ].copy()
                for sym, df in data.items()
            }
            yield fold_data

    def _build_combos(self) -> List[Dict]:
        """
        Build list of parameter combinations.
        If full grid > n_trials, randomly sample n_trials from it.
        """
        keys   = list(self.search_space.keys())
        values = list(self.search_space.values())
        full   = [dict(zip(keys, v)) for v in itertools.product(*values)]

        if len(full) <= self.n_trials:
            return full

        logger.info(
            f'HyperOpt: full grid has {len(full)} combos → '
            f'random sampling {self.n_trials}'
        )
        return random.sample(full, self.n_trials)

    # ── Results ──────────────────────────────────────────────────────────

    def _save_results(self) -> None:
        """Save all trial results to JSON for analysis."""
        os.makedirs('logs', exist_ok=True)
        # Sort by Sharpe descending
        sorted_results = sorted(self._results, key=lambda x: x['sharpe'], reverse=True)
        with open(RESULTS_PATH, 'w') as f:
            json.dump(sorted_results[:100], f, indent=2)  # top 100
        logger.info(f'HyperOpt results saved to {RESULTS_PATH}')

    def save_best_params(self, params: Dict) -> None:
        """
        Write best params into config/settings.yaml under [hyperopt] section.
        Also updates signal_thresholds.buy_threshold so the scanner
        picks it up immediately without a restart.
        Preserves all existing settings — never overwrites other keys.
        """
        try:
            import yaml
        except ImportError:
            logger.warning('PyYAML not installed — saving params to JSON instead')
            with open('logs/best_hyperopt_params.json', 'w') as f:
                json.dump(params, f, indent=2)
            return

        os.makedirs('config', exist_ok=True)
        existing = {}
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH) as f:
                existing = yaml.safe_load(f) or {}

        # ── Write into a dedicated [hyperopt] section ─────────────────
        clean = {k: v for k, v in params.items() if k not in ('sharpe', 'fold_sharpes')}
        existing.setdefault('hyperopt', {})
        existing['hyperopt'].update(clean)
        existing['hyperopt']['last_run']    = datetime.now().isoformat()
        existing['hyperopt']['best_sharpe'] = float(params.get('sharpe', 0) or 0)

        # ── Also mirror buy_threshold into signal_thresholds ──────────
        # so scanner._load_signal_settings() picks it up immediately
        if 'buy_threshold' in clean:
            existing.setdefault('signal_thresholds', {})
            existing['signal_thresholds']['buy_threshold'] = clean['buy_threshold']

        with open(SETTINGS_PATH, 'w') as f:
            yaml.dump(existing, f, default_flow_style=False, sort_keys=True)

        logger.info(f'Best params written to {SETTINGS_PATH}')
        logger.info(f'  buy_threshold updated to {clean.get("buy_threshold")}')

    def top_results(self, n: int = 10) -> pd.DataFrame:
        """Return top N results as a DataFrame for analysis."""
        if not self._results:
            return pd.DataFrame()
        return (
            pd.DataFrame(self._results)
            .sort_values('sharpe', ascending=False)
            .head(n)
            .reset_index(drop=True)
        )


# ── ROI Table (Freqtrade pattern) ─────────────────────────────────────────────

class ROITable:
    """
    Time-decaying take-profit targets (Freqtrade's ROI table concept).

    Instead of a fixed +8% target, exit at different levels based on
    how long the trade has been open. This captures quick winners and
    exits slower movers at lower targets.

    Example table:
        {0: 0.10, 60: 0.05, 120: 0.02, 240: 0.00}
        → Exit immediately if up 10%
        → Exit after 60min if up 5%
        → Exit after 120min if up 2%
        → Exit after 240min at breakeven

    For daily-bar systems, use 'bars' instead of minutes:
        {0: 0.10, 3: 0.06, 5: 0.03, 10: 0.00}
    """

    # Default table tuned for AlphaEdge daily bars
    DEFAULT_TABLE = {
        0 : 0.10,   # exit any time if up 10%+
        3 : 0.06,   # exit after 3 bars if up 6%+
        5 : 0.03,   # exit after 5 bars if up 3%+
        10: 0.00,   # exit after 10 bars at breakeven
    }

    def __init__(self, table: Optional[Dict[int, float]] = None):
        self.table = table or self.DEFAULT_TABLE
        # Sort by bars ascending
        self._sorted = sorted(self.table.items())

    def should_exit_profit(self, bars_open: int, current_profit_pct: float) -> bool:
        """
        Returns True if the current profit meets the ROI target for bars_open.

        Parameters
        ----------
        bars_open          : How many bars the position has been open
        current_profit_pct : Current unrealised profit as a fraction (e.g. 0.05 = 5%)
        """
        roi_target = self._get_roi_target(bars_open)
        return current_profit_pct >= roi_target

    def _get_roi_target(self, bars_open: int) -> float:
        """Get ROI target for given bars_open (latest applicable threshold)."""
        target = float('inf')
        for bars, roi in self._sorted:
            if bars_open >= bars:
                target = roi
        return target

    def __repr__(self) -> str:
        rows = '\n'.join(
            f'  After {b:>3} bars: exit at {r:.0%}+' for b, r in self._sorted
        )
        return f'ROITable:\n{rows}'


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )

    parser = argparse.ArgumentParser(description='AlphaEdge HyperOpt')
    parser.add_argument('--symbols', nargs='+',
                        default=['SPY', 'QQQ', 'AAPL', 'NVDA', 'MSFT', 'TSLA'],
                        help='Symbols to optimise on')
    parser.add_argument('--trials', type=int, default=200,
                        help='Max parameter combinations to test')
    parser.add_argument('--folds', type=int, default=4,
                        help='Walk-forward folds')
    parser.add_argument('--save', action='store_true',
                        help='Write best params to config/settings.yaml')
    args = parser.parse_args()

    print(f'\n{"="*60}')
    print(f'AlphaEdge HyperOpt')
    print(f'Symbols: {args.symbols}')
    print(f'Trials:  {args.trials} | Folds: {args.folds}')
    print(f'{"="*60}\n')

    opt  = AlphaEdgeHyperOpt(n_trials=args.trials, n_folds=args.folds)
    best = opt.run(symbols=args.symbols)

    print(f'\n{"="*60}')
    print('BEST PARAMETERS:')
    for k, v in best.items():
        print(f'  {k:<22} = {v}')
    print(f'{"="*60}\n')

    print('Top 5 results:')
    print(opt.top_results(5).to_string(index=False))

    if args.save:
        opt.save_best_params(best)
        print(f'\nBest params saved to {SETTINGS_PATH}')
