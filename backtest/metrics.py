# backtest/metrics.py
"""
Fix 5.1 + 5.2 — Backtest Performance Metrics Module
====================================================
Computes all institutional-grade metrics from an equity curve and trade list.
Includes regime breakdown (Fix 5.2).

Usage:
    from backtest.metrics import compute_metrics, regime_breakdown, print_metrics_report

    metrics = compute_metrics(equity_curve, trades)
    print_metrics_report(metrics)

    # With regime breakdown:
    breakdown = regime_breakdown(equity_curve, regime_series)
"""

import math
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Minimum thresholds for live deployment (Fix 5.1) ──────────────────────────
MINIMUM_THRESHOLDS = {
    'sharpe_ratio'    :  0.80,    # below this: no edge worth taking
    'max_drawdown_pct': -20.0,    # worse than -20%: unacceptable risk
    'win_rate_pct'    :  45.0,    # above 45% with good profit factor is OK
    'profit_factor'   :  1.30,    # minimum gross profit / gross loss
}

RISK_FREE_RATE = 0.05   # annualised, current T-bill rate
TRADING_DAYS   = 252


def compute_metrics(
    equity_curve,
    trades: list,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """
    Compute full set of performance metrics.

    Args:
        equity_curve: pd.Series of daily portfolio value, indexed by date.
                      OR a list of floats [day0_value, day1_value, ...]
        trades:       list of dicts, each with at minimum:
                      {'pnl_pct': float, 'pnl': float, 'entry_date': str, 'exit_date': str}
        risk_free_rate: annualised, default 5%

    Returns:
        dict of all metrics — see keys below.
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError as e:
        logger.error(f'backtest/metrics.py requires pandas and numpy: {e}')
        return _empty_metrics()

    # ── Coerce equity_curve to pd.Series ─────────────────────────────────────
    if not isinstance(equity_curve, pd.Series):
        equity_curve = pd.Series(equity_curve)
    if len(equity_curve) < 2:
        logger.warning('compute_metrics: equity curve too short (< 2 bars)')
        return _empty_metrics()

    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0:
        return _empty_metrics()

    # ── Return metrics ────────────────────────────────────────────────────────
    total_return = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1)
    n_years      = len(returns) / TRADING_DAYS
    cagr         = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0

    # ── Risk metrics ──────────────────────────────────────────────────────────
    daily_rfr = risk_free_rate / TRADING_DAYS
    excess    = returns - daily_rfr

    sharpe = float(
        excess.mean() / returns.std() * math.sqrt(TRADING_DAYS)
    ) if returns.std() > 0 else 0.0

    downside     = returns[returns < 0]
    down_std     = float(downside.std()) if len(downside) > 1 else 0.0
    sortino      = float(
        excess.mean() / down_std * math.sqrt(TRADING_DAYS)
    ) if down_std > 0 else 0.0

    # ── Drawdown ──────────────────────────────────────────────────────────────
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max
    max_dd      = float(drawdown.min())
    calmar      = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

    # ── Rolling 30-day Sharpe (for regime detection quality) ──────────────────
    rolling_sharpe_30 = (
        returns.rolling(30).mean() / returns.rolling(30).std() * math.sqrt(TRADING_DAYS)
    ).dropna()
    sharpe_stability  = float(rolling_sharpe_30.std()) if len(rolling_sharpe_30) > 0 else None

    # ── Trade metrics ─────────────────────────────────────────────────────────
    if trades:
        pnls   = [t.get('pnl_pct', t.get('pnl', 0)) for t in trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate      = len(wins) / len(pnls) if pnls else 0.0
        avg_win       = float(np.mean(wins))   if wins   else 0.0
        avg_loss      = float(np.mean(losses)) if losses else 0.0
        profit_factor = float(sum(wins) / abs(sum(losses))) if losses else float('inf')
        expectancy    = float(win_rate * avg_win + (1 - win_rate) * avg_loss)
        total_trades  = len(pnls)
    else:
        win_rate = profit_factor = expectancy = avg_win = avg_loss = 0.0
        total_trades = 0

    return {
        # Return
        'total_return_pct'   : round(total_return * 100, 2),
        'cagr_pct'           : round(cagr * 100, 2),
        # Risk-adjusted
        'sharpe_ratio'       : round(sharpe, 3),
        'sortino_ratio'      : round(sortino, 3),
        'calmar_ratio'       : round(calmar, 3),
        'max_drawdown_pct'   : round(max_dd * 100, 2),
        'sharpe_stability'   : round(sharpe_stability, 3) if sharpe_stability else None,
        # Trade
        'win_rate_pct'       : round(win_rate * 100, 2),
        'profit_factor'      : round(profit_factor, 3),
        'expectancy_pct'     : round(expectancy * 100, 4),
        'avg_win_pct'        : round(avg_win * 100, 4),
        'avg_loss_pct'       : round(avg_loss * 100, 4),
        'total_trades'       : total_trades,
        'n_years'            : round(n_years, 2),
    }


def regime_breakdown(equity_curve, regime_series) -> dict:
    """
    Fix 5.2 — Per-regime performance breakdown.

    Args:
        equity_curve:  pd.Series of daily portfolio value
        regime_series: pd.Series of regime labels ('bull','bear','sideways','crisis')
                       Must be index-aligned with equity_curve.

    Returns:
        dict of {regime_name: {'sharpe', 'n_days', 'total_return_pct'}}

    If Sharpe in bear regime < -0.5, the system is losing money when
    capital preservation matters most — do not deploy real capital.
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        return {}

    if not isinstance(equity_curve, pd.Series):
        equity_curve = pd.Series(equity_curve)

    returns   = equity_curve.pct_change().dropna()
    breakdown = {}

    for regime in ['bull', 'bear', 'sideways', 'crisis', 'unknown']:
        # Align regime_series with returns index
        regime_aligned = regime_series.reindex(returns.index)
        mask           = regime_aligned == regime
        regime_returns = returns[mask]

        if len(regime_returns) < 20:
            continue

        mean_r = float(regime_returns.mean())
        std_r  = float(regime_returns.std())
        sr     = (mean_r / std_r * math.sqrt(TRADING_DAYS)) if std_r > 0 else 0.0
        total  = float((1 + regime_returns).prod() - 1) * 100

        breakdown[regime] = {
            'sharpe'           : round(sr, 3),
            'n_days'           : int(mask.sum()),
            'total_return_pct' : round(total, 2),
        }

        # Warn on bear regime underperformance (Fix 5.2 threshold)
        if regime == 'bear' and sr < -0.5:
            logger.warning(
                f'regime_breakdown: BEAR regime Sharpe = {sr:.2f} '
                f'(threshold -0.5). System loses money when it matters most. '
                f'Do NOT deploy real capital.'
            )

    return breakdown


def print_metrics_report(metrics: dict, thresholds: dict = None) -> bool:
    """
    Print a formatted metrics report with PASS/FAIL against thresholds.

    Returns True if all thresholds pass, False if any fail.
    Call this at the end of run_backtest.py.
    """
    thresholds = thresholds or MINIMUM_THRESHOLDS

    print('\n' + '=' * 65)
    print('  BACKTEST PERFORMANCE REPORT')
    print('=' * 65)

    fmt_map = {
        'total_return_pct'  : ('Total Return',        '{:+.2f}%'),
        'cagr_pct'          : ('CAGR',                '{:+.2f}%'),
        'sharpe_ratio'      : ('Sharpe Ratio',         '{:.3f}'),
        'sortino_ratio'     : ('Sortino Ratio',        '{:.3f}'),
        'calmar_ratio'      : ('Calmar Ratio',         '{:.3f}'),
        'max_drawdown_pct'  : ('Max Drawdown',         '{:.2f}%'),
        'win_rate_pct'      : ('Win Rate',             '{:.1f}%'),
        'profit_factor'     : ('Profit Factor',        '{:.3f}'),
        'expectancy_pct'    : ('Expectancy/trade',     '{:+.4f}%'),
        'total_trades'      : ('Total Trades',         '{:d}'),
        'n_years'           : ('Years Backtested',     '{:.2f}'),
    }

    all_pass = True
    for key, (label, fmt) in fmt_map.items():
        value = metrics.get(key)
        if value is None:
            continue
        threshold = thresholds.get(key)

        try:
            val_str = fmt.format(value)
        except (ValueError, TypeError):
            val_str = str(value)

        if threshold is not None:
            passed = value >= threshold
            if not passed:
                all_pass = False
            status = '✅ PASS' if passed else '❌ FAIL'
            threshold_str = f'  (min: {threshold})'
        else:
            status        = '  INFO'
            threshold_str = ''

        print(f'  {status}  {label:<22} {val_str}{threshold_str}')

    print('=' * 65)
    if all_pass:
        print('  🟢 ALL THRESHOLDS PASSED — system qualifies for paper trading')
    else:
        print('  🔴 THRESHOLDS FAILED — DO NOT deploy real capital')
    print('=' * 65 + '\n')

    return all_pass


def _empty_metrics() -> dict:
    return {
        'total_return_pct'   : 0.0,
        'cagr_pct'           : 0.0,
        'sharpe_ratio'       : 0.0,
        'sortino_ratio'      : 0.0,
        'calmar_ratio'       : 0.0,
        'max_drawdown_pct'   : 0.0,
        'sharpe_stability'   : None,
        'win_rate_pct'       : 0.0,
        'profit_factor'      : 0.0,
        'expectancy_pct'     : 0.0,
        'avg_win_pct'        : 0.0,
        'avg_loss_pct'       : 0.0,
        'total_trades'       : 0,
        'n_years'            : 0.0,
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

    # Simulate a 2-year equity curve
    np.random.seed(42)
    daily_returns = np.random.normal(0.0004, 0.012, 504)   # ~2 years
    equity = pd.Series(
        10000 * (1 + daily_returns).cumprod(),
        index=pd.date_range('2023-01-01', periods=504, freq='B'),
    )

    trades = [
        {'pnl_pct':  0.05, 'pnl':  500},
        {'pnl_pct': -0.02, 'pnl': -200},
        {'pnl_pct':  0.08, 'pnl':  800},
        {'pnl_pct': -0.03, 'pnl': -300},
        {'pnl_pct':  0.04, 'pnl':  400},
    ]

    metrics = compute_metrics(equity, trades)
    all_pass = print_metrics_report(metrics)

    # Regime breakdown
    regimes = pd.Series(
        np.random.choice(['bull', 'bear', 'sideways'], size=len(equity)),
        index=equity.index,
    )
    breakdown = regime_breakdown(equity, regimes)
    print('\n  REGIME BREAKDOWN:')
    for regime, data in breakdown.items():
        print(f'    {regime:<10}: Sharpe={data["sharpe"]:+.2f}  '
              f'Days={data["n_days"]}  '
              f'Return={data["total_return_pct"]:+.2f}%')
