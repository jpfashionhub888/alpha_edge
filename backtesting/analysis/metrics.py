# backtesting/analysis/metrics.py
"""
Performance metrics.

All metrics are computed on the equity curve or returns series.
Annualisation assumes 252 trading days per year.

IC (Information Coefficient): correlation between signal scores and
subsequent returns. IC > 0.03 with t-stat > 2.0 is the minimum bar
for Phase 1 Gate 1 validation.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


TRADING_DAYS = 252


def returns_from_equity(equity: pd.Series) -> pd.Series:
    """Daily returns from equity curve."""
    return equity.pct_change().dropna()


def annualised_return(returns: pd.Series) -> float:
    """Geometric annualised return."""
    if len(returns) == 0:
        return 0.0
    total = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS
    if n_years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def annualised_volatility(returns: pd.Series) -> float:
    """Annualised volatility."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """
    Annualised Sharpe ratio.
    Uses 5% risk-free rate by default (approximate T-bill rate, mid-2026).
    """
    ann_ret  = annualised_return(returns)
    ann_vol  = annualised_volatility(returns)
    if ann_vol == 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / ann_vol)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Annualised Sortino ratio (downside deviation denominator)."""
    ann_ret = annualised_return(returns)
    daily_rf = risk_free_rate / TRADING_DAYS
    downside = returns[returns < daily_rf]
    if len(downside) < 2:
        return 0.0
    downside_std = float(downside.std() * np.sqrt(TRADING_DAYS))
    if downside_std == 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / downside_std)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (as a negative fraction)."""
    if len(equity) == 0:
        return 0.0
    roll_max   = equity.cummax()
    drawdown   = (equity - roll_max) / roll_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, equity: pd.Series) -> float:
    """Annualised return / abs(max drawdown)."""
    ann_ret = annualised_return(returns)
    mdd     = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(ann_ret / mdd)


def win_rate(trade_pnls: pd.Series) -> float:
    """Fraction of trades with positive PnL."""
    if len(trade_pnls) == 0:
        return 0.0
    return float((trade_pnls > 0).mean())


def profit_factor(trade_pnls: pd.Series) -> float:
    """Gross profit / gross loss."""
    gains  = trade_pnls[trade_pnls > 0].sum()
    losses = abs(trade_pnls[trade_pnls < 0].sum())
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return float(gains / losses)


# ── Information Coefficient ───────────────────────────────────────────────────

def information_coefficient(
    scores   : pd.Series,
    returns  : pd.Series,
    method   : str = 'spearman',
) -> float:
    """
    IC = rank correlation between signal scores and forward returns.

    Spearman (rank) correlation is preferred over Pearson because:
    - Less sensitive to outlier returns
    - Correctly captures monotonic but non-linear relationships
    - Directly measures whether top-ranked stocks outperform bottom-ranked

    Parameters
    ----------
    scores  : signal scores at time T (higher = more bullish)
    returns : forward returns at time T+1 (or holding period)
    method  : 'spearman' (default) or 'pearson'
    """
    common = scores.index.intersection(returns.index)
    if len(common) < 5:
        return 0.0
    s = scores.loc[common].dropna()
    r = returns.loc[common].dropna()
    common2 = s.index.intersection(r.index)
    if len(common2) < 5:
        return 0.0
    s, r = s.loc[common2], r.loc[common2]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if method == 'spearman':
            ic, _ = stats.spearmanr(s, r)
        else:
            ic, _ = stats.pearsonr(s, r)
    return float(ic) if not np.isnan(ic) else 0.0


def ic_series(
    signal_df : pd.DataFrame,    # index=date, columns=symbols, values=scores
    returns_df: pd.DataFrame,    # index=date, columns=symbols, values=fwd_returns
    method    : str = 'spearman',
) -> pd.Series:
    """
    IC computed cross-sectionally for each date.

    Returns a Series indexed by date with daily IC values.
    """
    dates = signal_df.index.intersection(returns_df.index)
    ics   = {}
    for date in dates:
        scores  = signal_df.loc[date].dropna()
        fwd_ret = returns_df.loc[date].dropna()
        ics[date] = information_coefficient(scores, fwd_ret, method)
    return pd.Series(ics).sort_index()


def ic_stats(ic: pd.Series) -> dict:
    """
    Summary statistics for an IC series.

    Gate 1 requirements (signal validation):
      - mean_ic > 0.03
      - ic_tstat > 2.0
      - pct_positive > 0.55
    """
    if len(ic) == 0:
        return {}
    mean_ic  = float(ic.mean())
    std_ic   = float(ic.std())
    n        = len(ic)
    tstat    = (mean_ic / std_ic * np.sqrt(n)) if std_ic > 0 else 0.0
    return {
        'mean_ic'     : mean_ic,
        'std_ic'      : std_ic,
        'ic_ir'       : mean_ic / std_ic if std_ic > 0 else 0.0,
        'ic_tstat'    : float(tstat),
        'pct_positive': float((ic > 0).mean()),
        'n_periods'   : n,
        'gate1_pass'  : (
            mean_ic > 0.03
            and abs(tstat) > 2.0
            and (ic > 0).mean() > 0.55
        ),
    }


# ── Quintile analysis ─────────────────────────────────────────────────────────

def quintile_returns(
    signal_df : pd.DataFrame,
    returns_df: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    Mean forward return by signal quantile, averaged across all dates.

    Gate 1 requires monotonic quintile spread: Q5 > Q4 > Q3 > Q2 > Q1.

    Returns DataFrame with columns: [quantile, mean_return, median_return, count]
    """
    rows = []
    for date in signal_df.index.intersection(returns_df.index):
        scores  = signal_df.loc[date].dropna()
        fwd_ret = returns_df.loc[date].dropna()
        common  = scores.index.intersection(fwd_ret.index)
        if len(common) < n_quantiles * 2:
            continue
        scores  = scores.loc[common]
        fwd_ret = fwd_ret.loc[common]
        labels  = pd.qcut(scores, n_quantiles, labels=False, duplicates='drop')
        for q in range(n_quantiles):
            mask = labels == q
            if mask.sum() > 0:
                rows.append({'quantile': q + 1, 'return': float(fwd_ret[mask].mean())})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    summary = (
        df.groupby('quantile')['return']
        .agg(['mean', 'median', 'count'])
        .reset_index()
        .rename(columns={'mean': 'mean_return', 'median': 'median_return'})
    )

    # Monotonicity check
    means = summary['mean_return'].values
    is_monotonic = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
    summary['monotonic'] = is_monotonic

    return summary


# ── Full performance summary ──────────────────────────────────────────────────

def performance_summary(
    equity  : pd.Series,
    trades  : Optional[pd.DataFrame] = None,
    ic_ser  : Optional[pd.Series]    = None,
) -> dict:
    """
    Full performance summary for tearsheet.

    Includes all Gate 1 + Gate 2 metrics.
    """
    ret = returns_from_equity(equity)

    summary = {
        # Returns
        'total_return'         : float((equity.iloc[-1] / equity.iloc[0]) - 1),
        'annualised_return'    : annualised_return(ret),
        'annualised_volatility': annualised_volatility(ret),
        'sharpe_ratio'         : sharpe_ratio(ret),
        'sortino_ratio'        : sortino_ratio(ret),
        'calmar_ratio'         : calmar_ratio(ret, equity),
        'max_drawdown'         : max_drawdown(equity),
        'n_trading_days'       : len(ret),
    }

    if trades is not None and len(trades) > 0:
        summary.update({
            'n_trades'             : len(trades),
            'win_rate'             : win_rate(trades['pnl']) if 'pnl' in trades else None,
            'profit_factor'        : profit_factor(trades['pnl']) if 'pnl' in trades else None,
            'avg_is_bps'           : float(trades['is_bps'].mean()) if 'is_bps' in trades else None,
            'median_is_bps'        : float(trades['is_bps'].median()) if 'is_bps' in trades else None,
        })

    if ic_ser is not None and len(ic_ser) > 0:
        stats_dict = ic_stats(ic_ser)
        summary.update({f'ic_{k}': v for k, v in stats_dict.items()})

    return summary


def print_summary(summary: dict) -> None:
    """Pretty-print performance summary."""
    print('\n' + '=' * 55)
    print('AlphaEdge Backtest Performance Summary')
    print('=' * 55)

    sections = {
        'Returns': [
            ('Total Return',          'total_return',          '{:.1%}'),
            ('Annualised Return',      'annualised_return',     '{:.1%}'),
            ('Annualised Volatility',  'annualised_volatility', '{:.1%}'),
            ('Sharpe Ratio',           'sharpe_ratio',          '{:.2f}'),
            ('Sortino Ratio',          'sortino_ratio',         '{:.2f}'),
            ('Calmar Ratio',           'calmar_ratio',          '{:.2f}'),
            ('Max Drawdown',           'max_drawdown',          '{:.1%}'),
        ],
        'Execution': [
            ('Trades',                 'n_trades',              '{:.0f}'),
            ('Win Rate',               'win_rate',              '{:.1%}'),
            ('Profit Factor',          'profit_factor',         '{:.2f}'),
            ('Avg IS (bps)',           'avg_is_bps',            '{:.1f}'),
            ('Median IS (bps)',        'median_is_bps',         '{:.1f}'),
        ],
        'Signal Quality (Gate 1)': [
            ('Mean IC',               'ic_mean_ic',            '{:.4f}'),
            ('IC t-stat',             'ic_ic_tstat',           '{:.2f}'),
            ('IC IR',                 'ic_ic_ir',              '{:.2f}'),
            ('% Positive IC',         'ic_pct_positive',       '{:.1%}'),
            ('Gate 1 PASS',           'ic_gate1_pass',         '{}'),
        ],
    }

    for section, fields in sections.items():
        print(f'\n{section}:')
        for label, key, fmt in fields:
            val = summary.get(key)
            if val is None:
                continue
            try:
                print(f'  {label:<30} {fmt.format(val)}')
            except Exception:
                print(f'  {label:<30} {val}')

    print()
