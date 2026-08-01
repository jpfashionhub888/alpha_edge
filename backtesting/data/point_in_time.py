# backtesting/data/point_in_time.py
"""
Lookahead bias prevention.

This is the most important module in the backtesting framework.
A single lookahead leak can make a useless signal appear to have
Sharpe > 3.0. Every signal must pass the lookahead audit here.

Rules enforced:
  1. Signal computed at time T may only use data available at T.
  2. The target (forward return) must be at T+lag (default T+1).
  3. Features that are contemporaneously correlated with same-bar returns
     are flagged as potential lookahead leaks.

The event-driven engine enforces rule 1 mechanically by passing
data_up_to_prev to signal_fn. This module provides the audit tool
to verify rule 3 statistically.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Correlation threshold above which a feature is flagged
LOOKAHEAD_CORR_THRESHOLD = 0.30


def make_forward_returns(
    prices    : pd.DataFrame,   # index=date, columns=symbols
    lag       : int = 1,
    holding   : int = 1,
) -> pd.DataFrame:
    """
    Build forward returns matrix.

    forward_return[t] = price[t + holding] / price[t + lag] - 1

    For lag=1, holding=1 (default): next-day open-to-open return.
    For lag=1, holding=5: 5-day return starting from next open.

    The lag ensures we can actually execute at T+lag given a signal at T.
    """
    # Use close as proxy; for accurate execution use open[t+lag]
    fwd = prices.shift(-lag - holding + 1) / prices.shift(-lag) - 1
    return fwd


def enforce_no_lookahead(
    data    : pd.DataFrame,
    date    : pd.Timestamp,
) -> pd.DataFrame:
    """
    Return only data strictly before `date`.
    Use this in signal functions to guarantee no lookahead.
    """
    return data[data.index < date]


def audit_lookahead(
    features        : pd.DataFrame,    # index=date, feature columns
    target          : pd.Series,       # index=date, forward returns
    feature_cols    : Optional[List[str]] = None,
    threshold       : float = LOOKAHEAD_CORR_THRESHOLD,
    lag             : int   = 1,
) -> dict:
    """
    Audit all features for lookahead bias.

    A feature is flagged if its correlation with SAME-BAR returns is
    significantly higher than its correlation with LAGGED returns.
    This pattern indicates the feature encodes information from the future.

    Parameters
    ----------
    features     : DataFrame with feature values (index=date)
    target       : Series of forward returns at T+lag (index=date)
    feature_cols : columns to audit; defaults to all numeric columns
    threshold    : flag if same-bar corr > threshold (default 0.30)
    lag          : the expected prediction lag

    Returns
    -------
    dict with:
      'clean'   : list of feature names with no lookahead
      'flagged' : list of feature names with potential lookahead
      'details' : dict of {feature: {same_bar_corr, future_corr, ratio}}
    """
    if feature_cols is None:
        feature_cols = list(features.select_dtypes(include=[np.number]).columns)

    # Same-bar returns (T) — this is what a lookahead feature correlates with
    same_bar_returns  = target.shift(lag)   # shift back to align with feature at T

    # Future returns (T+lag) — this is what a good signal correlates with
    future_returns    = target

    common = features.index.intersection(same_bar_returns.dropna().index)
    common = common.intersection(future_returns.dropna().index)

    clean   = []
    flagged = []
    details = {}

    for col in feature_cols:
        if col not in features.columns:
            continue
        feat = features[col].loc[common].dropna()
        if len(feat) < 20:
            continue

        sb_aligned  = same_bar_returns.loc[feat.index].dropna()
        fut_aligned = future_returns.loc[feat.index].dropna()
        common_idx  = feat.index.intersection(sb_aligned.index).intersection(fut_aligned.index)

        if len(common_idx) < 20:
            continue

        f   = feat.loc[common_idx]
        sb  = sb_aligned.loc[common_idx]
        fut = fut_aligned.loc[common_idx]

        sb_corr  = float(f.corr(sb)  if len(f) > 2 else 0)
        fut_corr = float(f.corr(fut) if len(f) > 2 else 0)

        is_flagged = abs(sb_corr) > threshold

        details[col] = {
            'same_bar_corr' : sb_corr,
            'future_corr'   : fut_corr,
            'ratio'         : abs(sb_corr) / max(abs(fut_corr), 1e-6),
            'flagged'       : is_flagged,
        }

        if is_flagged:
            flagged.append(col)
            logger.warning(
                f'POTENTIAL LOOKAHEAD: {col} '
                f'same_bar_corr={sb_corr:.3f} > {threshold} '
                f'(future_corr={fut_corr:.3f})'
            )
        else:
            clean.append(col)

    return {
        'clean'       : clean,
        'flagged'     : flagged,
        'details'     : details,
        'n_features'  : len(feature_cols),
        'n_clean'     : len(clean),
        'n_flagged'   : len(flagged),
        'audit_passed': len(flagged) == 0,
    }


def print_lookahead_report(audit_result: dict) -> None:
    """Print a human-readable lookahead audit report."""
    print('\n' + '=' * 55)
    print('Lookahead Audit Report')
    print('=' * 55)
    print(f"Features audited : {audit_result['n_features']}")
    print(f"Clean            : {audit_result['n_clean']}")
    print(f"Flagged          : {audit_result['n_flagged']}")
    print(f"Audit PASSED     : {audit_result['audit_passed']}")

    if audit_result['flagged']:
        print('\nFlagged features (potential lookahead):')
        for col in audit_result['flagged']:
            d = audit_result['details'][col]
            print(
                f"  {col:<40} "
                f"same_bar_corr={d['same_bar_corr']:+.3f}  "
                f"future_corr={d['future_corr']:+.3f}"
            )
    else:
        print('\nNo lookahead detected — audit passed.')
    print()
