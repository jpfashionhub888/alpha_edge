# models/ensemble.py

"""
EnsembleStrategy V2

Fixes applied:
- Vectorised signal generation replaces row iteration
  (Bug 1 — df.loc[idx] broke on duplicate datetime index)
- trend_strength column validated with safe default (Bug 2)
- Dead first strategy_return assignment removed (Bug 3)
- downtrend_brave removed — contradicted scanner.py AVOID (Flaw 1)
- Signal tiers aligned with PaperTrader._signal_to_size_multiplier
  (Flaw 2)
- Transaction costs applied to strategy_return (Flaw 3)
- Print statements replaced with logger calls (Flaw 4)
- managed_return validated after RiskManager.apply() (Risk 1)
- self.signals removed — caller owns the dataframe (Risk 3)
"""

import logging

import numpy as np
import pandas as pd

from risk.manager import RiskManager

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  CONSTANTS — kept in sync with paper_trader.py                     #
# ------------------------------------------------------------------ #

# Match PaperTrader._signal_to_size_multiplier() tiers exactly
SIGNAL_TIERS = {
    'full'    : 1.00,   # full position, high conviction
    'three_q' : 0.75,   # three-quarter position
    'half'    : 0.50,   # half position
    'quarter' : 0.25,   # minimum conviction
}

# Match PaperTrader defaults for cost simulation
SLIPPAGE_PCT     = 0.0005   # 0.05% per fill
COMMISSION       = 1.0      # $1 flat per trade
STARTING_CAPITAL = 10_000.0 # used to express commission as % return

# Round-trip cost as a fraction of capital
ROUND_TRIP_COST = (
    SLIPPAGE_PCT * 2                          # buy + sell slippage
    + (COMMISSION * 2) / STARTING_CAPITAL     # buy + sell commission
)


class EnsembleStrategy:
    """
    Adaptive signal generator.

    Regime rules
    ------------
    uptrend  : aggressive — lower prediction threshold
    sideways : cautious  — higher prediction threshold
    volatile : minimal   — small position only
    downtrend: no trade  — consistent with scanner.py AVOID

    Signal values use the same discrete tiers as PaperTrader
    so backtest position sizing matches live behaviour.
    """
    def __init__(
        self,
        base_stop_loss: float = 0.03,
        reward_risk_ratio: float = 2.5,
        trailing_stop_multiplier: float = 0.8,
        max_daily_loss: float = 0.02,
        max_portfolio_risk: float = 0.06,
    ):
        self.risk_manager = RiskManager(
            base_stop_loss           = base_stop_loss,
            reward_risk_ratio        = reward_risk_ratio,
            trailing_stop_multiplier = trailing_stop_multiplier,
            max_daily_loss           = max_daily_loss,
            max_portfolio_risk       = max_portfolio_risk,
        )

    # ---------------------------------------------------------------- #
    #  PUBLIC                                                            #
    # ---------------------------------------------------------------- #

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate adaptive trading signals for a fold dataframe.

        Parameters
        ----------
        df : output of WalkForwardBacktester fold — must contain:
             'prediction', 'regime', 'returns'
             'trend_strength' (optional — defaults to 50)

        Returns
        -------
        df with added columns:
            'signal'          — position size tier (0 / 0.25 / 0.75 / 1.0)
            'signal_reason'   — human-readable reason string
            'strategy_return' — risk-managed return after costs
            'managed_return'  — alias (same as strategy_return)
            'managed_signal'  — signal after risk manager filter
        """
        df = df.copy()

        # ── Validate required columns ─────────────────────────────
        required = ['prediction', 'regime', 'returns']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"generate_signals() missing required columns: "
                f"{missing}"
            )

        # ── Safe default for optional column ─────────────────────
        if 'trend_strength' not in df.columns:
            logger.warning(
                "trend_strength column missing — "
                "defaulting to 50 (neutral). "
                "Ensure RegimeDetector produces this column."
            )
            df['trend_strength'] = 50.0

        # ── Vectorised signal generation ──────────────────────────
        # Fix Bug 1: replaces row-by-row df.loc[idx] iteration
        # which broke silently on duplicate datetime indexes
        # (multiple stocks on same date after pd.concat)
        from config import settings

        df['signal']        = 0.0
        df['signal_reason'] = 'no_trade'

        # Calculate Kelly weights if enabled
        use_kelly = getattr(settings, 'KELLY_POSITION_SIZING', False)
        if use_kelly:
            p = df['prediction'].astype(float)
            b = getattr(settings, 'KELLY_REWARD_RISK_RATIO', 2.5)
            # Kelly formula: f* = (p * (b + 1) - 1) / b
            kelly_f = (p * (b + 1.0) - 1.0) / b
            kelly_f = kelly_f.clip(lower=0.0)
            
            # Apply Half-Kelly multiplier and cap at max position size
            kelly_base = kelly_f * getattr(settings, 'KELLY_MULTIPLIER', 0.5)
            kelly_base = kelly_base.clip(upper=settings.MAX_POSITION_SIZE)
        else:
            kelly_base = None

        # Helper to get signal size based on mode and multiplier
        def get_signal_size(multiplier, mask_size=None):
            if use_kelly:
                return kelly_base * multiplier
            return SIGNAL_TIERS[mask_size] if mask_size in SIGNAL_TIERS else multiplier

        # ── Uptrend — trade aggressively ──────────────────────────
        up = df['regime'] == 'uptrend'

        # Full position: strong prediction in uptrend
        mask = up & (df['prediction'] > 0.55)
        df.loc[mask, 'signal']        = get_signal_size(1.0, 'full')
        df.loc[mask, 'signal_reason'] = 'uptrend_strong'

        # Three-quarter: moderate prediction in uptrend
        mask = (
            up
            & (df['prediction'] > 0.50)
            & (df['prediction'] <= 0.55)
            & (df['trend_strength'] > 65)
        )
        df.loc[mask, 'signal']        = get_signal_size(0.75, 'three_q')
        df.loc[mask, 'signal_reason'] = 'uptrend_moderate'

        # ── Sideways — no trade ───────────────────────────────────
        # No edge in sideways — sitting out prevents drawdown
        # ── Sideways — trade only on very high confidence ─────────
        sw = df['regime'] == 'sideways'

        # High confidence sideways trade — reduced position size
        mask = sw & (df['prediction'] > 0.55)
        df.loc[mask, 'signal']        = get_signal_size(0.25, 'quarter')
        df.loc[mask, 'signal_reason'] = 'sideways_high_conf'

        # Low confidence sideways — no trade
        mask = sw & (df['prediction'] <= 0.55)
        df.loc[mask, 'signal']        = 0.0
        df.loc[mask, 'signal_reason'] = 'sideways_no_trade'

        # ── Volatile — minimal position only ─────────────────────
        vo   = df['regime'] == 'volatile'
        mask = vo & (df['prediction'] > 0.60)
        df.loc[mask, 'signal']        = get_signal_size(0.25, 'quarter')
        df.loc[mask, 'signal_reason'] = 'volatile_small'

        # ── Downtrend — no trade ──────────────────────────────────
        dn = df['regime'] == 'downtrend'
        df.loc[dn, 'signal']        = 0.0
        df.loc[dn, 'signal_reason'] = 'downtrend_avoid'

        # ── Apply transaction costs ───────────────────────────────
        df = self._apply_costs(df)

        # ── Apply risk management ─────────────────────────────────
        try:
            df = self.risk_manager.apply(df)

            if 'managed_return' not in df.columns:
                raise ValueError(
                    "RiskManager.apply() did not produce "
                    "'managed_return' column. "
                    "Check risk/manager.py."
                )
            if 'managed_signal' not in df.columns:
                raise ValueError(
                    "RiskManager.apply() did not produce "
                    "'managed_signal' column. "
                    "Check risk/manager.py."
                )

            # strategy_return IS the managed return — single
            # assignment (Bug 3 fix: removed dead first assignment)
            df['strategy_return'] = df['managed_return']

        except Exception as e:
            import traceback
            logger.error(
                "RiskManager failed: %s — "
                "falling back to cost-adjusted raw returns", e
            )
            logger.error("Full traceback: %s", traceback.format_exc())

            # Fallback: use cost-adjusted raw signal returns
            df['strategy_return'] = (
                df['signal'] * df['returns']
                - df['cost_return']
            )
            df['managed_return'] = df['strategy_return']
            df['managed_signal'] = df['signal']

        # ── Log stats (not print — called once per stock per fold) ─
        total  = len(df)
        active = int((df['managed_signal'] > 0).sum())
        pct    = active / total * 100 if total > 0 else 0.0

        logger.info(
            "Signal stats: %d total | %d active (%.1f%%)",
            total, active, pct,
        )

        reasons = df['signal_reason'].value_counts()
        for reason, count in reasons.items():
            logger.debug(
                "  %-30s %d (%.1f%%)",
                reason, count, count / total * 100,
            )

        # Fix Risk 3: do not store df in self.signals
        # walk_forward.py owns the returned dataframe
        return df

    # ---------------------------------------------------------------- #
    #  PRIVATE                                                           #
    # ---------------------------------------------------------------- #

    def _apply_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduct round-trip slippage and commission from each trade.

        Cost is applied on trade entry days only.
        ROUND_TRIP_COST covers both the entry and exit fill.

        Adds 'cost_return' column for transparency.
        """
        df = df.copy()

        # Detect entries: signal goes from 0 → positive
        prev_signal      = df['signal'].shift(1).fillna(0)
        entries          = (prev_signal == 0) & (df['signal'] > 0)

        df['cost_return']            = 0.0
        df.loc[entries, 'cost_return'] = ROUND_TRIP_COST

        return df