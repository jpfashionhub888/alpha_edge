# backtesting/engine/cost_model.py
"""
Transaction cost model.

Total cost = commission + spread + market impact + slippage.
All figures in basis points (bps = 0.01%).

Calibrated for Alpaca paper/live trading. Adjust for IBKR when moving to
Phase 2 live execution.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostBreakdown:
    commission_bps : float
    spread_bps     : float
    impact_bps     : float
    slippage_bps   : float
    total_bps      : float

    @property
    def total_pct(self) -> float:
        return self.total_bps / 1e4

    def __str__(self) -> str:
        return (
            f'Total: {self.total_bps:.1f}bps '
            f'(commission={self.commission_bps:.1f} '
            f'spread={self.spread_bps:.1f} '
            f'impact={self.impact_bps:.1f} '
            f'slippage={self.slippage_bps:.1f})'
        )


class TransactionCostModel:
    """
    Round-trip transaction cost model.

    Gate 2 of signal validation requires:
      - Gross Sharpe > 2.0
      - Net Sharpe (after costs) > 1.0
      - Transaction costs < 40% of gross return

    With a typical signal holding period of 5-20 days and ~5-10 trades/month,
    total costs of 15-25bps per trade are achievable and must be budgeted.
    """

    def __init__(
        self,
        commission_bps: float = 0.5,   # Alpaca/IBKR per leg
        spread_bps    : float = 3.0,   # typical mid-cap spread (half on each side)
        slippage_bps  : float = 1.0,   # timing slippage vs model signal price
        impact_factor : float = 10.0,  # Almgren-Chriss coefficient
        annual_volatility_ref: float = 0.02,  # 2% daily vol reference
    ):
        self.commission_bps = commission_bps
        self.spread_bps     = spread_bps
        self.slippage_bps   = slippage_bps
        self.impact_factor  = impact_factor
        self.vol_ref        = annual_volatility_ref

    def one_way_cost_bps(
        self,
        order_value      : float,
        adv              : float,
        daily_volatility : float = 0.02,
    ) -> CostBreakdown:
        """
        One-way transaction cost in bps.

        Parameters
        ----------
        order_value      : USD notional of the order
        adv              : average daily dollar volume of the stock
        daily_volatility : realized daily volatility of the stock

        Returns
        -------
        CostBreakdown with per-component and total bps
        """
        participation = order_value / adv if adv > 0 else 0.01
        participation = min(participation, 0.30)

        # Market impact scales with sqrt(participation) and relative vol
        vol_adj  = (daily_volatility / self.vol_ref) if self.vol_ref > 0 else 1.0
        impact   = self.impact_factor * (participation ** 0.5) * vol_adj

        total = (
            self.commission_bps
            + self.spread_bps
            + impact
            + self.slippage_bps
        )

        return CostBreakdown(
            commission_bps = self.commission_bps,
            spread_bps     = self.spread_bps,
            impact_bps     = impact,
            slippage_bps   = self.slippage_bps,
            total_bps      = total,
        )

    def round_trip_cost_bps(
        self,
        order_value      : float,
        adv              : float,
        daily_volatility : float = 0.02,
    ) -> float:
        """Full round-trip cost (entry + exit) in bps."""
        one_way = self.one_way_cost_bps(order_value, adv, daily_volatility)
        return one_way.total_bps * 2

    def passes_cost_gate(
        self,
        gross_return_bps : float,
        order_value      : float,
        adv              : float,
        daily_volatility : float = 0.02,
        max_cost_fraction: float = 0.40,
    ) -> tuple[bool, float]:
        """
        Gate 2 cost survival check.

        Returns (passes: bool, cost_fraction: float)
        where cost_fraction = round_trip_cost / gross_return.

        A signal must survive: cost_fraction < max_cost_fraction (default 40%).
        """
        rt_cost = self.round_trip_cost_bps(order_value, adv, daily_volatility)
        if gross_return_bps <= 0:
            return False, float('inf')
        fraction = rt_cost / gross_return_bps
        return fraction < max_cost_fraction, fraction
