# backtesting/engine/fill_model.py
"""
Realistic fill simulation.

Rule: NEVER use close price as fill price. Close price is unknowable at
order submission time and using it inflates returns by 20-50bps per trade.

The model here uses arrival price (bar open) as the decision benchmark,
then adds half-spread + Almgren-Chriss market impact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

# Default parameters calibrated for mid-cap US equities, Alpaca paper fills
DEFAULT_SPREAD_BPS   = 5.0   # half-spread per side, bps
DEFAULT_IMPACT_FACTOR = 0.1  # market impact scaling coefficient


@dataclass
class FillResult:
    symbol      : str
    side        : Literal['buy', 'sell']
    qty         : float
    arrival_price: float
    fill_price   : float
    spread_cost_bps : float
    impact_cost_bps : float
    total_cost_bps  : float

    @property
    def is_bps(self) -> float:
        """Implementation shortfall in basis points (positive = cost)."""
        if self.side == 'buy':
            return (self.fill_price - self.arrival_price) / self.arrival_price * 1e4
        else:
            return (self.arrival_price - self.fill_price) / self.arrival_price * 1e4


class FillModel:
    """
    Institutional-grade fill simulation.

    Fill price = arrival_price ± (half_spread + market_impact)

    Arrival price: bar['open'] — the best proxy for price at order submission.
    Half-spread: flat bps cost for crossing the bid-ask spread.
    Market impact: Almgren-Chriss simplified — scales with sqrt(participation).
    """

    def __init__(
        self,
        spread_bps          : float = DEFAULT_SPREAD_BPS,
        market_impact_factor: float = DEFAULT_IMPACT_FACTOR,
    ):
        self.spread_bps           = spread_bps
        self.market_impact_factor = market_impact_factor

    def simulate_fill(
        self,
        symbol       : str,
        side         : Literal['buy', 'sell'],
        qty          : float,
        bar          : dict,
    ) -> FillResult:
        """
        Simulate a single order fill.

        Parameters
        ----------
        symbol  : ticker
        side    : 'buy' or 'sell'
        qty     : number of shares (positive)
        bar     : OHLCV dict with keys 'open', 'close', 'volume'
                  ('high', 'low' used for sanity clamp)

        Returns
        -------
        FillResult with fill_price and cost breakdown
        """
        arrival = float(bar['open'])
        if arrival <= 0:
            logger.warning(f'{symbol}: invalid arrival price {arrival}, using close')
            arrival = float(bar.get('close', 1.0))

        # Half-spread cost (bps → price)
        spread_dollars = arrival * (self.spread_bps / 1e4)

        # Almgren-Chriss market impact:
        #   impact_bps = factor * price * sqrt(participation_rate)
        adv = float(bar.get('volume', 1e6)) * arrival   # dollar volume proxy
        order_value = abs(qty) * arrival
        participation = order_value / adv if adv > 0 else 0.01
        participation = min(participation, 0.3)          # cap at 30% ADV
        impact_dollars = (
            self.market_impact_factor * arrival * (participation ** 0.5)
        )

        if side == 'buy':
            fill = arrival + spread_dollars + impact_dollars
        else:
            fill = arrival - spread_dollars - impact_dollars

        # Sanity clamp: fill can't be worse than high (buy) or low (sell)
        bar_high = float(bar.get('high', fill + 1))
        bar_low  = float(bar.get('low',  fill - 1))
        if side == 'buy':
            fill = min(fill, bar_high)
        else:
            fill = max(fill, bar_low)

        spread_bps  = spread_dollars  / arrival * 1e4
        impact_bps  = impact_dollars  / arrival * 1e4
        total_bps   = spread_bps + impact_bps

        return FillResult(
            symbol          = symbol,
            side            = side,
            qty             = abs(qty),
            arrival_price   = arrival,
            fill_price      = fill,
            spread_cost_bps = spread_bps,
            impact_cost_bps = impact_bps,
            total_cost_bps  = total_bps,
        )
