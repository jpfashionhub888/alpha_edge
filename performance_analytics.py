# performance_analytics.py
# ALPHAEDGE - Performance Analytics Engine V2
#
# Fixes applied:
#   - Removed hardcoded ../bharat_edge/ and ../crypto_edge/ paths
#     (system no longer crashes if sibling projects don't exist)
#   - Added real Sharpe ratio calculation (annualised, risk-free=0)
#   - Added max drawdown calculation
#   - Report now includes unrealized P&L on open positions
#   - Trade date parsing made robust (handles missing/malformed dates)
#   - AlphaEdge-only by default; multi-system optional via config

import json
import os
import math
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PerformanceAnalytics:
    """
    Weekly performance analyzer.

    Now AlphaEdge-only by default.
    Multi-system reporting available if sibling project paths
    are explicitly configured (no silent failures on missing dirs).
    """

    def __init__(self, extra_systems=None):
        """
        extra_systems: optional list of dicts for multi-system reporting
          e.g. [
            {'name': 'BharatEdge', 'path': '/path/to/bharat_trades.json',
             'currency': 'Rs', 'starting_capital': 100000},
            {'name': 'CryptoEdge', 'path': '/path/to/crypto_trades.json',
             'currency': '$',  'starting_capital': 10000},
          ]
        """
        self.extra_systems = extra_systems or []

    def load_portfolio(self, filepath):
        """Load portfolio data from JSON file. Returns {} on missing/error."""
        if not os.path.exists(filepath):
            logger.debug(f"Portfolio file not found: {filepath}")
            return {}
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load portfolio {filepath}: {e}")
            return {}

    def _parse_date(self, date_str):
        """Parse ISO datetime string robustly. Returns None on failure."""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(str(date_str))
        except Exception:
            return None

    def calculate_metrics(self, trades, starting_capital,
                           current_value, open_positions=None, days_back=7):
        """
        Calculate performance metrics including Sharpe ratio and max drawdown.

        Parameters
        ----------
        trades          : full trade history list
        starting_capital: float
        current_value   : float (cash only)
        open_positions  : dict of open positions (for unrealized P&L)
        days_back       : int (window for recent-trade stats)
        """
        open_positions = open_positions or {}

        # Unrealized P&L on open positions
        unrealized_pnl = sum(
            pos.get('shares', pos.get('units', 0)) *
            (pos.get('current_price', pos.get('entry_price', 0)) - pos.get('entry_price', 0))
            for pos in open_positions.values()
        )
        total_value  = current_value + sum(
            pos.get('shares', pos.get('units', 0)) * pos.get('current_price', pos.get('entry_price', 0))
            for pos in open_positions.values()
        )
        total_pnl    = total_value - starting_capital
        total_pct    = total_pnl / starting_capital * 100 if starting_capital > 0 else 0

        if not trades:
            return self._empty_metrics(total_pnl, total_pct, unrealized_pnl)

        # Filter recent closed (SELL) trades
        cutoff = datetime.now() - timedelta(days=days_back)
        sells  = []
        for t in trades:
            if t.get('action') != 'SELL':
                continue
            trade_date = self._parse_date(t.get('date', ''))
            if trade_date and trade_date >= cutoff:
                sells.append(t)

        if not sells:
            return self._empty_metrics(total_pnl, total_pct, unrealized_pnl)

        wins   = [t for t in sells if t.get('pnl', 0) > 0]
        losses = [t for t in sells if t.get('pnl', 0) <= 0]
        total  = len(sells)

        win_rate     = len(wins) / total * 100 if total > 0 else 0
        avg_win      = sum(t.get('pnl', 0) for t in wins)   / len(wins)   if wins   else 0
        avg_loss     = abs(sum(t.get('pnl', 0) for t in losses) / len(losses)) if losses else 0
        total_wins_  = sum(t.get('pnl', 0) for t in wins)
        total_losses_= abs(sum(t.get('pnl', 0) for t in losses))
        profit_factor= total_wins_ / total_losses_ if total_losses_ > 0 else float('inf')

        best_trade  = max(sells, key=lambda x: x.get('pnl', 0))
        worst_trade = min(sells, key=lambda x: x.get('pnl', 0))

        # Sharpe ratio (annualised, risk-free rate = 0)
        # Uses per-trade returns as a proxy
        returns = [t.get('pnl_pct', 0) for t in sells if 'pnl_pct' in t]
        sharpe  = self._sharpe(returns)

        # Max drawdown from all SELL trade P&L sequence
        max_dd = self._max_drawdown_from_trades(trades)

        return {
            'total_trades'   : total,
            'wins'           : len(wins),
            'losses'         : len(losses),
            'win_rate'       : win_rate,
            'total_pnl'      : total_pnl,
            'total_pct'      : total_pct,
            'unrealized_pnl' : unrealized_pnl,
            'avg_win'        : avg_win,
            'avg_loss'       : avg_loss,
            'profit_factor'  : profit_factor,
            'sharpe_ratio'   : sharpe,
            'max_drawdown'   : max_dd,
            'best_trade'     : best_trade,
            'worst_trade'    : worst_trade,
        }

    def _empty_metrics(self, total_pnl, total_pct, unrealized_pnl):
        return {
            'total_trades'   : 0,
            'wins'           : 0,
            'losses'         : 0,
            'win_rate'       : 0,
            'total_pnl'      : total_pnl,
            'total_pct'      : total_pct,
            'unrealized_pnl' : unrealized_pnl,
            'avg_win'        : 0,
            'avg_loss'       : 0,
            'profit_factor'  : 0,
            'sharpe_ratio'   : 0,
            'max_drawdown'   : 0,
            'best_trade'     : None,
            'worst_trade'    : None,
        }

    @staticmethod
    def _sharpe(returns):
        """Annualised Sharpe ratio from a list of per-trade return fractions."""
        if len(returns) < 2:
            return 0.0
        n    = len(returns)
        mean = sum(returns) / n
        var  = sum((r - mean) ** 2 for r in returns) / (n - 1)
        std  = math.sqrt(var)
        if std == 0:
            return 0.0
        # Annualise assuming ~252 trading days / year
        return round((mean / std) * math.sqrt(252), 2)

    @staticmethod
    def _max_drawdown_from_trades(trades):
        """
        Compute max drawdown from cumulative P&L sequence of all SELL trades.
        Returns a negative fraction (e.g. -0.12 = -12% drawdown).
        """
        sells = [t for t in trades if t.get('action') == 'SELL']
        if not sells:
            return 0.0
        cumulative = 0.0
        peak       = 0.0
        max_dd     = 0.0
        for t in sells:
            cumulative += t.get('pnl', 0)
            if cumulative > peak:
                peak = cumulative
            dd = (cumulative - peak) / (abs(peak) + 1e-9)
            if dd < max_dd:
                max_dd = dd
        return round(max_dd, 4)

    def generate_report(self, days_back=7):
        """Generate performance report (AlphaEdge + any configured extras)."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M')

        # AlphaEdge (always present)
        alpha_portfolio = self.load_portfolio('logs/paper_trades.json')
        alpha_positions = alpha_portfolio.get('positions', {})
        alpha_value     = alpha_portfolio.get('capital', 10000)
        alpha_metrics   = self.calculate_metrics(
            trades           = alpha_portfolio.get('trade_history', []),
            starting_capital = alpha_portfolio.get('starting_capital', 10000),
            current_value    = alpha_value,
            open_positions   = alpha_positions,
            days_back        = days_back,
        )

        def fmt_pnl(pnl, pct, currency='$'):
            sign = '+' if pnl >= 0 else ''
            return f"{sign}{currency}{abs(pnl):,.2f} ({sign}{pct:.2f}%)"

        def fmt_trade(trade, currency='$'):
            if not trade:
                return 'N/A'
            pnl  = trade.get('pnl', 0)
            sym  = trade.get('symbol', '?')
            sign = '+' if pnl >= 0 else ''
            return f"{sym}: {sign}{currency}{abs(pnl):.2f}"

        m = alpha_metrics
        report = f"""ALPHAEDGE WEEKLY PERFORMANCE REPORT
=====================================
Period: Last {days_back} days | Date: {now}

📊 ALPHAEDGE (US Stocks & Crypto):
  Portfolio:     ${alpha_portfolio.get('capital', 0) + sum(p.get('shares', p.get('units', 0)) * p.get('current_price', p.get('entry_price', 0)) for p in alpha_positions.values()):,.2f}
  Realized P&L:  {fmt_pnl(m['total_pnl'], m['total_pct'])}
  Unrealized:    ${m['unrealized_pnl']:+,.2f}
  Trades:        {m['total_trades']} | Win Rate: {m['win_rate']:.1f}%
  Avg Win:       ${m['avg_win']:.2f} | Avg Loss: ${m['avg_loss']:.2f}
  Profit Factor: {m['profit_factor']:.2f}
  Sharpe Ratio:  {m['sharpe_ratio']:.2f}
  Max Drawdown:  {m['max_drawdown']:.2%}
  Best Trade:    {fmt_trade(m['best_trade'])}
  Worst Trade:   {fmt_trade(m['worst_trade'])}"""

        # Optional extra systems
        for sys in self.extra_systems:
            path = sys.get('path', '')
            if not os.path.exists(path):
                report += f"\n\n⚠️  {sys.get('name', 'Unknown')}: portfolio file not found at {path}"
                continue
            portfolio = self.load_portfolio(path)
            positions = portfolio.get('positions', {})
            value     = portfolio.get('capital', sys.get('starting_capital', 0))
            metrics   = self.calculate_metrics(
                trades           = portfolio.get('trade_history', []),
                starting_capital = portfolio.get('starting_capital', sys.get('starting_capital', 0)),
                current_value    = value,
                open_positions   = positions,
                days_back        = days_back,
            )
            cur = sys.get('currency', '$')
            sm  = metrics
            report += f"""

📊 {sys.get('name', 'System')}:
  Portfolio:     {cur}{value:,.2f}
  Realized P&L:  {fmt_pnl(sm['total_pnl'], sm['total_pct'], cur)}
  Unrealized:    {cur}{sm['unrealized_pnl']:+,.2f}
  Trades:        {sm['total_trades']} | Win Rate: {sm['win_rate']:.1f}%
  Sharpe Ratio:  {sm['sharpe_ratio']:.2f}
  Max Drawdown:  {sm['max_drawdown']:.2%}"""

        report += f"""

Next Report: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}
AlphaEdge Analytics V2"""

        return report

    def send_report(self, telegram_bot, days_back=7):
        print("\n  Generating Performance Report...")
        report = self.generate_report(days_back)
        print(report)
        if telegram_bot:
            telegram_bot.send_message(report)
            print("  Performance report sent! ✅")
        return report

    def should_run_today(self):
        """Run every Sunday."""
        return datetime.now().strftime('%A') == 'Sunday'


if __name__ == '__main__':
    print("\nTesting Performance Analytics V2...")
    analytics = PerformanceAnalytics()
    report    = analytics.generate_report(days_back=30)
    print(report)
