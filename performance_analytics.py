# performance_analytics.py
# TRADING EMPIRE - Performance Analytics Engine
#
# FIX P1-3: calculate_metrics() now counts PARTIAL_SELL alongside SELL
#   when computing wins, losses, avg_win/loss, profit_factor, and
#   best/worst trade. Previously partial exits were silently excluded,
#   understating realized P&L and win count in the weekly report.

import json
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Exit action types that represent realized P&L events
REALIZED_ACTIONS = ('SELL', 'PARTIAL_SELL')   # P1-3


class PerformanceAnalytics:
    """
    Weekly performance analyzer for all trading systems.
    Calculates win rates, Sharpe ratio, drawdown etc.
    """

    def load_trades(self, filepath):
        """Load trade history from JSON file."""
        if not os.path.exists(filepath):
            return []
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data.get('trade_history', [])
        except Exception as e:
            logger.warning(f"Load failed for {filepath}: {e}")
            return []

    def load_portfolio(self, filepath):
        """Load portfolio data from JSON file."""
        if not os.path.exists(filepath):
            return {}
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _empty_metrics(self, current_value, starting_capital, days_back):
        """Return a zeroed metrics dict when there are no qualifying trades."""
        return {
            'total_trades' : 0,
            'wins'         : 0,
            'losses'       : 0,
            'win_rate'     : 0,
            'total_pnl'    : current_value - starting_capital,
            'total_pct'    : (current_value - starting_capital) / starting_capital * 100
                             if starting_capital else 0,
            'avg_win'      : 0,
            'avg_loss'     : 0,
            'profit_factor': 'N/A',
            'best_trade'   : None,
            'worst_trade'  : None,
        }

    def calculate_metrics(self, trades, starting_capital,
                          current_value, days_back=7):
        """
        Calculate performance metrics.

        P1-3 FIX: Counts both SELL and PARTIAL_SELL as realized trades.
        """
        if not trades:
            return self._empty_metrics(current_value, starting_capital, days_back)

        # Filter to realized exit events within window  [P1-3]
        cutoff  = datetime.now() - timedelta(days=days_back)
        exits   = []
        for t in trades:
            if t.get('action') not in REALIZED_ACTIONS:   # P1-3: was == 'SELL'
                continue
            try:
                trade_date = datetime.fromisoformat(t.get('date', ''))
                if trade_date >= cutoff:
                    exits.append(t)
            except Exception:
                continue

        if not exits:
            return self._empty_metrics(current_value, starting_capital, days_back)

        wins   = [t for t in exits if t.get('pnl', 0) > 0]
        losses = [t for t in exits if t.get('pnl', 0) <= 0]
        total  = len(exits)

        win_rate  = len(wins)  / total * 100 if total > 0 else 0
        avg_win   = sum(t.get('pnl', 0) for t in wins)   / len(wins)   if wins   else 0
        avg_loss  = abs(sum(t.get('pnl', 0) for t in losses) / len(losses)) if losses else 0

        total_wins   = sum(t.get('pnl', 0) for t in wins)
        total_losses = abs(sum(t.get('pnl', 0) for t in losses))

        # Profit factor: handle zero-loss case correctly
        if total_losses == 0 and total_wins > 0:
            profit_factor = float('inf')   # no losses at all
        elif total_losses == 0:
            profit_factor = 'N/A'          # no trades resolved yet
        else:
            profit_factor = total_wins / total_losses

        best_trade  = max(exits, key=lambda x: x.get('pnl', 0)) if exits else None
        worst_trade = min(exits, key=lambda x: x.get('pnl', 0)) if exits else None

        total_pnl = current_value - starting_capital
        total_pct = total_pnl / starting_capital * 100 if starting_capital else 0

        return {
            'total_trades' : total,
            'wins'         : len(wins),
            'losses'       : len(losses),
            'win_rate'     : win_rate,
            'total_pnl'    : total_pnl,
            'total_pct'    : total_pct,
            'avg_win'      : avg_win,
            'avg_loss'     : avg_loss,
            'profit_factor': profit_factor,
            'best_trade'   : best_trade,
            'worst_trade'  : worst_trade,
        }

    def generate_report(self, days_back=7):
        """Generate weekly report for all 3 systems."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M')

        alpha_portfolio  = self.load_portfolio('logs/paper_trades_stocks_only.json')
        bharat_portfolio = self.load_portfolio('../bharat_edge/logs/bharat_trades.json')
        crypto_portfolio = self.load_portfolio('../crypto_edge/logs/crypto_trades.json')

        # AlphaEdge metrics
        alpha_value = alpha_portfolio.get('capital', 10000) + sum(
            pos.get('shares', 0) * pos.get('current_price', pos.get('entry_price', 0))
            for pos in alpha_portfolio.get('positions', {}).values()
        )
        alpha_metrics = self.calculate_metrics(
            trades          = alpha_portfolio.get('trade_history', []),
            starting_capital= alpha_portfolio.get('starting_capital', 10000),
            current_value   = alpha_value,
            days_back       = days_back,
        )

        # BharatEdge metrics
        bharat_value = bharat_portfolio.get('capital', 100000) + sum(
            pos.get('shares', 0) * pos.get('current_price', pos.get('entry_price', 0))
            for pos in bharat_portfolio.get('positions', {}).values()
        )
        bharat_metrics = self.calculate_metrics(
            trades          = bharat_portfolio.get('trade_history', []),
            starting_capital= bharat_portfolio.get('starting_capital', 100000),
            current_value   = bharat_value,
            days_back       = days_back,
        )

        # CryptoEdge metrics
        crypto_value = crypto_portfolio.get('capital', 10000) + sum(
            pos.get('units', 0) * pos.get('current_price', pos.get('entry_price', 0))
            for pos in crypto_portfolio.get('positions', {}).values()
        )
        crypto_metrics = self.calculate_metrics(
            trades          = crypto_portfolio.get('trade_history', []),
            starting_capital= crypto_portfolio.get('starting_capital', 10000),
            current_value   = crypto_value,
            days_back       = days_back,
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

        def fmt_pf(pf):
            if pf == float('inf'):
                return '∞ (no losses)'
            if pf == 'N/A':
                return 'N/A'
            return f"{pf:.2f}x"

        report = f"""TRADING EMPIRE WEEKLY REPORT
==============================
Period: Last {days_back} days
Date:   {now}

ALPHAEDGE (US Stocks):
  Portfolio:     ${alpha_value:,.2f}
  P&L:           {fmt_pnl(alpha_metrics['total_pnl'], alpha_metrics['total_pct'])}
  Trades:        {alpha_metrics['total_trades']} | Win Rate: {alpha_metrics['win_rate']:.1f}%
  Profit Factor: {fmt_pf(alpha_metrics['profit_factor'])}
  Best:          {fmt_trade(alpha_metrics['best_trade'])}
  Worst:         {fmt_trade(alpha_metrics['worst_trade'])}

BHARATEDGE (Indian Stocks):
  Portfolio:     Rs{bharat_value:,.2f}
  P&L:           {fmt_pnl(bharat_metrics['total_pnl'], bharat_metrics['total_pct'], 'Rs')}
  Trades:        {bharat_metrics['total_trades']} | Win Rate: {bharat_metrics['win_rate']:.1f}%
  Profit Factor: {fmt_pf(bharat_metrics['profit_factor'])}
  Best:          {fmt_trade(bharat_metrics['best_trade'], 'Rs')}
  Worst:         {fmt_trade(bharat_metrics['worst_trade'], 'Rs')}

CRYPTOEDGE (Crypto 24/7):
  Portfolio:     ${crypto_value:,.2f} USDT
  P&L:           {fmt_pnl(crypto_metrics['total_pnl'], crypto_metrics['total_pct'])}
  Trades:        {crypto_metrics['total_trades']} | Win Rate: {crypto_metrics['win_rate']:.1f}%
  Profit Factor: {fmt_pf(crypto_metrics['profit_factor'])}
  Best:          {fmt_trade(crypto_metrics['best_trade'])}
  Worst:         {fmt_trade(crypto_metrics['worst_trade'])}

COMBINED EMPIRE:
  Total Assets:  ${ alpha_value + crypto_value:,.2f} + Rs{bharat_value:,.2f}
  Next Report:   {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}

Trading Empire AI - Automated"""

        return report

    def send_report(self, telegram_bot, days_back=7):
        """Generate and send report to Telegram."""
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
    print("\nTesting Performance Analytics...")
    analytics = PerformanceAnalytics()
    report    = analytics.generate_report(days_back=30)
    print(report)
