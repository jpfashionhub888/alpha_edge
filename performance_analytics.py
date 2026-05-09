# performance_analytics.py
# TRADING EMPIRE - Performance Analytics Engine
# Analyzes ALL 3 systems weekly
# Sends comprehensive report to Telegram

import json
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


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

    def calculate_metrics(self, trades, starting_capital,
                          current_value, days_back=7):
        """Calculate performance metrics."""
        if not trades:
            return {
                'total_trades' : 0,
                'wins'         : 0,
                'losses'       : 0,
                'win_rate'     : 0,
                'total_pnl'    : current_value - starting_capital,
                'total_pct'    : (current_value - starting_capital) / starting_capital * 100,
                'avg_win'      : 0,
                'avg_loss'     : 0,
                'profit_factor': 0,
                'best_trade'   : None,
                'worst_trade'  : None,
            }

        # Filter recent trades
        cutoff = datetime.now() - timedelta(days=days_back)
        sells  = []

        for t in trades:
            if t.get('action') != 'SELL':
                continue
            try:
                trade_date = datetime.fromisoformat(
                    t.get('date', '')
                )
                if trade_date >= cutoff:
                    sells.append(t)
            except Exception:
                continue

        if not sells:
            return {
                'total_trades' : 0,
                'wins'         : 0,
                'losses'       : 0,
                'win_rate'     : 0,
                'total_pnl'    : current_value - starting_capital,
                'total_pct'    : (current_value - starting_capital) / starting_capital * 100,
                'avg_win'      : 0,
                'avg_loss'     : 0,
                'profit_factor': 0,
                'best_trade'   : None,
                'worst_trade'  : None,
            }

        wins   = [t for t in sells if t.get('pnl', 0) > 0]
        losses = [t for t in sells if t.get('pnl', 0) <= 0]
        total  = len(sells)

        win_rate = len(wins) / total * 100 if total > 0 else 0

        avg_win  = sum(t.get('pnl', 0) for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t.get('pnl', 0) for t in losses) / len(losses)) if losses else 0

        total_wins   = sum(t.get('pnl', 0) for t in wins)
        total_losses = abs(sum(t.get('pnl', 0) for t in losses))
        profit_factor= total_wins / total_losses if total_losses > 0 else 0

        best_trade  = max(sells, key=lambda x: x.get('pnl', 0)) if sells else None
        worst_trade = min(sells, key=lambda x: x.get('pnl', 0)) if sells else None

        total_pnl = current_value - starting_capital
        total_pct = total_pnl / starting_capital * 100

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

        # Load all portfolios
        alpha_portfolio  = self.load_portfolio(
            'logs/paper_trades.json'
        )
        bharat_portfolio = self.load_portfolio(
            '../bharat_edge/logs/bharat_trades.json'
        )
        crypto_portfolio = self.load_portfolio(
            '../crypto_edge/logs/crypto_trades.json'
        )

        # AlphaEdge metrics
        alpha_value   = alpha_portfolio.get('capital', 10000) + sum(
            pos.get('shares', 0) * pos.get('current_price', pos.get('entry_price', 0))
            for pos in alpha_portfolio.get('positions', {}).values()
        )
        alpha_metrics = self.calculate_metrics(
            trades           = alpha_portfolio.get('trade_history', []),
            starting_capital = alpha_portfolio.get('starting_capital', 10000),
            current_value    = alpha_value,
            days_back        = days_back,
        )

        # BharatEdge metrics
        bharat_value   = bharat_portfolio.get('capital', 100000) + sum(
            pos.get('shares', 0) * pos.get('current_price', pos.get('entry_price', 0))
            for pos in bharat_portfolio.get('positions', {}).values()
        )
        bharat_metrics = self.calculate_metrics(
            trades           = bharat_portfolio.get('trade_history', []),
            starting_capital = bharat_portfolio.get('starting_capital', 100000),
            current_value    = bharat_value,
            days_back        = days_back,
        )

        # CryptoEdge metrics
        crypto_value   = crypto_portfolio.get('capital', 10000) + sum(
            pos.get('units', 0) * pos.get('current_price', pos.get('entry_price', 0))
            for pos in crypto_portfolio.get('positions', {}).values()
        )
        crypto_metrics = self.calculate_metrics(
            trades           = crypto_portfolio.get('trade_history', []),
            starting_capital = crypto_portfolio.get('starting_capital', 10000),
            current_value    = crypto_value,
            days_back        = days_back,
        )

        # Build report
        def fmt_pnl(pnl, pct, currency='$'):
            sign = '+' if pnl >= 0 else ''
            return f"{sign}{currency}{abs(pnl):,.2f} ({sign}{pct:.2f}%)"

        def fmt_trade(trade, currency='$'):
            if not trade:
                return 'N/A'
            pnl = trade.get('pnl', 0)
            sym = trade.get('symbol', '?')
            sign = '+' if pnl >= 0 else ''
            return f"{sym}: {sign}{currency}{abs(pnl):.2f}"

        report = f"""TRADING EMPIRE WEEKLY REPORT
==============================
Period: Last {days_back} days
Date: {now}

ALPHAEDGE (US Stocks):
Portfolio: ${alpha_value:,.2f}
P&L: {fmt_pnl(alpha_metrics['total_pnl'], alpha_metrics['total_pct'])}
Trades: {alpha_metrics['total_trades']} | Win Rate: {alpha_metrics['win_rate']:.1f}%
Best: {fmt_trade(alpha_metrics['best_trade'])}
Worst: {fmt_trade(alpha_metrics['worst_trade'])}

BHARATEDGE (Indian Stocks):
Portfolio: Rs{bharat_value:,.2f}
P&L: {fmt_pnl(bharat_metrics['total_pnl'], bharat_metrics['total_pct'], 'Rs')}
Trades: {bharat_metrics['total_trades']} | Win Rate: {bharat_metrics['win_rate']:.1f}%
Best: {fmt_trade(bharat_metrics['best_trade'], 'Rs')}
Worst: {fmt_trade(bharat_metrics['worst_trade'], 'Rs')}

CRYPTOEDGE (Crypto 24/7):
Portfolio: ${crypto_value:,.2f} USDT
P&L: {fmt_pnl(crypto_metrics['total_pnl'], crypto_metrics['total_pct'])}
Trades: {crypto_metrics['total_trades']} | Win Rate: {crypto_metrics['win_rate']:.1f}%
Best: {fmt_trade(crypto_metrics['best_trade'])}
Worst: {fmt_trade(crypto_metrics['worst_trade'])}

COMBINED EMPIRE:
Total Assets: ${ alpha_value + crypto_value:,.2f} + Rs{bharat_value:,.2f}
Next Report: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}

Trading Empire AI - Automated"""

        return report

    def send_report(self, telegram_bot, days_back=7):
        """Generate and send report to Telegram."""
        print("\n   Generating Performance Report...")
        report = self.generate_report(days_back)
        print(report)

        if telegram_bot:
            telegram_bot.send_message(report)
            print("   Performance report sent! ✅")

        return report

    def should_run_today(self):
        """Run every Sunday."""
        return datetime.now().strftime('%A') == 'Sunday'


if __name__ == '__main__':
    print("\nTesting Performance Analytics...")
    analytics = PerformanceAnalytics()
    report    = analytics.generate_report(days_back=30)
    print(report)