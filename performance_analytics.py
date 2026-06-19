# performance_analytics.py
# TRADING EMPIRE - Performance Analytics Engine
#
# FIX P1-3: calculate_metrics() now counts PARTIAL_SELL alongside SELL
#   when computing wins, losses, avg_win/loss, profit_factor, and
#   best/worst trade. Previously partial exits were silently excluded,
#   understating realized P&L and win count in the weekly report.
#
# v2: Adds Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown
#   computed from the equity curve reconstructed from trade history.

import json
import os
import math
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Exit action types that represent realized P&L events
REALIZED_ACTIONS = ('SELL', 'PARTIAL_SELL')   # P1-3

RISK_FREE_RATE = 0.05   # annualised (approx 5% T-bill)
TRADING_DAYS  = 252     # annualisation factor


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
            'sharpe'       : 'N/A',
            'sortino'      : 'N/A',
            'calmar'       : 'N/A',
            'max_drawdown' : 0.0,
        }

    def _build_equity_curve(self, exits, starting_capital):
        """
        Build a daily equity curve from realized exits.
        Returns a list of daily returns (as fractions) sorted by date.
        Used for Sharpe, Sortino, and Calmar calculations.
        """
        if not exits:
            return []

        # Sort exits by date
        dated = []
        for t in exits:
            try:
                dated.append((datetime.fromisoformat(t['date']), t.get('pnl', 0)))
            except Exception:
                continue
        if not dated:
            return []

        dated.sort(key=lambda x: x[0])

        # Aggregate P&L by date (multiple trades same day collapse)
        daily_pnl = {}
        for dt, pnl in dated:
            day = dt.date()
            daily_pnl[day] = daily_pnl.get(day, 0) + pnl

        # Build equity curve and compute daily returns
        equity = starting_capital
        daily_returns = []
        for day in sorted(daily_pnl):
            equity_prev = equity
            equity += daily_pnl[day]
            if equity_prev > 0:
                daily_returns.append((equity - equity_prev) / equity_prev)

        return daily_returns

    def _risk_metrics(self, daily_returns, total_pct, days_back):
        """
        Compute Sharpe, Sortino, Calmar, and Max Drawdown.

        Sharpe  = (mean_return - daily_rfr) / std_return  * sqrt(252)
        Sortino = (mean_return - daily_rfr) / downside_std * sqrt(252)
        Calmar  = annualised_return / |max_drawdown|
        """
        if not daily_returns or len(daily_returns) < 2:
            return {'sharpe': 'N/A', 'sortino': 'N/A', 'calmar': 'N/A', 'max_drawdown': 0.0}

        # Daily risk-free rate
        daily_rfr = RISK_FREE_RATE / TRADING_DAYS

        mean_r   = sum(daily_returns) / len(daily_returns)
        excess   = [r - daily_rfr for r in daily_returns]
        variance = sum(e ** 2 for e in excess) / (len(excess) - 1)
        std_r    = math.sqrt(variance) if variance > 0 else 0

        # Sharpe
        sharpe = (mean_r - daily_rfr) / std_r * math.sqrt(TRADING_DAYS) if std_r > 0 else 'N/A'

        # Sortino — only downside deviations
        downside = [min(r - daily_rfr, 0) for r in daily_returns]
        down_var = sum(d ** 2 for d in downside) / max(len(downside) - 1, 1)
        down_std = math.sqrt(down_var) if down_var > 0 else 0
        sortino  = (mean_r - daily_rfr) / down_std * math.sqrt(TRADING_DAYS) if down_std > 0 else 'N/A'

        # Max Drawdown from equity curve
        peak = 1.0
        max_dd = 0.0
        equity = 1.0
        for r in daily_returns:
            equity *= (1 + r)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        # Calmar
        ann_return = (1 + mean_r) ** TRADING_DAYS - 1
        calmar = ann_return / max_dd if max_dd > 0 else 'N/A'

        return {
            'sharpe'      : round(sharpe,  2) if isinstance(sharpe,  float) else sharpe,
            'sortino'     : round(sortino, 2) if isinstance(sortino, float) else sortino,
            'calmar'      : round(calmar,  2) if isinstance(calmar,  float) else calmar,
            'max_drawdown': round(max_dd * 100, 2),   # as percentage
        }

    def calculate_metrics(self, trades, starting_capital,
                          current_value, days_back=7):
        """
        Calculate performance metrics.

        P1-3 FIX: Counts both SELL and PARTIAL_SELL as realized trades.
        v2: Adds Sharpe, Sortino, Calmar, Max Drawdown.
        """
        if not trades:
            return self._empty_metrics(current_value, starting_capital, days_back)

        # Filter to realized exit events within window  [P1-3]
        # FIX: count and log trades skipped due to unparseable dates
        # so the report never silently under-counts without a visible signal.
        cutoff   = datetime.now() - timedelta(days=days_back)
        exits    = []
        skipped  = 0
        for t in trades:
            if t.get('action') not in REALIZED_ACTIONS:   # P1-3: was == 'SELL'
                continue
            try:
                trade_date = datetime.fromisoformat(t.get('date', ''))
                if trade_date >= cutoff:
                    exits.append(t)
            except Exception:
                skipped += 1
                continue

        if skipped:
            logger.warning(
                f'calculate_metrics: {skipped} realized trade(s) skipped due to '
                f'unparseable date strings \u2014 weekly report may under-count'
            )

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

        # v2: Risk-adjusted metrics from equity curve
        daily_returns = self._build_equity_curve(exits, starting_capital)
        risk = self._risk_metrics(daily_returns, total_pct, days_back)

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
            'sharpe'       : risk['sharpe'],
            'sortino'      : risk['sortino'],
            'calmar'       : risk['calmar'],
            'max_drawdown' : risk['max_drawdown'],
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

        def fmt_ratio(r):
            if isinstance(r, (int, float)):
                return f"{r:.2f}"
            return str(r)

        def fmt_dd(dd):
            return f"{dd:.2f}%" if isinstance(dd, (int, float)) else 'N/A'

        report = f"""TRADING EMPIRE WEEKLY REPORT
==============================
Period: Last {days_back} days
Date:   {now}

ALPHAEDGE (US Stocks):
  Portfolio:     ${alpha_value:,.2f}
  P&L:           {fmt_pnl(alpha_metrics['total_pnl'], alpha_metrics['total_pct'])}
  Trades:        {alpha_metrics['total_trades']} | Win Rate: {alpha_metrics['win_rate']:.1f}%
  Profit Factor: {fmt_pf(alpha_metrics['profit_factor'])}
  Sharpe:        {fmt_ratio(alpha_metrics['sharpe'])} | Sortino: {fmt_ratio(alpha_metrics['sortino'])} | Calmar: {fmt_ratio(alpha_metrics['calmar'])}
  Max Drawdown:  {fmt_dd(alpha_metrics['max_drawdown'])}
  Best:          {fmt_trade(alpha_metrics['best_trade'])}
  Worst:         {fmt_trade(alpha_metrics['worst_trade'])}

BHARATEDGE (Indian Stocks):
  Portfolio:     Rs{bharat_value:,.2f}
  P&L:           {fmt_pnl(bharat_metrics['total_pnl'], bharat_metrics['total_pct'], 'Rs')}
  Trades:        {bharat_metrics['total_trades']} | Win Rate: {bharat_metrics['win_rate']:.1f}%
  Profit Factor: {fmt_pf(bharat_metrics['profit_factor'])}
  Sharpe:        {fmt_ratio(bharat_metrics['sharpe'])} | Sortino: {fmt_ratio(bharat_metrics['sortino'])} | Calmar: {fmt_ratio(bharat_metrics['calmar'])}
  Max Drawdown:  {fmt_dd(bharat_metrics['max_drawdown'])}
  Best:          {fmt_trade(bharat_metrics['best_trade'], 'Rs')}
  Worst:         {fmt_trade(bharat_metrics['worst_trade'], 'Rs')}

CRYPTOEDGE (Crypto 24/7):
  Portfolio:     ${crypto_value:,.2f} USDT
  P&L:           {fmt_pnl(crypto_metrics['total_pnl'], crypto_metrics['total_pct'])}
  Trades:        {crypto_metrics['total_trades']} | Win Rate: {crypto_metrics['win_rate']:.1f}%
  Profit Factor: {fmt_pf(crypto_metrics['profit_factor'])}
  Sharpe:        {fmt_ratio(crypto_metrics['sharpe'])} | Sortino: {fmt_ratio(crypto_metrics['sortino'])} | Calmar: {fmt_ratio(crypto_metrics['calmar'])}
  Max Drawdown:  {fmt_dd(crypto_metrics['max_drawdown'])}
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
