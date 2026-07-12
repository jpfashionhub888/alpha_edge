# critic_agent.py
# ALPHAEDGE - Self-Improving Critic Agent V2
#
# Fixes applied:
#   - `from groq import Groq` moved inside method (was at module top-level,
#     causing ImportError crash at startup if groq not installed)
#   - AI call errors now produce a graceful fallback report, not a crash
#   - Trade date parsing made robust
#   - Loss/win formatting safe against missing fields

import os
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CriticAgent:
    """
    AI-powered self-improvement system.
    Reviews wins/losses and suggests improvements every Sunday.
    Sends report to Telegram.
    """

    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY', '')
        self.enabled = bool(self.api_key)
        self.model   = 'llama-3.3-70b-versatile'

        # No top-level import of groq here — lazy import inside methods
        if not self.enabled:
            logger.warning("Critic Agent: GROQ_API_KEY not found — weekly review disabled")
        else:
            logger.info("Critic Agent: enabled (Groq/Llama3)")

    def _parse_date(self, date_str):
        """Parse ISO date string. Returns None on failure."""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(str(date_str))
        except Exception:
            return None

    def analyze_losses(self, trade_history, days_back=7):
        """Analyze trades from the past N days. Returns analysis dict or None."""
        if not trade_history:
            return None

        cutoff        = datetime.now() - timedelta(days=days_back)
        recent_trades = []
        for trade in trade_history:
            if trade.get('action') != 'SELL':
                continue
            trade_date = self._parse_date(trade.get('date', ''))
            if trade_date and trade_date >= cutoff:
                recent_trades.append(trade)

        if not recent_trades:
            return None

        wins   = [t for t in recent_trades if t.get('pnl', 0) > 0]
        losses = [t for t in recent_trades if t.get('pnl', 0) <= 0]

        return {
            'total_trades': len(recent_trades),
            'wins'        : wins,
            'losses'      : losses,
            'win_rate'    : len(wins) / len(recent_trades) * 100,
            'total_pnl'   : sum(t.get('pnl', 0) for t in recent_trades),
            'period_days' : days_back,
        }

    def _format_trades(self, trades, max_n=5):
        """Format a list of trades for the AI prompt."""
        lines = []
        for t in trades[:max_n]:
            pnl     = t.get('pnl', 0)
            pnl_pct = t.get('pnl_pct', 0) * 100
            symbol  = t.get('symbol', '?')
            reason  = t.get('reason', 'unknown')
            sign    = '+' if pnl >= 0 else ''
            lines.append(f"- {symbol}: {sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%) | reason: {reason}")
        return '\n'.join(lines) if lines else 'None'

    def generate_report(self, trade_history, portfolio_value,
                        starting_capital, days_back=7):
        """Generate AI-powered (or fallback) improvement report."""
        analysis = self.analyze_losses(trade_history, days_back)

        if not analysis:
            return self._no_trades_report()

        wins     = analysis['wins']
        losses   = analysis['losses']
        win_rate = analysis['win_rate']
        total_pnl= analysis['total_pnl']

        loss_summary = self._format_trades(losses)
        win_summary  = self._format_trades(wins)

        if not self.enabled:
            return self._basic_report(analysis)

        ai_analysis = "AI analysis unavailable."
        try:
            # Lazy import — safe even if groq not installed
            try:
                from groq import Groq
            except ImportError:
                logger.error("groq package not installed — falling back to basic report")
                return self._basic_report(analysis)

            client = Groq(api_key=self.api_key)
            prompt = f"""You are a quantitative trading analyst reviewing an AI trading system.

WEEKLY SUMMARY:
Period:       Last {days_back} days
Total Trades: {analysis['total_trades']}
Wins: {len(wins)} | Losses: {len(losses)}
Win Rate:     {win_rate:.1f}%
Total P&L:    ${total_pnl:+.2f}
Portfolio:    ${portfolio_value:,.2f}
Starting:     ${starting_capital:,.2f}

LOSSES:
{loss_summary if losses else 'No losses this week!'}

WINS:
{win_summary if wins else 'No wins this week!'}

Provide:
1. Patterns in the losses
2. What worked in the wins
3. Three specific improvements
4. Score this week 1-10

Be concise. Max 200 words."""

            response = client.chat.completions.create(
                model    = self.model,
                messages = [
                    {"role": "system", "content": "You are a quantitative trading analyst. Be concise."},
                    {"role": "user",   "content": prompt}
                ],
                temperature = 0.3,
                max_tokens  = 300,
                timeout     = 15,
            )
            ai_analysis = response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"Critic AI call failed: {e}")
            ai_analysis = f"AI analysis unavailable this week ({type(e).__name__})."

        pnl_sign = '+' if total_pnl >= 0 else ''
        return f"""ALPHAEDGE WEEKLY CRITIC REPORT
================================
Period: Last {days_back} days | Date: {datetime.now().strftime('%Y-%m-%d')}

PERFORMANCE:
  Trades:    {analysis['total_trades']}
  Win Rate:  {win_rate:.1f}%
  P&L:       {pnl_sign}${total_pnl:.2f}
  Portfolio: ${portfolio_value:,.2f}

WINS ({len(wins)}):
{win_summary if wins else '  No wins this week'}

LOSSES ({len(losses)}):
{loss_summary if losses else '  No losses this week!'}

AI ANALYSIS:
{ai_analysis}

Next review: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}
AlphaEdge Critic Agent V2"""

    def _basic_report(self, analysis):
        wins      = analysis['wins']
        losses    = analysis['losses']
        win_rate  = analysis['win_rate']
        total_pnl = analysis['total_pnl']
        pnl_sign  = '+' if total_pnl >= 0 else ''
        return f"""ALPHAEDGE WEEKLY REPORT (no AI)
================================
Trades:   {analysis['total_trades']}
Win Rate: {win_rate:.1f}%
P&L:      {pnl_sign}${total_pnl:.2f}

Losses:
{self._format_trades(losses) if losses else '  None!'}

AlphaEdge Critic Agent"""

    def _no_trades_report(self):
        return f"""ALPHAEDGE WEEKLY REPORT
========================
Date: {datetime.now().strftime('%Y-%m-%d')}

No closed trades this week.
System holding positions or waiting for signals.

AlphaEdge Critic Agent"""

    def should_run_today(self):
        return datetime.now().strftime('%A') == 'Sunday'

    def run_weekly_review(self, trade_history, portfolio_value,
                          starting_capital, telegram_bot):
        if not self.should_run_today():
            logger.debug("Critic Agent: not Sunday, skipping weekly review")
            return False

        logger.info("SUNDAY REVIEW — Running Critic Agent...")
        report = self.generate_report(
            trade_history    = trade_history,
            portfolio_value  = portfolio_value,
            starting_capital = starting_capital,
            days_back        = 7,
        )
        logger.info("Critic report:\n%s", report)

        if telegram_bot and report:
            telegram_bot.send_message(report)
            logger.info("Critic report sent to Telegram")

        return True


if __name__ == '__main__':
    print("\nTesting Critic Agent V2...")

    critic = CriticAgent()

    sample_trades = [
        {'action': 'SELL', 'symbol': 'TSLA', 'pnl': -45.20, 'pnl_pct': -0.035,
         'reason': 'stop_loss',   'date': (datetime.now() - timedelta(days=2)).isoformat()},
        {'action': 'SELL', 'symbol': 'AAPL', 'pnl':  78.50, 'pnl_pct':  0.042,
         'reason': 'take_profit', 'date': (datetime.now() - timedelta(days=3)).isoformat()},
        {'action': 'SELL', 'symbol': 'PFE',  'pnl': -22.10, 'pnl_pct': -0.028,
         'reason': 'stop_loss',   'date': (datetime.now() - timedelta(days=4)).isoformat()},
        {'action': 'SELL', 'symbol': 'CVX',  'pnl': 112.30, 'pnl_pct':  0.089,
         'reason': 'take_profit', 'date': (datetime.now() - timedelta(days=1)).isoformat()},
    ]

    report = critic.generate_report(
        trade_history    = sample_trades,
        portfolio_value  = 10245.00,
        starting_capital = 10000.00,
        days_back        = 7,
    )
    print("\n" + "="*50)
    print(report)
    print("="*50)
