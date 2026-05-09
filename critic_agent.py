# critic_agent.py
# ALPHAEDGE - Self-Improving Critic Agent
# Reviews losing trades every Sunday
# Sends improvement report to Telegram
# System gets smarter automatically!

import os
import json
import logging
from datetime import datetime, timedelta
from groq import Groq

logger = logging.getLogger(__name__)


class CriticAgent:
    """
    AI-powered self-improvement system.
    Reviews losses and suggests improvements.
    Runs every Sunday automatically.
    """

    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY', '')
        self.enabled = bool(self.api_key)
        self.model   = 'llama-3.3-70b-versatile'

        if not self.enabled:
            print("   Critic Agent: GROQ_API_KEY not found")
        else:
            print("   Critic Agent: Groq/Llama3 connected ✅")

    def analyze_losses(self, trade_history, days_back=7):
        """
        Analyze losing trades from the past week.
        Returns analysis and recommendations.
        """
        if not trade_history:
            return None

        # Get recent closed trades
        cutoff = datetime.now() - timedelta(days=days_back)
        recent_trades = []

        for trade in trade_history:
            if trade.get('action') != 'SELL':
                continue
            try:
                trade_date = datetime.fromisoformat(
                    trade.get('date', '')
                )
                if trade_date >= cutoff:
                    recent_trades.append(trade)
            except Exception:
                continue

        if not recent_trades:
            return None

        # Separate wins and losses
        wins   = [t for t in recent_trades if t.get('pnl', 0) > 0]
        losses = [t for t in recent_trades if t.get('pnl', 0) <= 0]

        return {
            'total_trades' : len(recent_trades),
            'wins'         : wins,
            'losses'       : losses,
            'win_rate'     : len(wins) / len(recent_trades) * 100 if recent_trades else 0,
            'total_pnl'    : sum(t.get('pnl', 0) for t in recent_trades),
            'period_days'  : days_back,
        }

    def generate_report(self, trade_history, portfolio_value,
                        starting_capital, days_back=7):
        """
        Generate AI-powered improvement report.
        Returns formatted report string.
        """
        analysis = self.analyze_losses(trade_history, days_back)

        if not analysis:
            return self._no_trades_report()

        wins   = analysis['wins']
        losses = analysis['losses']

        # Build trade summary for AI
        loss_summary = ""
        for t in losses[:5]:  # Max 5 losses
            pnl     = t.get('pnl', 0)
            pnl_pct = t.get('pnl_pct', 0) * 100
            loss_summary += (
                f"- {t.get('symbol', '?')}: "
                f"${pnl:.2f} ({pnl_pct:.1f}%) "
                f"Reason: {t.get('reason', 'unknown')}\n"
            )

        win_summary = ""
        for t in wins[:5]:  # Max 5 wins
            pnl     = t.get('pnl', 0)
            pnl_pct = t.get('pnl_pct', 0) * 100
            win_summary += (
                f"- {t.get('symbol', '?')}: "
                f"+${pnl:.2f} (+{pnl_pct:.1f}%) "
                f"Reason: {t.get('reason', 'unknown')}\n"
            )

        total_pnl = analysis['total_pnl']
        win_rate  = analysis['win_rate']

        # If no AI available, generate basic report
        if not self.enabled:
            return self._basic_report(analysis)

        try:
            client = Groq(api_key=self.api_key)

            prompt = f"""You are a quantitative trading analyst reviewing an AI trading system's performance.

WEEKLY PERFORMANCE SUMMARY:
Period: Last {days_back} days
Total Trades: {analysis['total_trades']}
Wins: {len(wins)} | Losses: {len(losses)}
Win Rate: {win_rate:.1f}%
Total P&L: ${total_pnl:+.2f}
Portfolio Value: ${portfolio_value:,.2f}
Starting Capital: ${starting_capital:,.2f}

LOSING TRADES:
{loss_summary if loss_summary else 'No losses this week!'}

WINNING TRADES:
{win_summary if win_summary else 'No wins this week!'}

Based on this data, provide:
1. What patterns caused the losses?
2. What worked well in the wins?
3. Three specific actionable recommendations to improve performance
4. Overall assessment (1-10 score)

Keep response concise and practical. Max 200 words."""

            response = client.chat.completions.create(
                model       = self.model,
                messages    = [
                    {
                        "role"   : "system",
                        "content": "You are a quantitative trading analyst. Be concise and specific."
                    },
                    {
                        "role"   : "user",
                        "content": prompt
                    }
                ],
                temperature = 0.3,
                max_tokens  = 300,
            )

            ai_analysis = response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            ai_analysis = "AI analysis unavailable this week."

        # Build final report
        pnl_sign = '+' if total_pnl >= 0 else ''
        report   = f"""ALPHAEDGE WEEKLY CRITIC REPORT
================================
Period: Last {days_back} days
Date: {datetime.now().strftime('%Y-%m-%d')}

PERFORMANCE SUMMARY:
Total Trades: {analysis['total_trades']}
Win Rate: {win_rate:.1f}%
Total P&L: {pnl_sign}${total_pnl:.2f}
Portfolio: ${portfolio_value:,.2f}

WINS ({len(wins)}):
{win_summary if win_summary else 'No wins this week'}
LOSSES ({len(losses)}):
{loss_summary if loss_summary else 'No losses this week!'}

AI ANALYSIS:
{ai_analysis}

Next review: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}
AlphaEdge Critic Agent"""

        return report

    def _basic_report(self, analysis):
        """Generate basic report without AI."""
        wins     = analysis['wins']
        losses   = analysis['losses']
        win_rate = analysis['win_rate']
        total_pnl= analysis['total_pnl']
        pnl_sign = '+' if total_pnl >= 0 else ''

        loss_summary = ""
        for t in losses[:5]:
            pnl = t.get('pnl', 0)
            loss_summary += f"- {t.get('symbol','?')}: ${pnl:.2f}\n"

        return f"""ALPHAEDGE WEEKLY REPORT
========================
Trades: {analysis['total_trades']}
Win Rate: {win_rate:.1f}%
P&L: {pnl_sign}${total_pnl:.2f}

Losses:
{loss_summary if loss_summary else 'None!'}

AlphaEdge Critic Agent"""

    def _no_trades_report(self):
        """Report when no trades in period."""
        return f"""ALPHAEDGE WEEKLY REPORT
========================
Date: {datetime.now().strftime('%Y-%m-%d')}

No closed trades this week.
System is holding positions
or waiting for signals.

Portfolio is being protected.
AlphaEdge Critic Agent"""

    def should_run_today(self):
        """Check if today is Sunday (weekly review day)."""
        return datetime.now().strftime('%A') == 'Sunday'

    def run_weekly_review(self, trade_history,
                          portfolio_value,
                          starting_capital,
                          telegram_bot):
        """
        Run weekly review if today is Sunday.
        Sends report to Telegram.
        """
        if not self.should_run_today():
            print("   Critic Agent: Not Sunday, skipping review")
            return False

        print("\n   SUNDAY REVIEW - Running Critic Agent...")
        report = self.generate_report(
            trade_history    = trade_history,
            portfolio_value  = portfolio_value,
            starting_capital = starting_capital,
            days_back        = 7,
        )

        print(report)

        # Send to Telegram
        if telegram_bot and report:
            telegram_bot.send_message(report)
            print("   Critic report sent to Telegram! ✅")
            return True

        return False


if __name__ == '__main__':
    print("\nTesting Critic Agent...")

    # Set API key for testing
    os.environ['GROQ_API_KEY'] = ''  # Add your key here for testing

    critic = CriticAgent()

    # Sample trade history for testing
    sample_trades = [
        {
            'action' : 'SELL',
            'symbol' : 'TSLA',
            'pnl'    : -45.20,
            'pnl_pct': -0.035,
            'reason' : 'stop_loss',
            'date'   : (datetime.now() - timedelta(days=2)).isoformat(),
        },
        {
            'action' : 'SELL',
            'symbol' : 'AAPL',
            'pnl'    : +78.50,
            'pnl_pct': 0.042,
            'reason' : 'take_profit',
            'date'   : (datetime.now() - timedelta(days=3)).isoformat(),
        },
        {
            'action' : 'SELL',
            'symbol' : 'PFE',
            'pnl'    : -22.10,
            'pnl_pct': -0.028,
            'reason' : 'stop_loss',
            'date'   : (datetime.now() - timedelta(days=4)).isoformat(),
        },
        {
            'action' : 'SELL',
            'symbol' : 'CVX',
            'pnl'    : +112.30,
            'pnl_pct': 0.089,
            'reason' : 'take_profit',
            'date'   : (datetime.now() - timedelta(days=1)).isoformat(),
        },
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