# analyze_performance.py
# Calculate win rate, avg win, avg loss from trade history

import json
import os

TRADES_FILE = 'logs/paper_trades.json'

def analyze():
    if not os.path.exists(TRADES_FILE):
        print("No trades file found!")
        return

    with open(TRADES_FILE, 'r') as f:
        data = json.load(f)

    history = data.get('trade_history', [])
    
    # Get all closed trades (SELL only)
    sells = [t for t in history if t.get('action') == 'SELL']
    
    if not sells:
        print("No closed trades yet!")
        return

    # Separate wins and losses
    wins   = [t for t in sells if t.get('pnl', 0) > 0]
    losses = [t for t in sells if t.get('pnl', 0) <= 0]

    total      = len(sells)
    win_count  = len(wins)
    loss_count = len(losses)
    win_rate   = win_count / total * 100 if total > 0 else 0

    # Average win and loss amounts
    avg_win  = sum(t.get('pnl', 0) for t in wins) / win_count if wins else 0
    avg_loss = sum(abs(t.get('pnl', 0)) for t in losses) / loss_count if losses else 0

    # Average win and loss percentages
    avg_win_pct  = sum(t.get('pnl_pct', 0) for t in wins) / win_count * 100 if wins else 0
    avg_loss_pct = sum(abs(t.get('pnl_pct', 0)) for t in losses) / loss_count * 100 if losses else 0

    # Profit factor
    total_wins   = sum(t.get('pnl', 0) for t in wins)
    total_losses = sum(abs(t.get('pnl', 0)) for t in losses)
    profit_factor= total_wins / total_losses if total_losses > 0 else 0

    # Kelly Criterion calculation
    # Kelly % = W - (1-W)/R
    # W = win rate, R = avg win / avg loss ratio
    if avg_loss > 0 and avg_win > 0:
        W = win_rate / 100
        R = avg_win / avg_loss
        kelly_pct = W - (1 - W) / R
        # Use half kelly for safety
        half_kelly = kelly_pct / 2
    else:
        kelly_pct  = 0.15
        half_kelly = 0.10

    print("\n" + "="*55)
    print("  ALPHAEDGE PERFORMANCE ANALYSIS")
    print("="*55)
    print(f"\n  Total Closed Trades : {total}")
    print(f"  Winning Trades      : {win_count}")
    print(f"  Losing Trades       : {loss_count}")
    print(f"\n  Win Rate            : {win_rate:.1f}%")
    print(f"  Avg Win Amount      : ${avg_win:.2f}")
    print(f"  Avg Loss Amount     : ${avg_loss:.2f}")
    print(f"  Avg Win %           : {avg_win_pct:.2f}%")
    print(f"  Avg Loss %          : {avg_loss_pct:.2f}%")
    print(f"\n  Win/Loss Ratio      : {avg_win/avg_loss:.2f}x" if avg_loss > 0 else "  Win/Loss Ratio: N/A")
    print(f"  Profit Factor       : {profit_factor:.2f}")
    print(f"\n  KELLY CRITERION:")
    print(f"  Full Kelly          : {kelly_pct:.1%}")
    print(f"  Half Kelly (Safe)   : {half_kelly:.1%}")
    print(f"\n  RECOMMENDATION:")
    if half_kelly <= 0:
        print(f"  Use minimum 5% per trade")
        print(f"  System needs more winning trades")
    elif half_kelly > 0.25:
        print(f"  Cap at 25% per trade (Half Kelly too high)")
        print(f"  Use 25% max position size")
    else:
        print(f"  Use {half_kelly:.1%} per trade (Half Kelly)")

    print(f"\n  INDIVIDUAL TRADES:")
    print(f"  {'Date':<12} {'Symbol':<10} {'Action':<6} {'PnL':>10} {'PnL%':>8} {'Reason'}")
    print(f"  {'-'*60}")
    for t in sells:
        pnl     = t.get('pnl', 0)
        pnl_pct = t.get('pnl_pct', 0) * 100
        date    = t.get('date', '')[:10]
        symbol  = t.get('symbol', '')
        reason  = t.get('reason', '')
        sign    = '+' if pnl > 0 else ''
        print(
            f"  {date:<12} {symbol:<10} {'SELL':<6}"
            f" {sign}${pnl:>8.2f} {sign}{pnl_pct:>6.1f}%"
            f" [{reason}]"
        )

    print(f"\n{'='*55}\n")


if __name__ == '__main__':
    analyze()