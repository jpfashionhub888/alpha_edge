# weight_backtest_report.py
# Purpose: Read logs/weight_results.json and print a clean report
#
# Run this AFTER run_weight_optimization.py completes
# Usage: python weight_backtest_report.py

import json
import sys
from pathlib import Path

RESULTS_PATH = Path("logs/weight_results.json")


def print_separator(char="─", width=65):
    print(char * width)


def format_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def format_float(val: float, decimals: int = 3) -> str:
    return f"{val:.{decimals}f}"


def main():
    # ── Load results ──────────────────────────────────────────────────
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found.")
        print("Run run_weight_optimization.py first.")
        sys.exit(1)

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    results = data.get("results", [])
    generated_at = data.get("generated_at", "unknown")

    if not results:
        print("No results found in file.")
        sys.exit(1)

    # ── Filter to results with enough trades ──────────────────────────
    valid = [r for r in results if r.get("trade_count", 0) >= 10]
    insufficient = len(results) - len(valid)

    # ── Header ────────────────────────────────────────────────────────
    print()
    print_separator("=")
    print("  SIGNAL WEIGHT OPTIMIZATION REPORT")
    print_separator("=")
    print(f"  Generated:    {generated_at}")
    print(f"  Combinations: {len(results)} tested | {len(valid)} valid | "
          f"{insufficient} insufficient trades")
    print()

    if not valid:
        print("ERROR: No combinations had >= 10 trades.")
        print("Increase your test universe or reduce walk-forward window.")
        sys.exit(1)

    # ── Sort by Sharpe ────────────────────────────────────────────────
    ranked = sorted(valid, key=lambda r: r["sharpe"], reverse=True)
    winner = ranked[0]
    current = next(
        (r for r in results
         if r["pred_w"] == 0.6 and r["sent_w"] == 0.2),
        None
    )

    # ── Current vs Winner comparison ──────────────────────────────────
    print_separator()
    print("  CURRENT CONFIG vs OPTIMAL")
    print_separator()
    print(f"  {'Metric':<20} {'Current (0.6/0.2/0.2)':<25} {'Winner'}")
    print_separator("─")

    metrics_to_show = [
        ("Weights", None, None),
        ("Sharpe Ratio", "sharpe", format_float),
        ("Win Rate", "win_rate", format_pct),
        ("Profit Factor", "profit_factor", format_float),
        ("Max Drawdown", "max_drawdown", format_pct),
        ("Avg Return/Trade", "avg_return_pct", format_pct),
        ("Trade Count", "trade_count", str),
    ]

    for label, key, fmt in metrics_to_show:
        if key is None:
            current_val = "0.6 / 0.2 / 0.2"
            winner_val = (
                f"{winner['pred_w']} / "
                f"{winner['sent_w']} / "
                f"{winner['sector_w']}"
            )
        elif current:
            current_val = fmt(current[key])
            winner_val = fmt(winner[key])
        else:
            current_val = "not tested"
            winner_val = fmt(winner[key])

        # Flag improvement
        improved = ""
        if key == "sharpe" and current:
            diff = winner["sharpe"] - current["sharpe"]
            improved = f"  ▲ +{diff:.3f}" if diff > 0 else f"  ▼ {diff:.3f}"
        elif key == "max_drawdown" and current:
            diff = current["max_drawdown"] - winner["max_drawdown"]
            improved = f"  ▲ -{diff:.1%} DD" if diff > 0 else ""

        print(f"  {label:<20} {current_val:<25} {winner_val}{improved}")

    # ── Top 10 table ──────────────────────────────────────────────────
    print()
    print_separator()
    print("  TOP 10 COMBINATIONS (ranked by Sharpe)")
    print_separator()
    print(
        f"  {'Rank':<5} {'pred_w':<8} {'sent_w':<8} {'sect_w':<8} "
        f"{'Sharpe':<10} {'WinRate':<10} {'PF':<8} {'MaxDD':<10} {'Trades'}"
    )
    print_separator("─")

    for i, r in enumerate(ranked[:10], 1):
        marker = " ◄ WINNER" if i == 1 else ""
        print(
            f"  {i:<5} {r['pred_w']:<8} {r['sent_w']:<8} {r['sector_w']:<8} "
            f"{r['sharpe']:<10.3f} {format_pct(r['win_rate']):<10} "
            f"{r['profit_factor']:<8.2f} {format_pct(r['max_drawdown']):<10} "
            f"{r['trade_count']}{marker}"
        )

    # ── Insight analysis ──────────────────────────────────────────────
    print()
    print_separator()
    print("  INSIGHTS")
    print_separator()

    # What weight range dominates the top 10?
    top10_pred = [r["pred_w"] for r in ranked[:10]]
    top10_sent = [r["sent_w"] for r in ranked[:10]]
    top10_sect = [r["sector_w"] for r in ranked[:10]]

    print(f"  Prediction weight in top 10:  "
          f"{min(top10_pred):.1f} – {max(top10_pred):.1f} "
          f"(avg {sum(top10_pred)/len(top10_pred):.2f})")
    print(f"  Sentiment weight in top 10:   "
          f"{min(top10_sent):.1f} – {max(top10_sent):.1f} "
          f"(avg {sum(top10_sent)/len(top10_sent):.2f})")
    print(f"  Sector weight in top 10:      "
          f"{min(top10_sect):.1f} – {max(top10_sect):.1f} "
          f"(avg {sum(top10_sect)/len(top10_sect):.2f})")

    # Is pure prediction better than blended?
    pure_pred = next(
        (r for r in results
         if r["pred_w"] == 1.0 and r["sent_w"] == 0.0),
        None
    )
    if pure_pred:
        print()
        if pure_pred["sharpe"] > winner["sharpe"] * 0.95:
            print("  ⚠  Pure prediction (1.0/0.0/0.0) is nearly as good as")
            print("     the blended winner. Sentiment/sector may not be adding")
            print("     value. Consider simplifying the signal formula.")
        else:
            print("  ✓  Blending improves over pure prediction — the formula")
            print(f"     is earning its complexity.")

    # ── Action items ──────────────────────────────────────────────────
    print()
    print_separator("=")
    print("  ACTION ITEMS")
    print_separator("=")
    print()
    print(f"  1. Update main.py with winning weights:")
    print()
    print(f"     PRED_WEIGHT   = {winner['pred_w']}")
    print(f"     SENT_WEIGHT   = {winner['sent_w']}")
    print(f"     SECTOR_WEIGHT = {winner['sector_w']}")
    print()
    print(f"  2. If Sharpe < 0.5, the model may not be predictive enough.")
    print(f"     Move to Task #3 (feature_engine.py look-ahead audit)")
    print(f"     before optimizing further.")
    print()
    print(f"  3. If top 10 all have similar Sharpe (< 0.1 spread),")
    print(f"     weights don't matter — the signal quality is the bottleneck.")
    print()
    print_separator("=")
    print()


if __name__ == "__main__":
    main()