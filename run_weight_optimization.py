# run_weight_optimization.py - V2
# 
# What changed from V1:
# Weight optimization is invalid when sentiment/sector are constants.
# This version optimizes what we actually have data for:
#   - buy threshold (when to enter)
#   - hold days (how long to hold)
# Uses pred_w=1.0 (prediction only) as the signal.
#
# Runtime: ~5-10 minutes
# Output:  logs/weight_results.json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backtest'))

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

from feature_engine import FeatureEngine
from walk_forward import WalkForwardBacktester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("peewee").setLevel(logging.WARNING)

# ── Output path ───────────────────────────────────────────────────────────
RESULTS_PATH = Path("logs/threshold_results.json")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Search grids ──────────────────────────────────────────────────────────
THRESHOLD_GRID = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
HOLD_DAYS_GRID = [3, 5, 7, 10]

# Fixed weights — prediction only
# sentiment/sector weights cannot be optimized without
# real historical per-bar sentiment and sector data
FIXED_WEIGHTS = (1.0, 0.0, 0.0)

# Symbols to test on
SYMBOLS = ['AAPL', 'MSFT', 'NVDA']


# ── Metrics ───────────────────────────────────────────────────────────────
def compute_metrics(trades: List[Dict]) -> Dict[str, float]:
    if not trades:
        return {
            "sharpe"          : -999.0,
            "win_rate"        : 0.0,
            "profit_factor"   : 0.0,
            "max_drawdown"    : 0.0,
            "avg_return_pct"  : 0.0,
            "total_return_pct": 0.0,
            "trade_count"     : 0,
        }

    returns = np.array([t["return_pct"] for t in trades])

    wins   = returns[returns > 0]
    losses = returns[returns <= 0]

    win_rate      = len(wins) / len(returns)
    gross_profit  = float(wins.sum())  if len(wins)   > 0 else 0.0
    gross_loss    = float(abs(losses.sum())) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss

    avg_held = np.mean([t.get("held_days", 5) for t in trades])
    periods_per_year = 252 / max(avg_held, 1)

    if len(returns) >= 2 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    cumulative  = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdowns   = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(abs(drawdowns.min()))
    total_return = float(cumulative[-1] - 1.0)

    return {
        "sharpe"          : round(float(sharpe), 4),
        "win_rate"        : round(win_rate, 4),
        "profit_factor"   : round(profit_factor, 4),
        "max_drawdown"    : round(max_drawdown, 4),
        "avg_return_pct"  : round(float(returns.mean()), 4),
        "total_return_pct": round(total_return, 4),
        "trade_count"     : len(trades),
    }


# ── Save atomically ───────────────────────────────────────────────────────
def save_results(results: List[Dict], path: Path) -> None:
    import shutil
    tmp = path.with_suffix(".tmp")
    payload = {
        "generated_at"       : datetime.now().isoformat(),
        "total_combinations" : len(results),
        "fixed_weights"      : {
            "pred_w"  : FIXED_WEIGHTS[0],
            "sent_w"  : FIXED_WEIGHTS[1],
            "sector_w": FIXED_WEIGHTS[2],
        },
        "results": results,
    }
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    shutil.move(str(tmp), str(path))
    logger.info(f"Results saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("THRESHOLD & HOLD-DAY OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Symbols:    {SYMBOLS}")
    logger.info(f"Thresholds: {THRESHOLD_GRID}")
    logger.info(f"Hold days:  {HOLD_DAYS_GRID}")
    logger.info(
        f"Combinations: "
        f"{len(THRESHOLD_GRID) * len(HOLD_DAYS_GRID)}"
    )

    feature_engine = FeatureEngine()
    all_results    = []
    combo_num      = 0
    total_combos   = len(THRESHOLD_GRID) * len(HOLD_DAYS_GRID)

    for hold_days in HOLD_DAYS_GRID:
        # Create validator with this hold period
        validator = WalkForwardBacktester(
            feature_engine=feature_engine,
        )
        # Override hold days for this combo
        validator._opt_hold_days = hold_days

        for threshold in THRESHOLD_GRID:
            combo_num += 1
            label = f"threshold={threshold} hold={hold_days}d"
            logger.info(
                f"Testing combo {combo_num}/{total_combos}: {label}"
            )

            all_trades = []
            errors     = 0

            for symbol in SYMBOLS:
                try:
                    result = validator.run_for_optimizer(
                        symbol=symbol,
                        weight_combo=FIXED_WEIGHTS,
                        compute_signal_fn=None,
                        buy_threshold=threshold,
                        hold_days_override=hold_days,
                    )
                    if result and result.get("trades"):
                        all_trades.extend(result["trades"])
                except Exception as e:
                    logger.warning(
                        f"[{symbol}] Failed for {label}: {e}"
                    )
                    errors += 1

            metrics = compute_metrics(all_trades)
            metrics.update({
                "threshold"   : threshold,
                "hold_days"   : hold_days,
                "label"       : label,
                "errors"      : errors,
                "symbols"     : SYMBOLS,
            })

            logger.info(
                f"{label} | "
                f"sharpe={metrics['sharpe']:.3f} | "
                f"win_rate={metrics['win_rate']:.1%} | "
                f"trades={metrics['trade_count']}"
            )

            all_results.append(metrics)
            save_results(all_results, RESULTS_PATH)

    # ── Rank results ──────────────────────────────────────────────
    valid = [r for r in all_results if r["trade_count"] >= 5]

    if not valid:
        logger.error(
            "No combinations produced >= 5 trades. "
            "Check feature_engine or increase symbol list."
        )
        return

    ranked = sorted(
        valid,
        key=lambda r: (r["sharpe"], r["win_rate"]),
        reverse=True,
    )

    # ── Print report ──────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info("\nTop 5 combinations:\n")

    for rank, r in enumerate(ranked[:5], 1):
        logger.info(
            f"#{rank}: threshold={r['threshold']} "
            f"hold={r['hold_days']}d\n"
            f"       Sharpe={r['sharpe']:.3f} | "
            f"WinRate={r['win_rate']:.1%} | "
            f"PF={r['profit_factor']:.2f} | "
            f"MaxDD={r['max_drawdown']:.1%} | "
            f"Trades={r['trade_count']}"
        )

    winner = ranked[0]
    logger.info(f"\n{'='*60}")
    logger.info("WINNER:")
    logger.info(f"  Buy threshold : {winner['threshold']}")
    logger.info(f"  Hold days     : {winner['hold_days']}")
    logger.info(f"  Sharpe        : {winner['sharpe']:.3f}")
    logger.info(f"  Win rate      : {winner['win_rate']:.1%}")
    logger.info(f"  Trades        : {winner['trade_count']}")
    logger.info(f"{'='*60}")
    logger.info(
        f"\nNext step: Update main.py with:\n"
        f"  BUY_THRESHOLD = {winner['threshold']}\n"
        f"  HOLD_DAYS     = {winner['hold_days']}\n"
        f"\nFull results: {RESULTS_PATH}"
    )

    save_results(ranked, RESULTS_PATH)


if __name__ == "__main__":
    main()