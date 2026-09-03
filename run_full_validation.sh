#!/usr/bin/env bash
# run_full_validation.sh
#
# Consolidated validation pass for alpha_edge.
# Runs on YOUR machine (VPS / dev box) where network access to
# yfinance/Yahoo Finance actually works — this cannot be run from
# a sandboxed chat environment.
#
# Usage:
#   1. Copy this file into the root of your alpha_edge repo checkout.
#   2. chmod +x run_full_validation.sh
#   3. ./run_full_validation.sh
#   4. Paste the ENTIRE contents of validation_results_<timestamp>.txt
#      back to Claude — not a summary, the raw file.
#
# What this does NOT do: it does not interpret the numbers for you,
# skip failures silently, or hide errors. Every section runs even if
# a prior one fails, and every failure is captured, not swallowed.

set -uo pipefail

TS=$(date +%Y%m%d_%H%M%S)
OUT="validation_results_${TS}.txt"

section () {
  echo "" >> "$OUT"
  echo "==================================================================" >> "$OUT"
  echo "  $1" >> "$OUT"
  echo "==================================================================" >> "$OUT"
}

run_step () {
  local label="$1"
  local cmd="$2"
  section "$label"
  echo "\$ $cmd" >> "$OUT"
  echo "--- started: $(date -u +%FT%TZ) ---" >> "$OUT"
  # Run, capture stdout+stderr, capture exit code, never abort the script.
  if eval "$cmd" >> "$OUT" 2>&1; then
    echo "--- exit code: 0 (success) ---" >> "$OUT"
  else
    echo "--- exit code: $? (FAILED — see output above) ---" >> "$OUT"
  fi
}

echo "AlphaEdge full validation run — $(date -u +%FT%TZ)" > "$OUT"
echo "Repo commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown — not a git checkout')" >> "$OUT"
echo "Repo status:" >> "$OUT"
git status --short >> "$OUT" 2>&1 || echo "  (not a git repo, or git not available)" >> "$OUT"

# ------------------------------------------------------------------
# 1. Signal validation — the Gate 1 IC diagnostics.
#    These answer: does any individual signal have real predictive
#    power, before it's wrapped in 9 layers of filters?
# ------------------------------------------------------------------
run_step "1a. SUE (earnings revision) IC diagnostic"        "python3 backtesting/diagnose_ic.py"
run_step "1b. SUE IC diagnostic v2 (entry delay/regime/surprise filters)" "python3 backtesting/diagnose_ic_v2.py"
run_step "1c. Momentum signal IC diagnostic"                 "python3 backtesting/diagnose_momentum_ic.py"
run_step "1d. Sector rotation IC diagnostic"                 "python3 backtesting/diagnose_sector_ic.py"
run_step "1e. SUE coverage/data-quality diagnostic"          "python3 backtesting/diagnose_sue.py"

# ------------------------------------------------------------------
# 2. The NEW proper backtest runner (event-driven engine, Gate 1/Gate 2
#    tearsheet). This is the rigorous path — use its numbers as
#    authoritative over anything in the old root-level runners.
# ------------------------------------------------------------------
run_step "2. New event-driven backtest runner (SUE signal, Gate 1+2 tearsheet)" \
  "python3 backtesting/run_backtest.py --start 2021-01-01"

run_step "2b. Momentum signal backtest runner"   "python3 backtesting/run_momentum.py"
run_step "2c. Sector momentum backtest runner"   "python3 backtesting/run_sector_momentum.py"
run_step "2d. Vol-targeted backtest runner"      "python3 backtesting/run_vol_targeted.py"

# ------------------------------------------------------------------
# 3. The OLD root-level backtest runners — same code path that
#    produced the June 14 result (Sharpe 2.65 / 31.65% annual) BEFORE
#    the July 16 overfit fix (167->20 features, 180->365 row window).
#    Re-running this now, on the corrected model, is how we find out
#    whether that number was real or an overfitting artifact.
# ------------------------------------------------------------------
run_step "3a. Legacy walk-forward backtest (post overfit-fix)" "python3 run_backtest.py"
run_step "3b. Legacy V6 ATR backtest (post overfit-fix, stocks only)" "python3 run_backtest_v2.py --stocks"
run_step "3c. Legacy V6 ATR backtest (post overfit-fix, crypto only)" "python3 run_backtest_v2.py --crypto"

# ------------------------------------------------------------------
# 4. Full test suite — confirms nothing is silently broken
#    (e.g. the CrashRecoveryTest / 0-collected-tests issue from the
#    prior audit should show up here if still unfixed).
# ------------------------------------------------------------------
run_step "4. Full pytest suite" "python3 -m pytest tests/ -v --tb=short"

# ------------------------------------------------------------------
section "DONE"
echo "Finished: $(date -u +%FT%TZ)" >> "$OUT"
echo "Full results written to: $OUT" >> "$OUT"
echo ""
echo "Validation complete. Results written to: $OUT"
echo "Paste the full contents of that file back to Claude for review."
