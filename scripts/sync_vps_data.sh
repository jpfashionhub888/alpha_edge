#!/bin/bash
# scripts/sync_vps_data.sh
#
# Syncs VPS scan output (logs/*.json) to docs/data/ and pushes to GitHub.
# Run after each scan cycle — either manually, via systemd timer, or from
# alpaca_live.py post-scan hook.
#
# Replaces the docs/data/ files that GitHub Actions cloud_scan.py would
# otherwise overwrite with empty data (no trained model on GH runners).
#
# Install as a timer:
#   cp /root/alpha_edge/scripts/sync_vps_data.sh /usr/local/bin/sync_vps_data.sh
#   chmod +x /usr/local/bin/sync_vps_data.sh
#   # See scripts/alphaedge-data-sync.timer / .service for systemd wiring

set -e

REPO=/root/alpha_edge
LOGS="$REPO/logs"
DATA="$REPO/docs/data"

cd "$REPO"

# ── 1. Copy logs → docs/data/ ──────────────────────────────────────────────
mkdir -p "$DATA"
FILES="paper_trades.json paper_trades_stocks_only.json latest_signals.json \
       closed_trades.json circuit_breaker.json model_auc.json scan_info.json sectors.json"

copied=0
for f in $FILES; do
  if [ -f "$LOGS/$f" ]; then
    size=$(wc -c < "$LOGS/$f")
    if [ "$size" -gt 2 ]; then          # skip bare {} files
      cp "$LOGS/$f" "$DATA/$f"
      echo "[sync] copied $f ($size bytes)"
      copied=$((copied + 1))
    else
      echo "[sync] skipped $f (empty, $size bytes)"
    fi
  fi
done

if [ "$copied" -eq 0 ]; then
  echo "[sync] nothing to sync — no non-empty log files found"
  exit 0
fi

# ── 2. Commit ──────────────────────────────────────────────────────────────
git config --local user.email "alphaedge-bot@github.com"
git config --local user.name "AlphaEdge Bot"

git add -f docs/data/

if git diff --staged --quiet; then
  echo "[sync] docs/data/ unchanged — no commit needed"
  exit 0
fi

git commit -m "Auto: VPS data sync $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "[sync] committed"

# ── 3. Push with retry ─────────────────────────────────────────────────────
push_ok=false
for attempt in 1 2 3; do
  if git push origin main; then
    push_ok=true
    echo "[sync] pushed (attempt $attempt)"
    break
  fi
  echo "[sync] push attempt $attempt failed — pulling and retrying..."
  # FIX: --strategy-option=ours made this VPS-local commit always win a
  # merge conflict against whatever was pushed to main in the meantime —
  # not scoped to docs/data/, the whole merge. Same defect FIX-01 already
  # found and fixed in daily_scan.yml; this script wasn't covered by that
  # audit since it's a VPS-side script, not a GitHub Actions workflow.
  # --rebase replays this sync commit on top of upstream instead of
  # silently discarding upstream's changes.
  git pull --no-edit --rebase origin main || true
  sleep 5
done

if [ "$push_ok" != "true" ]; then
  echo "[sync] ERROR: failed to push after 3 attempts"
  exit 1
fi

echo "[sync] done"
