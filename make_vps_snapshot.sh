#!/usr/bin/env bash
# Build a review snapshot of the AlphaEdge VPS tree.
# READ-ONLY. Writes one tarball to /tmp. Touches nothing in the repo.
#
#   bash make_vps_snapshot.sh
#   # then, from your Windows machine:
#   scp root@67.207.82.11:/tmp/alphaedge_snapshot_*.tar.gz C:\Users\giris\alpha_edge\
#
# EXCLUDES secrets, venv, model binaries, caches, and log payloads.
# A secret scan runs at the end and ABORTS if anything looks like a credential.

set -uo pipefail
SRC=/root/alpha_edge
STAMP=$(date -u +%Y%m%d_%H%M)
STAGE=/tmp/ae_snap_$STAMP
OUT=/tmp/alphaedge_snapshot_$STAMP.tar.gz

rm -rf "$STAGE"; mkdir -p "$STAGE/repo" "$STAGE/meta"
cd "$SRC" || exit 1

echo "==> Copying source (code + config + workflows only)"
rsync -a \
  --exclude='venv/' --exclude='.git/' --exclude='__pycache__/' \
  --exclude='.pytest_cache/' --exclude='catboost_info/' \
  --exclude='cache/' --exclude='model_cache/' --exclude='saved_models/' \
  --exclude='node_modules/' --exclude='alpha_edge/' \
  --exclude='*.env' --exclude='secrets*' --exclude='*.pem' \
  --exclude='*.key' --exclude='id_*' --exclude='*.joblib' \
  --exclude='*.pkl' --exclude='*.pt' --exclude='*.h5' --exclude='*.zip' \
  --exclude='*.log' --exclude='*.csv' --exclude='*.png' --exclude='*.ico' \
  --exclude='logs/audits/' --exclude='logs/backtest/' --exclude='logs/heartbeats/' \
  ./ "$STAGE/repo/"

echo "==> Capturing git metadata"
{
  echo "### HEAD";            git log -1 --format='%H %ad %an %s' --date=iso
  echo; echo "### LAST 40";   git --no-pager log --oneline -40
  echo; echo "### STATUS";    git status --porcelain
  echo; echo "### BRANCH";    git branch -vv
  echo; echo "### TRACKED logs/ + state";  git ls-files | grep -E '^logs/|\.json$' | head -40
  echo; echo "### STASH COUNT"; git stash list | wc -l
  echo; echo "### STASH LIST";  git stash list
} > "$STAGE/meta/git.txt" 2>&1

echo "==> Capturing runtime environment"
{
  echo "### HOST";     hostname; uname -a; date -u
  echo; echo "### DISK";    df -h /
  echo; echo "### MEM";     free -h
  echo; echo "### PY";      /root/alpha_edge/venv/bin/python -V
  echo; echo "### KEY PKG VERSIONS"
  /root/alpha_edge/venv/bin/pip list 2>/dev/null | grep -iE \
    'scikit-learn|xgboost|lightgbm|catboost|pandas|numpy|torch|alpaca|yfinance'
  echo; echo "### SERVICES"; systemctl is-active alphaedge alphaedge-dashboard 2>&1
  echo; echo "### UNIT FILE (running)"; systemctl cat alphaedge 2>&1 | grep -vE 'Token|SECRET|KEY|PASS'
  echo; echo "### TIMERS";  systemctl list-timers --no-pager 2>/dev/null | grep -i alpha
  echo; echo "### CRONTAB"; crontab -l 2>/dev/null | sed -E 's/token=[^&"]+/token=[REDACTED]/g'
  echo; echo "### FILE OWNERSHIP logs/"; stat -c '%U:%G %a %n' logs/*.json 2>/dev/null
  echo; echo "### LOG FRESHNESS"; ls -lt --time-style=+%Y-%m-%d_%H:%M logs/*.json 2>/dev/null
  echo; echo "### RECENT ERRORS"
  journalctl -u alphaedge --since '72 hours ago' -p err --no-pager 2>/dev/null | tail -40
} > "$STAGE/meta/runtime.txt" 2>&1

echo "==> Capturing state file SHAPES (structure only, no position values)"
/root/alpha_edge/venv/bin/python - <<'PY' > "$STAGE/meta/state_shapes.txt" 2>&1
import json, os, datetime
for fn in ('paper_trades_stocks_only.json','managed_positions.json','closed_trades.json',
           'circuit_breaker.json','kill_switch.json','latest_signals.json','model_auc.json'):
    p = os.path.join('logs', fn)
    try:
        st = os.stat(p)
        d  = json.load(open(p))
        mt = datetime.datetime.utcfromtimestamp(st.st_mtime).isoformat()
        if isinstance(d, dict):
            shape = f'dict keys={list(d)[:12]}'
        elif isinstance(d, list):
            shape = f'list len={len(d)} first_keys=' + (str(list(d[0])) if d and isinstance(d[0],dict) else 'n/a')
        else:
            shape = type(d).__name__
        print(f'{fn:38s} {st.st_size:>8}B  mtime={mt}  {shape}')
    except Exception as e:
        print(f'{fn:38s} ERROR: {e}')
PY

echo "==> Extracting stash@{36} (only copy of _recover_state_if_needed)"
git stash show -p 'stash@{36}' -- alpaca_live.py > "$STAGE/meta/stash36_alpaca_live.patch" 2>&1
git stash show -p 'stash@{115}' -- execution/alpaca_broker.py > "$STAGE/meta/stash115_broker.patch" 2>&1

echo "==> SECRET SCAN (aborts on hit)"
HITS=$(grep -rInE \
  '(ghp_|github_pat_|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xoxb-|BEGIN [A-Z ]*PRIVATE KEY|APCA-API|PK[A-Z0-9]{16,})' \
  "$STAGE" 2>/dev/null | head -20)
if [ -n "$HITS" ]; then
  echo "!! ABORTED — possible credentials found. Nothing packaged."
  echo "$HITS" | cut -c1-160
  echo "Remove or redact these, then re-run."
  exit 1
fi
echo "   clean"

tar -czf "$OUT" -C "$STAGE" . && rm -rf "$STAGE"
echo
echo "==> DONE: $OUT  ($(du -h "$OUT" | cut -f1))"
echo "Fetch with:"
echo "  scp root@67.207.82.11:$OUT ."
