#!/bin/bash
# scripts/p0_health_check.sh
# Run once on VPS when SSH is available.
# Covers P0-4 (VPS health verification) and P0-5 (crontab audit).
# Output is pass/fail for each check so issues are obvious at a glance.

set -uo pipefail
PASS=0; FAIL=0
ok()   { echo "  [PASS] $*"; PASS=$((PASS+1)); }
fail() { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
hdr()  { echo ""; echo "── $* ──────────────────────────────────────"; }

echo "╔══════════════════════════════════════════════╗"
echo "║   AlphaEdge P0 Health Check — $(date -u '+%Y-%m-%d %H:%M UTC')  ║"
echo "╚══════════════════════════════════════════════╝"

# ── P0-3: Service restart policy ──────────────────────────────────────────
hdr "P0-3: systemd restart policy"
RESTART=$(systemctl show alphaedge.service --property=Restart --value 2>/dev/null)
RSEC=$(systemctl show alphaedge.service --property=RestartUSec --value 2>/dev/null)
[ "$RESTART" = "on-failure" ] && ok "Restart=on-failure" || fail "Restart=$RESTART (want on-failure)"
echo "       RestartUSec=$RSEC"

# ── P0-4: VPS health ──────────────────────────────────────────────────────
hdr "P0-4: Service status"
STATE=$(systemctl is-active alphaedge.service 2>/dev/null)
[ "$STATE" = "active" ] && ok "alphaedge.service is active" || fail "alphaedge.service is $STATE"

hdr "P0-4: Disk space"
USAGE=$(df / --output=pcent | tail -1 | tr -d ' %')
[ "$USAGE" -lt 80 ] && ok "Disk ${USAGE}% used (< 80%)" || fail "Disk ${USAGE}% used — clean up!"

hdr "P0-4: Signal data freshness"
SIG=/root/alpha_edge/logs/latest_signals.json
if [ -f "$SIG" ]; then
  SIZE=$(wc -c < "$SIG")
  AGE_MIN=$(( ($(date +%s) - $(stat -c %Y "$SIG")) / 60 ))
  [ "$SIZE" -gt 100 ] && ok "latest_signals.json: ${SIZE} bytes" || fail "latest_signals.json is empty or tiny (${SIZE} bytes)"
  [ "$AGE_MIN" -lt 1440 ] && ok "Signal data ${AGE_MIN}m old (< 24h)" || fail "Signal data ${AGE_MIN}m old — stale!"
else
  fail "latest_signals.json not found"
fi

hdr "P0-4: Timers"
for t in alphaedge-audit.timer alphaedge-data-sync.timer; do
  STATE=$(systemctl is-active "$t" 2>/dev/null || echo "inactive")
  [ "$STATE" = "active" ] && ok "$t active" || fail "$t is $STATE"
done
# health timer may or may not be installed
HT=$(systemctl is-active alphaedge-health.timer 2>/dev/null || echo "not-found")
[ "$HT" = "active" ] && ok "alphaedge-health.timer active" || echo "  [INFO] alphaedge-health.timer: $HT (optional)"

hdr "P0-4: Heartbeat"
HB=/root/alpha_edge/logs/heartbeats/alpaca_bot.json
if [ -f "$HB" ]; then
  AGE_MIN=$(( ($(date +%s) - $(stat -c %Y "$HB")) / 60 ))
  [ "$AGE_MIN" -lt 60 ] && ok "Heartbeat ${AGE_MIN}m old" || fail "Heartbeat ${AGE_MIN}m old — service may be stuck"
else
  fail "Heartbeat file not found: $HB"
fi

hdr "P0-4: Model files"
MODEL_DIR=/root/alpha_edge/model_cache
if ls "$MODEL_DIR"/*.joblib 2>/dev/null | head -1 | grep -q joblib; then
  COUNT=$(ls "$MODEL_DIR"/*.joblib 2>/dev/null | wc -l)
  ok "model_cache: $COUNT .joblib files"
else
  fail "No .joblib model files in $MODEL_DIR"
fi

# ── P0-5: Crontab audit ───────────────────────────────────────────────────
hdr "P0-5: Crontab (root)"
CRON=$(crontab -l 2>/dev/null || echo "EMPTY")
if [ "$CRON" = "EMPTY" ]; then
  ok "No root crontab (clean)"
else
  echo "  [WARN] Root crontab has entries — review for conflicts:"
  echo "$CRON" | sed 's/^/         /'
  fail "Root crontab is non-empty — check for duplicate scan jobs"
fi

hdr "P0-5: System-wide cron"
SYSCRON=$(grep -r "alphaedge\|cloud_scan\|main\.py\|alpaca_live" /etc/cron* /etc/cron.d/ 2>/dev/null || echo "none")
[ "$SYSCRON" = "none" ] && ok "No system cron entries for AlphaEdge" || {
  fail "System cron entries found:"
  echo "$SYSCRON" | sed 's/^/         /'
}

hdr "P0-5: Active timers summary"
systemctl list-timers --no-pager | grep -E "alphaedge|NEXT|LAST" | head -20

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════"
echo "  PASS: $PASS   FAIL: $FAIL"
echo "══════════════════════════════════════"
[ "$FAIL" -eq 0 ] && echo "  All P0 checks passed" && exit 0 || exit 1
