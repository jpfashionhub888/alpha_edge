#!/bin/bash
# hardening/install_timers.sh
# Run once on VPS as root:
#   bash /root/alpha_edge/hardening/install_timers.sh
#
# Installs:
#   1. alphaedge-watchdog.timer  — checks heartbeat every 5 min, Telegram alert if stale
#   2. alphaedge-health.timer    — sends morning briefing at 08:00 ET (12:00 UTC) daily

set -e
VENV="/root/alpha_edge/venv/bin/python3"
DIR="/root/alpha_edge"

echo "Installing AlphaEdge systemd timers..."

# ── 1. Watchdog service + timer ────────────────────────────────────────────────

cat > /etc/systemd/system/alphaedge-watchdog.service << 'EOF'
[Unit]
Description=AlphaEdge Heartbeat Watchdog
After=network.target

[Service]
Type=oneshot
User=root
WorkingDirectory=/root/alpha_edge
ExecStart=/root/alpha_edge/venv/bin/python3 -m monitoring.heartbeat --check all
EnvironmentFile=-/etc/alphaedge/secrets
StandardOutput=journal
StandardError=journal
EOF

cat > /etc/systemd/system/alphaedge-watchdog.timer << 'EOF'
[Unit]
Description=AlphaEdge Watchdog — check heartbeat every 5 minutes
Requires=alphaedge-watchdog.service

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min
AccuracySec=30s

[Install]
WantedBy=timers.target
EOF

# ── 2. Health report service + timer ──────────────────────────────────────────

cat > /etc/systemd/system/alphaedge-health.service << 'EOF'
[Unit]
Description=AlphaEdge Daily Health Report
After=network.target

[Service]
Type=oneshot
User=root
WorkingDirectory=/root/alpha_edge
ExecStart=/root/alpha_edge/venv/bin/python3 -m monitoring.health_report
EnvironmentFile=-/etc/alphaedge/secrets
StandardOutput=journal
StandardError=journal
EOF

cat > /etc/systemd/system/alphaedge-health.timer << 'EOF'
[Unit]
Description=AlphaEdge Morning Health Report — 08:00 ET (12:00 UTC) daily
Requires=alphaedge-health.service

[Timer]
OnCalendar=*-*-* 12:00:00 UTC
AccuracySec=60s
Persistent=true

[Install]
WantedBy=timers.target
EOF

# ── Enable and start ───────────────────────────────────────────────────────────

systemctl daemon-reload

systemctl enable --now alphaedge-watchdog.timer
systemctl enable --now alphaedge-health.timer

echo ""
echo "Done. Timer status:"
systemctl list-timers alphaedge-* --no-pager
echo ""
echo "Test health report now:"
echo "  cd /root/alpha_edge && /root/alpha_edge/venv/bin/python3 -m monitoring.health_report"
echo ""
echo "Test watchdog now:"
echo "  cd /root/alpha_edge && /root/alpha_edge/venv/bin/python3 -m monitoring.heartbeat --check all"
