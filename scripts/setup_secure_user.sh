#!/bin/bash
# scripts/setup_secure_user.sh
# Phase 2A+2B: Create dedicated alphaedge system user and harden systemd units
#
# Run once as root on the production server:
#   bash /root/alpha_edge/scripts/setup_secure_user.sh
#
# What this does:
#   1. Creates a locked system user 'alphaedge' (no login shell, no home dir)
#   2. Moves secrets from flat config/secrets.env to /etc/alphaedge/secrets (chmod 600)
#   3. Rewrites all systemd unit files to run as 'alphaedge' with security hardening
#   4. Reloads systemd and restarts all services

set -euo pipefail

INSTALL_DIR="/root/alpha_edge"
SECRETS_DEST="/etc/alphaedge/secrets"
SERVICE_FILES=(alpaca gateio dashboard alphaedge-audit)

echo "=============================================="
echo " AlphaEdge — Security Hardening Setup"
echo "=============================================="

# ── 1. Create system user ──────────────────────────────────────────────────────
if id alphaedge &>/dev/null; then
    echo "✅ User 'alphaedge' already exists"
else
    useradd \
        --system \
        --no-create-home \
        --shell /usr/sbin/nologin \
        --comment "AlphaEdge trading bot service account" \
        alphaedge
    echo "✅ Created system user 'alphaedge'"
fi

# ── 2. Migrate secrets ─────────────────────────────────────────────────────────
mkdir -p /etc/alphaedge
chmod 750 /etc/alphaedge
chown root:alphaedge /etc/alphaedge

if [ -f "$INSTALL_DIR/config/secrets.env" ]; then
    # Strip comments and blank lines, write to system secrets file
    grep -v '^\s*#' "$INSTALL_DIR/config/secrets.env" \
        | grep -v '^\s*$' \
        > "$SECRETS_DEST"
    chmod 640 "$SECRETS_DEST"
    chown root:alphaedge "$SECRETS_DEST"
    echo "✅ Secrets migrated to $SECRETS_DEST (chmod 640, group: alphaedge)"
    echo "   ⚠️  You can now delete config/secrets.env from the repo directory"
else
    echo "⚠️  config/secrets.env not found — create $SECRETS_DEST manually"
    touch "$SECRETS_DEST"
    chmod 640 "$SECRETS_DEST"
    chown root:alphaedge "$SECRETS_DEST"
fi

# ── 3. Set ownership ───────────────────────────────────────────────────────────
chown -R alphaedge:alphaedge "$INSTALL_DIR"
# Logs directory writable by bot
mkdir -p "$INSTALL_DIR/logs"
chmod 755 "$INSTALL_DIR/logs"
echo "✅ Ownership set: $INSTALL_DIR → alphaedge:alphaedge"

# ── 4. Write hardened systemd unit files ───────────────────────────────────────

write_unit() {
    local name="$1"
    local exec_cmd="$2"
    local description="$3"

    cat > "/etc/systemd/system/${name}.service" <<EOF
[Unit]
Description=${description}
After=network.target
Wants=network.target

[Service]
Type=simple
User=alphaedge
Group=alphaedge
WorkingDirectory=${INSTALL_DIR}
ExecStart=${exec_cmd}
Restart=always
RestartSec=30

# ── Security hardening ─────────────────────────────────────────────────
EnvironmentFile=${SECRETS_DEST}
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=false
CapabilityBoundingSet=
AmbientCapabilities=

[Install]
WantedBy=multi-user.target
EOF
    echo "✅ Wrote /etc/systemd/system/${name}.service"
}

write_unit "alpaca"    "${INSTALL_DIR}/venv/bin/python ${INSTALL_DIR}/alpaca_live.py"   "AlphaEdge Alpaca Stock Bot"
write_unit "gateio"    "${INSTALL_DIR}/venv/bin/python ${INSTALL_DIR}/gateio_live.py"   "AlphaEdge Gate.io Crypto Bot"
write_unit "dashboard" "${INSTALL_DIR}/venv/bin/python -m http.server 8050 --directory ${INSTALL_DIR}/docs" "AlphaEdge Dashboard"

# Audit timer (one-shot, no Restart)
cat > /etc/systemd/system/alphaedge-audit.service <<EOF
[Unit]
Description=AlphaEdge Daily Deep Audit
After=network.target

[Service]
Type=oneshot
User=alphaedge
Group=alphaedge
WorkingDirectory=${INSTALL_DIR}
ExecStart=${INSTALL_DIR}/venv/bin/python ${INSTALL_DIR}/deep_audit.py
EnvironmentFile=${SECRETS_DEST}
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=${INSTALL_DIR}/logs
ProtectHome=true
CapabilityBoundingSet=
EOF

cat > /etc/systemd/system/alphaedge-audit.timer <<EOF
[Unit]
Description=AlphaEdge Daily Deep Audit — runs every day at 06:00 UTC

[Timer]
OnCalendar=*-*-* 06:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
EOF
echo "✅ Wrote audit service + timer"

# ── 5. Reload and restart ──────────────────────────────────────────────────────
systemctl daemon-reload

for svc in alpaca gateio dashboard; do
    systemctl enable "${svc}.service" 2>/dev/null || true
    systemctl restart "${svc}.service"
    STATUS=$(systemctl is-active "${svc}.service" 2>/dev/null)
    echo "  ${svc}.service → ${STATUS}"
done

systemctl enable alphaedge-audit.timer 2>/dev/null || true
systemctl restart alphaedge-audit.timer
echo "  alphaedge-audit.timer → $(systemctl is-active alphaedge-audit.timer)"

echo ""
echo "=============================================="
echo " Hardening complete!"
echo " All services now run as: alphaedge (not root)"
echo " Secrets loaded from:     $SECRETS_DEST"
echo " Next audit:              $(systemctl show alphaedge-audit.timer -p NextElapseUSecRealtime --value 2>/dev/null || echo 'check systemctl')"
echo "=============================================="
