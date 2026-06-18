# AlphaEdge Incident Runbook

> **Purpose**: Step-by-step response guide for every known failure mode.  
> **Audience**: You (the operator) at 3am with the bot down.  
> **Server**: `root@67.205.185.84` | SSH key: `~/.ssh/id_alpha_edge`

---

## Emergency Contacts

| Resource | Value |
|----------|-------|
| Server IP | `67.205.185.84` |
| SSH command | `ssh -i ~/.ssh/id_alpha_edge root@67.205.185.84` |
| Alpaca dashboard | https://app.alpaca.markets |
| Gate.io dashboard | https://www.gate.io |
| Webhook URL | `http://67.205.185.84:5001/webhook` |
| Kill switch URL | `http://67.205.185.84:5001/kill-switch` |
| Health check | `http://67.205.185.84:5001/health` |

---

## IMMEDIATE: Emergency Halt (Kill Switch)

Use this **first** when you see unexpected or dangerous behaviour.

### Activate via curl (from any machine)
```bash
curl -X POST http://67.205.185.84:5001/kill-switch \
  -H "Content-Type: application/json" \
  -d '{"secret": "YOUR_WEBHOOK_SECRET", "reason": "unexpected losses"}'
```

### Check status
```bash
curl http://67.205.185.84:5001/kill-switch
```

### Resume trading
```bash
curl -X POST http://67.205.185.84:5001/kill-switch/reset \
  -H "Content-Type: application/json" \
  -d '{"secret": "YOUR_WEBHOOK_SECRET"}'
```

> The kill switch fires a **Telegram CRITICAL alert** on activation and reset.

---

## Scenario 1: Bot is Silent (Telegram alert: "Bot Silent")

**Symptoms**: No Telegram messages for > 5 minutes during market hours.

### Step 1 — SSH and check service status
```bash
ssh -i ~/.ssh/id_alpha_edge root@67.205.185.84
systemctl status alpaca.service
systemctl status gateio.service
```

### Step 2 — Check recent logs
```bash
journalctl -u alpaca.service -n 50 --no-pager
journalctl -u gateio.service -n 50 --no-pager
```

### Step 3 — Check heartbeat file
```bash
cat /root/alpha_edge/logs/heartbeats/alpaca_bot.json
```
Look at `last_ping` timestamp and `status` field.

### Step 4 — Restart if needed
```bash
systemctl restart alpaca.service
systemctl restart gateio.service
```

### Step 5 — Verify restart
```bash
systemctl status alpaca.service   # should show "active (running)"
curl http://67.205.185.84:5001/health
```

---

## Scenario 2: Position Mismatch on Startup (Telegram: "Position Mismatch")

**Symptoms**: Telegram alert saying PHANTOM or ORPHAN or MISMATCH position.

> **NEVER** auto-trade until this is resolved.

### Step 1 — Read reconciliation log
```bash
ssh -i ~/.ssh/id_alpha_edge root@67.205.185.84
tail -20 /root/alpha_edge/logs/reconciliation.log | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(json.dumps(d, indent=2))
"
```

### Step 2 — Check actual Alpaca positions
Go to https://app.alpaca.markets → Portfolio → Positions.

### Step 3 — Check local state
```bash
cat /root/alpha_edge/logs/paper_trades_stocks_only.json | python3 -m json.tool | head -60
```

### Step 4 — Resolve

| Issue | Action |
|-------|--------|
| **PHANTOM** (broker has, local doesn't) | Manually add position to local JSON OR close the position on Alpaca |
| **ORPHAN** (local has, broker doesn't) | Remove position from local JSON (`positions` key) |
| **MISMATCH** (value differs) | Update `shares` or `current_price` in local JSON to match broker |

### Step 5 — Restart cleanly
```bash
systemctl restart alpaca.service
```
Bot will re-reconcile on startup — verify clean Telegram message.

---

## Scenario 3: Webhook Not Receiving Signals

**Symptoms**: TradingView alerts fire but no trades are placed.

### Step 1 — Check webhook health
```bash
curl http://67.205.185.84:5001/health
```
Expected: `{"status": "running", "kill_switch": false, ...}`

### Step 2 — Check if kill switch is on
```bash
curl http://67.205.185.84:5001/kill-switch
```
If `"active": true` → reset it (see Emergency Halt section above).

### Step 3 — Check webhook service
```bash
systemctl status webhook.service
journalctl -u webhook.service -n 30 --no-pager
```

### Step 4 — Test webhook manually
```bash
curl -X POST http://67.205.185.84:5001/webhook \
  -H "Content-Type: application/json" \
  -d '{"secret":"YOUR_SECRET","action":"BUY","symbol":"AAPL","price":150}'
```
Expected: `{"status": "received", ...}`

### Step 5 — View recent signals
```bash
curl http://67.205.185.84:5001/signals
```

---

## Scenario 4: Circuit Breaker Triggered (daily loss limit hit)

**Symptoms**: Telegram says "Circuit Breaker Active" or bot stops trading mid-day.

### Step 1 — Check circuit breaker state
```bash
cat /root/alpha_edge/logs/circuit_breaker.json
```

### Step 2 — Check daily P&L
```bash
cat /root/alpha_edge/logs/paper_trades_stocks_only.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Capital:', d.get('capital'))
print('Positions:', list(d.get('positions', {}).keys()))
"
```

### Step 3 — Let it auto-reset (preferred)
Circuit breaker auto-resets at midnight ET. Wait for the next trading day.

### Step 4 — Manual reset (emergency only)
```bash
python3 -c "
import json
with open('/root/alpha_edge/logs/circuit_breaker.json') as f:
    d = json.load(f)
d['halted'] = False
d['daily_loss'] = 0.0
with open('/root/alpha_edge/logs/circuit_breaker.json', 'w') as f:
    json.dump(d, f, indent=2)
print('Reset OK')
"
systemctl restart alpaca.service
```

---

## Scenario 5: Disk Full or Logs Growing Large

### Check disk usage
```bash
df -h /root
du -sh /root/alpha_edge/logs/
```

### Rotate logs
```bash
find /root/alpha_edge/logs/audits/ -name "*.md" -mtime +30 -delete
find /root/alpha_edge/logs/backtest/ -name "*.json" -mtime +30 -delete
```

### Permanent fix — add logrotate config
```bash
cat > /etc/logrotate.d/alphaedge << 'EOF'
/root/alpha_edge/logs/*.log {
    weekly
    rotate 4
    compress
    missingok
    notifempty
}
EOF
```

---

## Scenario 6: API Key Expired or Revoked

**Symptoms**: `ERROR: Alpaca not connected` or `Gate.io authentication failed`.

### Step 1 — Regenerate key on broker dashboard
- Alpaca: https://app.alpaca.markets → API Keys → Generate New Key
- Gate.io: https://www.gate.io → Account → API Management

### Step 2 — Update secrets on server
```bash
ssh -i ~/.ssh/id_alpha_edge root@67.205.185.84
nano /etc/alphaedge/secrets
# Update ALPACA_API_KEY, ALPACA_SECRET_KEY or GATEIO_API_KEY, GATEIO_SECRET
```

### Step 3 — Restart services
```bash
systemctl daemon-reload
systemctl restart alpaca.service gateio.service
```

### Step 4 — Verify connection
```bash
journalctl -u alpaca.service -n 20 --no-pager | grep -E 'connected|error|ERROR'
```

---

## Scenario 7: Server Unreachable

**Symptoms**: SSH fails, curl times out.

### Step 1 — Check DigitalOcean console
Go to https://cloud.digitalocean.com → Droplets → `alpha-edge` → Console

### Step 2 — Power cycle from dashboard
Droplets → `alpha-edge` → Power → Power Cycle

### Step 3 — Verify services on restart
Services are managed by systemd with `Restart=always` — they should
auto-start. Verify:
```bash
systemctl status alpaca.service gateio.service webhook.service
```

---

## Watchdog Cron (auto-installed)

The watchdog checks all heartbeat files every 5 minutes and fires a
Telegram alert if any bot goes silent:

```cron
*/5 * * * * /root/alpha_edge/venv/bin/python -m monitoring.heartbeat \
  --check all --dir /root/alpha_edge/logs/heartbeats \
  --stale-sec 300 >> /var/log/alphaedge-watchdog.log 2>&1
```

Check cron is installed:
```bash
crontab -l | grep heartbeat
```

Check watchdog log:
```bash
tail -20 /var/log/alphaedge-watchdog.log
```

---

## Health Check Summary

Run this at any time to see full system state:

```bash
echo "=== Services ===" && \
systemctl status alpaca.service gateio.service webhook.service --no-pager -l | grep -E "Active:|Main PID:|Exit code:" && \
echo "=== Heartbeats ===" && \
ls -la /root/alpha_edge/logs/heartbeats/ && \
echo "=== Kill Switch ===" && \
curl -s http://67.205.185.84:5001/kill-switch && echo && \
echo "=== Last Reconciliation ===" && \
tail -1 /root/alpha_edge/logs/reconciliation.log | python3 -m json.tool
```

---

## Runbook Maintenance

| Action | When |
|--------|------|
| Update secret references | After rotating API keys |
| Add new scenarios | After each incident post-mortem |
| Test kill switch | Monthly drill |
| Check cron | After server restart |
