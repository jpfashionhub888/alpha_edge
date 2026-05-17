# Key Rotation Runbook

After applying the security patches in this zip, do this — in order.
Estimated time: 15 minutes.

## 1. Regenerate Alpaca paper-trading keys

The keys hardcoded in `execution/alpaca_broker.py` must be considered
compromised. Even paper keys can be used by anyone to:
- Place paper trades on your account
- Read your portfolio and trading history
- Probe whether the same credentials work against the live URL

Steps:
1. Log in to https://app.alpaca.markets
2. Navigate to: Paper Trading → API Keys
3. Click "Regenerate"
4. Copy the new key + secret into `config/secrets.env`:
   ```
   ALPACA_API_KEY=PK...new_key...
   ALPACA_SECRET_KEY=...new_secret...
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ```
5. Update GitHub Actions secrets (Settings → Secrets → Actions):
   - `ALPACA_API_KEY`
   - `ALPACA_SECRET_KEY`

## 2. Regenerate Telegram bot token (if ever committed)

Check git history:
```bash
git log --all --source -p | grep -iE 'telegram.*token|TELEGRAM_BOT' | head
```

If you see a token starting with digits followed by `:AAH...` or similar
in any commit, it's leaked. Even if it was deleted in a later commit,
the value lives in history.

Rotate:
1. Open Telegram → @BotFather
2. `/mybots` → select your bot → API Token → Revoke current token
3. Generate new token
4. Update `config/secrets.env` and GitHub Actions secrets:
   ```
   TELEGRAM_BOT_TOKEN=new_token_here
   ```

## 3. Set WEBHOOK_SECRET

The new `webhook_server.py` requires this in env. Generate a random one:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Add to `config/secrets.env`:
```
WEBHOOK_SECRET=...output_from_above...
```

And configure the same value in your TradingView alert JSON payload.

## 4. (Optional but recommended) Scrub git history

If any of the rotated secrets ever appeared in a commit, they're
permanently in history accessible to anyone with a clone. Once rotated,
the OLD values are useless — but if you want a clean history:

```bash
# Install git-filter-repo (one-time)
pip install git-filter-repo --break-system-packages

# Back up the repo first!
cd /path/to/alpha_edge
git filter-repo --replace-text <(cat <<EOF
PK5TI3TNXIJLPTQ46UUPZIUHXM==>REDACTED
DUvwnAtnL49fZQ6RwiRDeXzb1EoNiVYs1TFvKG2w2M1A==>REDACTED
alphaedge_secret_2026==>REDACTED
EOF
)
# Force-push
git push --force origin main
```

This rewrites history. Anyone who has cloned the repo before will be
on a now-divergent branch. For a personal-use repo with 0 contributors,
that's fine.

## 5. Verify

After rotating, verify the new keys work:

```bash
# Set env vars (or export from secrets.env)
export ALPACA_API_KEY=new_key
export ALPACA_SECRET_KEY=new_secret

python -c "from execution.alpaca_broker import AlpacaBroker; b = AlpacaBroker(); b.get_summary()"
```

You should see `Alpaca connected (paper mode)` and your account
buying power. If you see `Alpaca not initialised`, env vars aren't set.
