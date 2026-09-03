# CRITICAL — Automated loop overwrites live trading state with CI simulation output

**Date:** 2026-07-24
**Severity:** P0 — corrupts the source of truth for open positions on a live-trading VPS
**Status:** Diagnosed from workflow + service files. **Nothing changed.** One command below confirms it in 5 seconds.

---

## Confirm it first — one command on the VPS

```bash
cd ~/alpha_edge && git stash list
```

If that returns a stack of entries — likely dozens, dated across weeks — the loop below is real and has been running since the deploy workflow was added. **Each entry is a snapshot of live position state that was silently discarded.**

Also run:

```bash
git ls-files | grep -E 'kill_switch|paper_trades|latest_signals'
```

Any output means `.gitignore` is inert for those paths. It only applies to *untracked* files; these are tracked, so the ignore rules have never done anything.

---

## The loop

Three environments write to one git branch. Here is what happens every weekday:

```
┌─ GitHub Actions ─ daily_scan.yml ─ cron 13:45, 16:30, 20:15 UTC ────────┐
│  Ephemeral ubuntu runner, NO broker connection                          │
│  runs cloud_scan.py → main.run_daily_scan() → PaperTrader               │
│  simulates a portfolio from whatever state is in git                    │
│                                                                          │
│  git add -f logs/paper_trades_stocks_only.json   ← force, overrides      │
│  git add -f logs/latest_signals.json                .gitignore           │
│  git add -f model_cache/                                                 │
│  git commit && git push origin main                                      │
│    └─ on conflict: git pull --strategy-option=ours  ← CI ALWAYS WINS     │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ push to main triggers…
┌──────────────────────────────▼───────────────────────────────────────────┐
│  deploy.yml — on: push: branches: [main]                                 │
│  ssh root@67.207.82.11:                                                  │
│      git stash          ← live VPS state removed from working tree       │
│      git pull origin main   ← CI's simulated state written in its place  │
│      python3 generate_dashboard.py                                       │
│      (stash is NEVER popped)                                             │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────────┐
│  VPS — alphaedge.service → alpaca_live.py                                │
│  Holds the REAL Alpaca positions.                                        │
│  On next startup, reconciliation.py reads                                │
│  logs/paper_trades_stocks_only.json — now containing CI's fake state —   │
│  and compares it against the real broker.                                │
└──────────────────────────────────────────────────────────────────────────┘
```

### The three defects that combine

**1. `deploy.yml` line ~49: `git stash` with no `git stash pop`.**
Live state files are stashed away and never restored. The stash stack grows by one entry per deploy, forever. This is the single most damaging line — it is where real position state goes to die.

**2. `daily_scan.yml`: `git add -f` on runtime state files.**
The `-f` flag exists specifically to override `.gitignore`. Your `.gitignore` says, in your own words:

> *"Runtime state files — VPS-local only, NEVER commit. Committing them causes git pull to overwrite live VPS state with stale repo data → shows fake/broken data on the dashboard."*

The workflow force-adds five of the exact files that comment names. Someone diagnosed this correctly and wrote the warning, then the fix landed in `.gitignore` — which cannot work, because the files were already tracked.

**3. `daily_scan.yml` push retry: `git pull --strategy-option=ours`.**
During a *pull*, `ours` means the local branch — the CI runner. So whenever CI's state conflicts with anything the VPS pushed, **CI's simulated state deterministically wins.** The retry loop was added to fix silent push failures, which was a real bug; the conflict strategy chosen to fix it makes CI authoritative over production.

---

## Why this matters more than a stale dashboard

Your own summary defines the blast radius:

> `logs/paper_trades_stocks_only.json` — **source of truth for open positions (what reconciler reads)**

And your reconciler's contract:

| Condition | Action |
|---|---|
| Local has position, broker doesn't | **ORPHAN → auto-correct** (silently removed) |
| Broker has position, local doesn't | **PHANTOM → halt trading** |
| Values disagree | **MISMATCH → halt trading** |

Feeding it CI-generated state produces exactly these, by construction:

- CI's simulated portfolio contains symbols Alpaca never traded → **ORPHAN → silently auto-deleted.** Real tracking is lost without a halt, because orphan handling is the one path designed *not* to stop.
- Real Alpaca positions absent from CI's state → **PHANTOM → halt.** Trading stops for a reason that has nothing to do with the market.

The safety layer isn't failing. It's working correctly on corrupt input.

**Timing:** the 20:15 UTC cron is 16:15 ET — the exact minute `alpaca_live.py` runs its daily cycle. The third CI run collides with the live trading window every weekday.

I'd note the hostname is `alphaedge-recovered`. If there was a prior incident that required recovery, this loop is a strong candidate for the cause.

---

## Secondary findings from the same workflows

### CI ignores `requirements.txt` entirely
`requirements.txt` pins 219 of 226 packages. `daily_scan.yml` ignores it and runs bare `pip install yfinance pandas numpy scikit-learn xgboost lightgbm catboost ...` with **no version constraints**, resolving to whatever is newest that day.

That runner then trains models and `git add -f model_cache/` commits the pickles. The VPS pulls them into a venv with different library versions. Scikit-learn pickles are not version-stable — this either raises on load (caught by one of the 133 silent exception handlers, so you'd never see it) or silently deserializes into subtly different behaviour. Either way the model actually scoring your trades is not the model you validated.

This also undermines your quality gate. `val_AUC ≥ 0.53` is enforced at training time in CI, under one set of library versions; inference happens on the VPS under another.

### `logs/kill_switch.json` is tracked
Your summary says it's gitignored. It is listed in `.gitignore` but **tracked in git**, so the rule has no effect. The kill switch is your fail-closed emergency halt — and its state travels through git and is subject to the same stash-and-overwrite path as everything else. Your `AUDIT_REPORT.md` already records one incident where it was committed as `active: true` and rejected every inbound webhook on a fresh deploy. Same root cause, still unfixed.

### Two claims in your summary that don't match this tree
Worth verifying against VPS HEAD (`bd5f17c`) rather than taking on trust — this tree is 6 commits behind, so these may already be fixed:

| Your claim | What I find here |
|---|---|
| "All JSON state writes: fsync before rename" | **`fsync` appears zero times** in the entire codebase. `atomic_json_write()` in `main.py:63` does tempfile + `os.replace` — atomic against a *killed process*, but **not** against power loss, because the data can still be in page cache when the rename lands. The rename half is right; the durability half is absent. |
| "6 inline write blocks → `_atomic_json_dump()`" | No function by that name exists here. **29 raw `json.dump()` calls remain**, including `risk_circuit_breaker.py:82` (circuit-breaker state) and `alpaca_live.py:802` (`managed_positions`) — both safety-critical, both non-atomic. |
| "systemd `Restart=always` (P0 fix pending)" | `alphaedge.service` here already reads `Restart=on-failure`, `RestartSec=30`, `StartLimitBurst=5`. Possibly already fixed, or the VPS unit file differs from the repo. |

The third one raises a broader question: **is the running unit file the one in the repo?** If `/etc/systemd/system/alphaedge.service` was hand-edited on the VPS, the repo copy is decoration. Worth checking with `systemctl cat alphaedge`.

### `docs/index.html` committed every run
1,100–1,600 lines of generated dashboard churned into history three times a weekday, plus `model_cache/` binaries. This is a large part of your 84% disk pressure and it makes `git log` useless for reviewing actual code changes.

---

## Proposed fix — for your approval, nothing done yet

Ordered so that each step is safe on its own.

### Step 1 — Stop the bleeding (5 minutes, VPS only, no code changes)
```bash
# Recover whatever live state is sitting in the stash graveyard
cd ~/alpha_edge
git stash list                    # inspect first
git stash show -p stash@{0}       # is this real position state?
```
Then **disable the deploy workflow** (Actions tab → deploy.yml → Disable workflow) so no further pulls clobber state while we fix it. The VPS keeps trading; it just stops receiving destructive deploys.

### Step 2 — Untrack the state files
```bash
git rm --cached logs/paper_trades_stocks_only.json logs/latest_signals.json \
                logs/sectors.json logs/earnings.json logs/model_auc.json \
                logs/kill_switch.json
git commit -m "Untrack runtime state — .gitignore was inert while these were tracked"
```
After this the existing `.gitignore` rules finally take effect. **Back up each file on the VPS first** — `git rm --cached` leaves the working copy, but the next pull on another machine will delete its copy.

### Step 3 — Fix the workflows
- `deploy.yml`: replace `git stash` with a targeted `git checkout -- docs/` for generated files only, or add `git stash pop`. Never blanket-stash on a machine holding live state.
- `daily_scan.yml`: delete every `git add -f` line for `logs/` and `model_cache/`. Keep the `upload-artifact` step — that's the right way to retain CI output.
- `daily_scan.yml`: replace the dependency block with `pip install -r requirements.txt`.
- Reconsider whether CI should run `run_daily_scan()` at all. If the VPS is the real trader, CI running the full pipeline is duplicated work that produces a competing portfolio. Running the *scan* for signals is defensible; running `PaperTrader` and persisting positions is not.

### Step 4 — Decide the ownership rule
One sentence, then enforce it in code: **the VPS owns runtime state; git owns code; CI owns neither.** Everything above follows from it.

---

## What I need from you

1. Run `git stash list` and `git ls-files | grep -E 'kill_switch|paper_trades'` and paste the output — confirms the loop and tells us how much state is recoverable.
2. Confirm whether the VPS is genuinely paper-only. Your summary says Alpaca paper account; if any live capital is attached anywhere, Step 1 becomes urgent rather than important.
3. Approve or amend the four steps. I've made no changes and won't without your go-ahead.
