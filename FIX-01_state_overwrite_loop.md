# FIX-01 — Break the CI → deploy → VPS state-overwrite loop

**Status:** Plan only. **Nothing applied.** Approve step by step.
**Why this first:** it is the only finding that actively destroys live position state, and it is the reason eleven previous fix attempts failed.

---

## Why the previous eleven attempts failed

Your stash list contains eleven separate commits titled some variant of *"untrack runtime state files."* Every one of them was the correct diagnosis and the wrong sequence.

```
You run:      git rm --cached logs/paper_trades_stocks_only.json
              → file is now untracked. Fixed. ✅

Hours later:  daily_scan.yml runs
              git add -f logs/paper_trades_stocks_only.json
              → the -f flag exists SPECIFICALLY to override .gitignore
              → file is tracked again. ❌

Then:         git commit && git push
              → push to main triggers deploy.yml

Then:         deploy.yml SSHes to VPS
              git stash          → live state removed from working tree
              git pull origin main → CI's simulated state written in its place
              (no git stash pop)  → live state stranded in the stash stack forever
```

`git rm --cached` cannot survive the next cron run. **The order must be: disarm the writers, then disarm the destroyer, then untrack.** Reverse it and this becomes attempt twelve.

---

## Ordering rule

| # | Action | Must come before | Because |
|---|---|---|---|
| 1 | Remove `git add -f` from workflows | untracking | otherwise CI re-tracks within hours |
| 2 | Remove `git stash` from `deploy.yml` | untracking | otherwise the next deploy strands state again |
| 3 | Back up live state off the repo | untracking | `git rm --cached` is safe locally but the next pull elsewhere deletes it |
| 4 | `git rm --cached` | — | now it holds |
| 5 | Verify over one full cron cycle | closing this out | proves it held |

---

## STEP 0 — Freeze (do this before anything, ~2 min)

Disable the deploy workflow so nothing pulls while we work:

**GitHub → Actions → "Deploy to VPS" → ⋯ → Disable workflow**

Then snapshot live state outside the repo:

```bash
mkdir -p /root/alphaedge_state_backup
cp -av /root/alpha_edge/logs/*.json /root/alphaedge_state_backup/
ls -la /root/alphaedge_state_backup/
```

The trading service keeps running throughout. Nothing here touches it.

**Checkpoint:** deploy workflow shows "Disabled", backup directory has ~11 JSON files.

---

## STEP 1 — Stop CI from force-adding state

### `.github/workflows/daily_scan.yml`

In the `Save trading state and generate dashboard` step, **delete these six lines**:

```diff
           python generate_dashboard.py || true
-          git add -f logs/paper_trades_stocks_only.json || true
-          git add -f logs/latest_signals.json || true
-          git add -f logs/sectors.json || true
-          git add -f logs/earnings.json || true
-          git add -f logs/model_auc.json || true
-          git add -f model_cache/ || true
           git add -f docs/index.html || true
           git diff --staged --quiet || git commit -m "Auto: Trading state + dashboard update $(date -u '+%Y-%m-%d %H:%M UTC')"
```

Keep `docs/index.html` **only if** you want the GitHub Pages dashboard to update from CI. Note the VPS cron already regenerates it every 5 minutes, so you have two writers for one file — see the open question at the end.

The existing `upload-artifact` step already retains `logs/` for 30 days. That is the correct way to keep CI output, and it stays.

### `.github/workflows/weekly_maintenance.yml`

```diff
-          git add -f logs/key_ages.json || true
-          git add -f logs/feature_drift_baseline.json || true
           git diff --staged --quiet || git commit -m "Auto: Weekly maintenance state update ..."
```

### Also fix the conflict strategy

In `daily_scan.yml`'s push-retry loop:

```diff
-            git pull --no-edit --strategy-option=ours origin main || true
+            git pull --no-edit --rebase origin main || true
```

On a **pull**, `--strategy-option=ours` means *keep the local (CI runner) version on conflict* — which is what makes CI authoritative over production. `--rebase` replays CI's commits on top of whatever is upstream instead of overwriting it. Once state files are untracked (Step 4), conflicts should stop occurring at all.

**Checkpoint:** `grep -rn 'add -f' .github/workflows/` returns only the `docs/index.html` line, or nothing.

---

## STEP 2 — Stop deploy from stranding live state

### `.github/workflows/deploy.yml`

```diff
             echo "=== Stashing any local generated files ==="
             cd /root/alpha_edge
-            git stash
+            # Do NOT blanket-stash: this machine holds live trading state and
+            # stashes were never popped (117 entries accumulated). Discard only
+            # the generated dashboard, which is rebuilt below anyway.
+            git checkout -- docs/index.html 2>/dev/null || true
 
             echo ""
             echo "=== Pulling latest code ==="
             git pull origin main
```

`git checkout -- docs/index.html` discards local changes to exactly one regenerated file. Everything else in the working tree is left alone.

**If `git pull` then fails** with "local changes would be overwritten," that is the system telling you a state file is still tracked — which Step 4 resolves. Treat that error as informative, not as something to force past. **Do not** add `git reset --hard`; that reintroduces the same data loss by another route.

**Checkpoint:** no `git stash` anywhere in `deploy.yml`.

---

## STEP 3 — Recover what is in the stash stack

Six stashes hold `.py` changes; `stash@{36}` is the only place `_recover_state_if_needed` has ever existed — `git log -S` confirms it was never committed.

```bash
cd /root/alpha_edge
git stash show -p 'stash@{36}' > /root/alphaedge_state_backup/stash36.patch
git stash show -p 'stash@{115}' > /root/alphaedge_state_backup/stash115.patch
```

Review those patches before applying anything. **Do not `git stash pop`** — the stashes are based on old commits and will conflict. Cherry-pick the wanted hunks by hand.

The remaining 111 state-only stashes can be dropped once the six are resolved, but leave them until Step 5 passes. They cost 154MB total and are not urgent.

---

## STEP 4 — Untrack the state files (now it holds)

```bash
cd /root/alpha_edge
git rm --cached logs/paper_trades_stocks_only.json \
                logs/latest_signals.json \
                logs/sectors.json \
                logs/earnings.json \
                logs/model_auc.json \
                logs/kill_switch.json
git commit -m "Untrack runtime state — .gitignore was inert while these were tracked

.gitignore only applies to untracked files. These were tracked, so eleven
prior untracking attempts were undone by 'git add -f' in daily_scan.yml
within hours. That force-add is removed as of the preceding commit."
```

`logs/kill_switch.json` matters most — it is the fail-closed emergency halt, and `AUDIT_REPORT.md` records an incident where it was committed as `active: true` and rejected every inbound webhook on a fresh deploy.

Add the two missing entries to `.gitignore`:

```diff
 logs/circuit_breaker.json
+logs/kill_switch.json
+logs/paper_trades_merged.json
```

**Do not push yet.** Re-enable the deploy workflow first, then push, so the deploy that fires runs the *fixed* version.

**Checkpoint:** `git ls-files | grep -E 'logs/.*\.json'` returns only `logs/backtest/*` historical files.

---

## STEP 5 — Verify across one full cycle

Re-enable the deploy workflow, push, then check after the next scheduled scan (13:45, 16:30, or 20:15 UTC):

```bash
cd /root/alpha_edge
git stash list | wc -l        # must NOT have grown
git ls-files | grep -c 'logs/.*\.json'   # expect only backtest history
git log --oneline -5          # auto-commits should no longer touch logs/
ls -la logs/paper_trades_stocks_only.json  # mtime should be from the VPS, not a pull
```

**Pass:** stash count unchanged, no `logs/*.json` in new commits, state file mtime tracks VPS scans.
**Fail:** stash count grew → a `git stash` survives somewhere. `grep -rn 'git stash' .github/`.

---

## The deeper question — should CI run the trading scan at all?

Everything above stops CI from *persisting* state. It does not stop CI from *generating* it.

`daily_scan.yml` runs `cloud_scan.py` → `run_daily_scan()` → `PaperTrader`, three times each weekday on an ephemeral runner with no broker connection. It builds a portfolio that exists only for the life of that runner. Meanwhile the VPS runs `alpaca_live.py` against real Alpaca. Two systems, two portfolios, one repo.

The 20:15 UTC cron is 16:15 ET — the exact minute `alpaca_live.py` runs its daily cycle.

Three options:

| Option | What it means | Trade-off |
|---|---|---|
| **A. CI stops trading entirely** | Delete the scan cron from `daily_scan.yml`; keep `ci.yml`, `daily_audit.yml`, `security.yml` | Cleanest. Loses the cloud dashboard refresh. |
| **B. CI runs signals only** | Split `run_daily_scan()` so CI computes signals without instantiating `PaperTrader` | Keeps the Pages dashboard live; needs a code change |
| **C. CI writes to its own namespace** | `logs/ci/` prefix, never touching VPS paths | Least invasive; leaves two portfolios drifting |

I would push for **A**. The VPS is your trader; CI's job is tests and audits. Running the full pipeline in CI produces a second portfolio whose only effect on the world has been to corrupt the first one. But this is your architecture call, and B is defensible if the GitHub Pages dashboard matters to you.

There is a related question underneath it: **the VPS cron regenerates `docs/index.html` every 5 minutes** (288×/day) while CI regenerates and commits the same file. Even after this fix, two writers share one file. Worth resolving alongside the option above.

---

## Rollback

Every step is reversible.

- Steps 1, 2: `git revert` the workflow commits.
- Step 4: `git add -f <file>` restores tracking (that is precisely why it worked against you).
- Step 0's backup at `/root/alphaedge_state_backup/` is the authority on state as of the freeze.

The trading service is never stopped and no source code outside `.github/workflows/` is modified.

---

## What I need before applying anything

1. Approval, step by step — I will not run these.
2. Your call on A / B / C above.
3. Confirmation that no scan is mid-flight when you start Step 0.
