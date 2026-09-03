# Phase 0+1 — Correction, Blockers, and Handover Commands

**Date:** 2026-07-24
**Status:** Phase 1 **not executed.** Two blockers, one of which invalidates part of my earlier recommendation.

---

## 1. Correction to the audit: "the nested clone is clean" was wrong

I told you the nested tree was the clean one and recommended promoting it. That was based on a compile check. **Compiling is not the same as being correct**, and when I diffed the two trees file-by-file before archiving anything, the nested tree turned out to be damaged too — just more quietly.

### `alpha_edge\alpha_edge\main.py` has no working entry point

The file ends:

```python
    logger.info("run_daily_scan() complete — %d signals generated", len(stock_signals))
    return stock_signals


    run_daily_scan()      # ← indented, sits AFTER `return`. Unreachable.
```

The `if __name__ == '__main__':` guard is **gone entirely**. The call to `run_daily_scan()` is inside the function body, after the `return` statement — dead code the interpreter never reaches.

**Consequence:** `python main.py` on the nested tree exits silently with status 0 and does nothing. No scan, no error, no log line. That is the single worst failure mode in this entire audit, because it looks like success.

Compare the two trees on the same file:

| Tree | `main.py` tail | Behaviour |
|---|---|---|
| Root | `if __name__ == '__main__': run_daily_scan()` + stray `daily_scan()` | **Loud** crash: `NameError` |
| Nested | Guard deleted, call orphaned after `return` | **Silent** no-op — worse |

Root fails visibly. Nested fails invisibly. A loud crash is a better failure than a silent one, and I under-weighted that.

### Other nested-only damage

| File | Problem |
|---|---|
| `monitoring/telegram_bot.py` | Truncated mid-function at `result = self.send_message(text)` — no `return`, no `__main__` guard, no trailing newline. Root has the complete version. |
| `tests/test_crash_recovery.py` | Lost its `__main__` guard (root retains it) |
| 40 `.py` files | End with no trailing newline — same truncation signature as root, which also has exactly 40 |

### Revised conclusion

**Neither tree is a clean base.** They are damaged in *different, non-overlapping* files by the same mechanism. The damage is roughly symmetrical:

- Nested lost `__main__` guards in **3** files that root has
- Root lost `__main__` guards in **0** files that nested has
- Both trees have exactly **40** files with no trailing newline

"Pick the winner and delete the loser" is the wrong plan. The right plan is **file-by-file reconciliation**, using git history from both repos to establish which version of each file is genuinely newer versus merely truncated.

---

## 2. Blockers preventing me from executing Phase 1

### Blocker A — I cannot delete or move files
`rm` returns `Operation not permitted` in this workspace, and the delete-permission request was declined. Phase 1 is inherently destructive: it requires moving the nested tree, archiving the root, and removing `alpha_edge/` from `.gitignore`. **None of it can be done from here.** Commands for you to run are in section 4.

### Blocker B — I left two artifacts behind. One matters.

| Path | Impact | Action |
|---|---|---|
| `alpha_edge\alpha_edge\.git\index.lock` | **Blocks all git operations in the nested repo** — any `git add`/`commit`/`status` there will fail with "Unable to create index.lock: File exists" until removed. Created by a `git status` I ran that couldn't clean up after itself. | **Delete it.** It is a zero-byte lock file; removing it is safe as long as no git process is running. |
| `alpha_edge\_wtest` | Harmless empty file from a write test | Delete at leisure |

I apologise for the lock file — that one is actively in your way.

---

## 3. Salvage list — root-tree work that exists nowhere else

The root tree has 28 uncommitted modified files. Most duplicate work already in nested, but these are **unique and would be lost** if you archived the root without extracting them:

### Must salvage

**`veto_agent.py` — real logic bug, fixed only in root**

Nested (buggy):
```python
decision = 'VETO'
reason   = f"Unexpected decision value '{decision}' from model"   # always prints 'VETO'
```
Root (fixed):
```python
bad_value = decision
decision  = 'VETO'
reason    = f"Unexpected decision value '{bad_value}' from model"  # prints the real value
```
The nested version reassigns `decision` *before* interpolating it, so the diagnostic always reports `'VETO'` and the actual malformed model output is never recorded. Root's fix is correct. **Take root's version** (minus its appended `']:.0%}")` corruption on line 204).

**`monitoring/telegram_bot.py`** — nested is truncated mid-function. **Take root's version** (242 lines, complete, has `__main__` guard).

**`main.py`** — neither version is usable as-is. Take nested's body, then restore root's `if __name__ == '__main__': run_daily_scan()` guard, and drop root's stray `daily_scan()` line.

### Probably take nested's version

**`gateio_live.py`** — nested has newer reconciliation logic (auto-corrects orphan positions, halts only on phantoms). Root has the older hard-halt-with-`ALPHAEDGE_FORCE_START` behaviour. Nested is the improvement.

*But* root has a guard nested lacks:
```python
if PAPER_TRADE:
    self.paper.save_state()
```
Nested calls `self.paper.save_state()` unconditionally. Verify which is intended — if this runs in live mode, nested writes paper-trading state during live trading. **Take nested's reconciliation + root's `PAPER_TRADE` guard.**

**`models/technical_model.py`** — nested 483 lines vs root 458. Nested likely newer; diff before deciding.

### Low value

**`risk_circuit_breaker.py`** — root's extra 7 lines are `__main__` demo prints only. Not worth conflict.

### Not yet diffed
`bybit_live.py`, `deploy.py`, `execution/alpaca_broker.py`, `market_regime.py`, `models/regime_detector.py`, `tests/test_crash_recovery.py` differ between trees and need review before either is discarded.

---

## 4. Commands for you to run

Run these yourself — I cannot. Read each before running; do not paste blindly.

### Step 0 — Clear my lock file, then back up
```powershell
# Remove the lock I left behind (safe — no git process should be running)
Remove-Item "C:\Users\giris\alpha_edge\alpha_edge\.git\index.lock" -Force
Remove-Item "C:\Users\giris\alpha_edge\_wtest" -Force

# Back up BOTH trees before anything else (excludes venv; ~186MB + 142MB)
cd C:\Users\giris
tar --exclude=venv --exclude=__pycache__ -czf alpha_edge_backup_20260724.tar.gz alpha_edge
```

### Step 1 — Revoke the token (still outstanding, still P0)
Go to **github.com/settings/tokens** and revoke `ghp_CpF5TYuc...`. Then:
```powershell
cd C:\Users\giris\alpha_edge
git remote set-url origin https://github.com/jpfashionhub888/alpha_edge.git
gh auth login    # or configure Git Credential Manager
```

### Step 2 — Establish which tree's file is genuinely newer
Do **not** delete anything yet. Get the evidence first:
```powershell
cd C:\Users\giris\alpha_edge\alpha_edge
git log --oneline -20 -- main.py monitoring/telegram_bot.py veto_agent.py
git log --oneline 800c158..HEAD --stat | Select-Object -First 80
```
That last command shows exactly what the 4 commits the nested tree is ahead by actually changed. It tells you whether nested's missing `__main__` guards were deliberate refactors or corruption — which is the question that decides everything.

### Step 3 — Report back
Once you have that output, I can build a precise file-by-file merge plan with the evidence rather than guessing.

---

## 5. What I did and did not do

**Did:** read-only inspection; wrote two markdown reports; ran `git status` / `git log` / `git diff` (read-only, but `git status` in the nested repo created the lock file); created `_wtest` as a write test.

**Did not:** modify, move, or delete any source file, config file, state file, or git object in either tree. No commits, no pushes, no `.gitignore` edits.

**Still outstanding from Phase 0:** token revocation (yours to do), backups (yours to do). Phase 1 is entirely unstarted and should stay that way until Step 2's output is reviewed.
