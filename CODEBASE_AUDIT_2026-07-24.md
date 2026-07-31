# AlphaEdge — Full Codebase Audit, Performance Check & Remediation Roadmap

**Date:** 2026-07-24
**Scope:** Complete tree at `C:\Users\giris\alpha_edge` (140 tracked .py files, ~37.6k LOC) plus the nested clone at `alpha_edge\alpha_edge` (150 .py files)
**Method:** OODA — structural inventory → static compilation → AST-level defect scan → security scan → state/staleness audit → trading-logic review
**Status:** Findings only. **No code has been changed.**

---

## Executive summary — the headline

You do not have one codebase. You have **two**, and the one you are looking at in this folder is the broken, abandoned one.

| | Root (`alpha_edge\`) | Nested (`alpha_edge\alpha_edge\`) |
|---|---|---|
| HEAD commit | `800c158` (Jul 12) | `84bda84` (Jul 19) — **4 commits ahead** |
| Files that compile | **136 / 140** — 3 syntax errors + 1 NameError | **150 / 150 clean** |
| Working tree | 28 uncommitted modified files, 4 deleted | **Clean** |
| Newest state/log files | Jul 12 | Jul 19 |
| Visible to git? | Yes | **No — `.gitignore` line 11 is `alpha_edge/`** |
| Python files differing between the two | — | **82 differ, 11 exist only here** |

The nested copy is newer, cleaner, fully compiles, and has a clean working tree. It is also **completely invisible to git and to any tooling that respects `.gitignore`** — including most editors' search, most linters, and every audit that has been run on this project before this one.

Everything below is triaged with this in mind. **Nothing should be fixed until you confirm which tree is canonical**, because fixing the wrong one is worse than fixing neither.

---

## P0 — Act today

### P0-1. Live GitHub Personal Access Token committed in plaintext
**Severity:** Critical — credential compromise
**Location:** root `.git/config`, `remote.origin.url`

```
https://ghp_[REDACTED—token revoked]@github.com/jpfashionhub888/alpha_edge.git
```

A classic-format GitHub PAT is embedded in the remote URL. Classic PATs are typically broad-scope (`repo`, often `workflow`, sometimes `delete_repo`). Anyone with read access to this machine, any backup, any screen share, or any process that dumps `.git/config` has full write access to the repository.

Note the *nested* clone uses a clean tokenless remote — so this is specific to the root tree.

**Action:** Revoke at github.com/settings/tokens **now**, before any other work. Then re-auth via `gh auth login` or a credential helper. Assume the token is burned; rotating is not optional even if you believe the machine is private.

**Also check:** whether this token or the repo history contains any broker API keys. `config/secrets.env` is correctly gitignored and not tracked — that part is right.

---

### P0-2. Decide which tree is canonical — then delete the other
**Severity:** Critical — this is the root cause of most other findings

Two divergent clones of the same repo, one nested inside the other, 82 differing files. The nested one is gitignored, so:

- Every previous audit (`AUDIT_REPORT.md`, `deep_audit.py`) scanned only the root — i.e. the *stale* copy.
- Edits made in one tree are silently invisible in the other.
- `git status` in the root will never tell you the nested tree has uncommitted work.
- The 11 nested-only files (`backtesting/diagnose_ic*.py`, `run_momentum.py`, `signals/library/*.py`) exist nowhere in git history reachable from the root.

This structure will keep manufacturing "mystery" bugs indefinitely. It has to go.

---

## P1 — Blocks the system from running at all

### P1-1. Root tree does not compile — 4 files, all with the same corruption signature
**Severity:** Critical (root tree only; nested tree is clean)

Four files in the root working tree have a **fragment appended after the true end of file**:

| File | Appended garbage | Failure |
|---|---|---|
| `main.py:837` | `daily_scan()` | `NameError` on every CLI run |
| `veto_agent.py:204` | `']:.0%}")` | `SyntaxError: unterminated string literal` |
| `alpaca_live.py:928` | duplicate of the final 10-line block | `SyntaxError: unterminated string literal` |
| `models/regime_detector.py:98` | ` df` | `IndentationError` |

These are not four unrelated bugs. Each is a **partial line of the file's own content re-appended at EOF** — the signature of a patch/sed/edit tool that failed to seek correctly, almost certainly because of the line-ending problem in P1-3. The same signature produced the `options_analyzer.py` and `alpaca_broker.py` failures documented in your existing `AUDIT_REPORT.md`, meaning **this has now happened at least six times and the underlying cause has never been addressed.**

### P1-2. Root `HEAD` is *also* broken — differently
`data/feature_engine.py` at commit `800c158` ends mid-comment:

```python
df['target'] = (df['future_return'] > 0).astype(int)

# CRITICAL:          ← file ends here, no newline
```

`_add_target_variable()` therefore **never returns** — it returns `None`, and `future_return` is never dropped, meaning if it did return it would leak the target into the feature matrix as lookahead bias.

The uncommitted working-tree version fixes this correctly. So:

> **The root tree is broken at HEAD *and* broken in the working tree — in different files.** There is no commit in the root tree from which the system runs. The only runnable state is the nested clone.

### P1-3. Line-ending chaos is the mechanism behind the recurring corruption
**Severity:** High — root cause, not a symptom

- **67** `.py` files are pure CRLF
- **9** `.py` files have **mixed CRLF and LF within the same file** — including `model_cache.py` (447 CRLF / 8 LF), `execution/alpaca_broker.py` (340/82), `risk/position_sizer.py` (322/6), `performance_analytics.py` (302/54)
- `.gitattributes` exists but git still reports "CRLF will be replaced by LF" on ~100 files on every `git diff`

Mixed line endings inside a single file break byte-offset-based editing, confuse patch application, and make diffs unreadable. Every one of the six historical truncation incidents is consistent with this. Normalising line endings is not cosmetic housekeeping here — it is the fix for the bug class.

---

## P2 — Runtime crashes and silent failures (present in BOTH trees)

### P2-1. `NameError: logger` inside exception handlers
`monitoring/dashboard.py:323` and `:506` reference `logger`, which is never defined in that module. Both references sit **inside `except` blocks** — so the module works fine until something goes wrong, at which point the error handler itself raises, converting a recoverable error into an unhandled crash of the dashboard process. Worst possible placement.

### P2-2. `NameError: all_returns`
`backtest/hyperopt.py:270` uses an undefined `all_returns`. Any hyperparameter optimisation run reaching that line dies.

### P2-3. `NameError: generate_deployment_f` (nested tree)
`deploy.py:161` calls a truncated identifier — same corruption signature as P1-1, and it is in the **deployment** path.

### P2-4. 133 exception handlers swallow errors with zero logging
AST scan of the nested tree found **133** `except` blocks whose entire body is `pass` / `continue` / `return` / a bare assignment, with **no logger call, no print, no re-raise**. (Root tree: 122.) Worst offenders:

```
 24  deep_audit.py
 12  scripts/live_readiness_check.py
  6  monitoring/walkforward_monitor.py
  5  monitoring/model_watchdog.py
  4  generate_dashboard.py
  3  execution/webhook_server.py, execution/bybit_client.py,
     models/meta_labeler.py, monitoring/dashboard.py, model_cache.py, ...
```

`live_readiness_check.py` with 12 silent handlers is especially bad: a readiness gate that silently swallows its own failures will report "ready" when it is not. Same for `model_watchdog.py` — a watchdog that cannot report is worse than no watchdog, because it manufactures false confidence.

**This is your "silent crash" category.** The system does not crash; it quietly does nothing and reports success.

### P2-5. Crash-recovery tests still collect zero tests
`tests/test_crash_recovery.py:49` — `class CrashRecoveryTest:`. Pytest only auto-collects classes prefixed `Test`. This was flagged in your own `AUDIT_REPORT.md` on 2026-07-12 as issue #7 and is **still unfixed in both trees 12 days later**. The file matches `test_*.py` so CI reports it as "collected", giving a green tick over a class that never executes. Crash recovery — the thing that protects live positions across a restart — has no test coverage in CI.

---

## P3 — Stale data

### P3-1. Nothing has run in 5 days
Newest state files in the nested (live) tree: **2026-07-19**. Today is **2026-07-24**.
Root tree is worse: newest is **2026-07-12** (12 days).

Any dashboard reading `logs/latest_signals.json` is displaying signals that are 5 days old with no visible staleness indicator. The dashboard has no "data as of" freshness gate that I could find.

### P3-2. Circuit-breaker baselines are stale and will misfire on restart
`logs/circuit_breaker.json`:

```json
{ "peak_value": 10000.0,
  "daily_start": "2026-07-12",  "daily_start_val": 10000.0,
  "weekly_start": "2026-07-06", "weekly_start_val": 9400.0 }
```

`daily_start` is 12 days old. On restart, daily drawdown is computed against a **12-day-old baseline**, not today's open. Depending on the sign of the intervening move this either (a) trips the breaker instantly on a portfolio that is fine, or (b) fails to trip on a genuinely bad day because the stale baseline is below current value. Both are dangerous; (b) is the one that costs money.

There is no staleness check on load. Recommend: if `daily_start != today`, re-baseline before evaluating.

### P3-3. 30 non-atomic JSON writes
You have a correct `atomic_json_write()` helper — defined once, in `main.py:63`. Meanwhile there are **30 raw `json.dump()` calls** across the codebase writing state files directly. A crash or kill mid-write leaves a truncated JSON file, and the next read raises `JSONDecodeError` — which, given P2-4, will likely be silently swallowed and treated as "no positions."

Given your `.gitignore` comments explicitly warn that stale state files cause "fake/broken data on the dashboard," this is a known-class risk with a half-applied fix.

---

## P4 — Performance

### P4-1. Watchlist grew 78% with no parallelisation change
Uncommitted change to `main.py::get_full_watchlist()` expands the list from ~40 to **71 symbols**. `main.py:104` still fetches via a **sequential** `yf.Ticker(symbol)` loop with `time.sleep(delay)` backoff. `scanner.py` correctly uses `ThreadPoolExecutor` for the earnings calendar — but `main.py`'s own fetch loop was never converted.

At 5 models per symbol (XGB + LGB + RF + CatBoost + LSTM), 71 symbols means **355 model fits per full scan**, fed by a serial network loop. Expect scan wall-time to roughly double vs the 40-symbol baseline. This change is uncommitted and untested against a real scan.

**Recommendation:** parallelise the `main.py:104` fetch with the same `ThreadPoolExecutor` pattern already proven in `scanner.py:426` before expanding the watchlist — not after.

### P4-2. Dead weight in the tree
~35 files that are diffs, one-off patches, or superseded scripts: `*.diff` (6 files), `hardening/patch_*.py` (6), `patch_dashboard.py`, `find_signal_logic.py`, `test_predictor.py`, `live.py` vs `live_trading.py` (duplicates), `run_backtest.py` vs `run_backtest_v2.py`, `backtest/` vs `backtesting/` (two parallel backtest frameworks). Plus an empty file literally named `python`.

Two backtest frameworks is the one that matters — it is not obvious which produces the numbers you trust.

### P4-3. Lint noise
~110 unused imports and ~40 f-strings with no placeholders. Individually trivial; collectively they are why real defects like `undefined name 'logger'` went unnoticed — the signal is buried.

---

## What is actually in good shape

Not everything is broken, and it is worth knowing where not to spend effort:

- **Webhook authentication** (`execution/webhook_server.py`) is properly done: HMAC-SHA256, `hmac.compare_digest()` for timing-attack resistance, refuses to import without a real secret rather than falling back to a default, signature check before rate-limit counter. This is better than most production code.
- **Secret hygiene in files** is correct — `config/secrets.env` and `.env` are gitignored and untracked. No hardcoded API keys, AWS keys, or private keys found anywhere in the tree. The P0-1 leak is in git config, not source.
- **Lookahead-bias guards** in `data/feature_engine.py` are thoughtful — `_add_target_variable` explicitly refuses to overwrite a pre-computed target (comment correctly notes that `shift(-5)` on test rows is lookahead), and `future_return` is dropped from features.
- **Model cache TTL logic** (`model_cache.py`) is sound: version check, feature-name check, and 7-day TTL, with an explicit note that a previous bypass of those checks caused problems.
- **Crypto circuit-breaker bypass** (issue #6 in your July audit) **is now fixed** — `main.py:705-709` applies the `can_trade` gate to crypto. Good.
- **Target-labelling bug** (issue #8) **is fixed** in the working tree — `df = df[df['future_return'].notna()]` now drops the mislabelled tail rows.

---

## Remediation roadmap

Ordered by dependency, not just severity. Each phase gates the next.

### Phase 0 — Containment (today, ~30 min, no code changes)
| # | Action | Why first |
|---|---|---|
| 0.1 | **Revoke the GitHub PAT** at github.com/settings/tokens | Every minute it lives is exposure |
| 0.2 | Re-auth git via `gh auth login` / credential manager | Restores push access safely |
| 0.3 | **Full backup of both trees** before anything else | Phase 1 deletes a tree |
| 0.4 | **You decide:** which tree is canonical? | Everything downstream depends on this |

### Phase 1 — Collapse to one codebase (half a day)
| # | Action |
|---|---|
| 1.1 | Confirm nested tree is canonical (recommended — it compiles, root does not) |
| 1.2 | Move nested tree to a sibling directory, outside the root repo |
| 1.3 | Commit the 11 nested-only files (`backtesting/diagnose_*`, `run_momentum.py`, `signals/library/*`) — they exist in no reachable git history |
| 1.4 | Push nested `84bda84` to `origin` so remote and local agree |
| 1.5 | Archive the root tree read-only; do not delete until Phase 3 passes |
| 1.6 | Remove `alpha_edge/` from `.gitignore` so this cannot recur |

**Exit criterion:** exactly one working tree, clean `git status`, `git log` matches `origin`, everything compiles.

### Phase 2 — Kill the corruption mechanism (half a day)
| # | Action |
|---|---|
| 2.1 | Normalise all line endings to LF (`git add --renormalize .`) |
| 2.2 | Fix `.gitattributes` to enforce `*.py text eol=lf` |
| 2.3 | Add a pre-commit hook that runs `python -m compileall` and rejects any commit containing a file that does not compile |
| 2.4 | Add `pyflakes` to the same hook, failing only on `undefined name` and syntax errors (not unused imports) |

**Exit criterion:** it becomes structurally impossible to commit a file that does not parse. This is what stops incident #7.

### Phase 3 — Fix the crashers (1 day)
| # | Action | Ref |
|---|---|---|
| 3.1 | Define `logger` in `monitoring/dashboard.py` | P2-1 |
| 3.2 | Fix `all_returns` in `backtest/hyperopt.py:270` | P2-2 |
| 3.3 | Fix `generate_deployment_f` in `deploy.py:161` | P2-3 |
| 3.4 | Rename `CrashRecoveryTest` → `TestCrashRecovery`; verify tests actually run and pass | P2-5 |
| 3.5 | Run the full pytest suite locally with real deps and record the true baseline | — |

**Exit criterion:** clean pyflakes on `undefined name`; documented pass/fail count from a real local test run, not a CI green tick.

### Phase 4 — Stop the silent failures (2–3 days)
| # | Action | Ref |
|---|---|---|
| 4.1 | Triage all 133 silent handlers into: legitimate (add a one-line comment), should-log (add `logger.warning`), should-raise (remove the handler) | P2-4 |
| 4.2 | Prioritise `live_readiness_check.py` (12) and `model_watchdog.py` (5) — these produce false confidence | P2-4 |
| 4.3 | Replace all 30 raw `json.dump()` state writes with `atomic_json_write()`; move that helper out of `main.py` into a shared util | P3-3 |
| 4.4 | Add staleness guards: circuit breaker re-baselines if `daily_start != today`; dashboard shows a visible warning if `latest_signals.json` is older than one trading day | P3-1, P3-2 |

**Exit criterion:** no state file can be written non-atomically; no error path exits without a log line; stale data is visible rather than silent.

### Phase 5 — Performance & cleanup (2 days)
| # | Action | Ref |
|---|---|---|
| 5.1 | Parallelise `main.py:104` fetch loop with `ThreadPoolExecutor` (mirror `scanner.py:426`) | P4-1 |
| 5.2 | Benchmark full scan at 40 vs 71 symbols; only then confirm the watchlist expansion | P4-1 |
| 5.3 | Decide `backtest/` vs `backtesting/` — delete the loser | P4-2 |
| 5.4 | Delete `*.diff`, `hardening/patch_*.py`, `patch_dashboard.py`, `find_signal_logic.py`, the empty `python` file; reconcile `live.py` / `live_trading.py` | P4-2 |
| 5.5 | Sweep unused imports and placeholder-less f-strings | P4-3 |

### Phase 6 — Verification (1 day)
| # | Action |
|---|---|
| 6.1 | Full test suite green, with crash-recovery tests actually executing |
| 6.2 | End-to-end paper-trading scan on the unified tree, timed |
| 6.3 | Deliberately corrupt a state file mid-write; confirm the system logs it and degrades safely rather than reporting zero positions |
| 6.4 | Restart with a stale `circuit_breaker.json`; confirm it re-baselines |
| 6.5 | Re-run this audit end-to-end and diff against this document |

---

## Findings index

| ID | Severity | Finding | Tree |
|---|---|---|---|
| P0-1 | Critical | GitHub PAT in plaintext in `.git/config` | Root |
| P0-2 | Critical | Two divergent codebases; newer one is gitignored | Both |
| P1-1 | Critical | 4 files with appended-fragment corruption; will not compile | Root WT |
| P1-2 | Critical | `feature_engine._add_target_variable` truncated, returns `None` | Root HEAD |
| P1-3 | High | 67 CRLF + 9 mixed-ending files — cause of the corruption class | Both |
| P2-1 | High | `NameError: logger` inside two `except` blocks in dashboard | Both |
| P2-2 | High | `NameError: all_returns` in hyperopt | Both |
| P2-3 | High | `NameError: generate_deployment_f` in deploy path | Nested |
| P2-4 | High | 133 error handlers swallow failures with no logging | Both |
| P2-5 | High | Crash-recovery tests collect zero tests; flagged Jul 12, still open | Both |
| P3-1 | High | No scan has run in 5 days; dashboard shows stale data silently | Both |
| P3-2 | High | Circuit-breaker daily baseline 12 days stale; misfires on restart | Both |
| P3-3 | Medium | 30 non-atomic JSON state writes vs 1 atomic helper | Both |
| P4-1 | Medium | Watchlist +78% (40→71) with serial fetch loop, uncommitted | Both |
| P4-2 | Medium | Two parallel backtest frameworks; ~35 dead files | Both |
| P4-3 | Low | ~110 unused imports, ~40 empty f-strings burying real signal | Both |

---

## Open questions for you

1. **Which tree is canonical?** Evidence points to the nested clone. Confirm before anything is touched.
2. **Is this system currently live** with real capital, or paper only? Changes the urgency of P3-2 substantially.
3. **`backtest/` or `backtesting/`** — which one produced the performance numbers you trust?
4. **Was the watchlist expansion to 71 symbols deliberate**, and has a full scan been run and timed since?

---

*No files were modified in the production of this audit.*
