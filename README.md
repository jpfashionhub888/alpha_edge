# AlphaEdge Fix Package

This zip contains patched files and tests for the alpha_edge repo.

## Start here

1. **Read `AUDIT.md` first.** It explains every change. ~10 minute read.
2. **`scripts/rotate_keys.md`** is the only thing that needs to happen
   in production right now — your Alpaca API keys are in the public
   repo's git history and must be rotated.
3. **Apply the patches** by copying files from this zip over the
   corresponding paths in your repo:

```
   alpaca_edge_fixes/
   ├── execution/alpaca_broker.py      → alpha_edge/execution/alpaca_broker.py
   ├── execution/paper_trader.py       → alpha_edge/execution/paper_trader.py
   ├── execution/webhook_server.py     → alpha_edge/execution/webhook_server.py
   ├── monitoring/telegram_bot.py      → alpha_edge/monitoring/telegram_bot.py
   ├── scanner.py                      → alpha_edge/scanner.py
   ├── model_cache.py                  → alpha_edge/model_cache.py
   ├── live.py                         → alpha_edge/live.py
   └── tests/                          → alpha_edge/tests/         (new dir)
       ├── conftest.py
       ├── test_paper_trader.py
       └── test_signal_logic.py
```

4. **Run the tests** to verify the patches:

```bash
   cd alpha_edge
   pip install pytest --break-system-packages
   pytest tests/ -v
```

Expected: 29/29 passing (15 paper_trader + 14 signal_logic).

5. **Compare paper-trading behavior** before going to production:

   For 1 week, run both the old and new code side-by-side (different
   log files). They should produce equivalent or strictly-better
   signals. The new code will:
   - Skip slightly more BUY signals (more conservative sentiment)
   - Have fewer ATR fallback false-positives (raised from 2% to 3%)
   - Correctly count P&L on partial exits (this was wrong before)
   - Refuse to write state if accounting drifts

## Files NOT in this zip (keep your originals)

- `main.py` — already in good shape, no fixes needed
- `risk/manager.py` — used only by backtest, see AUDIT.md L5
- `models/*.py` — not part of this audit
- `data/feature_engine.py` — already clean (audited it, found no bugs)
- All other files at root and in subdirs not listed above

## Verification of test pass rate

The two test files in this package were both run against the patched
code during creation:

```
tests/test_paper_trader.py:    15 passed, 0 failed
tests/test_signal_logic.py:    14 passed, 0 failed
                              ────────────────────
                          TOTAL: 29 passed
```

The `test_paper_trader.py::test_reconcile_after_partial_exit` test
actually CAUGHT the H5 partial-exit P&L bug during patching. That's
the value of having invariant tests — it's already paid for itself.

## If you need to revert

Every patched file is a drop-in replacement for the original. To
revert any single file, just restore the original from git:

```bash
   git checkout HEAD -- execution/paper_trader.py
```

The new tests in `tests/` are additions, not replacements — leave
them in place even if you revert other changes. They're useful for
catching regressions in future edits.

## What's still on your plate (not fixed here)

See AUDIT.md "What this audit didn't cover" section:

- Statistical significance of your paper-trading P&L
- Whether LSTM helps the ensemble or hurts it
- Whether each of the 9 filter layers earns its place
- Backtest validity beyond a cursory review
- Whether your reported Sharpe matches walk-forward Sharpe
