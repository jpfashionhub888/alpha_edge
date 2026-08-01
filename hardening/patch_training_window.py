#!/usr/bin/env python3
"""
hardening/patch_training_window.py
Fix ML overfit by doubling the training data window in alpaca_live.py.

Root cause (discovered after ML regularization patch):
  alpaca_live.py line ~369:
      train = df.iloc[max(0, split-180):split]

  730 days of bars are fetched but training is HARDCODED to the last 180 rows
  (~9 months).  The other ~340 rows are discarded.  Result: ~144 training
  samples after val split → all models overfit even with reduced complexity.

Fix:
  Change window from 180 → 365 rows (~18 months).

  With 365 training samples on 20 features: p/n ≈ 0.055 (well below the
  safe < 0.15 threshold).  Expected train_AUC: 0.65-0.80, gap < 0.30.

Run from /root/alpha_edge:
    python3 hardening/patch_training_window.py
    find /root/alpha_edge -name "*.pyc" -delete
    systemctl restart alpaca.service
    journalctl -u alpaca.service -f | grep -E 'Overfit check OK|OVERFIT HARD|train_AUC'
"""

import ast
import os
import shutil
import sys
from pathlib import Path

os.chdir(Path(__file__).parent.parent)

TARGET = Path('alpaca_live.py')
BACKUP = Path('alpaca_live.py.pre_training_window')


def abort(msg):
    print(f'\nERROR: {msg}')
    sys.exit(1)


def check_syntax(src, label=''):
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f'Syntax error {label}: {e}')
        return False


# ── PATCH: extend training window from 180 → 365 rows ────────────────────────
# Before:
#   split = len(df) - 30
#   train = df.iloc[max(0, split-180):split]
#
# After:
#   split = len(df) - 30
#   train = df.iloc[max(0, split-365):split]
#
# The lookback_days=730 fetch gives ~520 clean bars per symbol.
# Using 365 of them for training gives ~300 training samples after val split.
# The most recent 30 bars remain held out (for prediction and val alignment).

OLD_WINDOW = 'train      = df.iloc[max(0, split-180):split]'
NEW_WINDOW = 'train      = df.iloc[max(0, split-365):split]'

# Also update the guard threshold: if < 100 samples, use all available.
# With a 365-row window, this guard now only triggers for very new symbols
# (< 100 trading days of history) — keep it as-is, no change needed.


def patch_window(src):
    # Allow alternate whitespace in case file was reformatted
    alt_old = 'train = df.iloc[max(0, split-180):split]'
    alt_new = 'train = df.iloc[max(0, split-365):split]'

    if 'split-365' in src:
        print('  [1] Training window already 365 — skipping')
        return src

    if OLD_WINDOW in src:
        src = src.replace(OLD_WINDOW, NEW_WINDOW, 1)
        print('  [1] Training window: 180 → 365 rows (~9 months → ~18 months)')
        return src

    if alt_old in src:
        src = src.replace(alt_old, alt_new, 1)
        print('  [1] Training window: 180 → 365 rows (~9 months → ~18 months)')
        return src

    # Fallback: search line-by-line
    lines = src.splitlines(keepends=True)
    patched_lines = []
    found = False
    for line in lines:
        if 'split-180' in line and 'train' in line and 'iloc' in line:
            line = line.replace('split-180', 'split-365')
            found = True
            print(f'  [1] Patched line: {line.rstrip()}')
        patched_lines.append(line)

    if not found:
        abort(
            'Cannot find "split-180" training window line in alpaca_live.py.\n'
            '  Expected: train = df.iloc[max(0, split-180):split]\n'
            '  Check alpaca_live.py around line 369 manually.'
        )

    return ''.join(patched_lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if not TARGET.exists():
        abort(f'{TARGET} not found — run from /root/alpha_edge')

    print('\nAlphaEdge — training window expansion patch')
    print('=' * 50)
    print('Root cause: df.iloc[max(0, split-180):split] discards 340 of 520 bars.')
    print('Fix: expand to split-365 to use ~300 training samples instead of ~144.')
    print()

    src = TARGET.read_text(encoding='utf-8')

    if not check_syntax(src, 'BEFORE patch'):
        abort('alpaca_live.py has a syntax error — fix it first')

    shutil.copy2(TARGET, BACKUP)
    print(f'Backup saved → {BACKUP}\n')

    src = patch_window(src)

    if not check_syntax(src, 'AFTER patch'):
        print(f'Rolling back to {BACKUP}')
        shutil.copy2(BACKUP, TARGET)
        abort('Patch introduced syntax error — rolled back')

    TARGET.write_text(src, encoding='utf-8')

    # Confirm the change landed
    patched = TARGET.read_text(encoding='utf-8')
    if 'split-365' not in patched:
        abort('Write verification failed — split-365 not found in file after write')

    print(f'\n✓ Patch applied successfully.')
    print(f'  Before: df.iloc[max(0, split-180):split]  → ~144 training rows')
    print(f'  After:  df.iloc[max(0, split-365):split]  → ~300 training rows')
    print(f'  p/n ratio: 20/144 ≈ 0.14 → 20/300 ≈ 0.07')
    print()
    print('Next steps:')
    print('  find /root/alpha_edge -name "*.pyc" -delete')
    print('  systemctl restart alpaca.service')
    print('  journalctl -u alpaca.service -f | grep -E '
          '"Overfit check OK|OVERFIT HARD|train_AUC"')
    print()
    print('Expected: most symbols should now pass with gap < 0.30.')
    print('If train_AUC is still > 0.90 for most symbols, delete the model cache:')
    print('  rm -f /root/alpha_edge/models/cache/*.pkl')


if __name__ == '__main__':
    main()
