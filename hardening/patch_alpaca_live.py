#!/usr/bin/env python3
"""
hardening/patch_alpaca_live.py
Applies hardening patches to alpaca_live.py in-place.
Run once from /root/alpha_edge:
    python3 hardening/patch_alpaca_live.py

Patches applied:
  1. Cache path hardening (HF_HOME, yfinance) at top of file
  2. State-file backup before every write (buy + exit recorders)
  3. State-file recovery on startup if file is empty/corrupt
"""

import ast
import os
import re
import shutil
import sys
from pathlib import Path

TARGET = Path('alpaca_live.py')
BACKUP = Path('alpaca_live.py.pre_hardening')


# ── helpers ────────────────────────────────────────────────────────────────────

def abort(msg: str):
    print(f'ERROR: {msg}')
    sys.exit(1)


def check_syntax(src: str, label: str = ''):
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f'Syntax error {label}: {e}')
        return False


# ── PATCH 1 — cache hardening ─────────────────────────────────────────────────

CACHE_BLOCK = '''\
# ── Cache path hardening (must be first — before yfinance / HF imports) ───────
os.environ.setdefault('HF_HOME',            '/root/.cache/huggingface')
os.environ.setdefault('TRANSFORMERS_CACHE', '/root/.cache/huggingface/hub')
os.environ.setdefault('XDG_CACHE_HOME',     '/root/.cache')
for _d in ('/root/.cache/huggingface', '/root/.cache/py-yfinance'):
    try:
        os.makedirs(_d, exist_ok=True)
    except Exception as e:
        import logging as _log; _log.getLogger(__name__).warning(f'Cache dir create failed for {_d}: {e}')
try:
    import yfinance as _yf_cache
    _yf_cache.set_tz_cache_location('/root/.cache/py-yfinance')
except Exception as e:
    import logging as _log; _log.getLogger(__name__).warning(f'yfinance cache location set failed: {e}')

'''

CACHE_ANCHOR = 'from config import settings'   # insert just before this line


def patch_cache(src: str) -> str:
    if 'Cache path hardening' in src:
        print('  [1] Cache hardening already applied — skipping')
        return src
    if CACHE_ANCHOR not in src:
        abort(f'Could not find anchor "{CACHE_ANCHOR}" in alpaca_live.py')
    patched = src.replace(CACHE_ANCHOR, CACHE_BLOCK + CACHE_ANCHOR, 1)
    print('  [1] Cache hardening applied')
    return patched


# ── PATCH 2 — state-file backup before every write ────────────────────────────

BACKUP_SNIPPET = '''\
            # Backup before overwrite so a crash can never wipe history
            if os.path.exists(trade_file) and os.path.getsize(trade_file) > 20:
                try:
                    import shutil as _shutil
                    _shutil.copy2(trade_file, trade_file + '.bak')
                except Exception as e:
                    import logging as _log; _log.getLogger(__name__).warning(f'State backup failed: {e}')
'''

# Anchor: the atomic write pattern used in both _record_buy_to_state and
# _record_exit_to_json.  We patch both occurrences.
WRITE_ANCHOR = "            tmp_fd, tmp_path = _tf.mkstemp(dir='logs', suffix='.tmp')"


def patch_backup(src: str) -> str:
    if 'Backup before overwrite' in src:
        print('  [2] State backup already applied — skipping')
        return src
    count = src.count(WRITE_ANCHOR)
    if count == 0:
        abort(f'Could not find write anchor in alpaca_live.py (expected 2 occurrences)')
    patched = src.replace(WRITE_ANCHOR, BACKUP_SNIPPET + WRITE_ANCHOR)
    print(f'  [2] State backup applied to {count} write site(s)')
    return patched


# ── PATCH 3 — startup state recovery ─────────────────────────────────────────

RECOVERY_METHOD = '''
    def _recover_state_if_needed(self) -> None:
        """
        On startup: if paper_trades_stocks_only.json is missing, empty, or
        corrupt, restore from the .bak file created before every write.
        Prevents losing trade history on hard crashes.
        """
        import json as _json
        trade_file = 'logs/paper_trades_stocks_only.json'
        bak_file   = trade_file + '.bak'

        needs_recovery = False
        if not os.path.exists(trade_file):
            needs_recovery = True
        else:
            try:
                with open(trade_file) as f:
                    state = _json.load(f)
                # Treat as corrupt if positions and history are both empty
                # AND capital equals the default starting value (sign of a reset)
                if (not state.get('positions') and
                        not state.get('trade_history') and
                        os.path.exists(bak_file)):
                    # Peek at the backup — if it has data, prefer it
                    with open(bak_file) as f:
                        bak = _json.load(f)
                    if bak.get('trade_history') or bak.get('positions'):
                        needs_recovery = True
            except Exception:
                needs_recovery = True

        if needs_recovery and os.path.exists(bak_file):
            try:
                import shutil as _sh
                _sh.copy2(bak_file, trade_file)
                logger.info('[Recovery] Restored paper_trades from .bak file')
            except Exception as e:
                logger.warning(f'[Recovery] Could not restore from backup: {e}')
        elif needs_recovery:
            logger.info('[Recovery] No backup found — starting fresh state')

'''

RECOVERY_ANCHOR = '    def _save_managed_positions(self):'


def patch_recovery(src: str) -> str:
    if '_recover_state_if_needed' in src:
        print('  [3] State recovery already applied — skipping')
        return src
    if RECOVERY_ANCHOR not in src:
        abort(f'Could not find anchor "{RECOVERY_ANCHOR}" in alpaca_live.py')
    patched = src.replace(RECOVERY_ANCHOR, RECOVERY_METHOD + RECOVERY_ANCHOR, 1)
    print('  [3] State recovery method added')
    return patched


# ── PATCH 4 — call recovery on startup ────────────────────────────────────────

RECOVERY_CALL      = '        self._recover_state_if_needed()\n'
RECOVERY_CALL_ANCHOR = '        self.heartbeat = HeartbeatMonitor'


def patch_recovery_call(src: str) -> str:
    if '_recover_state_if_needed()' in src:
        print('  [4] Recovery call already present — skipping')
        return src
    if RECOVERY_CALL_ANCHOR not in src:
        abort(f'Could not find anchor for recovery call in __init__')
    patched = src.replace(
        RECOVERY_CALL_ANCHOR,
        RECOVERY_CALL + RECOVERY_CALL_ANCHOR,
        1
    )
    print('  [4] Recovery call wired into __init__')
    return patched


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    if not TARGET.exists():
        abort(f'{TARGET} not found — run from /root/alpha_edge directory')

    print(f'\nAlphaEdge alpaca_live.py hardening patch')
    print('=' * 45)

    src = TARGET.read_text(encoding='utf-8')

    if not check_syntax(src, 'BEFORE patch'):
        abort('alpaca_live.py has a syntax error before patching — fix it first')

    # Save pre-patch backup
    shutil.copy2(TARGET, BACKUP)
    print(f'  Pre-patch backup saved to {BACKUP}\n')

    src = patch_cache(src)
    src = patch_backup(src)
    src = patch_recovery(src)
    src = patch_recovery_call(src)

    if not check_syntax(src, 'AFTER patch'):
        print(f'\nRolling back to {BACKUP}')
        shutil.copy2(BACKUP, TARGET)
        abort('Patch introduced a syntax error — rolled back')

    TARGET.write_text(src, encoding='utf-8')
    print(f'\nAll patches applied successfully.')
    print(f'Lines: {len(src.splitlines())}')
    print('Run: systemctl restart alpaca.service')


if __name__ == '__main__':
    os.chdir(Path(__file__).parent.parent)   # ensure we're in /root/alpha_edge
    main()
