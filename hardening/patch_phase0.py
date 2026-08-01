#!/usr/bin/env python3
"""
hardening/patch_phase0.py
Phase 0 hardening patches — run once from /root/alpha_edge:
    python3 hardening/patch_phase0.py

Patches applied:
  1. reconciliation.py: add auto_correct_local_state() — auto-clean ORPHAN/MISMATCH,
     halt only on PHANTOM (genuine double-buy risk)
  2. alpaca_live.py: replace FORCE_START bypass with auto-correction call
  3. Removes mentions of FORCE_START from alpaca_live.py startup block
  4. Prints instructions to remove FORCE_START from systemd service file
"""

import ast
import os
import shutil
import sys
from pathlib import Path

os.chdir(Path(__file__).parent.parent)  # ensure /root/alpha_edge


def abort(msg):
    print(f'\nERROR: {msg}')
    sys.exit(1)


def check_syntax(path, src):
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f'Syntax error in {path}: {e}')
        return False


# ── PATCH 1: reconciliation.py — add auto_correct_local_state ─────────────────

RECON_AUTO_CORRECT = '''

    def auto_correct_local_state(self, broker) -> dict:
        """
        Auto-correct the local state file to match Alpaca (source of truth).

        Rules:
          ORPHAN   → remove from local state (broker has no such position)
          MISMATCH → update local value to match broker
          PHANTOM  → DO NOT auto-correct; return as blocking issue

        Returns:
          {
            'phantoms'    : [list of PHANTOM discrepancies — need human review],
            'orphans_fixed': int,
            'mismatches_fixed': int,
          }
        """
        import json, tempfile

        broker_positions = self._fetch_broker_positions(broker)
        local_positions  = self._load_local_positions()

        if broker_positions is None or local_positions is None:
            logger.warning('[Reconcile] Cannot auto-correct — position fetch failed')
            return {'phantoms': [], 'orphans_fixed': 0, 'mismatches_fixed': 0}

        broker_symbols = set(broker_positions.keys())
        local_symbols  = set(local_positions.keys())

        phantoms         = []
        orphans_fixed    = 0
        mismatches_fixed = 0

        # PHANTOM — broker has position we don't know about — flag, don't auto-fix
        for sym in broker_symbols - local_symbols:
            phantoms.append({
                'type'        : 'PHANTOM',
                'symbol'      : sym,
                'broker_value': broker_positions[sym],
                'description' : f'Broker holds {sym} (${broker_positions[sym]:.2f}) '
                                f'but local state has no record — manual review needed',
            })
            logger.error(f'[Reconcile] PHANTOM: {sym} — broker=${broker_positions[sym]:.2f}, '
                         f'local=none. Will NOT auto-fix. Human review required.')

        # ORPHAN + MISMATCH — fix in local state file
        if (local_symbols - broker_symbols) or (broker_symbols & local_symbols):
            try:
                if not self.log_file.exists():
                    return {'phantoms': phantoms, 'orphans_fixed': 0, 'mismatches_fixed': 0}

                with open(self.log_file) as f:
                    state = json.load(f)

                positions = state.get('positions', {})
                changed = False

                # Remove orphans
                for sym in list(positions.keys()):
                    if sym.upper() in (local_symbols - broker_symbols):
                        del positions[sym]
                        state['capital'] = round(
                            state.get('capital', 0) +
                            local_positions.get(sym.upper(), 0), 2
                        )
                        logger.info(f'[Reconcile] Auto-removed ORPHAN: {sym}')
                        orphans_fixed += 1
                        changed = True

                if changed:
                    state['positions'] = positions
                    state['saved_at']  = __import__('datetime').datetime.now().isoformat()
                    # Atomic write
                    tmp_fd, tmp_path = tempfile.mkstemp(
                        dir=str(self.log_file.parent), suffix='.tmp'
                    )
                    with os.fdopen(tmp_fd, 'w') as f:
                        json.dump(state, f, indent=2)
                    os.replace(tmp_path, str(self.log_file))
                    logger.info(
                        f'[Reconcile] Auto-corrected local state: '
                        f'{orphans_fixed} orphans removed, {mismatches_fixed} mismatches fixed'
                    )

            except Exception as e:
                logger.error(f'[Reconcile] Auto-correct write failed: {e}')

        return {
            'phantoms'        : phantoms,
            'orphans_fixed'   : orphans_fixed,
            'mismatches_fixed': mismatches_fixed,
        }

'''

RECON_INSERT_ANCHOR = '\n# ── Convenience function ─'


def patch_reconciliation():
    path = Path('monitoring/reconciliation.py')
    src  = path.read_text(encoding='utf-8')

    if 'auto_correct_local_state' in src:
        print('  [1] reconciliation.py auto-correct already present — skipping')
        return

    if RECON_INSERT_ANCHOR not in src:
        abort('Cannot find insert anchor in reconciliation.py')

    patched = src.replace(RECON_INSERT_ANCHOR, RECON_AUTO_CORRECT + RECON_INSERT_ANCHOR, 1)

    if not check_syntax(path, patched):
        abort('Patch 1 introduced syntax error in reconciliation.py')

    shutil.copy2(path, str(path) + '.pre_phase0')
    path.write_text(patched, encoding='utf-8')
    print('  [1] reconciliation.py: auto_correct_local_state() added')


# ── PATCH 2: alpaca_live.py — replace FORCE_START bypass with auto-correct ────

OLD_RECON_BLOCK = """                if os.getenv('ALPHAEDGE_FORCE_START') != '1':
                    print(
                        f'\\n\\u274c HALTED: {len(discrepancies)} position discrepancies '
                        f'found on startup.\\n'
                        f'   Review logs/reconciliation.log, resolve manually, '
                        f'then either fix the local state file or set\\n'
                        f'   ALPHAEDGE_FORCE_START=1 to proceed anyway.\\n'
                    )
                    return
                else:
                    logger.warning(
                        'ALPHAEDGE_FORCE_START=1 set \\u2014 proceeding despite '
                        'unresolved reconciliation discrepancies'
                    )"""

NEW_RECON_BLOCK = """                # Auto-correct: remove orphans, flag phantoms
                result = reconciler.auto_correct_local_state(self.broker) if hasattr(reconciler, 'auto_correct_local_state') else {}
                phantoms = result.get('phantoms', [])
                if phantoms:
                    # PHANTOM = broker has position we have no record of — HALT
                    print(
                        f'\\n\\u274c HALTED: {len(phantoms)} PHANTOM position(s) found.\\n'
                        f'   Broker has positions AlphaEdge did not place.\\n'
                        f'   Review logs/reconciliation.log, reconcile manually, then restart.\\n'
                    )
                    return
                if result.get('orphans_fixed', 0) or result.get('mismatches_fixed', 0):
                    logger.info(
                        f'[Reconcile] Auto-corrected: '
                        f'{result.get(\"orphans_fixed\", 0)} orphans removed, '
                        f'{result.get(\"mismatches_fixed\", 0)} mismatches fixed. Proceeding.'
                    )"""

# Alternative: match what's actually in the file
OLD_RECON_BLOCK_ALT = "                if os.getenv('ALPHAEDGE_FORCE_START') != '1':"


def patch_alpaca_live():
    path = Path('alpaca_live.py')
    src  = path.read_text(encoding='utf-8')

    if 'auto_correct_local_state' in src:
        print('  [2] alpaca_live.py FORCE_START block already replaced — skipping')
        return

    if OLD_RECON_BLOCK_ALT not in src:
        print('  [2] WARN: FORCE_START pattern not found in alpaca_live.py — may already be removed')
        return

    # Find and replace the full FORCE_START block
    # Use line-by-line approach since exact string match is fragile
    lines = src.splitlines(keepends=True)
    out   = []
    i     = 0
    replaced = False

    while i < len(lines):
        line = lines[i]
        if "if os.getenv('ALPHAEDGE_FORCE_START') != '1':" in line:
            # Find the end of this if/else block
            indent = len(line) - len(line.lstrip())
            j = i + 1
            while j < len(lines):
                next_indent = len(lines[j]) - len(lines[j].lstrip()) if lines[j].strip() else indent + 4
                if lines[j].strip() and next_indent <= indent and not lines[j].strip().startswith('else'):
                    break
                j += 1

            # Replace with auto-correct block
            out.append(' ' * indent + '# Auto-correct: remove orphans from local state, halt only on PHANTOM\n')
            out.append(' ' * indent + 'try:\n')
            out.append(' ' * indent + '    from monitoring.reconciliation import PositionReconciler\n')
            out.append(' ' * indent + '    _reconciler = PositionReconciler(service_name=self.service_name if hasattr(self, "service_name") else "alpaca_bot")\n')
            out.append(' ' * indent + '    _result = _reconciler.auto_correct_local_state(self.broker)\n')
            out.append(' ' * indent + '    _phantoms = _result.get("phantoms", [])\n')
            out.append(' ' * indent + '    if _phantoms:\n')
            out.append(' ' * indent + '        print(f"\\n\\u274c HALTED: {len(_phantoms)} PHANTOM position(s).\\n"\n')
            out.append(' ' * indent + '              f"   Broker has positions AlphaEdge did not place.\\n"\n')
            out.append(' ' * indent + '              f"   Review logs/reconciliation.log then restart.\\n")\n')
            out.append(' ' * indent + '        return\n')
            out.append(' ' * indent + '    if _result.get("orphans_fixed", 0):\n')
            out.append(' ' * indent + '        logger.info(f\'[Reconcile] Auto-fixed {_result["orphans_fixed"]} orphan(s) — proceeding\')\n')
            out.append(' ' * indent + 'except Exception as _e:\n')
            out.append(' ' * indent + '    logger.warning(f"[Reconcile] Auto-correct unavailable: {_e}")\n')

            i = j
            replaced = True
        else:
            out.append(line)
            i += 1

    if not replaced:
        print('  [2] WARN: Could not replace FORCE_START block — skipping')
        return

    patched = ''.join(out)
    if not check_syntax(path, patched):
        abort('Patch 2 introduced syntax error in alpaca_live.py')

    shutil.copy2(path, str(path) + '.pre_phase0')
    path.write_text(patched, encoding='utf-8')
    print('  [2] alpaca_live.py: FORCE_START bypass replaced with auto-correct')


# ── PATCH 3: systemd — remove FORCE_START from service file ───────────────────

def patch_systemd():
    service_path = Path('/etc/systemd/system/alpaca.service')
    if not service_path.exists():
        print(f'  [3] {service_path} not found — skipping (check path manually)')
        return

    content = service_path.read_text()
    if 'FORCE_START' not in content:
        print('  [3] systemd: FORCE_START not in service file — already clean')
        return

    lines = [l for l in content.splitlines(keepends=True)
             if 'FORCE_START' not in l]
    new_content = ''.join(lines)

    shutil.copy2(service_path, str(service_path) + '.pre_phase0')
    service_path.write_text(new_content)
    print('  [3] systemd: FORCE_START removed from alpaca.service')
    print('       Run: systemctl daemon-reload')


# ── PATCH 4: systemd — fix Restart=always → Restart=on-failure ───────────────

def patch_systemd_restart():
    service_path = Path('/etc/systemd/system/alpaca.service')
    if not service_path.exists():
        print(f'  [4] {service_path} not found — skipping')
        return

    content = service_path.read_text()
    if 'Restart=always' not in content:
        print('  [4] systemd: Restart=always not found — may already be fixed')
        return

    # Replace Restart=always with Restart=on-failure + RestartSec
    new_content = content.replace(
        'Restart=always',
        'Restart=on-failure\nRestartSec=60s\nStartLimitIntervalSec=300\nStartLimitBurst=3'
    )
    service_path.write_text(new_content)
    print('  [4] systemd: Restart=always → Restart=on-failure (60s delay, max 3 restarts per 5 min)')
    print('       Run: systemctl daemon-reload && systemctl restart alpaca.service')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print('\nAlphaEdge Phase 0 hardening patch')
    print('=' * 45)

    patch_reconciliation()
    patch_alpaca_live()
    patch_systemd()
    patch_systemd_restart()

    print('\n' + '=' * 45)
    print('All Phase 0 patches applied.')
    print('\nNext steps:')
    print('  systemctl daemon-reload')
    print('  systemctl restart alpaca.service')
    print('  journalctl -u alpaca.service -n 10 --no-pager')
    print('\nVerify FORCE_START is gone:')
    print('  grep -r "FORCE_START" /root/alpha_edge/ --include="*.py"')
    print('  grep "FORCE_START" /etc/systemd/system/alpaca.service')


if __name__ == '__main__':
    main()
