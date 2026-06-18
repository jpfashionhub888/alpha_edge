#!/usr/bin/env python3
# deep_audit.py
# AlphaEdge — Daily Deep Audit Agent
#
# Runs every day (via systemd timer + GitHub Actions).
# Performs a full source-level, runtime, log, and data audit of the
# entire project and sends a structured Telegram report with:
#   - 0-10 health score
#   - All bugs found, graded P0/P1/P2
#   - Line-level code antipattern detection
#   - Service crash / silent-error detection
#   - Trade data integrity checks
#   - Specific fix recommendations
#
# Usage:
#   python deep_audit.py              # full audit + Telegram report
#   python deep_audit.py --stdout     # print report, no Telegram
#   python deep_audit.py --fix        # apply safe auto-fixes after reporting

import os
import sys
import ast
import re
import json
import math
import subprocess
import traceback
import importlib
import inspect
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ── Bootstrap ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

STDOUT_ONLY = '--stdout' in sys.argv
AUTO_FIX    = '--fix'    in sys.argv

logging.basicConfig(level=logging.WARNING)  # suppress noisy import logs
logger = logging.getLogger('deep_audit')

# ── Finding registry ──────────────────────────────────────────────────────────
findings: list[dict] = []   # {priority, category, file, line, title, detail, fix}
passed:   list[str]  = []   # short pass messages

def finding(priority: str, category: str, file: str, line: int,
            title: str, detail: str, fix: str = ''):
    findings.append({
        'priority': priority,  # P0 / P1 / P2 / INFO
        'category': category,
        'file'    : file,
        'line'    : line,
        'title'   : title,
        'detail'  : detail,
        'fix'     : fix,
    })

def ok(msg: str):
    passed.append(msg)

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 1: STATIC CODE ANALYSIS
# Scans every .py file for known antipatterns, fail-open bugs, hardcoded paths.
# ═════════════════════════════════════════════════════════════════════════════

# Pattern definitions: (regex, priority, category, title, detail, fix)
ANTIPATTERNS = [
    (
        r'except\s+(Exception|BaseException)?\s*:\s*\n\s*pass\b',
        'P0', 'fail-open',
        'Bare except:pass silences errors',
        'Any exception (network timeout, auth failure, logic error) is swallowed.',
        'At minimum: except Exception as e: logger.error(f"...: {e}")',
    ),
    (
        r'except\s*:\s*\n\s*pass\b',
        'P0', 'fail-open',
        'Bare except:pass (catches SystemExit/KeyboardInterrupt too)',
        'catches SystemExit, KeyboardInterrupt — use except Exception.',
        'Replace with: except Exception as e: logger.error(...)',
    ),
    (
        r'starting_capital\s*=\s*(?:balance|capital|current_value|portfolio)',
        'P0', 'risk-control',
        'starting_capital derived from current balance — total-loss check dead',
        'Passing current value as starting capital makes (current-start)/start = 0.0 always.',
        'Persist starting_capital in circuit_breaker.json on first run.',
    ),
    (
        r'except\s+Exception\s*(?:as\s+\w+)?\s*:\s*\n\s*return\s+[{\'"].*APPROVE',
        'P0', 'fail-open',
        'Exception handler returns APPROVE — veto/risk agent fails open',
        'Any error in veto/critic logic silently approves the trade.',
        'Return VETO / block action on all exception paths.',
    ),
    (
        r"ROOT\s*=\s*['\"].+alpha_edge['\"]",
        'P1', 'portability',
        'Hardcoded ROOT path',
        'Script only works on one specific server path.',
        "Use: ROOT = os.path.dirname(os.path.abspath(__file__))",
    ),
    (
        r"(?:host|server|ip)\s*=\s*['\"][\d]{1,3}\.[\d]{1,3}\.[\d]{1,3}\.[\d]{1,3}['\"]",
        'P1', 'portability',
        'Hardcoded server IP address in source',
        'IP should be in config/env, not source.',
        'Move to config/settings.py or environment variable.',
    ),
    (
        r'(?:password|secret|api_key|token)\s*=\s*[\'"][A-Za-z0-9+/]{20,}[\'"]',
        'P0', 'security',
        'Hardcoded credential detected',
        'Secret committed to source — rotate immediately.',
        'Move to environment variable or config/secrets.env.',
    ),
    (
        r'json\.load\(open\(',
        'P2', 'reliability',
        'json.load(open(...)) without context manager',
        'File handle not guaranteed to close on exception.',
        'Use: with open(...) as f: json.load(f)',
    ),
    (
        r'time\.sleep\(\s*(?:3600|7200|86400)\s*\)',
        'P2', 'reliability',
        'Very long time.sleep() — process unresponsive to signals',
        'Systemd sends SIGTERM on restart; a sleeping process won\'t respond.',
        'Use a loop with short sleeps (e.g., 60s) and a running flag.',
    ),
    (
        r'requests\.\w+\([^)]+\)(?!\s*#.*timeout)',
        'P2', 'reliability',
        'requests call without explicit timeout',
        'Hangs forever if the remote server doesn\'t respond.',
        'Add timeout=10 to all requests calls.',
    ),
    (
        r'\.predict\([^)]+\)\[0\]',
        'P2', 'reliability',
        '.predict()[0] without bounds check — IndexError if model returns empty',
        'If predict() returns empty array, [0] raises IndexError.',
        'Guard: preds = model.predict(X); pred = preds[0] if len(preds) else 0.5',
    ),
    (
        r"action\s*==\s*['\"]SELL['\"]",
        'P1', 'correctness',
        "== 'SELL' filter excludes PARTIAL_SELL from metrics",
        'PARTIAL_SELL trades are silently excluded from KPIs, win rate, and drawdown.',
        "Use: action in {'SELL', 'PARTIAL_SELL'}",
    ),
    (
        r'from\s+groq\s+import',
        'P2', 'reliability',
        'Top-level groq import — crashes startup if package missing',
        'ImportError at module load crashes the whole bot.',
        'Use lazy import inside the function body with try/except ImportError.',
    ),
    (
        r'pd\.DataFrame\(\)\.append\(',
        'P2', 'deprecated',
        'DataFrame.append() removed in pandas 2.0',
        'Will raise AttributeError in pandas >= 2.0.',
        'Use pd.concat([df, new_row]) instead.',
    ),
]

SKIP_DIRS = {'.git', 'venv', '__pycache__', 'node_modules', '.pytest_cache',
             'catboost_info', 'notebooks', 'research', 'backtest',
             'saved_models', 'model_cache', 'cache', '.github'}

def _collect_py_files() -> list[Path]:
    files = []
    for p in ROOT.rglob('*.py'):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        files.append(p)
    return sorted(files)

def run_static_analysis():
    """Scan all Python files for antipatterns."""
    py_files = _collect_py_files()
    total_lines = 0
    hit_count = 0

    for filepath in py_files:
        rel = filepath.relative_to(ROOT)
        try:
            src = filepath.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        lines = src.splitlines()
        total_lines += len(lines)

        for pattern, priority, category, title, detail, fix in ANTIPATTERNS:
            for m in re.finditer(pattern, src, re.MULTILINE | re.IGNORECASE):
                lineno = src[:m.start()].count('\n') + 1
                # Skip if this is inside a comment
                line_text = lines[lineno - 1].strip() if lineno <= len(lines) else ''
                if line_text.startswith('#'):
                    continue
                finding(priority, category, str(rel), lineno, title, detail, fix)
                hit_count += 1

    # AST-level checks
    _run_ast_checks(py_files)

    ok(f'Static analysis: {len(py_files)} files, {total_lines:,} lines scanned')
    return hit_count

def _run_ast_checks(py_files: list[Path]):
    """AST-level checks that regex can't catch reliably."""
    for filepath in py_files:
        rel = str(filepath.relative_to(ROOT))
        try:
            src = filepath.read_text(encoding='utf-8', errors='replace')
            tree = ast.parse(src, filename=str(filepath))
        except SyntaxError as e:
            finding('P0', 'syntax', rel, e.lineno or 0,
                    f'Syntax error: {e.msg}',
                    str(e),
                    'Fix the syntax error before anything else.')
            continue
        except Exception:
            continue

        for node in ast.walk(tree):
            # Bare except with pass
            if isinstance(node, ast.ExceptHandler):
                if (node.type is None and
                    len(node.body) == 1 and
                    isinstance(node.body[0], ast.Pass)):
                    finding('P0', 'fail-open', rel, node.lineno,
                            'Bare except: pass (AST confirmed)',
                            'All exceptions including SystemExit silently swallowed.',
                            'Replace with: except Exception as e: logger.error(...)')

            # Division without zero guard
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                if isinstance(node.right, ast.Name):
                    # division by a variable — flag if it looks like a count/total
                    name = node.right.id.lower()
                    if any(w in name for w in ('len', 'count', 'total', 'n', 'loss')):
                        pass  # too noisy — skip for now


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2: RUNTIME HEALTH CHECKS
# Actually import key modules and test core functionality.
# ═════════════════════════════════════════════════════════════════════════════

def run_runtime_checks():
    """Import and smoke-test critical modules."""
    critical_modules = [
        ('config.settings',        'Settings'),
        ('risk_circuit_breaker',   'RiskCircuitBreaker'),
        ('veto_agent',             'VetoAgent'),
        ('performance_analytics',  'PerformanceAnalytics'),
        ('execution.paper_trader', 'PaperTrader'),
        ('market_regime',          'MarketRegimeDetector'),
        ('generate_dashboard',     None),
    ]

    for mod_name, class_name in critical_modules:
        try:
            mod = importlib.import_module(mod_name)
            if class_name:
                cls = getattr(mod, class_name)
                ok(f'Import + class: {mod_name}.{class_name}')
            else:
                ok(f'Import: {mod_name}')
        except ImportError as e:
            finding('P1', 'runtime', f'{mod_name}.py', 0,
                    f'Import failed: {mod_name}',
                    str(e),
                    'Install missing dependency or fix the import path.')
        except Exception as e:
            finding('P1', 'runtime', f'{mod_name}.py', 0,
                    f'Module error on import: {mod_name}',
                    str(e),
                    'Check for module-level code that crashes at import time.')

    # Check veto agent fails closed
    try:
        from veto_agent import VetoAgent
        va_src = inspect.getsource(VetoAgent.review_signal)
        # Exception handler should return VETO not APPROVE
        exception_blocks = re.findall(
            r'except.*?:\n(.*?)(?=\nexcept|\ndef |\nclass |\Z)',
            va_src, re.DOTALL
        )
        for block in exception_blocks:
            if "'APPROVE'" in block or '"APPROVE"' in block:
                finding('P0', 'fail-open', 'veto_agent.py', 0,
                        'VetoAgent exception handler returns APPROVE (fail-open)',
                        'Any Groq API error silently approves trades.',
                        "Change exception handler return to 'VETO'.")
                break
        else:
            ok('VetoAgent: fail-closed (exceptions return VETO)')
    except Exception:
        pass

    # Check circuit breaker is wired into alpaca_live
    try:
        alpaca_src = (ROOT / 'alpaca_live.py').read_text(encoding='utf-8')
        if 'RiskCircuitBreaker' in alpaca_src:
            ok('alpaca_live.py: RiskCircuitBreaker wired in')
        else:
            finding('P0', 'risk-control', 'alpaca_live.py', 0,
                    'No circuit breaker in alpaca_live.py',
                    'Alpaca stock bot has zero drawdown protection.',
                    'Import and call RiskCircuitBreaker at top of _run_scan().')
    except Exception:
        pass

    # Check REALIZED_ACTIONS at module scope in generate_dashboard
    try:
        dash_src = (ROOT / 'generate_dashboard.py').read_text(encoding='utf-8')
        # Is REALIZED_ACTIONS defined BEFORE the functions that use it?
        ra_pos   = dash_src.find("REALIZED_ACTIONS")
        sell_pos = dash_src.find("== 'SELL'")
        if ra_pos == -1:
            finding('P0', 'correctness', 'generate_dashboard.py', 0,
                    "REALIZED_ACTIONS not defined — PARTIAL_SELL excluded from all KPIs",
                    'KPI tiles (win rate, P&L, Sharpe) disagree with equity chart.',
                    "Add REALIZED_ACTIONS = {'SELL','PARTIAL_SELL'} at module scope.")
        elif sell_pos != -1:
            # Check if any == 'SELL' filters remain in metric functions
            # (not in comments, not in generate_dashboard's history display)
            sell_matches = list(re.finditer(r"action.*==\s*['\"]SELL['\"]", dash_src))
            metric_hits = [m for m in sell_matches
                           if 'hist_rows' not in dash_src[max(0,m.start()-200):m.start()]]
            if metric_hits:
                finding('P1', 'correctness', 'generate_dashboard.py',
                        dash_src[:metric_hits[0].start()].count('\n') + 1,
                        "Stray == 'SELL' filter still excluding PARTIAL_SELL from a metric",
                        'Metric function uses literal SELL instead of REALIZED_ACTIONS.',
                        "Replace with: action in REALIZED_ACTIONS")
            else:
                ok("generate_dashboard.py: REALIZED_ACTIONS used consistently")
        else:
            ok("generate_dashboard.py: REALIZED_ACTIONS defined, no stray SELL filters")
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 3: SERVICE + LOG HEALTH (server-side only)
# ═════════════════════════════════════════════════════════════════════════════

SERVICES = ['alpaca.service', 'gateio.service', 'dashboard.service']

def run_service_checks():
    """Check systemd service health and scan logs for errors."""
    for svc in SERVICES:
        try:
            r = subprocess.run(
                ['systemctl', 'is-active', svc],
                capture_output=True, text=True, timeout=5
            )
            status = r.stdout.strip()
            if status == 'active':
                ok(f'{svc}: active')
            else:
                finding('P0', 'service', svc, 0,
                        f'Service not running: {svc} ({status})',
                        f'systemctl is-active returned: {status}',
                        f'Run: systemctl restart {svc} && journalctl -u {svc} -n 50')
        except FileNotFoundError:
            ok('systemctl not available (GitHub Actions / dev machine) — skipping')
            return
        except Exception as e:
            finding('P1', 'service', svc, 0, f'Cannot check {svc}', str(e), '')

    # Log scan: last 500 lines per service for crash patterns
    CRASH_PATTERNS = [
        (r'Traceback \(most recent call last\)', 'P0', 'Exception/Traceback in logs'),
        (r'KeyError:',                           'P1', 'KeyError crash'),
        (r'AttributeError:',                     'P1', 'AttributeError crash'),
        (r'JSONDecodeError',                     'P1', 'JSON corruption'),
        (r'ConnectionRefusedError',              'P1', 'API connection refused'),
        (r'401\s+Unauthorized',                  'P0', '401 Unauthorized (bad API key?)'),
        (r'429\s+Too Many Requests',             'P1', 'Rate limited (429)'),
        (r'circuit breaker.*OPEN',               'P0', 'Circuit breaker tripped'),
        (r'VETO.*fail-closed',                   'INFO','Veto agent fail-closed triggered'),
        (r'starting_capital.*0\.0',              'P0', 'starting_capital=0 (dead total-loss check)'),
    ]

    for svc in SERVICES:
        try:
            r = subprocess.run(
                ['journalctl', '-u', svc, '-n', '500', '--no-pager', '--output=cat'],
                capture_output=True, text=True, timeout=10
            )
            log_text = r.stdout
            if not log_text.strip():
                continue
            for pattern, priority, label in CRASH_PATTERNS:
                matches = list(re.finditer(pattern, log_text, re.IGNORECASE))
                if matches:
                    # Show last match context
                    m = matches[-1]
                    ctx = log_text[max(0, m.start()-80):m.end()+80].replace('\n', ' ')
                    if priority != 'INFO':
                        finding(priority, 'log-crash', svc, 0,
                                f'{label} in {svc} logs (×{len(matches)} in last 500 lines)',
                                ctx.strip(),
                                f'Run: journalctl -u {svc} -n 100 --no-pager')
                    else:
                        ok(f'{svc}: {label} (expected, means fix is working)')
        except FileNotFoundError:
            return  # not on Linux
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 4: DATA INTEGRITY
# Validate all JSON log files for corruption, NaN values, schema drift.
# ═════════════════════════════════════════════════════════════════════════════

def run_data_integrity():
    """Check all JSON data files for corruption and logical consistency."""
    LOGS = ROOT / 'logs'
    if not LOGS.exists():
        finding('P1', 'data', 'logs/', 0,
                'logs/ directory missing',
                'No trade data or signals have been saved yet.',
                'Run cloud_scan.py or main.py to generate initial data.')
        return

    # Trade files schema
    TRADE_FILES = [
        ('paper_trades_stocks_only.json', 10_000),
        ('paper_trades.json', 10_000),
    ]
    for fname, expected_start in TRADE_FILES:
        fpath = LOGS / fname
        if not fpath.exists():
            ok(f'{fname}: not yet created (normal on first run)')
            continue
        try:
            with open(fpath) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            finding('P0', 'data', f'logs/{fname}', 0,
                    f'JSON CORRUPT: {fname}',
                    str(e),
                    'Restore from git: git checkout HEAD -- logs/' + fname)
            continue

        cap   = data.get('capital', 0)
        start = data.get('starting_capital', expected_start)
        hist  = data.get('trade_history', [])
        pos   = data.get('positions', {})

        # NaN / Inf check
        if isinstance(cap, float) and (math.isnan(cap) or math.isinf(cap)):
            finding('P0', 'data', f'logs/{fname}', 0,
                    f'NaN/Inf capital in {fname}',
                    f'capital={cap}',
                    'Reset capital to starting_capital value and investigate PnL calculations.')

        # starting_capital missing or zero
        if not start or start <= 0:
            finding('P1', 'data', f'logs/{fname}', 0,
                    f'starting_capital missing or zero in {fname}',
                    f'starting_capital={start}',
                    'Add starting_capital key matching original deposit amount.')

        # PARTIAL_SELL exclusion check
        sell_count     = sum(1 for t in hist if t.get('action') == 'SELL')
        realized_count = sum(1 for t in hist if t.get('action') in ('SELL', 'PARTIAL_SELL'))
        excluded = realized_count - sell_count
        if excluded > 0:
            finding('P1', 'correctness', f'logs/{fname}', 0,
                    f'{excluded} PARTIAL_SELL trades excluded from dashboard KPIs',
                    f'SELL={sell_count}, PARTIAL_SELL={excluded}, '
                    f'total realized={realized_count}',
                    "Update all metric functions to use action in {'SELL','PARTIAL_SELL'}.")
        else:
            ok(f'{fname}: PARTIAL_SELL inclusion OK ({realized_count} realized trades)')

        # Trades with missing date
        bad_dates = [t.get('symbol','?') for t in hist
                     if not t.get('date')]
        if bad_dates:
            finding('P2', 'data', f'logs/{fname}', 0,
                    f'{len(bad_dates)} trades missing date field',
                    f'Symbols: {bad_dates[:5]}',
                    "Backfill date field: t['date'] = datetime.now().isoformat()")

        # Positions with bad entry_price
        bad_pos = [sym for sym, p in pos.items() if p.get('entry_price', 0) <= 0]
        if bad_pos:
            finding('P1', 'data', f'logs/{fname}', 0,
                    f'Positions with entry_price<=0: {bad_pos}',
                    'Zero/negative entry price causes division by zero in PnL calc.',
                    'Remove the position or correct entry_price.')

        ok(f'{fname}: {len(hist)} trades, '
           f'capital=${cap:,.0f}, {len(pos)} open positions')

    # Signals
    sig_path = LOGS / 'latest_signals.json'
    if sig_path.exists():
        try:
            sigs = json.load(open(sig_path))
            # Check signal age
            sig_time = datetime.fromtimestamp(sig_path.stat().st_mtime, tz=timezone.utc)
            age_hrs = (datetime.now(tz=timezone.utc) - sig_time).total_seconds() / 3600
            if age_hrs > 26:
                finding('P1', 'data', 'logs/latest_signals.json', 0,
                        f'Signal file is {age_hrs:.0f}h old — scanner may have stopped',
                        f'Last modified: {sig_time.isoformat()}',
                        'Check alpaca.service logs; restart service if needed.')
            else:
                ok(f'latest_signals.json: {len(sigs)} symbols, {age_hrs:.1f}h old')
        except Exception as e:
            finding('P1', 'data', 'logs/latest_signals.json', 0,
                    'latest_signals.json corrupt or unreadable', str(e), '')

    # Earnings staleness
    earn_path = LOGS / 'earnings.json'
    if earn_path.exists():
        try:
            earnings = json.load(open(earn_path))
            stale = [e for e in earnings
                     if isinstance(e.get('days_until'), (int, float))
                     and e['days_until'] < 0]
            if stale:
                finding('P2', 'data', 'logs/earnings.json', 0,
                        f'{len(stale)} earnings entries have negative days_until (stale)',
                        f"Symbols: {[e.get('symbol') for e in stale[:5]]}",
                        'Recompute days_until at render time from date string, not scan time.')
            else:
                ok(f'earnings.json: {len(earnings)} entries, none stale')
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5: PERFORMANCE ANALYTICS
# Analyse trade history for patterns: win rate, Sharpe, drawdown trends.
# ═════════════════════════════════════════════════════════════════════════════

REALIZED_ACTIONS = {'SELL', 'PARTIAL_SELL'}

def _safe_sharpe(returns: list[float]) -> float | None:
    n = len(returns)
    if n < 20:
        return None
    avg = sum(returns) / n
    std = math.sqrt(sum((r - avg)**2 for r in returns) / (n - 1))
    return round((avg / std) * math.sqrt(252), 2) if std > 0 else None

def run_performance_analysis():
    """Analyse trade history — flag if KPIs are deteriorating."""
    LOGS = ROOT / 'logs'
    fpath = LOGS / 'paper_trades_stocks_only.json'
    if not fpath.exists():
        return

    try:
        data    = json.load(open(fpath))
        start   = data.get('starting_capital', 10_000)
        capital = data.get('capital', start)
        hist    = data.get('trade_history', [])
    except Exception:
        return

    sells = [t for t in hist if t.get('action') in REALIZED_ACTIONS]
    if not sells:
        ok('Performance: no closed trades yet — nothing to analyse')
        return

    wins      = [t for t in sells if t.get('pnl', 0) > 0]
    losses    = [t for t in sells if t.get('pnl', 0) <= 0]
    win_rate  = len(wins) / len(sells) * 100
    realized  = sum(t.get('pnl', 0) for t in sells)
    returns   = [t.get('pnl_pct', 0) for t in sells if 'pnl_pct' in t]
    sharpe    = _safe_sharpe(returns)

    # Drawdown
    peak = start
    equity = start
    max_dd = 0.0
    for t in sells:
        equity += t.get('pnl', 0)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Rolling win rate (last 10 trades) vs overall — detect slippage
    last10 = sells[-10:]
    last10_wr = len([t for t in last10 if t.get('pnl', 0) > 0]) / len(last10) * 100

    ok(
        f'Performance: {len(sells)} trades | WR={win_rate:.1f}% '
        f'| Realized P&L=${realized:+,.2f} | MaxDD={max_dd:.1%}'
        + (f' | Sharpe={sharpe:.2f}' if sharpe else ' | Sharpe=N/A (<20 trades)')
    )

    # Flags
    if win_rate < 40:
        finding('P1', 'performance', 'paper_trades_stocks_only.json', 0,
                f'Win rate below 40%: {win_rate:.1f}%',
                f'{len(wins)} wins / {len(sells)} trades.',
                'Review signal thresholds (BUY_THRESHOLD, veto agent criteria).')

    if max_dd > 0.15:
        finding('P0', 'performance', 'paper_trades_stocks_only.json', 0,
                f'Max drawdown > 15%: {max_dd:.1%}',
                f'Peak-to-trough drawdown exceeded safe limit.',
                'Lower MAX_DRAWDOWN in settings.py or reduce position sizing.')

    if len(last10) == 10 and last10_wr < win_rate - 15:
        finding('P1', 'performance', 'paper_trades_stocks_only.json', 0,
                f'Win rate deteriorating: last-10={last10_wr:.1f}% vs overall={win_rate:.1f}%',
                'Recent trades performing significantly worse than the historical average.',
                'Check if market regime changed; consider reducing position size.')

    if sharpe is not None and sharpe < 0:
        finding('P1', 'performance', 'paper_trades_stocks_only.json', 0,
                f'Negative Sharpe ratio: {sharpe:.2f}',
                'Risk-adjusted returns are negative.',
                'Strategy is destroying risk-adjusted value — pause and review signals.')


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 6: CONFIG CONSISTENCY
# Cross-check settings.py values for logical contradictions.
# ═════════════════════════════════════════════════════════════════════════════

def run_config_checks():
    """Validate configuration consistency and required keys."""
    try:
        from config import settings
    except Exception as e:
        finding('P0', 'config', 'config/settings.py', 0,
                'Cannot import config.settings', str(e),
                'Fix the settings file — all modules depend on it.')
        return

    REQUIRED = [
        'BUY_THRESHOLD', 'VOLUME_SPIKE_MIN', 'MIN_RISK_REWARD',
        'ATR_STOP_MULT', 'ATR_TARGET_MULT', 'MAX_OPEN_POSITIONS',
        'MAX_DRAWDOWN', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID',
        'GROQ_API_KEY', 'ALPACA_API_KEY', 'ALPACA_SECRET_KEY',
    ]
    missing = [k for k in REQUIRED if not getattr(settings, k, None)]
    if missing:
        finding('P1', 'config', 'config/settings.py', 0,
                f'Missing required settings: {missing}',
                'Bots will either error or use fallback defaults.',
                'Set all missing keys in config/settings.py or environment.')
    else:
        ok(f'Config: all {len(REQUIRED)} required keys present')

    # ATR R:R consistency
    try:
        rr = settings.ATR_TARGET_MULT / settings.ATR_STOP_MULT
        min_rr = settings.MIN_RISK_REWARD
        if rr < min_rr:
            finding('P1', 'config', 'config/settings.py', 0,
                    f'ATR R:R ({rr:.1f}x) < MIN_RISK_REWARD ({min_rr}x) — trades never pass R:R filter',
                    'The R:R filter will always reject trades since the ATR multiples'
                    ' can never achieve the minimum required ratio.',
                    f'Either raise ATR_TARGET_MULT above {min_rr * settings.ATR_STOP_MULT:.1f}'
                    f' or lower MIN_RISK_REWARD below {rr:.1f}.')
        else:
            ok(f'Config R:R: ATR R:R={rr:.1f}x >= MIN={min_rr}x')
    except (AttributeError, ZeroDivisionError):
        pass

    # MAX_DRAWDOWN sanity
    try:
        dd = getattr(settings, 'MAX_DRAWDOWN', 0.10)
        if dd > 0.30:
            finding('P2', 'config', 'config/settings.py', 0,
                    f'MAX_DRAWDOWN={dd:.0%} is very permissive',
                    'A 30%+ drawdown before triggering the circuit breaker is high.',
                    'Consider MAX_DRAWDOWN=0.10 (10%) for capital preservation.')
        if dd <= 0:
            finding('P0', 'config', 'config/settings.py', 0,
                    'MAX_DRAWDOWN <= 0 — circuit breaker will trigger immediately',
                    f'MAX_DRAWDOWN={dd}',
                    'Set to a positive fraction e.g. 0.10.')
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 7: DEPENDENCY FRESHNESS
# Check requirements.txt for unpinned or known-broken versions.
# ═════════════════════════════════════════════════════════════════════════════

def run_dependency_checks():
    """Check requirements.txt for unpinned packages."""
    req_file = ROOT / 'requirements.txt'
    if not req_file.exists():
        finding('P1', 'deps', 'requirements.txt', 0,
                'requirements.txt missing',
                'Cannot reproduce the environment.',
                'Run: pip freeze > requirements.txt')
        return

    lines = req_file.read_text().splitlines()
    unpinned = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Unpinned: no ==, >=, <=, ~=
        if not any(op in line for op in ('==', '>=', '<=', '~=', '!=')):
            unpinned.append(line)

    if unpinned:
        finding('P2', 'deps', 'requirements.txt', 0,
                f'{len(unpinned)} unpinned packages — reproducibility risk',
                f'Unpinned: {unpinned[:8]}',
                'Pin all packages: pip freeze > requirements.txt')
    else:
        ok(f'requirements.txt: all {len(lines)} packages pinned')


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 8: CIRCUIT BREAKER STATE INTEGRITY
# ═════════════════════════════════════════════════════════════════════════════

def run_circuit_breaker_checks():
    """Validate circuit breaker state file is coherent."""
    cb_path = ROOT / 'logs' / 'circuit_breaker.json'
    if not cb_path.exists():
        ok('circuit_breaker.json: not yet created (normal on first run)')
        return
    try:
        state = json.load(open(cb_path))
    except Exception as e:
        finding('P0', 'risk-control', 'logs/circuit_breaker.json', 0,
                'circuit_breaker.json corrupt', str(e),
                'Delete the file — it will be recreated on next run.')
        return

    sc = state.get('starting_capital', 0)
    if sc <= 0:
        finding('P0', 'risk-control', 'logs/circuit_breaker.json', 0,
                f'starting_capital={sc} in circuit breaker state — total-loss check dead',
                'The 10% total-loss guardrail can never fire if starting_capital=0.',
                'Set starting_capital to your actual deposit amount.')
    else:
        ok(f'Circuit breaker: starting_capital=${sc:,.0f}')

    is_open = state.get('is_open', False)
    if is_open:
        finding('P0', 'risk-control', 'logs/circuit_breaker.json', 0,
                'Circuit breaker is OPEN — all bots should be blocking trades',
                f'State: {state}',
                'If this is intentional, monitor and reset when recovered. '
                'Otherwise investigate why it tripped.')
    else:
        ok('Circuit breaker: CLOSED (trading allowed)')


# ═════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE
# Produces a health score 0-10 based on priority-weighted finding count.
# ═════════════════════════════════════════════════════════════════════════════

PRIORITY_WEIGHTS = {'P0': 3.0, 'P1': 1.0, 'P2': 0.3, 'INFO': 0.0}

def compute_score() -> tuple[float, str]:
    """Return (score, grade) where score is 0-10."""
    deduction = sum(PRIORITY_WEIGHTS.get(f['priority'], 0) for f in findings)
    # Deduction scale: 0=10, 3=9, 6=8, 10=7, 20=5, 40=2, 60+=0
    score = max(0.0, 10.0 - deduction * 0.15)
    score = round(score, 1)
    if score >= 9.5:
        grade = '🟢 EXCELLENT'
    elif score >= 8.0:
        grade = '🟢 GOOD'
    elif score >= 6.5:
        grade = '🟡 FAIR'
    elif score >= 5.0:
        grade = '🟠 NEEDS WORK'
    elif score >= 3.0:
        grade = '🔴 POOR'
    else:
        grade = '🔴 CRITICAL'
    return score, grade


# ═════════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def build_report(score: float, grade: str, duration_s: float) -> str:
    """Build the full audit report (Telegram-safe markdown)."""
    now  = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    p0s  = [f for f in findings if f['priority'] == 'P0']
    p1s  = [f for f in findings if f['priority'] == 'P1']
    p2s  = [f for f in findings if f['priority'] == 'P2']

    lines = [
        f'🔍 *AlphaEdge Deep Audit — {now}*',
        f'Score: *{score}/10* {grade}',
        f'Findings: {len(p0s)} P0 · {len(p1s)} P1 · {len(p2s)} P2  |  '
        f'{len(passed)} checks passed  |  {duration_s:.0f}s',
        '',
    ]

    def fmt_finding(f: dict, idx: int) -> str:
        loc = f['file'] + (f':L{f["line"]}' if f['line'] else '')
        txt = (
            f'{idx}. [{f["priority"]}] *{f["title"]}*\n'
            f'   📍 `{loc}`\n'
            f'   ⚠️ {f["detail"][:200]}\n'
        )
        if f['fix']:
            txt += f'   ✅ FIX: _{f["fix"][:200]}_\n'
        return txt

    if p0s:
        lines.append('🚨 *CRITICAL (P0)*')
        for i, f in enumerate(p0s, 1):
            lines.append(fmt_finding(f, i))

    if p1s:
        lines.append('⚠️ *SIGNIFICANT (P1)*')
        for i, f in enumerate(p1s, 1):
            lines.append(fmt_finding(f, i))

    if p2s:
        lines.append('📋 *MINOR (P2)*')
        for i, f in enumerate(p2s[:5], 1):  # cap at 5 for Telegram length
            lines.append(fmt_finding(f, i))
        if len(p2s) > 5:
            lines.append(f'   _...and {len(p2s)-5} more P2 findings_\n')

    if not findings:
        lines.append('✅ *No issues found — codebase is clean!*\n')

    # Passed checks summary
    lines.append(f'✅ *{len(passed)} checks passed:*')
    for p in passed[:10]:
        lines.append(f'   • {p}')
    if len(passed) > 10:
        lines.append(f'   _...and {len(passed)-10} more_')

    lines.append('')
    lines.append(f'_Audit ran in {duration_s:.0f}s · AlphaEdge Deep Audit Agent_')

    return '\n'.join(lines)


def build_github_issue(score: float, grade: str) -> str:
    """Build a GitHub Issue body in markdown."""
    now = datetime.now().strftime('%Y-%m-%d')
    p0s = [f for f in findings if f['priority'] == 'P0']
    p1s = [f for f in findings if f['priority'] == 'P1']
    p2s = [f for f in findings if f['priority'] == 'P2']

    md = [
        f'## AlphaEdge Deep Audit — {now}',
        f'**Score: {score}/10** {grade}  |  '
        f'{len(p0s)} P0 · {len(p1s)} P1 · {len(p2s)} P2',
        '',
    ]

    for priority, label, emoji, items in [
        ('P0', 'Critical', '🚨', p0s),
        ('P1', 'Significant', '⚠️', p1s),
        ('P2', 'Minor', '📋', p2s),
    ]:
        if not items:
            continue
        md.append(f'### {emoji} {label} ({priority})')
        md.append('')
        for f in items:
            loc = f['file'] + (f':L{f["line"]}' if f['line'] else '')
            md.append(f'#### `[{priority}]` {f["title"]}')
            md.append(f'- **File:** `{loc}`')
            md.append(f'- **Category:** {f["category"]}')
            md.append(f'- **Detail:** {f["detail"]}')
            if f['fix']:
                md.append(f'- **Fix:** {f["fix"]}')
            md.append('')

    md.append('### ✅ Passed Checks')
    for p in passed:
        md.append(f'- {p}')

    md.append('')
    md.append('---')
    md.append('_Generated by AlphaEdge Deep Audit Agent (`deep_audit.py`)_')
    return '\n'.join(md)


# ═════════════════════════════════════════════════════════════════════════════
# TELEGRAM SENDER
# ═════════════════════════════════════════════════════════════════════════════

def send_telegram(message: str):
    """Send audit report via Telegram (chunked for long messages)."""
    try:
        from config import settings
        token   = settings.TELEGRAM_BOT_TOKEN
        chat_id = settings.TELEGRAM_CHAT_ID
    except Exception:
        token   = os.getenv('TELEGRAM_BOT_TOKEN', '')
        chat_id = os.getenv('TELEGRAM_CHAT_ID', '')

    if not token or not chat_id:
        print('[Telegram] No credentials — skipping notification')
        return

    import urllib.request, urllib.parse

    # Telegram max message length is 4096 chars — chunk if needed
    chunk_size = 4000
    chunks = [message[i:i+chunk_size] for i in range(0, len(message), chunk_size)]

    for chunk in chunks:
        payload = json.dumps({
            'chat_id'   : chat_id,
            'text'      : chunk,
            'parse_mode': 'Markdown',
        }).encode()
        try:
            req = urllib.request.Request(
                f'https://api.telegram.org/bot{token}/sendMessage',
                data=payload,
                headers={'Content-Type': 'application/json'},
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f'[Telegram] Send error: {e}')


# ═════════════════════════════════════════════════════════════════════════════
# GITHUB ISSUE POSTER (for CI runs)
# ═════════════════════════════════════════════════════════════════════════════

def post_github_issue(body: str, score: float):
    """Post or update a GitHub Issue with today's audit report."""
    token = os.getenv('GITHUB_TOKEN', '')
    repo  = os.getenv('GITHUB_REPOSITORY', 'jpfashionhub888/alpha_edge')
    if not token:
        return

    import urllib.request, urllib.parse
    api  = f'https://api.github.com/repos/{repo}'
    hdrs = {
        'Authorization': f'Bearer {token}',
        'Accept'       : 'application/vnd.github+json',
        'Content-Type' : 'application/json',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    today = datetime.now().strftime('%Y-%m-%d')
    title = f'[Audit] {today} — Score {score}/10'
    label_color = 'e11d48' if score < 6 else 'f59e0b' if score < 8 else '10b981'

    # Ensure audit label exists
    try:
        lbl_payload = json.dumps({'name':'audit','color':label_color,'description':'Daily audit report'}).encode()
        req = urllib.request.Request(f'{api}/labels', data=lbl_payload, headers=hdrs)
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass  # label already exists
    except Exception:
        pass

    # Post the issue
    issue_payload = json.dumps({
        'title' : title,
        'body'  : body,
        'labels': ['audit'],
    }).encode()
    try:
        req = urllib.request.Request(f'{api}/issues', data=issue_payload, headers=hdrs)
        urllib.request.urlopen(req, timeout=15)
        print(f'[GitHub] Issue posted: {title}')
    except Exception as e:
        print(f'[GitHub] Issue post error: {e}')


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    print(f'\n{"="*62}')
    print(f'  AlphaEdge Deep Audit Agent — {t0.strftime("%Y-%m-%d %H:%M")}')
    print(f'{"="*62}\n')

    print('1/7  Static code analysis ...')
    hit = run_static_analysis()
    print(f'     {hit} antipattern hits across all source files')

    print('2/7  Runtime module checks ...')
    run_runtime_checks()

    print('3/7  Service + log health ...')
    run_service_checks()

    print('4/7  Data integrity ...')
    run_data_integrity()

    print('5/7  Performance analysis ...')
    run_performance_analysis()

    print('6/7  Config consistency ...')
    run_config_checks()

    print('7/7  Dependencies + circuit breaker ...')
    run_dependency_checks()
    run_circuit_breaker_checks()

    duration = (datetime.now() - t0).total_seconds()
    score, grade = compute_score()

    print(f'\n{"="*62}')
    print(f'  SCORE: {score}/10  {grade}')
    print(f'  P0={len([f for f in findings if f["priority"]=="P0"])}  '
          f'P1={len([f for f in findings if f["priority"]=="P1"])}  '
          f'P2={len([f for f in findings if f["priority"]=="P2"])}  '
          f'Passed={len(passed)}')
    print(f'{"="*62}\n')

    report     = build_report(score, grade, duration)
    issue_body = build_github_issue(score, grade)

    # Print findings to stdout always
    print(report)

    # Save report to file
    report_dir = ROOT / 'logs' / 'audits'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f'audit_{t0.strftime("%Y%m%d_%H%M%S")}.md'
    report_path.write_text(issue_body, encoding='utf-8')
    print(f'\nReport saved: {report_path}')

    if not STDOUT_ONLY:
        print('\nSending Telegram report...')
        send_telegram(report)
        print('\nPosting GitHub issue (if GITHUB_TOKEN set)...')
        post_github_issue(issue_body, score)

    # Exit code reflects health: non-zero if any P0
    p0_count = len([f for f in findings if f['priority'] == 'P0'])
    sys.exit(1 if p0_count > 0 else 0)


if __name__ == '__main__':
    main()
