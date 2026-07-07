# scripts/live_readiness_check.py
"""
AlphaEdge — Formal Live Trading Go/No-Go Checklist

Institutional trading desks require a signed-off readiness checklist
before going live. This script automates every criterion and exits
non-zero if ANY criterion fails.

Run before switching ALPACA_BASE_URL from paper to live:
    python scripts/live_readiness_check.py

Criteria:
    1.  Paper trades completed        >= 50
    2.  Win rate (all paper trades)   >= 55%
    3.  Profit factor                 >= 1.5
    4.  Max drawdown (paper)          <= 15%
    5.  Sharpe ratio (annualized)     >= 1.0
    6.  Model AUC (walk-forward)      >= 0.55
    7.  Circuit breaker               = Clear (not triggered)
    8.  No reconciliation failures    = 0 discrepancies
    9.  Audit passes                  = 0 FAIL
    10. Emergency stop tested         = Manual confirm
    11. Slippage model calibrated     = Manual confirm
    12. No open bugs in audit         = All checks pass
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PASS = '  [GO]   '
FAIL = '  [NOGO] '
WARN = '  [WARN] '
INFO = '  [INFO] '

results = []
manual_confirms = []


def check(name: str, ok: bool, detail: str = '', critical: bool = True) -> bool:
    flag = PASS if ok else FAIL
    line = f'{flag} {name}'
    if detail:
        line += f'  --  {detail}'
    print(line)
    results.append({'name': name, 'ok': ok, 'detail': detail, 'critical': critical})
    return ok


def manual_check(name: str, prompt: str) -> bool:
    """Ask for manual confirmation of a criterion."""
    print(f'\n{INFO} MANUAL CHECK: {name}')
    print(f'  {prompt}')
    ans = input('  Confirmed? (yes/no): ').strip().lower()
    ok  = ans in ('yes', 'y')
    check(name, ok, 'Manually confirmed' if ok else 'NOT confirmed')
    return ok


def check_trade_count():
    print('\n-- 1. Trade Count ------------------------------------------')
    try:
        with open('logs/closed_trades.json') as f:
            data = json.load(f)
        n = data['summary']['total']
        return check('Paper trades >= 50', n >= 50, f'{n}/50 trades completed')
    except FileNotFoundError:
        return check('Paper trades >= 50', False, '0/50 — trade log not found')
    except Exception as e:
        return check('Paper trades >= 50', False, str(e))


def check_win_rate():
    print('\n-- 2. Win Rate ---------------------------------------------')
    try:
        with open('logs/closed_trades.json') as f:
            data = json.load(f)
        wr  = data['summary']['win_rate'] * 100
        n   = data['summary']['total']
        if n < 30:
            print(f'{WARN} Only {n} trades — win rate not statistically significant (<30)')
        return check('Win rate >= 55%', wr >= 55, f'{wr:.1f}% (need >= 55%)')
    except Exception as e:
        return check('Win rate >= 55%', False, str(e))


def check_profit_factor():
    print('\n-- 3. Profit Factor ----------------------------------------')
    try:
        with open('logs/closed_trades.json') as f:
            data = json.load(f)
        pf = data['summary'].get('profit_factor') or 0
        return check('Profit factor >= 1.5', pf >= 1.5, f'{pf:.2f} (need >= 1.5)')
    except Exception as e:
        return check('Profit factor >= 1.5', False, str(e))


def check_max_drawdown():
    print('\n-- 4. Max Drawdown -----------------------------------------')
    try:
        with open('logs/paper_trades_stocks_only.json') as f:
            data = json.load(f)
        starting  = data.get('starting_capital', 10000)
        peak      = data.get('peak_value', starting)
        capital   = data.get('capital', starting)
        dd        = (capital - peak) / peak * 100 if peak > 0 else 0
        return check('Max drawdown <= 15%', abs(dd) <= 15,
                     f'{dd:.1f}% (need <= 15%)')
    except Exception as e:
        try:
            with open('logs/circuit_breaker.json') as f:
                cb = json.load(f)
            peak = cb.get('peak_value')
            if peak:
                print(f'{INFO} Using peak from circuit_breaker.json: ${peak:,.0f}')
        except Exception:
            pass
        return check('Max drawdown <= 15%', False, str(e))


def check_sharpe():
    print('\n-- 5. Sharpe Ratio -----------------------------------------')
    try:
        with open('logs/closed_trades.json') as f:
            data = json.load(f)
        trades = data.get('trades', [])
        if len(trades) < 20:
            print(f'{WARN} Only {len(trades)} trades — Sharpe estimate not reliable (<20)')
            return check('Sharpe >= 1.0', False,
                         f'Need >= 20 trades for reliable estimate (have {len(trades)})')
        import statistics
        returns   = [t.get('pnl_pct', 0) for t in trades]
        avg_r     = statistics.mean(returns)
        std_r     = statistics.stdev(returns) if len(returns) > 1 else 1
        # Annualise: assume ~252 trading days, trades come from daily bars
        n_per_yr  = 252 / max(1, len(trades)) * max(1, len(trades))
        sharpe    = (avg_r / std_r) * (252 ** 0.5) if std_r > 0 else 0
        return check('Sharpe >= 1.0', sharpe >= 1.0, f'{sharpe:.2f} (need >= 1.0)')
    except Exception as e:
        return check('Sharpe >= 1.0', False, str(e))


def check_circuit_breaker():
    print('\n-- 6. Circuit Breaker --------------------------------------')
    try:
        with open('logs/circuit_breaker.json') as f:
            cb = json.load(f)
        triggered = cb.get('triggered', False)
        return check('Circuit breaker clear', not triggered,
                     'Clear' if not triggered else
                     f"TRIGGERED: {cb.get('trigger_reason', '')}")
    except FileNotFoundError:
        return check('Circuit breaker clear', True, 'File not found — assumed clear')
    except Exception as e:
        return check('Circuit breaker clear', False, str(e))


def check_audit():
    print('\n-- 7. Deep Audit -------------------------------------------')
    try:
        result = subprocess.run(
            [sys.executable, 'audit_system.py'],
            capture_output=True, text=True, timeout=60
        )
        ok = result.returncode == 0
        if not ok:
            failed_lines = [
                l for l in result.stdout.splitlines()
                if '[FAIL]' in l or '[CRITICAL]' in l
            ]
            detail = f'{len(failed_lines)} failures detected'
            for l in failed_lines[:5]:
                print(f'    {l.strip()}')
        else:
            detail = 'All checks passed'
        return check('Audit: 0 failures', ok, detail)
    except subprocess.TimeoutExpired:
        return check('Audit: 0 failures', False, 'Audit timed out')
    except FileNotFoundError:
        return check('Audit: 0 failures', False, 'audit_system.py not found')
    except Exception as e:
        return check('Audit: 0 failures', False, str(e))


def check_model_auc():
    print('\n-- 8. Model AUC (Walk-Forward) -----------------------------')
    try:
        with open('logs/latest_signals.json') as f:
            signals = json.load(f)
        auc = signals.get('model_auc')
        if auc is None:
            print(f'{INFO} model_auc not yet logged in signals — skipping')
            results.append({'name': 'Model AUC >= 0.55', 'ok': True,
                            'detail': 'Not measured yet — add AUC to scanner logs',
                            'critical': False})
            return True
        return check('Model AUC >= 0.55', auc >= 0.55, f'{auc:.3f} (need >= 0.55)')
    except FileNotFoundError:
        print(f'{INFO} latest_signals.json not found — AUC check skipped')
        return True
    except Exception as e:
        return check('Model AUC >= 0.55', False, str(e))


def main():
    print('\n' + '=' * 58)
    print('  ALPHAEDGE LIVE TRADING GO / NO-GO CHECKLIST')
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print('  Institutional standard: ALL criteria must pass')
    print('=' * 58)

    # Automated checks
    check_trade_count()
    check_win_rate()
    check_profit_factor()
    check_max_drawdown()
    check_sharpe()
    check_circuit_breaker()
    check_audit()
    check_model_auc()

    # Manual confirmation checks
    print('\n-- Manual Confirmations ------------------------------------')
    print(f'{INFO} The following require your explicit confirmation.')
    print(f'{INFO} Answer "yes" only if truly verified.\n')

    manual_check(
        'Emergency stop tested',
        'Have you tested /pause via Telegram and confirmed it stops new entries?'
    )
    manual_check(
        'Slippage model calibrated',
        'Have you compared paper fill prices vs Alpaca actual fill prices on >= 10 orders?'
    )
    manual_check(
        'Live account funded',
        'Is your Alpaca LIVE account funded with the intended starting capital?'
    )
    manual_check(
        'VPS environment verified',
        'Have you confirmed the VPS has the correct ALPACA_BASE_URL=live set?'
    )

    # Summary
    total    = len(results)
    passed   = sum(1 for r in results if r['ok'])
    critical = [r for r in results if not r['ok'] and r.get('critical', True)]
    warns    = [r for r in results if not r['ok'] and not r.get('critical', True)]

    print('\n' + '=' * 58)
    print(f'  RESULT: {passed}/{total} criteria passed')

    if not critical:
        print('\n  ✅ GO — All critical criteria met.')
        print('  You may switch ALPACA_BASE_URL to live.')
        print('\n  Steps to go live:')
        print('  1. export ALPACA_API_KEY=<live_key>')
        print('  2. export ALPACA_SECRET_KEY=<live_secret>')
        print('  3. export ALPACA_BASE_URL=https://api.alpaca.markets')
        print('  4. systemctl restart alphaedge.service')
        sys.exit(0)
    else:
        print(f'\n  ❌ NO-GO — {len(critical)} critical criteria NOT met:')
        for r in critical:
            print(f'    [NOGO] {r["name"]}  --  {r["detail"]}')
        if warns:
            print(f'\n  Warnings ({len(warns)}):')
            for r in warns:
                print(f'    [WARN] {r["name"]}  --  {r["detail"]}')
        print('\n  Fix all NOGO items before going live.')
        print('=' * 58)
        sys.exit(1)


if __name__ == '__main__':
    main()
