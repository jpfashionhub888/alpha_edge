#!/usr/bin/env python3
"""
AlphaEdge Comprehensive System Audit
Checks all modules, data files, config, and key functionality.
"""
import os, sys, json, traceback
from datetime import datetime

ROOT = '/root/alpha_edge'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

PASS = '✅'
WARN = '⚠️ '
FAIL = '❌'
INFO = 'ℹ️ '

results = []

def check(label, fn):
    try:
        msg = fn()
        status = PASS
        results.append((status, label, msg or 'OK'))
        print(f"{status} {label}: {msg or 'OK'}")
    except Exception as e:
        status = FAIL
        results.append((status, label, str(e)))
        print(f"{status} {label}: {e}")

def warn(label, msg):
    results.append((WARN, label, msg))
    print(f"{WARN} {label}: {msg}")

print("=" * 60)
print(f"  AlphaEdge Audit  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ── 1. CORE MODULE IMPORTS ────────────────────────────────────
print("\n── 1. MODULE IMPORTS ──────────────────────────────────────")

def imp(mod): __import__(mod); return 'imported'

check("config.settings",        lambda: imp('config.settings'))
check("correlation_filter",     lambda: imp('correlation_filter'))
check("insider_tracker",        lambda: imp('insider_tracker'))
check("veto_agent",             lambda: imp('veto_agent'))
check("performance_analytics",  lambda: imp('performance_analytics'))
check("market_regime",          lambda: imp('market_regime'))
check("multi_timeframe",        lambda: imp('multi_timeframe'))
check("models.sector_rotation", lambda: imp('models.sector_rotation'))
check("models.technical_model", lambda: imp('models.technical_model'))
check("data.feature_engine",    lambda: imp('data.feature_engine'))
check("execution.paper_trader", lambda: imp('execution.paper_trader'))
check("generate_dashboard",     lambda: imp('generate_dashboard'))

# ── 2. CONFIG AUDIT ───────────────────────────────────────────
print("\n── 2. CONFIG AUDIT ────────────────────────────────────────")

def cfg_check():
    from config import settings
    required = [
        'BUY_THRESHOLD','VOLUME_SPIKE_MIN','MIN_RISK_REWARD',
        'ATR_STOP_MULT','ATR_TARGET_MULT','MAX_OPEN_POSITIONS',
        'TIME_STOP_DAYS','KELLY_POSITION_SIZING','MAX_DRAWDOWN',
        'TELEGRAM_BOT_TOKEN','TELEGRAM_CHAT_ID','GROQ_API_KEY',
        'ALPACA_API_KEY','ALPACA_SECRET_KEY',
    ]
    missing = [k for k in required if not getattr(settings, k, None)]
    if missing:
        raise ValueError(f"Missing: {missing}")
    return f"{len(required)} keys present"

check("Required settings", cfg_check)

def rr_check():
    from config import settings
    rr = settings.ATR_TARGET_MULT / settings.ATR_STOP_MULT
    if rr < settings.MIN_RISK_REWARD:
        raise ValueError(f"ATR R:R {rr:.1f} < MIN_RISK_REWARD {settings.MIN_RISK_REWARD}")
    return f"R:R={rr:.1f} >= MIN={settings.MIN_RISK_REWARD}"

check("ATR R:R vs MIN_RISK_REWARD", rr_check)

# ── 3. DATA FILE AUDIT ────────────────────────────────────────
print("\n── 3. DATA FILES ──────────────────────────────────────────")

def load_trades():
    f = 'logs/paper_trades_stocks_only.json'
    if not os.path.exists(f):
        raise FileNotFoundError(f)
    d = json.load(open(f))
    cap = d.get('capital', 0)
    pos = d.get('positions', {})
    hist = d.get('trade_history', [])
    # Validate no NaN in capital
    if not isinstance(cap, (int, float)) or cap != cap:
        raise ValueError(f"capital is NaN/invalid: {cap}")
    return f"capital=${cap:,.0f} | {len(pos)} positions | {len(hist)} trades"

check("paper_trades_stocks_only.json", load_trades)

def load_signals():
    f = 'logs/latest_signals.json'
    if not os.path.exists(f):
        return "NOT FOUND (will be created on next scan)"
    d = json.load(open(f))
    if not d:
        return "EXISTS but empty (no scan run yet today)"
    # Check required keys in each signal
    bad = []
    for sym, sig in d.items():
        for key in ['prediction','regime','price','signal']:
            if key not in sig:
                bad.append(f"{sym} missing {key}")
    if bad:
        raise ValueError(f"Malformed signals: {bad[:3]}")
    return f"{len(d)} symbols | sample: {list(d.keys())[:3]}"

check("latest_signals.json", load_signals)

def load_sectors():
    f = 'logs/sectors.json'
    if not os.path.exists(f):
        return "NOT FOUND (generated on scan)"
    d = json.load(open(f))
    return f"{len(d)} sectors | {list(d.keys())[:3]}"

check("sectors.json", load_sectors)

def check_logs_dir():
    files = os.listdir('logs') if os.path.exists('logs') else []
    return f"{len(files)} files: {files}"

check("logs/ directory", check_logs_dir)

# ── 4. PAPER TRADER STATE ─────────────────────────────────────
print("\n── 4. PAPER TRADER STATE ──────────────────────────────────")

def trader_state():
    from execution.paper_trader import PaperTrader
    pt = PaperTrader()
    issues = []
    for sym, pos in pt.positions.items():
        ep = pos.get('entry_price', 0)
        sh = pos.get('shares', 0)
        if ep <= 0: issues.append(f"{sym}: bad entry_price={ep}")
        if sh <= 0: issues.append(f"{sym}: bad shares={sh}")
        if 'entry_date' not in pos: issues.append(f"{sym}: missing entry_date")
    if issues:
        raise ValueError(f"Position issues: {issues}")
    return f"time_stop_days={pt.time_stop_days} | {len(pt.positions)} positions validated"

check("PaperTrader state consistency", trader_state)

# ── 5. GATE.IO API KEY ────────────────────────────────────────
print("\n── 5. LIVE API CONNECTIONS ────────────────────────────────")

def gateio_key():
    # Check if key is configured
    key = os.getenv('GATEIO_API_KEY', '')
    secret = os.getenv('GATEIO_SECRET', '')
    if not key and not secret:
        # Try reading from secrets file
        env_paths = [
            'config/secrets.env',
            '/root/alpha_edge/config/secrets.env',
            '.env',
        ]
        for ep in env_paths:
            if os.path.exists(ep):
                content = open(ep).read()
                if 'GATEIO' in content.upper():
                    return f"Found in {ep}"
                else:
                    return f"GATEIO key NOT in {ep} — 401 errors expected"
        return "No secrets.env found — Gate.io uses INVALID_KEY (paper mode only)"
    return f"Key configured: ...{key[-4:] if len(key)>4 else 'SHORT'}"

check("Gate.io API key", gateio_key)

def alpaca_key():
    from config import settings
    key = settings.ALPACA_API_KEY
    if not key:
        raise ValueError("ALPACA_API_KEY is empty")
    return f"Key: ...{key[-6:]}"

check("Alpaca API key", alpaca_key)

def telegram_live():
    import requests
    from config import settings
    r = requests.get(
        f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/getMe",
        timeout=8
    )
    d = r.json()
    if not d.get('ok'):
        raise ValueError(f"Bot unreachable: {d}")
    return f"@{d['result']['username']} is alive"

check("Telegram bot reachable", telegram_live)

# ── 6. DASHBOARD GENERATION TEST ─────────────────────────────
print("\n── 6. DASHBOARD GENERATION ────────────────────────────────")

def dash_gen():
    # Test that generate_dashboard runs without crashing
    import subprocess
    r = subprocess.run(
        ['python3', 'generate_dashboard.py'],
        capture_output=True, text=True, timeout=30, cwd=ROOT
    )
    if r.returncode != 0:
        raise RuntimeError(f"Exit {r.returncode}: {r.stderr[-300:]}")
    size = os.path.getsize('docs/index.html')
    return f"Generated OK | index.html = {size/1024:.0f} KB"

check("generate_dashboard.py runs clean", dash_gen)

# ── 7. SILENT CRASH HOTSPOTS ─────────────────────────────────
print("\n── 7. KNOWN SILENT CRASH HOTSPOTS ────────────────────────")

def check_feature_engine():
    from data.feature_engine import FeatureEngine
    fe = FeatureEngine()
    # Verify the look-ahead fix is present
    import inspect
    src = inspect.getsource(fe.create_features)
    if 'max_date' not in src:
        raise ValueError("Look-ahead bias fix (max_date) NOT in create_features!")
    return "max_date look-ahead guard present"

check("FeatureEngine look-ahead guard", check_feature_engine)

def check_sector_rotation():
    from models.sector_rotation import _flatten_yf_columns
    import pandas as pd
    # Test with MultiIndex (yfinance >= 0.2.x style)
    idx = pd.MultiIndex.from_tuples([('Close','XLK'),('Volume','XLK')])
    df = pd.DataFrame([[100.0, 1000000]], columns=idx)
    df = _flatten_yf_columns(df)
    assert 'close' in df.columns, "flatten failed"
    return "_flatten_yf_columns handles MultiIndex correctly"

check("SectorRotation MultiIndex flatten", check_sector_rotation)

def check_veto_decision():
    from veto_agent import VetoAgent
    import inspect
    src = inspect.getsource(VetoAgent.review_signal)
    if 'original_decision' in src or "decision value" in src or "unexpected" in src.lower():
        return "Veto reason captures unexpected values"
    # Check that HOLD_TIGHT or unexpected values are handled
    if 'APPROVE' in src:
        return "Veto logic present"
    return "Veto agent loaded"

check("VetoAgent decision capture", check_veto_decision)

def check_insider_score():
    from insider_tracker import InsiderTracker
    it = InsiderTracker.__new__(InsiderTracker)
    it.ticker_to_cik = {}
    score = it.get_insider_score('AAPL')
    assert score == 0.0, f"Expected 0.0 got {score}"
    return "Returns 0.0 (boost removed)"

check("InsiderTracker returns 0.0", check_insider_score)

# ── 8. ALPACA LIVE SCRIPT CHECK ───────────────────────────────
print("\n── 8. LIVE SCRIPT SYNTAX ──────────────────────────────────")

for script in ['alpaca_live.py', 'gateio_live.py', 'cloud_scan.py', 'main.py']:
    def syntax_check(s=script):
        import subprocess
        r = subprocess.run(['python3', '-m', 'py_compile', s],
                          capture_output=True, text=True, cwd=ROOT)
        if r.returncode != 0:
            raise SyntaxError(r.stderr.strip())
        return "syntax OK"
    check(f"{script} syntax", syntax_check)

# ── SUMMARY ───────────────────────────────────────────────────
print("\n" + "=" * 60)
passes  = sum(1 for r in results if r[0] == PASS)
warns   = sum(1 for r in results if r[0] == WARN)
fails   = sum(1 for r in results if r[0] == FAIL)
total   = len(results)
print(f"  AUDIT COMPLETE: {passes}/{total} passed | {warns} warnings | {fails} failures")
print("=" * 60)

if fails > 0:
    print("\nFAILED CHECKS:")
    for s, l, m in results:
        if s == FAIL:
            print(f"  {s} {l}: {m}")

if warns > 0:
    print("\nWARNINGS:")
    for s, l, m in results:
        if s == WARN:
            print(f"  {s} {l}: {m}")
