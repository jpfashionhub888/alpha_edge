# monitoring/dashboard.py
"""
AlphaEdge Live Dashboard -- "Neural Trading Core" (single consolidated UI)

Replaces THREE previously separate dashboards:
  1. jpfashionhub888.github.io/alpha_edge/         (static docs/index.html)
  2. jpfashionhub888.github.io/alpha_edge/app.html  (static)
  3. alphaedgetrading.duckdns.org                   (old Dash "Bloomberg V7" UI)

This IS now the one live dashboard, served at alphaedgetrading.duckdns.org
(same port 8050, same nginx Basic Auth in front -- unchanged). The two
GitHub Pages URLs should be turned into redirect stubs pointing here
(see docs/index.html / docs/app.html after this change).

Architecture: plain Flask, not Dash. The UI (sci-fi "Neural Trading Core"
theme) is a static HTML/CSS/JS document -- Dash's component-tree model
doesn't fit it well, and Flask is already a dependency (Dash sits on top
of Flask). The server's only job is:
  - GET /            -> serve the static HTML shell (DASH_HTML, below)
  - GET /api/data    -> serve one JSON payload with everything the page needs
  - GET /assets/...  -> serve icons/manifest (same files as before)

The client polls /api/data every 60s (same cadence as the old Dash
dcc.Interval) and re-renders in place. No server-side templating of data
into HTML -- keeps the server simple and the page fast.
"""

import json, os, time, logging, threading
from datetime import datetime

from flask import Flask, jsonify, send_from_directory

logger = logging.getLogger(__name__)

# ── Eastern Time (auto-handles EDT/EST daylight saving) ─────────────────────
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')
except ImportError:
    try:
        import pytz
        _ET = pytz.timezone('America/New_York')
    except ImportError:
        from datetime import timezone, timedelta
        _ET = timezone(timedelta(hours=-4))  # fallback: EDT


def now_et():
    """Return current datetime in US/Eastern (handles EDT and EST)."""
    return datetime.now(tz=_ET)


# ── Settings load (same pattern as the previous dashboard.py) ───────────────
try:
    from config import settings
    kelly_active    = getattr(settings, 'KELLY_POSITION_SIZING', True)
    kelly_mult      = getattr(settings, 'KELLY_MULTIPLIER', 0.5)
    kelly_rr        = getattr(settings, 'KELLY_REWARD_RISK_RATIO', 2.5)
    max_pos_pct     = getattr(settings, 'MAX_POSITION_SIZE', 0.15)
    buy_threshold   = getattr(settings, 'BUY_THRESHOLD', 0.63)
    max_dd_limit    = getattr(settings, 'MAX_DRAWDOWN', 0.10)
    max_daily_loss  = getattr(settings, 'MAX_DAILY_LOSS', 0.02)
    max_positions   = getattr(settings, 'MAX_OPEN_POSITIONS', 5)
    vol_spike_min   = getattr(settings, 'VOLUME_SPIKE_MIN', 1.3)
    min_rr          = getattr(settings, 'MIN_RISK_REWARD', 2.0)
    try:
        from main import get_full_watchlist as _gwl
        WATCHLIST = _gwl()
    except Exception:
        WATCHLIST = list(getattr(settings, 'STOCK_WATCHLIST', [])) or []
except Exception as e:
    logger.warning(f'Settings load failed, using fallback defaults: {e}')
    kelly_active = True;  kelly_mult = 0.5;  kelly_rr = 2.5
    max_pos_pct = 0.15;   buy_threshold = 0.63; max_dd_limit = 0.10
    max_daily_loss = 0.02; max_positions = 8
    vol_spike_min = 1.3; min_rr = 2.0
    WATCHLIST = []

watchlist_len = len(WATCHLIST) or 66

TRADES_FILE          = 'logs/paper_trades_stocks_only.json'
SIGNALS_FILE          = 'logs/latest_signals.json'
SECTORS_FILE          = 'logs/sectors.json'
EARNINGS_FILE         = 'logs/earnings.json'
CIRCUIT_BREAKER_FILE  = 'logs/circuit_breaker.json'
MODEL_AUC_FILE        = 'logs/model_auc.json'


def _jload(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f'Failed to read {path}: {e}')
        return default


def load_portfolio():   return _jload(TRADES_FILE,  {'capital': 10000.0, 'starting_capital': 10000.0, 'positions': {}, 'trade_history': []})
def load_signals():     return _jload(SIGNALS_FILE, {})
def load_sectors():     return _jload(SECTORS_FILE, {})
def load_earnings():    return _jload(EARNINGS_FILE, [])
def load_circuit_breaker(): return _jload(CIRCUIT_BREAKER_FILE, {})
def load_model_auc():       return _jload(MODEL_AUC_FILE, {})


def kelly_sizing(pred):
    if not kelly_active:
        return 0.0
    p = float(pred)
    f = max(0.0, (p * (kelly_rr + 1) - 1) / kelly_rr)
    return min(f * kelly_mult, max_pos_pct)


# ── Live quotes (Market Movers panel) ────────────────────────────────────────
# AlphaEdge has no existing "current price + 52w range" feed for the
# dashboard -- the trading loop fetches OHLCV per symbol during its own
# scan but doesn't persist it anywhere the dashboard can read cheaply.
# Rather than add a new third-party dependency, this reuses AlphaEdge's
# own DataLoader (backtesting/data/loader.py, yfinance w/ Polygon
# fallback) on a background thread with a 5-minute cache, so a dashboard
# page load / 60s poll never blocks on a network call. Best-effort: if
# the fetch fails (no network, rate limit, etc.) the panel just shows
# "no data" rather than taking the rest of the page down with it.
_quotes_cache      = {'data': {}, 'fetched_at': 0.0, 'fetching': False}
_QUOTES_TTL_SECONDS = 300
_QUOTES_MAX_SYMBOLS = 60  # cap to keep the background fetch bounded


def _quotes_watchlist():
    """Symbols worth fetching quotes for: today's signals ∪ open positions,
    capped and falling back to the static watchlist if signals are empty."""
    syms = set()
    try:
        sigs = load_signals()
        syms |= {s for s in sigs.keys() if not s.startswith('_')}
    except Exception:
        pass
    try:
        pos = load_portfolio().get('positions', {})
        syms |= set(pos.keys())
    except Exception:
        pass
    if not syms:
        syms = set(WATCHLIST)
    return sorted(syms)[:_QUOTES_MAX_SYMBOLS]


def _fetch_quotes_once():
    try:
        from backtesting.data.loader import DataLoader
    except Exception as e:
        logger.warning(f'Quotes: DataLoader unavailable ({e}) -- Market Movers will be empty')
        return {}
    symbols = _quotes_watchlist()
    if not symbols:
        return {}
    try:
        from datetime import timedelta
        end   = datetime.now()
        start = end - timedelta(days=370)
        loader = DataLoader()
        data = loader.get_ohlcv(symbols, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), '1d')
    except Exception as e:
        logger.warning(f'Quotes: get_ohlcv failed: {e}')
        return {}
    out = {}
    for sym, df in (data or {}).items():
        try:
            if df is None or len(df) < 2:
                continue
            price      = float(df['close'].iloc[-1])
            prev_close = float(df['close'].iloc[-2])
            change_pct = (price - prev_close) / prev_close * 100 if prev_close else 0.0
            w52_high   = float(df['high'].max())
            w52_low    = float(df['low'].min())
            out[sym] = {
                'price': price, 'change_pct': change_pct,
                'w52_high': w52_high, 'w52_low': w52_low,
            }
        except Exception:
            continue
    return out


def _quotes_refresh_loop():
    while True:
        try:
            _quotes_cache['fetching'] = True
            fresh = _fetch_quotes_once()
            if fresh:
                _quotes_cache['data'] = fresh
            _quotes_cache['fetched_at'] = time.time()
        except Exception as e:
            logger.warning(f'Quotes refresh loop error: {e}')
        finally:
            _quotes_cache['fetching'] = False
        time.sleep(_QUOTES_TTL_SECONDS)


def get_quotes():
    return _quotes_cache['data']


# ── Payload builder ──────────────────────────────────────────────────────────
def build_payload():
    p         = load_portfolio()
    signals   = load_signals()
    sectors   = load_sectors()
    earnings  = load_earnings()
    cb_raw    = load_circuit_breaker()
    auc_raw   = load_model_auc()

    capital   = p.get('capital', 10000)
    starting  = p.get('starting_capital', 10000)
    positions = p.get('positions', {})
    history   = p.get('trade_history', [])

    pos_out = {}
    for sym, pos in positions.items():
        entry = pos.get('entry_price', 0)
        curr  = pos.get('current_price', entry)
        pos_out[sym] = {
            'shares'      : pos.get('shares', 0),
            'entry_price' : entry,
            'market_price': curr,
            'signal'      : pos.get('signal', 0.5),
        }
    pos_value = sum(v['shares'] * v['market_price'] for v in pos_out.values())
    total     = capital + pos_value

    sells  = [t for t in history if t.get('action') in ('SELL', 'PARTIAL_SELL')]
    wins   = sum(1 for t in sells if t.get('pnl', 0) > 0)
    losses = sum(1 for t in sells if t.get('pnl', 0) <= 0)
    closed = wins + losses
    gp     = sum(t.get('pnl', 0) for t in sells if t.get('pnl', 0) > 0)
    gl     = sum(t.get('pnl', 0) for t in sells if t.get('pnl', 0) <= 0)
    closed_trades_out = [{
        'symbol' : t.get('symbol', ''),
        'pnl_usd': t.get('pnl', 0),
        'pnl_pct': (t.get('pnl_pct', 0) or 0) * 100,
        'reason' : (t.get('reason', '') or '').upper(),
    } for t in reversed(sells[-25:])]

    # Circuit breaker: peak/daily/weekly baselines are None until the first
    # check() call establishes them (fresh install / just after a reset).
    # Fall back to current total so the risk bars render 0% instead of NaN.
    cb_out = {
        'triggered'      : bool(cb_raw.get('triggered', False)),
        'peak_value'     : cb_raw.get('peak_value') if cb_raw.get('peak_value') is not None else total,
        'daily_start_val': cb_raw.get('daily_start_val') if cb_raw.get('daily_start_val') is not None else total,
        'weekly_start_val': cb_raw.get('weekly_start_val') if cb_raw.get('weekly_start_val') is not None else total,
        'baseline_established': cb_raw.get('peak_value') is not None,
    }

    # Model AUC: the live pipeline currently only ever writes training_auc
    # (see main.py) -- there's no "live/recent AUC" field being populated
    # despite walkforward_monitor.py expecting one. Reporting that honestly
    # (null) rather than inventing a number.
    auc_out = {
        'training_auc'        : auc_raw.get('training_auc'),
        'model_auc'           : auc_raw.get('model_auc'),
        'n_symbols_retrained' : auc_raw.get('n_symbols_retrained'),
        'updated_at'          : auc_raw.get('updated_at'),
    }

    # Clean signals: drop any non-dict / meta entries
    signals_out = {k: v for k, v in signals.items() if isinstance(v, dict) and not k.startswith('_')}

    # Scan staleness (ported from the old dashboard's P3-1 fix)
    saved_at = signals.get('_meta', {}).get('saved_at', '') if isinstance(signals.get('_meta'), dict) else ''
    if not saved_at:
        saved_at = next((v.get('saved_at', '') for v in signals_out.values() if v.get('saved_at')), '')
    scan_age_hours = None
    scan_label     = 'NO SCAN DATA'
    scan_color     = 'red'
    if saved_at:
        try:
            st = datetime.fromisoformat(saved_at.replace('Z', '+00:00')).astimezone(_ET)
            scan_age_hours = (datetime.now(_ET) - st).total_seconds() / 3600
            ts = st.strftime('%m/%d %H:%M ET')
            if scan_age_hours >= 120:
                scan_color = 'red';    scan_label = f'LAST SCAN {ts} (STALE - {scan_age_hours/24:.1f}d old)'
            elif scan_age_hours >= 72:
                scan_color = 'amber';  scan_label = f'LAST SCAN {ts} ({scan_age_hours/24:.1f}d old)'
            else:
                scan_color = 'green';  scan_label = f'LAST SCAN {ts}'
        except Exception as e:
            logger.warning(f'Scan timestamp parse failed: {e}')

    return {
        'meta': {
            'server_time_et'  : now_et().strftime('%Y-%m-%d %H:%M:%S ET'),
            'scan_saved_at'   : saved_at,
            'scan_age_hours'  : scan_age_hours,
            'scan_label'      : scan_label,
            'scan_color'      : scan_color,
            'watchlist_len'   : watchlist_len,
            'max_positions'   : max_positions,
            'buy_threshold'   : buy_threshold,
            'quotes_fetching' : _quotes_cache['fetching'],
            'quotes_age_sec'  : (time.time() - _quotes_cache['fetched_at']) if _quotes_cache['fetched_at'] else None,
        },
        'portfolio': {
            'capital'         : capital,
            'starting_capital': starting,
            'positions'       : pos_out,
            'total'           : total,
            'pos_value'       : pos_value,
        },
        'closed_trades': {
            'trades' : closed_trades_out,
            'summary': {
                'total'        : closed,
                'wins'         : wins,
                'losses'       : losses,
                'win_rate'     : (wins / closed) if closed else None,
                'total_pnl'    : gp + gl,
                'profit_factor': (abs(gp / gl) if gl else None),
            },
        },
        'history'        : list(reversed(history[-50:])),
        'signals'        : signals_out,
        'sectors'        : sectors,
        'earnings'       : earnings,
        'circuit_breaker': cb_out,
        'model_auc'      : auc_out,
        'quotes'         : get_quotes(),
    }


# ── Flask app ─────────────────────────────────────────────────────────────
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')


def create_app():
    app = Flask(__name__)

    @app.route('/')
    def index():
        return DASHBOARD_HTML

    @app.route('/api/data')
    def api_data():
        try:
            return jsonify(build_payload())
        except Exception as e:
            logger.exception('build_payload failed')
            return jsonify({'error': str(e)}), 500

    @app.route('/assets/<path:filename>')
    def assets(filename):
        return send_from_directory(_ASSETS_DIR, filename)

    return app


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ALPHA EDGE // NEURAL TRADING CORE</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<link rel="manifest" href="/assets/manifest.json">
<meta name="theme-color" content="#0ef7ff">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="AlphaEdge">
<link rel="apple-touch-icon" href="/assets/icon-v2-192.png">
<link rel="icon" type="image/png" sizes="192x192" href="/assets/icon-v2-192.png">
<link rel="icon" type="image/png" sizes="512x512" href="/assets/icon-v2-512.png">
<style>
  :root{
    --bg:#02030a; --bg2:#050814;
    --cyan:#0ef7ff; --cyan-dim:#0ef7ff55;
    --magenta:#ff2fd0; --amber:#ffb020; --green:#00ffa8; --red:#ff2b4e;
    --panel:rgba(8,16,32,0.72); --panel-border:rgba(14,247,255,0.25);
    --text:#d9f9ff; --sub:#5f8fa3;
    --font-head:'Orbitron',sans-serif;
    --font-mono:'JetBrains Mono','Share Tech Mono',monospace;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  html,body{background:var(--bg);color:var(--text);font-family:var(--font-mono);overflow-x:hidden;}
  body{min-height:100vh;position:relative;}

  #starfield{position:fixed;inset:0;z-index:0;}
  .grid-overlay{
    position:fixed;inset:0;z-index:1;pointer-events:none;
    background-image:linear-gradient(rgba(14,247,255,0.045) 1px, transparent 1px),
      linear-gradient(90deg, rgba(14,247,255,0.045) 1px, transparent 1px);
    background-size:42px 42px;
    mask-image:radial-gradient(ellipse 90% 70% at 50% 0%, black 30%, transparent 90%);
  }
  .scanlines{
    position:fixed;inset:0;z-index:2;pointer-events:none;opacity:.5;
    background:repeating-linear-gradient(to bottom, rgba(0,0,0,0) 0px, rgba(0,0,0,0) 2px, rgba(14,247,255,0.02) 3px, rgba(0,0,0,0) 4px);
    animation:scan 9s linear infinite;
  }
  @keyframes scan{0%{background-position-y:0}100%{background-position-y:400px}}
  .vignette{position:fixed;inset:0;z-index:2;pointer-events:none;
    background:radial-gradient(ellipse 80% 80% at 50% 40%, transparent 40%, #02030a 100%);}

  .wrap{position:relative;z-index:3;max-width:1560px;margin:0 auto;padding:22px 26px 60px;}

  header{
    display:flex;justify-content:space-between;align-items:flex-end;
    border-bottom:1px solid var(--panel-border);
    padding-bottom:16px;margin-bottom:14px;flex-wrap:wrap;gap:14px;
  }
  .brand{display:flex;flex-direction:column;gap:6px;}
  .brand-row{display:flex;align-items:center;gap:12px;}
  .logo-hex{
    width:34px;height:34px;position:relative;
    background:conic-gradient(from 90deg, var(--cyan), var(--magenta), var(--cyan));
    clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
    animation:spin 6s linear infinite;
    box-shadow:0 0 18px var(--cyan-dim);
  }
  @keyframes spin{to{transform:rotate(360deg)}}
  h1{
    font-family:var(--font-head);font-weight:900;font-size:26px;letter-spacing:3px;
    color:#eafffe;text-shadow:0 0 6px var(--cyan), 0 0 22px var(--cyan-dim);
  }
  .tagline{font-size:10.5px;letter-spacing:3px;color:var(--sub);text-transform:uppercase;padding-left:46px;}
  .badge-row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;}
  .badge{
    font-size:10px;letter-spacing:1.4px;padding:5px 11px;border-radius:3px;
    border:1px solid;text-transform:uppercase;font-weight:600;
    display:flex;align-items:center;gap:6px;
  }
  .badge-live{color:var(--green);border-color:#00ffa866;background:#00ffa811;}
  .badge-scan{color:var(--sub);border-color:#5f8fa355;background:#5f8fa311;}
  .badge-scan.amber{color:var(--amber);border-color:#ffb02066;background:#ffb02011;}
  .badge-scan.red{color:var(--red);border-color:#ff2b4e66;background:#ff2b4e11;}
  .badge-scan.green{color:var(--green);border-color:#00ffa866;background:#00ffa811;}
  .dot{width:6px;height:6px;border-radius:50%;background:currentColor;box-shadow:0 0 8px currentColor;animation:pulse 2s ease-in-out infinite;}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
  .clock{font-family:var(--font-mono);font-size:13px;color:var(--cyan);text-align:right;letter-spacing:1px;}
  .clock .date{display:block;font-size:10px;color:var(--sub);letter-spacing:2px;margin-top:2px;}

  .maintabs{display:flex;gap:8px;margin-bottom:22px;border-bottom:1px solid var(--panel-border);}
  .maintab-btn{
    font-family:var(--font-head);font-size:11px;letter-spacing:2px;padding:11px 20px;
    background:none;border:none;border-bottom:2px solid transparent;color:var(--sub);
    cursor:pointer;text-transform:uppercase;transition:.15s;
  }
  .maintab-btn:hover{color:var(--text);}
  .maintab-btn.active{color:var(--cyan);border-bottom-color:var(--cyan);text-shadow:0 0 8px var(--cyan-dim);}
  .tabpage{display:none;}
  .tabpage.active{display:block;}

  .panel{
    position:relative;background:var(--panel);
    border:1px solid var(--panel-border);border-radius:6px;
    backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
    padding:18px 20px;
  }
  .panel::before,.panel::after{
    content:'';position:absolute;width:14px;height:14px;border:2px solid var(--cyan);opacity:.7;
  }
  .panel::before{top:-1px;left:-1px;border-right:none;border-bottom:none;}
  .panel::after{bottom:-1px;right:-1px;border-left:none;border-top:none;}
  .panel-title{
    font-family:var(--font-head);font-size:11.5px;letter-spacing:2.5px;color:var(--cyan);
    text-transform:uppercase;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;
  }
  .panel-title .n{color:var(--sub);font-family:var(--font-mono);font-weight:400;letter-spacing:1px;font-size:10px;}
  .src-tag{
    font-size:8px;letter-spacing:.5px;color:var(--green);border:1px solid #00ffa855;
    background:#00ffa80d;border-radius:2px;padding:2px 6px;text-transform:none;font-weight:600;
  }

  .kpi-strip{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin-bottom:22px;}
  .kpi{padding:16px 16px 14px;}
  .kpi-label{font-size:9.5px;letter-spacing:1.8px;color:var(--sub);text-transform:uppercase;margin-bottom:8px;}
  .kpi-val{font-family:var(--font-head);font-size:21px;font-weight:700;letter-spacing:.5px;}
  .kpi-sub{font-size:10px;color:var(--sub);margin-top:5px;letter-spacing:.5px;}
  .c-cyan{color:var(--cyan);text-shadow:0 0 10px var(--cyan-dim);}
  .c-green{color:var(--green);text-shadow:0 0 10px #00ffa855;}
  .c-red{color:var(--red);text-shadow:0 0 10px #ff2b4e55;}
  .c-amber{color:var(--amber);text-shadow:0 0 10px #ffb02055;}
  .c-magenta{color:var(--magenta);text-shadow:0 0 10px #ff2fd055;}
  .c-sub{color:var(--sub);}

  .main-grid{display:grid;grid-template-columns:1.6fr 1fr;gap:18px;margin-bottom:18px;align-items:start;}
  .stack{display:flex;flex-direction:column;gap:18px;}

  .toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;align-items:center;}
  .search-box{
    flex:1;min-width:140px;background:rgba(4,10,22,0.7);border:1px solid var(--panel-border);
    border-radius:4px;padding:7px 12px;color:var(--text);font-family:var(--font-mono);
    font-size:11.5px;outline:none;
  }
  .search-box:focus{border-color:var(--cyan);}
  .search-box::placeholder{color:var(--sub);}
  select.sort-select{
    background:rgba(4,10,22,0.7);border:1px solid var(--panel-border);border-radius:4px;
    color:var(--cyan);font-family:var(--font-mono);font-size:10.5px;padding:7px 10px;outline:none;cursor:pointer;
  }
  select.sort-select option{background:#050814;}

  .filter-row{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;}
  .filter-btn{
    font-family:var(--font-mono);font-size:9.5px;letter-spacing:1px;padding:5px 10px;
    border-radius:3px;border:1px solid var(--panel-border);background:transparent;color:var(--sub);
    cursor:pointer;text-transform:uppercase;transition:.15s;
  }
  .filter-btn:hover,.filter-btn.active{background:var(--cyan-dim);color:var(--cyan);border-color:var(--cyan);}

  .sig-grid{
    display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:9px;
    max-height:660px;overflow-y:auto;padding-right:4px;
  }
  .sig-grid::-webkit-scrollbar{width:5px;}
  .sig-grid::-webkit-scrollbar-thumb{background:var(--panel-border);border-radius:3px;}
  .sig-tile{
    border-radius:5px;padding:10px 11px 9px;border:1px solid var(--panel-border);
    background:rgba(4,10,22,0.6);position:relative;overflow:hidden;transition:transform .15s;cursor:pointer;
  }
  .sig-tile:hover{transform:translateY(-2px);border-color:var(--cyan);}
  .sig-tile.buy{border-color:var(--green);box-shadow:0 0 14px #00ffa833;animation:tilepulse 2.4s ease-in-out infinite;}
  @keyframes tilepulse{0%,100%{box-shadow:0 0 10px #00ffa822;}50%{box-shadow:0 0 20px #00ffa855;}}
  .sig-tile.avoid{border-color:#ff2b4e55;}
  .sig-tile.vetoed{border-color:var(--magenta);opacity:.85;}
  .sig-top{display:flex;justify-content:space-between;align-items:flex-start;}
  .sig-sym{font-family:var(--font-head);font-size:13px;font-weight:700;letter-spacing:.5px;color:#eafffe;}
  .sig-chg{font-size:9.5px;font-weight:700;font-family:var(--font-mono);}
  .sig-sec{font-size:8.5px;color:var(--sub);letter-spacing:.5px;text-transform:uppercase;margin-top:1px;margin-bottom:6px;}
  .sig-bar-bg{height:3px;border-radius:2px;background:#0e1c2e;margin-bottom:6px;overflow:hidden;}
  .sig-bar-fill{height:100%;border-radius:2px;}
  .sig-pred{font-size:11px;font-weight:700;font-family:var(--font-mono);}
  .sig-tag{
    display:inline-block;margin-top:6px;font-size:8px;letter-spacing:.8px;font-weight:700;
    padding:2px 6px;border-radius:2px;text-transform:uppercase;
  }
  .tag-BUY{background:#00ffa822;color:var(--green);}
  .tag-HOLD{background:#5f8fa322;color:var(--sub);}
  .tag-AVOID{background:#ff2b4e22;color:var(--red);}
  .tag-CAUTION{background:#ffb02022;color:var(--amber);}
  .tag-VETOED{background:#ff2fd022;color:var(--magenta);}
  .tag-MTF_HOLD{background:#0ef7ff18;color:var(--cyan);}
  .tag-CORR_HOLD{background:#0ef7ff18;color:var(--cyan);}
  .tag-EARNINGS_HOLD{background:#ffb02022;color:var(--amber);}

  .range-wrap{margin-top:7px;}
  .range-lbl{display:flex;justify-content:space-between;font-size:7.5px;color:var(--sub);margin-bottom:2px;letter-spacing:.3px;}
  .range-track{position:relative;height:5px;border-radius:3px;background:linear-gradient(90deg,#ff2b4e44,#0e1c2e 30%,#0e1c2e 70%,#00ffa844);overflow:visible;}
  .range-dot{
    position:absolute;top:50%;width:7px;height:7px;border-radius:50%;background:var(--cyan);
    transform:translate(-50%,-50%);box-shadow:0 0 6px var(--cyan);border:1px solid #eafffe;
  }

  .radar-wrap{display:flex;justify-content:center;padding:6px 0 2px;}
  .radar-legend{display:flex;justify-content:center;gap:16px;margin-top:6px;font-size:9.5px;letter-spacing:1px;color:var(--sub);}
  .legend-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px;vertical-align:middle;}
  .radar-hint{text-align:center;font-size:9px;color:var(--sub);margin-top:4px;letter-spacing:.5px;}
  .radar-hint b{color:var(--cyan);}

  .gauge-row{display:flex;align-items:center;gap:20px;}
  .gauge-readout{flex:1;}
  .gauge-line{display:flex;justify-content:space-between;font-size:10.5px;padding:5px 0;border-bottom:1px dashed rgba(14,247,255,0.12);}
  .gauge-line .l{color:var(--sub);letter-spacing:.5px;}
  .gauge-line .v{font-weight:700;}

  .shield-status{
    display:flex;align-items:center;gap:12px;padding:10px 12px;border-radius:5px;
    background:#00ffa80d;border:1px solid #00ffa844;margin-bottom:14px;
  }
  .shield-status.tripped{background:#ff2b4e0d;border-color:#ff2b4e66;}
  .shield-icon{font-size:20px;}
  .risk-bar-row{margin-bottom:11px;}
  .risk-bar-label{display:flex;justify-content:space-between;font-size:9.5px;color:var(--sub);margin-bottom:4px;letter-spacing:.5px;}
  .risk-bar-bg{height:6px;border-radius:3px;background:#0e1c2e;overflow:hidden;position:relative;}
  .risk-bar-fill{height:100%;border-radius:3px;position:relative;}
  .risk-bar-fill::after{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.35),transparent);animation:shimmer 2.4s linear infinite;}
  @keyframes shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}

  table{width:100%;border-collapse:collapse;font-size:11.5px;}
  th{
    text-align:left;font-size:9px;letter-spacing:1.4px;color:var(--sub);text-transform:uppercase;
    padding:6px 10px;border-bottom:1px solid var(--panel-border);
  }
  td{padding:8px 10px;border-bottom:1px solid rgba(14,247,255,0.07);}
  tr:last-child td{border-bottom:none;}
  .sym-cell{font-family:var(--font-head);font-weight:700;color:#eafffe;letter-spacing:.5px;}
  .empty-row td{color:var(--sub);text-align:center;padding:20px 10px;font-size:11px;letter-spacing:.5px;}
  .tbl-scroll{max-height:640px;overflow-y:auto;}

  .ticker-wrap{overflow:hidden;border-top:1px solid var(--panel-border);border-bottom:1px solid var(--panel-border);
    padding:10px 0;margin:22px 0;background:rgba(4,10,22,.5);}
  .ticker-track{display:flex;gap:48px;white-space:nowrap;animation:ticker 30s linear infinite;width:max-content;}
  @keyframes ticker{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
  .tick-item{font-size:12px;letter-spacing:.5px;display:flex;gap:8px;align-items:center;}
  .tick-item .lbl{color:var(--sub);}

  .regime-row{display:flex;align-items:center;gap:10px;margin-bottom:10px;}
  .regime-name{width:78px;font-size:10px;color:var(--sub);text-transform:uppercase;letter-spacing:1px;}
  .regime-track{flex:1;height:14px;background:#0e1c2e;border-radius:3px;overflow:hidden;}
  .regime-fill{height:100%;display:flex;align-items:center;justify-content:flex-end;padding-right:6px;font-size:9px;font-weight:700;}

  .mover-col-title{font-size:9px;letter-spacing:1.4px;color:var(--sub);text-transform:uppercase;margin-bottom:8px;}
  .mover-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid rgba(14,247,255,0.06);font-size:11.5px;cursor:pointer;}
  .mover-row:hover{background:rgba(14,247,255,0.04);}
  .mover-row:last-child{border-bottom:none;}
  .movers-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;}

  .modal-backdrop{
    position:fixed;inset:0;z-index:50;background:rgba(2,3,10,0.82);backdrop-filter:blur(4px);
    display:none;align-items:center;justify-content:center;padding:20px;
  }
  .modal-backdrop.open{display:flex;}
  .modal{
    width:100%;max-width:600px;max-height:88vh;overflow-y:auto;
    background:#070c18;border:1px solid var(--cyan);border-radius:8px;
    box-shadow:0 0 40px rgba(14,247,255,0.25);padding:26px 28px;position:relative;
  }
  .modal::before,.modal::after{
    content:'';position:absolute;width:18px;height:18px;border:2px solid var(--cyan);opacity:.8;
  }
  .modal::before{top:-1px;left:-1px;border-right:none;border-bottom:none;}
  .modal::after{bottom:-1px;right:-1px;border-left:none;border-top:none;}
  .modal-close{
    position:absolute;top:14px;right:16px;background:none;border:none;color:var(--sub);
    font-size:20px;cursor:pointer;font-family:var(--font-mono);line-height:1;
  }
  .modal-close:hover{color:var(--red);}
  .modal-head{display:flex;align-items:baseline;gap:14px;margin-bottom:4px;flex-wrap:wrap;}
  .modal-sym{font-family:var(--font-head);font-size:28px;font-weight:900;color:#eafffe;text-shadow:0 0 10px var(--cyan-dim);}
  .modal-price{font-size:20px;font-weight:700;}
  .modal-sub{font-size:10.5px;color:var(--sub);letter-spacing:1px;text-transform:uppercase;margin-bottom:18px;}
  .modal-section-title{font-size:10px;letter-spacing:2px;color:var(--cyan);text-transform:uppercase;margin:18px 0 10px;border-bottom:1px solid var(--panel-border);padding-bottom:6px;}
  .modal-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
  .modal-stat{background:rgba(255,255,255,0.02);border:1px solid var(--panel-border);border-radius:5px;padding:10px 12px;}
  .modal-stat-lbl{font-size:9px;color:var(--sub);letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;}
  .modal-stat-val{font-family:var(--font-head);font-size:15px;font-weight:700;}
  .modal-range-track{position:relative;height:8px;border-radius:4px;background:linear-gradient(90deg,#ff2b4e33,#0e1c2e 25%,#0e1c2e 75%,#00ffa833);margin:14px 0 6px;}
  .modal-range-dot{position:absolute;top:50%;width:12px;height:12px;border-radius:50%;background:var(--cyan);transform:translate(-50%,-50%);box-shadow:0 0 10px var(--cyan);border:2px solid #eafffe;}
  .modal-range-lbls{display:flex;justify-content:space-between;font-size:9.5px;color:var(--sub);}

  footer{
    margin-top:30px;padding-top:16px;border-top:1px solid var(--panel-border);
    font-size:10px;color:var(--sub);letter-spacing:.5px;text-align:center;line-height:1.7;
  }
  footer b{color:var(--amber);}

  @media(max-width:1100px){
    .kpi-strip{grid-template-columns:repeat(3,1fr);}
    .main-grid{grid-template-columns:1fr;}
    .movers-grid{grid-template-columns:1fr;}
  }
  @media(max-width:600px){
    .kpi-strip{grid-template-columns:repeat(2,1fr);}
  }
</style>
</head>
<body>

<canvas id="starfield"></canvas>
<div class="grid-overlay"></div>
<div class="scanlines"></div>
<div class="vignette"></div>

<div class="wrap">

  <header>
    <div class="brand">
      <div class="brand-row">
        <div class="logo-hex"></div>
        <h1>ALPHA EDGE // NEURAL TRADING CORE</h1>
      </div>
      <div class="tagline">Autonomous Signal Intelligence &middot; Multi-Factor Regime Engine</div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:8px;">
      <div class="badge-row">
        <span class="badge badge-live"><span class="dot"></span>LIVE &middot; Alpaca Paper Trading</span>
        <span class="badge badge-scan" id="scanBadge"><span class="dot"></span><span id="scanBadgeText">Loading...</span></span>
      </div>
      <div class="clock"><span id="clockTime">--:--:--</span><span class="date" id="clockDate">-- --- ----</span></div>
    </div>
  </header>

  <div class="maintabs">
    <button class="maintab-btn active" data-tab="dashboard">Trading Dashboard</button>
    <button class="maintab-btn" data-tab="history">Trade History</button>
    <button class="maintab-btn" data-tab="earnings">Earnings Calendar</button>
  </div>

  <!-- ══════════════════════════════════ TAB 1: TRADING DASHBOARD ══ -->
  <div class="tabpage active" id="tabDashboard">

  <div class="kpi-strip">
    <div class="panel kpi">
      <div class="kpi-label">Portfolio Value</div>
      <div class="kpi-val c-cyan" id="kpiValue">--</div>
      <div class="kpi-sub" id="kpiValueSub">--</div>
    </div>
    <div class="panel kpi">
      <div class="kpi-label">Total P&amp;L</div>
      <div class="kpi-val" id="kpiPnl">--</div>
      <div class="kpi-sub" id="kpiPnlSub">--</div>
    </div>
    <div class="panel kpi">
      <div class="kpi-label">Win Rate</div>
      <div class="kpi-val c-cyan" id="kpiWr">--</div>
      <div class="kpi-sub" id="kpiWrSub">--</div>
    </div>
    <div class="panel kpi">
      <div class="kpi-label">Model AUC (Training)</div>
      <div class="kpi-val c-amber" id="kpiAuc">--</div>
      <div class="kpi-sub" id="kpiAucSub">--</div>
    </div>
    <div class="panel kpi">
      <div class="kpi-label">Circuit Breaker</div>
      <div class="kpi-val c-green" id="kpiCb">--</div>
      <div class="kpi-sub" id="kpiCbSub">--</div>
    </div>
    <div class="panel kpi">
      <div class="kpi-label">Open Positions</div>
      <div class="kpi-val c-magenta" id="kpiPos">--</div>
      <div class="kpi-sub" id="kpiPosSub">--</div>
    </div>
  </div>

  <div class="main-grid">
    <div class="panel">
      <div class="panel-title">Signal Matrix <span class="n" id="sigCount">-- symbols tracked</span> <span class="src-tag">SOURCE: logs/latest_signals.json</span></div>
      <div class="toolbar">
        <input class="search-box" id="searchBox" type="text" placeholder="Search symbol or sector... ( / )">
        <select class="sort-select" id="sortSelect">
          <option value="combined">Sort: Combined Score</option>
          <option value="prediction">Sort: ML Prediction</option>
          <option value="alpha">Sort: A &rarr; Z</option>
          <option value="change">Sort: Day % Change</option>
          <option value="range">Sort: Closest to 52w High</option>
        </select>
      </div>
      <div class="filter-row" id="filterRow"></div>
      <div class="sig-grid" id="sigGrid"></div>
    </div>

    <div class="stack">
      <div class="panel">
        <div class="panel-title">Sector Flow Radar <span class="src-tag">SOURCE: logs/sectors.json</span></div>
        <div class="radar-wrap"><svg id="radarSvg" width="300" height="300"></svg></div>
        <div class="radar-legend">
          <span><span class="legend-dot" style="background:var(--green)"></span>Inflow</span>
          <span><span class="legend-dot" style="background:var(--sub)"></span>Neutral</span>
          <span><span class="legend-dot" style="background:var(--red)"></span>Outflow</span>
        </div>
        <div class="radar-hint">Click a sector node to filter the Signal Matrix — currently <b id="sectorActiveLbl">ALL SECTORS</b></div>
      </div>

      <div class="panel">
        <div class="panel-title">Model Health <span class="src-tag">SOURCE: logs/model_auc.json</span></div>
        <div class="gauge-row">
          <svg id="gaugeSvg" width="120" height="120"></svg>
          <div class="gauge-readout">
            <div class="gauge-line"><span class="l">Training AUC</span><span class="v c-cyan" id="aucTrain">--</span></div>
            <div class="gauge-line"><span class="l">Live AUC</span><span class="v c-amber" id="aucLive">--</span></div>
            <div class="gauge-line"><span class="l">Degradation</span><span class="v" id="aucDelta">--</span></div>
            <div class="gauge-line" style="border-bottom:none;"><span class="l">Symbols Retrained</span><span class="v c-cyan" id="aucN">--</span></div>
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-title">Risk Shield <span class="src-tag">SOURCE: logs/circuit_breaker.json</span></div>
        <div class="shield-status" id="shieldStatus">
          <div class="shield-icon" id="shieldIcon">◆</div>
          <div>
            <div style="font-family:var(--font-head);font-size:12px;letter-spacing:1px;" id="shieldText">--</div>
            <div style="font-size:9.5px;color:var(--sub);margin-top:2px;" id="shieldSub">--</div>
          </div>
        </div>
        <div class="risk-bar-row">
          <div class="risk-bar-label"><span>Daily Drawdown</span><span id="ddDailyLbl">--</span></div>
          <div class="risk-bar-bg"><div class="risk-bar-fill" id="ddDailyBar" style="background:var(--green);width:0%"></div></div>
        </div>
        <div class="risk-bar-row">
          <div class="risk-bar-label"><span>Weekly Drawdown</span><span id="ddWeeklyLbl">--</span></div>
          <div class="risk-bar-bg"><div class="risk-bar-fill" id="ddWeeklyBar" style="background:var(--green);width:0%"></div></div>
        </div>
        <div class="risk-bar-row" style="margin-bottom:0;">
          <div class="risk-bar-label"><span>Drawdown From Peak</span><span id="ddPeakLbl">--</span></div>
          <div class="risk-bar-bg"><div class="risk-bar-fill" id="ddPeakBar" style="background:var(--green);width:0%"></div></div>
        </div>
      </div>
    </div>
  </div>

  <div class="ticker-wrap">
    <div class="ticker-track" id="tickerTrack"></div>
  </div>

  <div class="main-grid">
    <div class="panel">
      <div class="panel-title">Open Positions <span class="n" id="posCount">--</span> <span class="src-tag">SOURCE: logs/paper_trades_stocks_only.json</span></div>
      <table>
        <thead><tr><th>Symbol</th><th>Shares</th><th>Entry</th><th>Market</th><th>Unrealized</th><th>Signal Strength</th></tr></thead>
        <tbody id="posBody"></tbody>
      </table>
    </div>
    <div class="panel">
      <div class="panel-title">Regime Distribution</div>
      <div id="regimeBars"></div>
    </div>
  </div>

  <div class="panel" style="margin-bottom:18px;">
    <div class="panel-title">Market Movers <span class="n">by day % change</span> <span class="src-tag">SOURCE: AlphaEdge's own price fetcher (yfinance/Polygon)</span></div>
    <div class="movers-grid">
      <div>
        <div class="mover-col-title" style="color:var(--green)">▲ Top Gainers</div>
        <div id="gainersList"></div>
      </div>
      <div>
        <div class="mover-col-title" style="color:var(--red)">▼ Top Losers</div>
        <div id="losersList"></div>
      </div>
    </div>
  </div>

  <footer>
    Live AlphaEdge trading system &mdash; this is the single dashboard, replacing the three that used to exist separately.<br>
    Every panel reads directly from AlphaEdge's own <span style="color:var(--cyan)">logs/*.json</span> files, refreshed every 60 seconds. Market Movers uses AlphaEdge's own price fetcher on a 5-minute cache and may show "no data" briefly after a restart while the first fetch completes. Model AUC shows training-time AUC only — the live pipeline does not currently compute a separate live/recent AUC. Click any signal tile for detail. Paper trading only. Not investment advice.
  </footer>

  </div>
  <!-- ══════════════════════════════════ /TAB 1 ══════════════════════ -->

  <!-- ══════════════════════════════════ TAB 2: TRADE HISTORY ══════ -->
  <div class="tabpage" id="tabHistory">
    <div class="panel">
      <div class="panel-title">Trade Execution Log <span class="n" id="histCount">-- trades</span> <span class="src-tag">SOURCE: logs/paper_trades_stocks_only.json</span></div>
      <div class="tbl-scroll">
        <table>
          <thead><tr><th>Date</th><th>Action</th><th>Symbol</th><th>Shares</th><th>Fill Price</th><th>P&amp;L</th><th>P&amp;L %</th><th>Reason</th></tr></thead>
          <tbody id="histBody"></tbody>
        </table>
      </div>
    </div>
    <footer>Last 50 trades, most recent first.</footer>
  </div>
  <!-- ══════════════════════════════════ /TAB 2 ══════════════════════ -->

  <!-- ══════════════════════════════════ TAB 3: EARNINGS CALENDAR ══ -->
  <div class="tabpage" id="tabEarnings">
    <div class="panel">
      <div class="panel-title">Earnings Safety Calendar <span class="n" id="earnCount">-- upcoming</span> <span class="src-tag">SOURCE: logs/earnings.json</span></div>
      <table>
        <thead><tr><th>Symbol</th><th>Date</th><th>Days Until</th><th>Time</th><th>Risk Level</th><th>EPS Est</th></tr></thead>
        <tbody id="earnBody"></tbody>
      </table>
    </div>
    <footer>AlphaEdge avoids opening new positions within 2 days of an earnings release.</footer>
  </div>
  <!-- ══════════════════════════════════ /TAB 3 ══════════════════════ -->

</div>

<div class="modal-backdrop" id="modalBackdrop">
  <div class="modal" id="modalBody"></div>
</div>

<script>
/* ══════════════════════════════════════════════════════════════════
   LIVE STATE — populated from /api/data, refreshed every 60s.
   ══════════════════════════════════════════════════════════════════ */
let PORTFOLIO = {capital:10000, starting_capital:10000, positions:{}, total:10000, pos_value:0};
let CLOSED_TRADES = {trades:[], summary:{total:0,wins:0,losses:0,win_rate:null,total_pnl:0,profit_factor:null}};
let CIRCUIT_BREAKER = {triggered:false, peak_value:10000, daily_start_val:10000, weekly_start_val:10000, baseline_established:false};
let MODEL_AUC = {training_auc:null, model_auc:null, n_symbols_retrained:null};
let SECTORS = {};
let SIGNALS = {};
let QUOTES = {};
let HISTORY = [];
let EARNINGS = [];
let META = {};

/* ══════════════════════════════════════════════════════════════════
   STARFIELD BACKGROUND (purely visual, no data dependency)
   ══════════════════════════════════════════════════════════════════ */
(function(){
  const c = document.getElementById('starfield');
  const ctx = c.getContext && c.getContext('2d');
  if(!ctx) return;
  let w,h,stars=[];
  function resize(){
    w = c.width = window.innerWidth;
    h = c.height = window.innerHeight;
    stars = [];
    const n = Math.floor((w*h)/9000);
    for(let i=0;i<n;i++){
      stars.push({
        x:Math.random()*w, y:Math.random()*h,
        r:Math.random()*1.3+0.2, s:Math.random()*0.4+0.05,
        tw:Math.random()*Math.PI*2,
        hue: Math.random()<0.15 ? '176,247,255' : '160,200,220'
      });
    }
  }
  function draw(){
    ctx.fillStyle='#02030a';
    ctx.fillRect(0,0,w,h);
    for(const st of stars){
      st.y += st.s; st.tw += 0.02;
      if(st.y>h){st.y=0;st.x=Math.random()*w;}
      const a = 0.4 + Math.sin(st.tw)*0.35;
      ctx.beginPath();
      ctx.fillStyle = `rgba(${st.hue},${Math.max(0.05,a)})`;
      ctx.arc(st.x, st.y, st.r, 0, Math.PI*2);
      ctx.fill();
    }
    requestAnimationFrame(draw);
  }
  window.addEventListener('resize', resize);
  resize(); draw();
})();

function tickClock(){
  const now = new Date();
  document.getElementById('clockTime').textContent = now.toLocaleTimeString('en-US',{hour12:false});
  document.getElementById('clockDate').textContent = now.toLocaleDateString('en-US',{weekday:'short',year:'numeric',month:'short',day:'2-digit'}).toUpperCase();
}
tickClock(); setInterval(tickClock,1000);

/* ══════════════════════════════════════════════════════════════════
   MAIN TAB SWITCHING
   ══════════════════════════════════════════════════════════════════ */
const TAB_ID_MAP = {dashboard:'tabDashboard', history:'tabHistory', earnings:'tabEarnings'};
document.querySelectorAll('.maintab-btn').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    document.querySelectorAll('.maintab-btn').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tabpage').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(TAB_ID_MAP[btn.dataset.tab]).classList.add('active');
  });
});

/* ══════════════════════════════════════════════════════════════════
   KPI STRIP
   ══════════════════════════════════════════════════════════════════ */
function renderKPIs(){
  const total = PORTFOLIO.total;
  const pnl = total - PORTFOLIO.starting_capital;
  const pnlPct = PORTFOLIO.starting_capital ? pnl/PORTFOLIO.starting_capital*100 : 0;

  document.getElementById('kpiValue').textContent = '$' + total.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
  document.getElementById('kpiValueSub').textContent = `Cash $${PORTFOLIO.capital.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})} · Invested $${PORTFOLIO.pos_value.toFixed(2)}`;

  const pnlEl = document.getElementById('kpiPnl');
  pnlEl.textContent = (pnl>=0?'+':'') + '$' + pnl.toFixed(2);
  pnlEl.className = 'kpi-val ' + (pnl>=0?'c-green':'c-red');
  document.getElementById('kpiPnlSub').textContent = `${pnl>=0?'+':''}${pnlPct.toFixed(2)}% vs start`;

  const wrEl = document.getElementById('kpiWr');
  const wrSubEl = document.getElementById('kpiWrSub');
  if(CLOSED_TRADES.summary.total === 0){
    wrEl.textContent = 'N/A';
    wrEl.className = 'kpi-val c-sub';
    wrSubEl.textContent = 'No closed trades yet';
  } else {
    wrEl.textContent = (CLOSED_TRADES.summary.win_rate*100).toFixed(0)+'%';
    wrEl.className = 'kpi-val c-cyan';
    wrSubEl.textContent = `${CLOSED_TRADES.summary.wins}W / ${CLOSED_TRADES.summary.losses}L · ${CLOSED_TRADES.summary.total} closed`;
  }

  const aucEl = document.getElementById('kpiAuc');
  const aucSubEl = document.getElementById('kpiAucSub');
  if(MODEL_AUC.training_auc == null){
    aucEl.textContent = 'N/A';
    aucSubEl.textContent = 'Not tracked yet';
  } else {
    aucEl.textContent = MODEL_AUC.training_auc.toFixed(3);
    if(MODEL_AUC.model_auc != null){
      const aucDrop = MODEL_AUC.training_auc - MODEL_AUC.model_auc;
      aucSubEl.textContent = `Live ${MODEL_AUC.model_auc.toFixed(3)} · -${(aucDrop*100).toFixed(1)}pp`;
    } else {
      aucSubEl.textContent = 'Live AUC not tracked by pipeline yet';
    }
  }

  const cbEl = document.getElementById('kpiCb');
  cbEl.textContent = CIRCUIT_BREAKER.triggered ? 'TRIPPED' : 'ARMED';
  cbEl.className = 'kpi-val ' + (CIRCUIT_BREAKER.triggered?'c-red':'c-green');
  document.getElementById('kpiCbSub').textContent = CIRCUIT_BREAKER.triggered ? 'Trading halted' : 'All thresholds nominal';

  document.getElementById('kpiPos').textContent = Object.keys(PORTFOLIO.positions).length + '/' + (META.max_positions||'-');
  const buyCount = Object.values(SIGNALS).filter(s=>s.signal==='BUY').length;
  document.getElementById('kpiPosSub').textContent = `${buyCount} active BUY signal${buyCount===1?'':'s'} this scan`;

  const badge = document.getElementById('scanBadge');
  badge.className = 'badge badge-scan ' + (META.scan_color||'');
  document.getElementById('scanBadgeText').textContent = META.scan_label || 'Loading...';
}

/* ══════════════════════════════════════════════════════════════════
   SIGNAL MATRIX
   ══════════════════════════════════════════════════════════════════ */
const SIG_COLORS = {
  BUY:'var(--green)', HOLD:'var(--sub)', AVOID:'var(--red)',
  CAUTION:'var(--amber)', VETOED:'var(--magenta)', EARNINGS_HOLD:'var(--amber)',
  MTF_HOLD:'var(--cyan)', CORR_HOLD:'var(--cyan)'
};
const REGIME_ICON = {uptrend:'▲',downtrend:'▼',sideways:'▬',volatile:'◈'};

let currentFilter = 'ALL';
let currentSector = null;
let currentSearch = '';
let currentSort = 'combined';

function rangePct(sym){
  const q = QUOTES[sym];
  if(!q) return 50;
  const span = q.w52_high - q.w52_low;
  if(span<=0) return 50;
  return Math.max(0, Math.min(100, (q.price - q.w52_low)/span*100));
}

function sortedEntries(){
  let entries = Object.entries(SIGNALS);
  if(currentSearch){
    const s = currentSearch.toUpperCase();
    entries = entries.filter(([sym,d])=> sym.includes(s) || (d.sector||'').toUpperCase().includes(s));
  }
  if(currentFilter!=='ALL') entries = entries.filter(([,d])=>d.signal===currentFilter);
  if(currentSector) entries = entries.filter(([,d])=>d.sector===currentSector);

  const cmp = {
    combined:(a,b)=>(b[1].combined||0)-(a[1].combined||0),
    prediction:(a,b)=>(b[1].prediction||0)-(a[1].prediction||0),
    alpha:(a,b)=>a[0].localeCompare(b[0]),
    change:(a,b)=>(QUOTES[b[0]]?.change_pct??-999)-(QUOTES[a[0]]?.change_pct??-999),
    range:(a,b)=>rangePct(b[0])-rangePct(a[0]),
  };
  entries.sort(cmp[currentSort] || cmp.combined);
  return entries;
}

function renderSignals(){
  const all = Object.entries(SIGNALS);
  const entries = sortedEntries();
  const grid = document.getElementById('sigGrid');
  grid.innerHTML='';
  for(const [sym,d] of entries){
    const cls = d.signal==='BUY'?'buy':(d.signal==='AVOID'?'avoid':(d.signal==='VETOED'?'vetoed':''));
    const pct = Math.round((d.prediction||0)*100);
    const barColor = SIG_COLORS[d.signal] || 'var(--sub)';
    const q = QUOTES[sym];
    const chg = q ? q.change_pct : null;
    const chgColor = chg==null ? 'var(--sub)' : (chg>=0?'var(--green)':'var(--red)');
    const chgStr = chg==null ? '' : `${chg>=0?'+':''}${chg.toFixed(1)}%`;
    const rPct = q ? rangePct(sym) : null;

    const tile = document.createElement('div');
    tile.className = 'sig-tile '+cls;
    tile.onclick = ()=>openModal(sym);
    tile.innerHTML = `
      <div class="sig-top">
        <div class="sig-sym">${sym}</div>
        <div class="sig-chg" style="color:${chgColor}">${chgStr}</div>
      </div>
      <div class="sig-sec">${d.sector||''}</div>
      <div class="sig-bar-bg"><div class="sig-bar-fill" style="width:${pct}%;background:${barColor}"></div></div>
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span class="sig-pred" style="color:${barColor}">${pct}%</span>
        <span style="font-size:11px;color:var(--sub);" title="${d.regime||''}">${REGIME_ICON[d.regime]||'—'}</span>
      </div>
      <span class="sig-tag tag-${d.signal}">${d.signal||'HOLD'}</span>
      ${q ? `
      <div class="range-wrap">
        <div class="range-lbl"><span>52wL $${q.w52_low.toFixed(0)}</span><span>52wH $${q.w52_high.toFixed(0)}</span></div>
        <div class="range-track"><div class="range-dot" style="left:${rPct}%"></div></div>
      </div>` : ''}
    `;
    grid.appendChild(tile);
  }
  document.getElementById('sigCount').textContent = `${entries.length} / ${all.length} symbols`;
}

function buildFilterRow(){
  const counts = {};
  Object.values(SIGNALS).forEach(d=>{counts[d.signal]=(counts[d.signal]||0)+1;});
  const order = ['ALL','BUY','HOLD','AVOID','CAUTION','VETOED','MTF_HOLD','CORR_HOLD','EARNINGS_HOLD'];
  const row = document.getElementById('filterRow');
  row.innerHTML = '';
  order.forEach(f=>{
    if(f!=='ALL' && !counts[f]) return;
    const btn = document.createElement('button');
    btn.className = 'filter-btn' + (f===currentFilter?' active':'');
    btn.textContent = f + (f==='ALL' ? ` (${Object.keys(SIGNALS).length})` : ` (${counts[f]})`);
    btn.onclick = ()=>{
      currentFilter = f;
      buildFilterRow();
      renderSignals();
    };
    row.appendChild(btn);
  });
}

document.getElementById('searchBox').addEventListener('input', e=>{
  currentSearch = e.target.value.trim();
  renderSignals();
});
document.addEventListener('keydown', e=>{
  if(e.key==='/' && document.activeElement.id!=='searchBox'){
    e.preventDefault();
    document.getElementById('searchBox').focus();
  }
  if(e.key==='Escape') closeModal();
});
document.getElementById('sortSelect').addEventListener('change', e=>{
  currentSort = e.target.value;
  renderSignals();
});

/* ══════════════════════════════════════════════════════════════════
   MODAL
   ══════════════════════════════════════════════════════════════════ */
function openModal(sym){
  const d = SIGNALS[sym];
  if(!d) return;
  const q = QUOTES[sym];
  const backdrop = document.getElementById('modalBackdrop');
  const body = document.getElementById('modalBody');
  const chg = q ? q.change_pct : null;
  const chgColor = chg==null ? 'var(--sub)' : (chg>=0?'var(--green)':'var(--red)');
  const rPct = q ? rangePct(sym) : 50;
  const barColor = SIG_COLORS[d.signal] || 'var(--sub)';

  body.innerHTML = `
    <button class="modal-close" onclick="closeModal()">✕</button>
    <div class="modal-head">
      <div class="modal-sym">${sym}</div>
      ${q ? `<div class="modal-price">$${q.price.toFixed(2)}</div><div style="color:${chgColor};font-weight:700;">${chg>=0?'+':''}${chg.toFixed(2)}%</div>` : ''}
    </div>
    <div class="modal-sub">${d.sector||''} &middot; ${REGIME_ICON[d.regime]||''} ${d.regime||''} regime &middot; <span class="sig-tag tag-${d.signal}" style="margin-top:0;">${d.signal||'HOLD'}</span></div>

    ${q ? `
    <div class="modal-section-title">52-Week Range</div>
    <div class="modal-range-track"><div class="modal-range-dot" style="left:${rPct}%"></div></div>
    <div class="modal-range-lbls"><span>$${q.w52_low.toFixed(2)} (52w low)</span><span>$${q.w52_high.toFixed(2)} (52w high)</span></div>
    <div style="text-align:center;font-size:10px;color:var(--sub);margin-top:8px;">Currently ${rPct.toFixed(0)}% of the way through its 52-week range</div>
    ` : '<div style="font-size:10px;color:var(--sub);margin-top:8px;">No live quote available yet for this symbol.</div>'}

    <div class="modal-section-title">AlphaEdge Model Signal</div>
    <div class="modal-grid">
      <div class="modal-stat">
        <div class="modal-stat-lbl">ML Prediction</div>
        <div class="modal-stat-val" style="color:${barColor}">${((d.prediction||0)*100).toFixed(1)}%</div>
      </div>
      <div class="modal-stat">
        <div class="modal-stat-lbl">Combined Score</div>
        <div class="modal-stat-val c-cyan">${((d.combined||0)*100).toFixed(1)}%</div>
      </div>
      <div class="modal-stat">
        <div class="modal-stat-lbl">Signal at Scan Time</div>
        <div class="modal-stat-val">$${(d.price||0).toFixed(2)}</div>
      </div>
      <div class="modal-stat">
        <div class="modal-stat-lbl">Regime</div>
        <div class="modal-stat-val">${REGIME_ICON[d.regime]||''} ${d.regime||''}</div>
      </div>
    </div>
  `;
  backdrop.classList.add('open');
}
function closeModal(){
  document.getElementById('modalBackdrop').classList.remove('open');
}
document.getElementById('modalBackdrop').addEventListener('click', e=>{
  if(e.target.id==='modalBackdrop') closeModal();
});

/* ══════════════════════════════════════════════════════════════════
   SECTOR RADAR
   ══════════════════════════════════════════════════════════════════ */
function renderRadar(){
  const svg = document.getElementById('radarSvg');
  svg.innerHTML = '';
  const NS = 'http://www.w3.org/2000/svg';
  const cx=150, cy=150, maxR=105;
  const sectors = Object.entries(SECTORS);
  const n = sectors.length;
  if(n===0) return;
  const maxMom = Math.max(...sectors.map(s=>Math.abs(s[1].momentum_21d||0))) * 1.3 || 1;

  function pt(i, val){
    const angle = (Math.PI*2*i/n) - Math.PI/2;
    const r = maxR * (0.15 + 0.85*(val/maxMom));
    return [cx + r*Math.cos(angle), cy + r*Math.sin(angle)];
  }

  [0.25,0.5,0.75,1.0].forEach(f=>{
    const ring = document.createElementNS(NS,'circle');
    ring.setAttribute('cx',cx); ring.setAttribute('cy',cy); ring.setAttribute('r',maxR*f);
    ring.setAttribute('fill','none'); ring.setAttribute('stroke','rgba(14,247,255,0.12)');
    ring.setAttribute('stroke-width','1');
    svg.appendChild(ring);
  });

  sectors.forEach((s,i)=>{
    const [ex,ey] = pt(i, maxMom);
    const line = document.createElementNS(NS,'line');
    line.setAttribute('x1',cx); line.setAttribute('y1',cy);
    line.setAttribute('x2',ex); line.setAttribute('y2',ey);
    line.setAttribute('stroke','rgba(14,247,255,0.15)');
    svg.appendChild(line);

    const angle = (Math.PI*2*i/n) - Math.PI/2;
    const lx = cx + (maxR+20)*Math.cos(angle);
    const ly = cy + (maxR+20)*Math.sin(angle);
    const label = document.createElementNS(NS,'text');
    label.setAttribute('x',lx); label.setAttribute('y',ly);
    label.setAttribute('text-anchor','middle'); label.setAttribute('dominant-baseline','middle');
    label.setAttribute('font-size','8.5'); label.setAttribute('font-family','JetBrains Mono, monospace');
    label.setAttribute('cursor','pointer');
    const flow = s[1].flow;
    label.setAttribute('fill', flow==='INFLOW'?'#00ffa8':flow==='OUTFLOW'?'#ff2b4e':'#5f8fa3');
    label.textContent = s[0].length>10? s[0].slice(0,9)+'…' : s[0];
    label.addEventListener('click', ()=>{
      currentSector = (currentSector===s[0]) ? null : s[0];
      document.getElementById('sectorActiveLbl').textContent = currentSector || 'ALL SECTORS';
      renderSignals();
    });
    svg.appendChild(label);
  });

  const pts = sectors.map((s,i)=> pt(i, (s[1].momentum_21d||0) + maxMom*0.15)).map(p=>p.join(',')).join(' ');
  const poly = document.createElementNS(NS,'polygon');
  poly.setAttribute('points', pts);
  poly.setAttribute('fill','rgba(14,247,255,0.14)');
  poly.setAttribute('stroke','#0ef7ff');
  poly.setAttribute('stroke-width','1.6');
  svg.appendChild(poly);

  sectors.forEach((s,i)=>{
    const [px,py] = pt(i, (s[1].momentum_21d||0) + maxMom*0.15);
    const dot = document.createElementNS(NS,'circle');
    dot.setAttribute('cx',px); dot.setAttribute('cy',py); dot.setAttribute('r','3.5');
    dot.setAttribute('cursor','pointer');
    const flow = s[1].flow;
    dot.setAttribute('fill', flow==='INFLOW'?'#00ffa8':flow==='OUTFLOW'?'#ff2b4e':'#5f8fa3');
    dot.addEventListener('click', ()=>{
      currentSector = (currentSector===s[0]) ? null : s[0];
      document.getElementById('sectorActiveLbl').textContent = currentSector || 'ALL SECTORS';
      renderSignals();
    });
    svg.appendChild(dot);
  });
}

/* ══════════════════════════════════════════════════════════════════
   MODEL HEALTH GAUGE
   ══════════════════════════════════════════════════════════════════ */
function renderGauge(){
  const svg = document.getElementById('gaugeSvg');
  svg.innerHTML = '';
  const NS = 'http://www.w3.org/2000/svg';
  const cx=60, cy=60, r=48;
  const circumference = 2*Math.PI*r;
  const trainAuc = MODEL_AUC.training_auc;
  const liveAuc = MODEL_AUC.model_auc;

  const bg = document.createElementNS(NS,'circle');
  bg.setAttribute('cx',cx); bg.setAttribute('cy',cy); bg.setAttribute('r',r);
  bg.setAttribute('fill','none'); bg.setAttribute('stroke','#0e1c2e'); bg.setAttribute('stroke-width','9');
  svg.appendChild(bg);

  if(trainAuc != null){
    const trainArc = document.createElementNS(NS,'circle');
    trainArc.setAttribute('cx',cx); trainArc.setAttribute('cy',cy); trainArc.setAttribute('r',r);
    trainArc.setAttribute('fill','none'); trainArc.setAttribute('stroke','rgba(14,247,255,0.25)');
    trainArc.setAttribute('stroke-width','9');
    trainArc.setAttribute('stroke-dasharray', `${circumference*trainAuc} ${circumference}`);
    trainArc.setAttribute('stroke-linecap','round');
    trainArc.setAttribute('transform', `rotate(-90 ${cx} ${cy})`);
    svg.appendChild(trainArc);
  }

  if(liveAuc != null){
    const liveArc = document.createElementNS(NS,'circle');
    liveArc.setAttribute('cx',cx); liveArc.setAttribute('cy',cy); liveArc.setAttribute('r',r);
    liveArc.setAttribute('fill','none'); liveArc.setAttribute('stroke','#ffb020');
    liveArc.setAttribute('stroke-width','9');
    liveArc.setAttribute('stroke-dasharray', `${circumference*liveAuc} ${circumference}`);
    liveArc.setAttribute('stroke-linecap','round');
    liveArc.setAttribute('transform', `rotate(-90 ${cx} ${cy})`);
    liveArc.style.filter = 'drop-shadow(0 0 5px #ffb02099)';
    svg.appendChild(liveArc);
  }

  const txt = document.createElementNS(NS,'text');
  txt.setAttribute('x',cx); txt.setAttribute('y',cy+5);
  txt.setAttribute('text-anchor','middle'); txt.setAttribute('font-family','Orbitron, sans-serif');
  txt.setAttribute('font-size','16'); txt.setAttribute('font-weight','700'); txt.setAttribute('fill','#eafffe');
  txt.textContent = liveAuc != null ? liveAuc.toFixed(2) : (trainAuc != null ? trainAuc.toFixed(2) : 'N/A');
  svg.appendChild(txt);

  document.getElementById('aucTrain').textContent = trainAuc != null ? trainAuc.toFixed(4) : 'N/A';
  document.getElementById('aucLive').textContent = liveAuc != null ? liveAuc.toFixed(4) : 'N/A (not tracked)';
  const deltaEl = document.getElementById('aucDelta');
  if(trainAuc != null && liveAuc != null){
    const delta = trainAuc - liveAuc;
    deltaEl.textContent = `-${(delta*100).toFixed(1)}pp`;
    deltaEl.className = 'v ' + (delta>0.15?'c-red':delta>0.08?'c-amber':'c-green');
  } else {
    deltaEl.textContent = 'N/A';
    deltaEl.className = 'v c-sub';
  }
  document.getElementById('aucN').textContent = MODEL_AUC.n_symbols_retrained != null ? MODEL_AUC.n_symbols_retrained : 'N/A';
}

/* ══════════════════════════════════════════════════════════════════
   RISK SHIELD
   ══════════════════════════════════════════════════════════════════ */
function renderShield(){
  const statusEl = document.getElementById('shieldStatus');
  const iconEl = document.getElementById('shieldIcon');
  const textEl = document.getElementById('shieldText');
  const subEl = document.getElementById('shieldSub');

  statusEl.classList.remove('tripped');
  if(CIRCUIT_BREAKER.triggered){
    statusEl.classList.add('tripped');
    iconEl.textContent = '⛔'; iconEl.style.color='var(--red)';
    textEl.textContent = 'CIRCUIT BREAKER TRIPPED'; textEl.style.color='var(--red)';
    subEl.textContent = 'Trading halted pending review';
  } else {
    iconEl.textContent = '◆'; iconEl.style.color='var(--green)';
    textEl.textContent = 'ALL SYSTEMS ARMED'; textEl.style.color='var(--green)';
    subEl.textContent = CIRCUIT_BREAKER.baseline_established ? 'Daily / weekly / drawdown limits nominal' : 'Baseline not yet established (fresh state)';
  }

  const total = PORTFOLIO.total;

  function bar(id,label,pct,limitPct){
    const clamped = Math.min(100, Math.abs(pct)/limitPct*100);
    document.getElementById(id+'Lbl').textContent = `${pct.toFixed(2)}% (limit ${limitPct}%)`;
    const fill = document.getElementById(id+'Bar');
    fill.style.width = clamped+'%';
    fill.style.background = clamped>80?'var(--red)':clamped>50?'var(--amber)':'var(--green)';
  }

  const dailyDD = ((CIRCUIT_BREAKER.daily_start_val - total)/CIRCUIT_BREAKER.daily_start_val)*100;
  const weeklyDD = ((CIRCUIT_BREAKER.weekly_start_val - total)/CIRCUIT_BREAKER.weekly_start_val)*100;
  const peakDD = ((CIRCUIT_BREAKER.peak_value - total)/CIRCUIT_BREAKER.peak_value)*100;

  bar('ddDaily','Daily', dailyDD, 2);
  bar('ddWeekly','Weekly', weeklyDD, 6);
  bar('ddPeak','Peak', peakDD, 10);
}

/* ══════════════════════════════════════════════════════════════════
   POSITIONS TABLE
   ══════════════════════════════════════════════════════════════════ */
function renderPositions(){
  const body = document.getElementById('posBody');
  body.innerHTML = '';
  const entries = Object.entries(PORTFOLIO.positions);
  document.getElementById('posCount').textContent = entries.length + ' open';
  if(entries.length === 0){
    const tr = document.createElement('tr');
    tr.className = 'empty-row';
    tr.innerHTML = `<td colspan="6">No open positions — watching ${META.watchlist_len||'--'} stocks</td>`;
    body.appendChild(tr);
    return;
  }
  entries.forEach(([sym,p])=>{
    const pnl = (p.market_price - p.entry_price) * p.shares;
    const pnlPct = p.entry_price ? (p.market_price - p.entry_price)/p.entry_price*100 : 0;
    const pnlColor = pnl>=0 ? 'var(--green)' : 'var(--red)';
    const strengthPct = Math.round((p.signal||0)*100);
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="sym-cell">${sym}</td>
      <td>${p.shares}</td>
      <td>$${p.entry_price.toFixed(2)}</td>
      <td>$${p.market_price.toFixed(2)}</td>
      <td style="color:${pnlColor}">${pnl>=0?'+':''}$${pnl.toFixed(2)} <span style="color:var(--sub);font-size:10px;">(${pnlPct>=0?'+':''}${pnlPct.toFixed(2)}%)</span></td>
      <td>
        <div style="display:flex;align-items:center;gap:8px;">
          <div class="sig-bar-bg" style="flex:1;margin-bottom:0;"><div class="sig-bar-fill" style="width:${strengthPct}%;background:var(--cyan)"></div></div>
          <span style="font-size:10px;color:var(--cyan);">${strengthPct}%</span>
        </div>
      </td>
    `;
    body.appendChild(tr);
  });
}

/* ══════════════════════════════════════════════════════════════════
   TICKER
   ══════════════════════════════════════════════════════════════════ */
function renderTicker(){
  const track = document.getElementById('tickerTrack');
  const items = [];
  if(CLOSED_TRADES.trades.length === 0){
    items.push(`<div class="tick-item"><span class="lbl">TRADES</span><b>NONE CLOSED YET</b></div>`);
  }
  CLOSED_TRADES.trades.slice(0,20).forEach(t=>{
    const c = t.pnl_usd>=0?'var(--green)':'var(--red)';
    items.push(`<div class="tick-item"><span class="lbl">CLOSED</span><b>${t.symbol}</b><span style="color:${c}">${t.pnl_usd>=0?'+':''}$${t.pnl_usd.toFixed(2)} (${t.pnl_pct>=0?'+':''}${t.pnl_pct.toFixed(2)}%)</span><span class="lbl">${t.reason}</span></div>`);
  });
  Object.entries(SECTORS).forEach(([name,s])=>{
    const c = s.flow==='INFLOW'?'var(--green)':s.flow==='OUTFLOW'?'var(--red)':'var(--sub)';
    items.push(`<div class="tick-item"><span class="lbl">SECTOR</span><b>${name}</b><span style="color:${c}">${s.flow}</span><span class="lbl">${((s.momentum_21d||0)*100).toFixed(2)}% mom</span></div>`);
  });
  Object.entries(QUOTES).slice(0,15).forEach(([sym,q])=>{
    const c = q.change_pct>=0?'var(--green)':'var(--red)';
    items.push(`<div class="tick-item"><span class="lbl">LIVE</span><b>${sym}</b><span style="color:${c}">$${q.price.toFixed(2)} (${q.change_pct>=0?'+':''}${q.change_pct.toFixed(1)}%)</span></div>`);
  });
  items.push(`<div class="tick-item"><span class="lbl">RISK</span><b>CIRCUIT BREAKER</b><span style="color:${CIRCUIT_BREAKER.triggered?'var(--red)':'var(--green)'}">${CIRCUIT_BREAKER.triggered?'TRIPPED':'ARMED · NOMINAL'}</span></div>`);
  if(items.length===0){
    items.push(`<div class="tick-item"><span class="lbl">STATUS</span><b>WAITING ON DATA</b></div>`);
  }
  const html = items.join('');
  track.innerHTML = html + html;
}

/* ══════════════════════════════════════════════════════════════════
   REGIME DISTRIBUTION
   ══════════════════════════════════════════════════════════════════ */
function renderRegimeBars(){
  const counts = {uptrend:0,downtrend:0,sideways:0,volatile:0};
  Object.values(SIGNALS).forEach(d=>{ if(counts[d.regime]!==undefined) counts[d.regime]++; });
  const total = Object.values(counts).reduce((a,b)=>a+b,0);
  const colors = {uptrend:'#00ffa8',downtrend:'#ff2b4e',sideways:'#5f8fa3',volatile:'#ff2fd0'};
  const wrap = document.getElementById('regimeBars');
  wrap.innerHTML = '';
  Object.entries(counts).forEach(([name,c])=>{
    const pct = total? (c/total*100) : 0;
    const row = document.createElement('div');
    row.className = 'regime-row';
    row.innerHTML = `
      <div class="regime-name">${REGIME_ICON[name]} ${name}</div>
      <div class="regime-track"><div class="regime-fill" style="width:${pct}%;background:${colors[name]}22;color:${colors[name]};">${c}</div></div>
      <div style="width:42px;text-align:right;font-size:10.5px;color:var(--sub);">${pct.toFixed(0)}%</div>
    `;
    wrap.appendChild(row);
  });
}

/* ══════════════════════════════════════════════════════════════════
   MARKET MOVERS
   ══════════════════════════════════════════════════════════════════ */
function renderMovers(){
  const gWrap = document.getElementById('gainersList');
  const lWrap = document.getElementById('losersList');
  gWrap.innerHTML = ''; lWrap.innerHTML = '';
  const entries = Object.entries(QUOTES);
  if(entries.length === 0){
    const msg = META.quotes_fetching ? 'Fetching live quotes…' : 'No quote data yet (refreshes every 5 min)';
    gWrap.innerHTML = `<div style="color:var(--sub);font-size:11px;padding:8px 0;">${msg}</div>`;
    lWrap.innerHTML = `<div style="color:var(--sub);font-size:11px;padding:8px 0;">${msg}</div>`;
    return;
  }
  const ranked = entries.sort((a,b)=>b[1].change_pct-a[1].change_pct);
  const gainers = ranked.slice(0,5);
  const losers = ranked.slice(-5).reverse();

  function row(sym,q){
    const c = q.change_pct>=0?'var(--green)':'var(--red)';
    const d = document.createElement('div');
    d.className = 'mover-row';
    d.onclick = ()=>{ if(SIGNALS[sym]) openModal(sym); };
    d.innerHTML = `<span class="sym-cell">${sym}</span><span>$${q.price.toFixed(2)}</span><span style="color:${c};font-weight:700;">${q.change_pct>=0?'+':''}${q.change_pct.toFixed(2)}%</span>`;
    return d;
  }
  gainers.forEach(([sym,q])=>gWrap.appendChild(row(sym,q)));
  losers.forEach(([sym,q])=>lWrap.appendChild(row(sym,q)));
}

/* ══════════════════════════════════════════════════════════════════
   TRADE HISTORY TAB
   ══════════════════════════════════════════════════════════════════ */
function renderHistory(){
  const body = document.getElementById('histBody');
  body.innerHTML = '';
  document.getElementById('histCount').textContent = HISTORY.length + ' trades';
  if(HISTORY.length === 0){
    const tr = document.createElement('tr');
    tr.className = 'empty-row';
    tr.innerHTML = `<td colspan="8">No trades executed yet — system live and watching</td>`;
    body.appendChild(tr);
    return;
  }
  HISTORY.forEach(t=>{
    const pnl = t.pnl;
    const fillPrice = t.fill_price != null ? t.fill_price : (t.price||0);
    const pnlPct = t.pnl_pct != null ? t.pnl_pct*100 : null;
    const pnlColor = pnl>0?'var(--green)':(pnl<0?'var(--red)':'var(--sub)');
    const actionColor = t.action==='BUY'?'var(--green)':(t.action==='SELL'?'var(--red)':'var(--amber)');
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${(t.date||'').slice(0,16)}</td>
      <td style="color:${actionColor};font-weight:700;">${t.action||''}</td>
      <td class="sym-cell">${t.symbol||''}</td>
      <td>${t.shares||0}</td>
      <td>$${fillPrice.toFixed(2)}</td>
      <td style="color:${pnlColor}">${pnl!=null?('$'+pnl.toFixed(2)):'--'}</td>
      <td style="color:${pnlColor}">${pnlPct!=null?(pnlPct.toFixed(2)+'%'):'--'}</td>
      <td style="color:var(--sub);">${(t.reason||'').toUpperCase()}</td>
    `;
    body.appendChild(tr);
  });
}

/* ══════════════════════════════════════════════════════════════════
   EARNINGS CALENDAR TAB
   ══════════════════════════════════════════════════════════════════ */
function renderEarnings(){
  const body = document.getElementById('earnBody');
  body.innerHTML = '';
  document.getElementById('earnCount').textContent = EARNINGS.length + ' upcoming';
  if(EARNINGS.length === 0){
    const tr = document.createElement('tr');
    tr.className = 'empty-row';
    tr.innerHTML = `<td colspan="6">No earnings this week — all clear to trade</td>`;
    body.appendChild(tr);
    return;
  }
  const sorted = [...EARNINGS].sort((a,b)=>(a.days_until||99)-(b.days_until||99));
  sorted.forEach(e=>{
    const days = e.days_until||0;
    const risk = days<=1?'AVOID NOW':(days<=3?'CAUTION':'MONITOR');
    const riskColor = days<=1?'var(--red)':(days<=3?'var(--amber)':'var(--green)');
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="sym-cell">${e.symbol||''}</td>
      <td>${e.date||''}</td>
      <td>${days}</td>
      <td>${e.time||'TBD'}</td>
      <td style="color:${riskColor};font-weight:700;">${risk}</td>
      <td>${e.eps_estimate!=null?e.eps_estimate:'N/A'}</td>
    `;
    body.appendChild(tr);
  });
}

/* ══════════════════════════════════════════════════════════════════
   REFRESH LOOP
   ══════════════════════════════════════════════════════════════════ */
function renderAll(){
  renderKPIs();
  buildFilterRow();
  renderSignals();
  renderRadar();
  renderGauge();
  renderShield();
  renderPositions();
  renderTicker();
  renderRegimeBars();
  renderMovers();
  renderHistory();
  renderEarnings();
}

async function refreshData(){
  try{
    const res = await fetch('/api/data');
    const d = await res.json();
    if(d.error){ console.error('API error:', d.error); return; }
    PORTFOLIO = d.portfolio;
    CLOSED_TRADES = d.closed_trades;
    CIRCUIT_BREAKER = d.circuit_breaker;
    MODEL_AUC = d.model_auc;
    SECTORS = d.sectors;
    SIGNALS = d.signals;
    QUOTES = d.quotes;
    HISTORY = d.history;
    EARNINGS = d.earnings;
    META = d.meta;
    renderAll();
  }catch(e){
    console.error('Failed to refresh dashboard data:', e);
  }
}

refreshData();
setInterval(refreshData, 60000);
</script>
</body>
</html>
"""


if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  ALPHAEDGE // NEURAL TRADING CORE  (live, single dashboard)')
    print('=' * 60)
    print('  http://localhost:8050\n')
    threading.Thread(target=_quotes_refresh_loop, daemon=True).start()
    app = create_app()
    app.run(debug=False, host='0.0.0.0', port=8050)
