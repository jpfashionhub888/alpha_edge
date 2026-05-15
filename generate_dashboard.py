# generate_dashboard.py
# AlphaEdge V5 — Institutional-Grade Dashboard (Full Width)
# Upgrades: Drawdown curve, Weight results tab, Signal sort,
#           Per-symbol P&L chart, Daily returns histogram,
#           Regime badge, Unrealized P&L total, Sector charts,
#           Dark/light mode, CSS bug fixes
# Run: python generate_dashboard.py
# Auto-refreshes every 5 minutes when open in browser

import os
import json
import math
from datetime import datetime

TRADES_FILE   = 'logs/paper_trades.json'
SIGNALS_FILE  = 'logs/latest_signals.json'
SECTORS_FILE  = 'logs/sectors.json'
EARNINGS_FILE = 'logs/earnings.json'
WEIGHTS_FILE  = 'logs/weight_results.json'
DASHBOARD_DIR = 'docs'
DASHBOARD_FILE = f'{DASHBOARD_DIR}/index.html'


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_json(filepath, default):
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception:
        return default


# ── Metric calculators ────────────────────────────────────────────────────────

def calculate_sharpe(trade_history):
    sells   = [t for t in trade_history if t.get('action') == 'SELL']
    returns = [t.get('pnl_pct', 0) for t in sells if 'pnl_pct' in t]
    if len(returns) < 2:
        return 0.0
    n   = len(returns)
    avg = sum(returns) / n
    std = math.sqrt(sum((r - avg) ** 2 for r in returns) / (n - 1))
    return round((avg / std) * math.sqrt(252), 2) if std > 0 else 0.0


def calculate_drawdown_series(trade_history, starting_capital):
    """Returns list of (label, drawdown_pct) for charting."""
    sells  = [t for t in trade_history if t.get('action') == 'SELL']
    if not sells:
        return [0.0], ['Start']
    peak   = starting_capital
    equity = starting_capital
    dds    = [0.0]
    labels = ['Start']
    for t in sells:
        equity += t.get('pnl', 0)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        dds.append(round(dd, 4))
        labels.append(t.get('date', '')[:10])
    return dds, labels


def calculate_max_drawdown(dd_series):
    return round(max(dd_series) / 100, 4) if dd_series else 0.0


def calculate_profit_factor(trade_history):
    sells        = [t for t in trade_history if t.get('action') == 'SELL']
    gross_wins   = sum(t.get('pnl', 0) for t in sells if t.get('pnl', 0) > 0)
    gross_losses = abs(sum(t.get('pnl', 0) for t in sells if t.get('pnl', 0) < 0))
    return round(gross_wins / gross_losses, 2) if gross_losses > 0 else 0.0


def calculate_daily_returns(trade_history):
    """Bucket closed trade P&L % into daily bins for histogram."""
    sells = [t for t in trade_history if t.get('action') == 'SELL']
    by_date = {}
    for t in sells:
        d = t.get('date', '')[:10]
        by_date.setdefault(d, 0)
        by_date[d] += t.get('pnl_pct', 0)
    if not by_date:
        return [], []
    vals   = list(by_date.values())
    labels = list(by_date.keys())
    return vals, labels


def calculate_symbol_pnl(trade_history):
    """Per-symbol realized P&L for bar chart."""
    by_sym = {}
    for t in trade_history:
        if t.get('action') == 'SELL':
            s = t.get('symbol', 'UNK')
            by_sym[s] = by_sym.get(s, 0) + t.get('pnl', 0)
    return dict(sorted(by_sym.items(), key=lambda x: x[1], reverse=True))


# ── Dashboard generator ───────────────────────────────────────────────────────

def generate_dashboard():
    os.makedirs(DASHBOARD_DIR, exist_ok=True)

    portfolio = load_json(TRADES_FILE, {
        'capital': 10000.0, 'starting_capital': 10000.0,
        'positions': {}, 'trade_history': [], 'saved_at': 'Never',
    })
    signals      = load_json(SIGNALS_FILE, {})
    sectors      = load_json(SECTORS_FILE, {})
    earnings     = load_json(EARNINGS_FILE, [])
    weight_data_raw = load_json(WEIGHTS_FILE, [])

    # Handle both list and dict formats
    # Load weight results — handle dict or list format
    weight_data_raw = load_json(WEIGHTS_FILE, [])
    if isinstance(weight_data_raw, dict):
        weight_data = list(weight_data_raw.values())
    elif isinstance(weight_data_raw, list):
        weight_data = weight_data_raw
    else:
        weight_data = []
    weight_data = [w for w in weight_data if isinstance(w, dict)]

    capital   = portfolio.get('capital', 10000)
    starting  = portfolio.get('starting_capital', 10000)
    positions = portfolio.get('positions', {})
    history   = portfolio.get('trade_history', [])
    saved_at  = portfolio.get('saved_at', 'Never')[:16]

    pos_value   = sum(
        p.get('shares', 0) * p.get('current_price', p.get('entry_price', 0))
        for p in positions.values()
    )
    total_value  = capital + pos_value
    total_pnl    = total_value - starting
    total_pct    = (total_pnl / starting) * 100 if starting > 0 else 0
    unrealized   = sum(
        (p.get('current_price', p.get('entry_price', 0)) - p.get('entry_price', 0))
        * p.get('shares', 0)
        for p in positions.values()
    )

    sells     = [t for t in history if t.get('action') == 'SELL']
    wins      = [t for t in sells if t.get('pnl', 0) > 0]
    losses    = [t for t in sells if t.get('pnl', 0) <= 0]
    win_rate  = len(wins) / len(sells) * 100 if sells else 0
    realized  = sum(t.get('pnl', 0) for t in sells)

    dd_series, dd_labels = calculate_drawdown_series(history, starting)
    max_dd    = calculate_max_drawdown(dd_series)
    sharpe    = calculate_sharpe(history)
    pf        = calculate_profit_factor(history)
    avg_win   = sum(t.get('pnl', 0) for t in wins)   / len(wins)   if wins   else 0
    avg_loss  = abs(sum(t.get('pnl', 0) for t in losses) / len(losses)) if losses else 0
    expectancy = round((win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss), 2)

    daily_ret_vals, daily_ret_labels = calculate_daily_returns(history)
    sym_pnl  = calculate_symbol_pnl(history)

    now_str  = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Detect current regime from signals
    regime_counts = {}
    for d in signals.values():
        r = d.get('regime', '')
        if r:
            regime_counts[r] = regime_counts.get(r, 0) + 1
    dominant_regime = max(regime_counts, key=regime_counts.get) if regime_counts else 'unknown'
    regime_icon = {'uptrend': '↑ UPTREND', 'downtrend': '↓ DOWNTREND',
                   'sideways': '→ SIDEWAYS', 'volatile': '⚡ VOLATILE'}.get(dominant_regime, '— UNKNOWN')
    regime_color = {'uptrend': '#00d4aa', 'downtrend': '#f87171',
                    'sideways': '#f0b429', 'volatile': '#a855f7'}.get(dominant_regime, '#6b7280')

    # Equity curve
    chart_vals   = [starting]
    chart_labels = ['Start']
    running = starting
    for t in history:
        if t.get('action') == 'SELL':
            running += t.get('pnl', 0)
            chart_vals.append(round(running, 2))
            chart_labels.append(t.get('date', '')[:10])
    chart_vals.append(round(total_value, 2))
    chart_labels.append('Now')

    # Allocation donut
    alloc_labels = ['Cash'] + list(positions.keys())
    alloc_vals   = [round(capital, 2)] + [
        round(p.get('shares', 0) * p.get('current_price', p.get('entry_price', 0)), 2)
        for p in positions.values()
    ]

    # Signal buckets
    buy_sigs     = [(s, d) for s, d in signals.items() if d.get('signal') == 'BUY']
    avoid_sigs   = [(s, d) for s, d in signals.items() if d.get('signal') == 'AVOID']
    hold_sigs    = [(s, d) for s, d in signals.items() if d.get('signal') == 'HOLD']
    caution_sigs = [(s, d) for s, d in signals.items() if d.get('signal') == 'CAUTION']

    # Weight results — parse and rank
    weight_rows_html = ''
    best_weight = None
    if weight_data:
        sorted_weights = sorted(weight_data, key=lambda x: x.get('sharpe', 0), reverse=True)
        best_weight    = sorted_weights[0] if sorted_weights else None
        for i, w in enumerate(sorted_weights):
            rank      = i + 1
            pw        = w.get('pred_w', 0)
            sw        = w.get('sent_w', 0)
            secw      = w.get('sector_w', 0)
            sh        = w.get('sharpe', 0)
            wr        = w.get('win_rate', 0) * 100
            ret       = w.get('total_return', 0) * 100
            trades    = w.get('total_trades', 0)
            dd        = w.get('max_drawdown', 0) * 100
            pf2       = w.get('profit_factor', 0)
            is_best   = rank == 1
            row_cls   = 'wt-best' if is_best else ''
            rank_html = '🏆' if is_best else f'#{rank}'
            sh_c      = '#00d4aa' if sh >= 1.5 else '#f0b429' if sh >= 0.5 else '#f87171'
            wr_c      = '#00d4aa' if wr >= 60 else '#f0b429' if wr >= 45 else '#f87171'
            ret_c     = '#00d4aa' if ret >= 0 else '#f87171'
            weight_rows_html += f'''
            <tr class="trow {row_cls}">
              <td class="px-5 py-4 tc fw mono">{rank_html}</td>
              <td class="px-5 py-4">
                <div class="wt-bars">
                  <div class="wt-bar-row">
                    <span class="wt-lbl">PRED</span>
                    <div class="bar-bg"><div class="bar-fill" style="width:{int(pw*100)}%;background:#0ea5e9"></div></div>
                    <span class="mono fw" style="color:#0ea5e9">{pw:.1f}</span>
                  </div>
                  <div class="wt-bar-row">
                    <span class="wt-lbl">SENT</span>
                    <div class="bar-bg"><div class="bar-fill" style="width:{int(sw*100)}%;background:#a855f7"></div></div>
                    <span class="mono fw" style="color:#a855f7">{sw:.1f}</span>
                  </div>
                  <div class="wt-bar-row">
                    <span class="wt-lbl">SECT</span>
                    <div class="bar-bg"><div class="bar-fill" style="width:{int(secw*100)}%;background:#f0b429"></div></div>
                    <span class="mono fw" style="color:#f0b429">{secw:.1f}</span>
                  </div>
                </div>
              </td>
              <td class="px-5 py-4 tc mono fw" style="color:{sh_c}">{sh:.2f}</td>
              <td class="px-5 py-4 tc mono" style="color:{wr_c}">{wr:.1f}%</td>
              <td class="px-5 py-4 tc mono" style="color:{ret_c}">{ret:+.2f}%</td>
              <td class="px-5 py-4 tc mono">{trades}</td>
              <td class="px-5 py-4 tc mono" style="color:#f87171">{dd:.2f}%</td>
              <td class="px-5 py-4 tc mono">{pf2:.2f}x</td>
            </tr>'''
    else:
        weight_rows_html = '''<tr><td colspan="8" class="empty-state">
          <div class="empty-icon">⚖</div>
          <p>No weight optimization results yet</p>
          <p class="sub">Run: python run_weight_optimization.py</p>
        </td></tr>'''

    # Best weight banner
    best_banner = ''
    if best_weight:
        bp  = best_weight.get('pred_w', 0)
        bs  = best_weight.get('sent_w', 0)
        bsc = best_weight.get('sector_w', 0)
        bsh = best_weight.get('sharpe', 0)
        bwr = best_weight.get('win_rate', 0) * 100
        formula = f"combined = pred × {bp:.1f} + (sent + 0.5) × {bs:.1f} + (sect - 0.5) × {bsc:.1f}"
        best_banner = f'''
        <div class="best-banner">
          <div class="best-left">
            <span class="best-crown">🏆</span>
            <div>
              <div class="best-title">Optimal Signal Weights Found</div>
              <div class="best-formula mono">{formula}</div>
            </div>
          </div>
          <div class="best-right">
            <div class="best-stat">
              <div class="best-stat-val" style="color:#00d4aa">{bsh:.2f}</div>
              <div class="best-stat-lbl">Sharpe</div>
            </div>
            <div class="best-stat">
              <div class="best-stat-val" style="color:#0ea5e9">{bwr:.1f}%</div>
              <div class="best-stat-lbl">Win Rate</div>
            </div>
            <div class="best-stat">
              <div class="best-stat-val mono" style="color:#f0b429">{bp:.1f} / {bs:.1f} / {bsc:.1f}</div>
              <div class="best-stat-lbl">Pred / Sent / Sect</div>
            </div>
          </div>
        </div>'''

    # ── HTML fragments ────────────────────────────────────────────────────────

    def kpi_color(val, low_bad=True, thresholds=(0, 0)):
        lo, hi = thresholds
        if low_bad:
            return '#00d4aa' if val >= hi else '#f0b429' if val >= lo else '#f87171'
        else:
            return '#f87171' if val >= hi else '#f0b429' if val >= lo else '#00d4aa'

    pnl_c   = '#00d4aa' if total_pnl >= 0 else '#f87171'
    pnl_sgn = '+' if total_pnl >= 0 else ''
    wr_c    = kpi_color(win_rate,   True,  (45, 60))
    sh_c    = kpi_color(sharpe,     True,  (0.5, 1.5))
    dd_c    = kpi_color(max_dd * 100, False, (10, 20))
    pf_c    = kpi_color(pf,         True,  (1.0, 1.5))
    unr_c   = '#00d4aa' if unrealized >= 0 else '#f87171'
    unr_sgn = '+' if unrealized >= 0 else ''

    # Positions table
    pos_rows = ''
    for sym, pos in positions.items():
        shares  = pos.get('shares', 0)
        entry   = pos.get('entry_price', 0)
        current = pos.get('current_price', entry)
        pnl_pos = (current - entry) * shares
        pnl_pct = (current - entry) / entry * 100 if entry > 0 else 0
        stop    = entry * (1 - pos.get('stop_loss_pct', 0.03))
        target  = entry * (1 + pos.get('take_profit_pct', 0.08))
        ml      = pos.get('signal', 0.5)
        ml_pct  = int(ml * 100)
        p_c     = '#00d4aa' if pnl_pos >= 0 else '#f87171'
        sgn     = '+' if pnl_pos >= 0 else ''
        days    = 0
        try:
            days = (datetime.now() - datetime.fromisoformat(pos.get('entry_date', ''))).days
        except Exception:
            pass
        bar_c = '#00d4aa' if ml_pct >= 65 else '#f0b429' if ml_pct >= 55 else '#f87171'
        pos_rows += f'''
        <tr class="trow">
          <td class="px-5 py-4">
            <div class="sym">{sym}</div>
            <div class="sub">{pos.get('reason', '—')}</div>
          </td>
          <td class="px-5 py-4 tc mono">{shares}</td>
          <td class="px-5 py-4 tc mono">${entry:.2f}</td>
          <td class="px-5 py-4 tc mono fw">${current:.2f}</td>
          <td class="px-5 py-4 tc mono fw" style="color:{p_c}">
            {sgn}${pnl_pos:.2f}<br><span class="sub">{sgn}{pnl_pct:.1f}%</span>
          </td>
          <td class="px-5 py-4 tc">
            <div class="sub stop">${stop:.2f}</div>
            <div class="sub tgt">${target:.2f}</div>
          </td>
          <td class="px-5 py-4">
            <div class="bar-row">
              <div class="bar-bg"><div class="bar-fill" style="width:{ml_pct}%;background:{bar_c}"></div></div>
              <span class="bar-lbl mono" style="color:{bar_c}">{ml_pct}%</span>
            </div>
          </td>
          <td class="px-5 py-4 tc sub">{days}d</td>
        </tr>'''

    # Unrealized P&L total row
    pos_total_row = ''
    if positions:
        pos_total_row = f'''
        <tr style="background:var(--surface2);border-top:2px solid var(--border2)">
          <td class="px-5 py-3" colspan="4">
            <span class="sub fw">TOTAL UNREALIZED</span>
          </td>
          <td class="px-5 py-3 tc mono fw" style="color:{unr_c}">
            {unr_sgn}${unrealized:,.2f}
          </td>
          <td colspan="3" class="px-5 py-3 sub tc">
            Invested ${pos_value:,.2f} · {len(positions)} positions
          </td>
        </tr>'''

    if not pos_rows:
        pos_rows = '''<tr><td colspan="8" class="empty-state">
          <div class="empty-icon">◎</div>
          <p>No open positions</p>
          <p class="sub">Waiting for high-confidence signals</p>
        </td></tr>'''

    # Signals table (sortable via JS)
    sig_rows = ''
    for sym, d in sorted(signals.items(), key=lambda x: x[1].get('prediction', 0), reverse=True):
        sig    = d.get('signal', 'HOLD')
        pred   = d.get('prediction', 0)
        price  = d.get('price', 0)
        regime = d.get('regime', '—')
        sent   = d.get('sentiment', 0)
        sector = d.get('sector', '—')
        comb   = d.get('combined', pred)
        ml_pct = int(pred * 100)
        sig_cls = {'BUY': 'badge-buy', 'AVOID': 'badge-avoid', 'CAUTION': 'badge-caution',
                   'EARNINGS_HOLD': 'badge-earn', 'VETOED': 'badge-veto',
                   'MTF_HOLD': 'badge-hold', 'CORR_HOLD': 'badge-hold'}.get(sig, 'badge-hold')
        row_cls = {'BUY': 'row-buy', 'AVOID': 'row-avoid'}.get(sig, '')
        bar_c   = '#00d4aa' if ml_pct >= 65 else '#f0b429' if ml_pct >= 55 else '#6b7280'
        sent_c  = '#00d4aa' if sent > 0.1 else '#f87171' if sent < -0.1 else '#9ca3af'
        sent_sgn = '+' if sent > 0 else ''
        reg_icon = {'uptrend': '↑', 'downtrend': '↓', 'sideways': '→',
                    'volatile': '⚡'}.get(regime, '—')
        layers  = min(9, max(0, int(comb * 9)))
        squares = ''.join(
            f'<span class="sq" style="background:{"#00d4aa" if i < layers else "#1f2937"}"></span>'
            for i in range(9)
        )
        sig_rows += f'''
        <tr class="trow {row_cls}" data-pred="{pred}" data-sent="{sent}"
            data-price="{price}" data-sig="{sig}" data-sym="{sym}">
          <td class="px-5 py-4">
            <div class="sym">{sym}</div>
            <div class="sub">{sector}</div>
          </td>
          <td class="px-5 py-4 tc"><span class="badge {sig_cls}">{sig}</span></td>
          <td class="px-5 py-4" style="min-width:140px">
            <div class="bar-row">
              <div class="bar-bg"><div class="bar-fill" style="width:{ml_pct}%;background:{bar_c}"></div></div>
              <span class="bar-lbl mono" style="color:{bar_c}">{ml_pct}%</span>
            </div>
          </td>
          <td class="px-5 py-4 tc">
            <span class="regime-tag">{reg_icon} {regime}</span>
          </td>
          <td class="px-5 py-4 tc mono" style="color:{sent_c}">{sent_sgn}{sent:.2f}</td>
          <td class="px-5 py-4 tc mono fw">${price:.2f}</td>
          <td class="px-5 py-4">
            <div class="squares">{squares}</div>
            <div class="sub mt-1">{layers}/9 passed</div>
          </td>
        </tr>'''

    # Sector cards
    sector_cards = ''
    for name, d in sectors.items():
        flow  = d.get('flow', 'NEUTRAL')
        score = d.get('score', 0)
        mom   = d.get('momentum_21d', 0)
        fc    = {'INFLOW': '#00d4aa', 'OUTFLOW': '#f87171'}.get(flow, '#9ca3af')
        bc    = {'INFLOW': 'card-inflow', 'OUTFLOW': 'card-outflow'}.get(flow, 'card-neutral')
        bar_w = min(100, abs(score) * 2000)
        mom_c = '#00d4aa' if mom >= 0 else '#f87171'
        mom_s = '+' if mom >= 0 else ''
        sector_cards += f'''
        <div class="sector-card {bc}">
          <div class="sec-header">
            <span class="sec-name">{name}</span>
            <span class="sec-badge" style="color:{fc};border-color:{fc}">{flow}</span>
          </div>
          <div class="sec-bar-bg mt-3">
            <div class="sec-bar-fill" style="width:{bar_w}%;background:{fc}"></div>
          </div>
          <div class="sec-footer">
            <span class="mono sub">Score: {score:.4f}</span>
            <span class="mono" style="color:{mom_c}">{mom_s}{mom:.2f}%</span>
          </div>
        </div>'''
    if not sector_cards:
        sector_cards = '<div class="empty-state" style="grid-column:1/-1">No sector data available</div>'

    # Earnings
    earn_rows = ''
    for e in earnings:
        days = e.get('days_until', 0)
        sym  = e.get('symbol', '')
        dt   = e.get('date', '')
        if days == 0:
            lbl, lc, bc2 = 'TODAY',    '#f87171', 'earn-today'
        elif days <= 2:
            lbl, lc, bc2 = f'In {days}d', '#f0b429', 'earn-soon'
        else:
            lbl, lc, bc2 = f'In {days}d', '#9ca3af', 'earn-later'
        earn_rows += f'''
        <div class="earn-row {bc2}">
          <div><span class="sym">{sym}</span><span class="sub ml-3">{dt}</span></div>
          <span class="mono fw" style="color:{lc}">{lbl}</span>
        </div>'''
    if not earn_rows:
        earn_rows = '<div class="empty-state"><p>No earnings this week</p></div>'

    # Trade history
    hist_rows = ''
    for t in reversed(history[-30:]):
        action = t.get('action', '')
        pnl    = t.get('pnl', 0)
        sym    = t.get('symbol', '')
        price  = t.get('price', 0)
        dt     = t.get('date', '')[:10]
        reason = t.get('reason', '')
        pnl_c2 = '#00d4aa' if pnl >= 0 else '#f87171'
        sgn    = '+' if pnl >= 0 else ''
        act_c  = 'badge-buy' if action == 'BUY' else 'badge-avoid'
        pnl_txt = f'{sgn}${pnl:.2f}' if action == 'SELL' else '—'
        hist_rows += f'''
        <div class="hist-row">
          <span class="badge {act_c} mr-3">{action}</span>
          <div class="flex-1">
            <span class="sym">{sym}</span>
            <span class="mono sub ml-2">${price:.2f}</span>
            <div class="sub">{dt} · {reason}</div>
          </div>
          <span class="mono fw" style="color:{pnl_c2}">{pnl_txt}</span>
        </div>'''
    if not hist_rows:
        hist_rows = '<div class="empty-state"><p>No trades yet</p></div>'

    # Overview top BUY table
    buy_table_rows = ''.join(f'''
    <tr class="trow row-buy">
      <td class="px-5 py-3"><div class="sym">{s}</div></td>
      <td class="px-5 py-3 sub">{d.get("sector", "—")}</td>
      <td class="px-5 py-3">
        <div class="bar-row">
          <div class="bar-bg">
            <div class="bar-fill" style="width:{int(d.get("prediction",0)*100)}%;background:#00d4aa"></div>
          </div>
          <span class="bar-lbl mono" style="color:#00d4aa">{int(d.get("prediction",0)*100)}%</span>
        </div>
      </td>
      <td class="px-5 py-3 tc"><span class="regime-tag">{d.get("regime","—")}</span></td>
      <td class="px-5 py-3 tc mono"
          style="color:{('#00d4aa' if d.get('sentiment',0)>0.1 else '#f87171' if d.get('sentiment',0)<-0.1 else '#9ca3af')}">
        {('+' if d.get('sentiment',0)>0 else '')}{d.get('sentiment',0):.2f}
      </td>
      <td class="px-5 py-3 tc mono fw">${d.get("price",0):.2f}</td>
    </tr>''' for s, d in buy_sigs[:8]) or \
    '<tr><td colspan="6" class="empty-state">No BUY signals today</td></tr>'

    # ── JSON for JS charts ────────────────────────────────────────────────────
    import json as _json

    up_trend   = chart_vals[-1] >= chart_vals[0] if len(chart_vals) > 1 else True
    line_color = '#00d4aa' if up_trend else '#f87171'

    sym_pnl_keys = _json.dumps(list(sym_pnl.keys()))
    sym_pnl_vals = _json.dumps([round(v, 2) for v in sym_pnl.values()])
    sym_pnl_colors = _json.dumps([
        '#00d4aa' if v >= 0 else '#f87171' for v in sym_pnl.values()
    ])

    daily_vals_json   = _json.dumps([round(v, 4) for v in daily_ret_vals])
    daily_labels_json = _json.dumps(daily_ret_labels)
    dd_vals_json      = _json.dumps(dd_series)
    dd_labels_json    = _json.dumps(dd_labels)

    # ── FULL HTML ─────────────────────────────────────────────────────────────
    html = f'''<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <meta http-equiv="refresh" content="300">
  <title>AlphaEdge V5 — Institutional Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Sora:wght@300;400;600;700;800&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    /* ── Reset ───────────────────────────────────────────────────── */
    *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}

    /* ── Theme tokens ────────────────────────────────────────────── */
    :root {{
      --bg:        #070b0f;
      --surface:   #0d1117;
      --surface2:  #111820;
      --border:    #1a2433;
      --border2:   #243040;
      --text:      #e2eaf5;
      --sub:       #5a7090;
      --accent:    #00d4aa;
      --accent2:   #0ea5e9;
      --warn:      #f0b429;
      --danger:    #f87171;
      --purple:    #a855f7;
      --font-ui:   'Sora', sans-serif;
      --font-mono: 'IBM Plex Mono', monospace;
    }}
    [data-theme="light"] {{
      --bg:       #f0f4f8;
      --surface:  #ffffff;
      --surface2: #f8fafc;
      --border:   #e2e8f0;
      --border2:  #cbd5e1;
      --text:     #0f172a;
      --sub:      #64748b;
    }}

    /* ── Base ────────────────────────────────────────────────────── */
    html {{ scroll-behavior:smooth; }}
    body {{
      font-family: var(--font-ui);
      background: var(--bg);
      color: var(--text);
      font-size: 13px;
      line-height: 1.5;
      min-height: 100vh;
      overflow-x: hidden;
    }}
    ::-webkit-scrollbar {{ width:5px; height:5px; }}
    ::-webkit-scrollbar-track {{ background:var(--bg); }}
    ::-webkit-scrollbar-thumb {{ background:var(--border2); border-radius:3px; }}

    /* ── Utility ─────────────────────────────────────────────────── */
    .mono   {{ font-family: var(--font-mono); }}
    .fw     {{ font-weight: 600; }}
    .sym    {{ font-weight: 700; font-size: 13px; color: var(--text); }}
    .sub    {{ font-size: 11px; color: var(--sub); }}
    .tc     {{ text-align: center; }}
    .stop   {{ color: var(--danger); }}
    .tgt    {{ color: var(--accent); }}
    .ml-2   {{ margin-left: 8px; }}
    .ml-3   {{ margin-left: 12px; }}
    .mr-3   {{ margin-right: 12px; }}
    .mt-1   {{ margin-top: 4px; }}
    .mt-3   {{ margin-top: 12px; }}
    .flex-1 {{ flex: 1; }}

    /* ── Spacing (fixed from V4) ─────────────────────────────────── */
    .px-5   {{ padding-left: 20px; padding-right: 20px; }}
    .py-3   {{ padding-top: 10px; padding-bottom: 10px; }}
    .py-3-5  {{ padding-top: 12px; padding-bottom: 12px; }}
    .py-4   {{ padding-top: 14px; padding-bottom: 14px; }}

    /* ── Header ──────────────────────────────────────────────────── */
    .header {{
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      position: sticky; top:0; z-index:100;
      padding: 0 28px;
      height: 54px;
      display: flex; align-items:center; justify-content:space-between;
    }}
    .logo {{ display:flex; align-items:center; gap:10px; }}
    .logo-mark {{
      width:30px; height:30px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      border-radius: 7px;
      display:flex; align-items:center; justify-content:center;
      font-size:15px; font-weight:800; color:#070b0f;
    }}
    .logo-text {{
      font-size:16px; font-weight:800; letter-spacing:-0.3px;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }}
    .header-right {{ display:flex; align-items:center; gap:14px; }}
    .pulse-dot {{
      width:7px; height:7px; border-radius:50%;
      background:var(--accent);
      animation: pulse 2.5s ease-in-out infinite;
    }}
    @keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:.4;transform:scale(.85)}} }}
    .chip {{
      padding:2px 8px; border-radius:4px;
      font-size:10px; font-weight:700; letter-spacing:.5px;
      border:1px solid;
    }}
    .chip-paper {{ background:#0ea5e920; color:#0ea5e9; border-color:#0ea5e940; }}
    .chip-v5    {{ background:#00d4aa20; color:#00d4aa; border-color:#00d4aa40; }}
    .regime-chip {{
      padding:3px 10px; border-radius:4px; font-size:10px; font-weight:700;
      border:1px solid; letter-spacing:.4px;
    }}
    .theme-btn {{
      background: var(--surface2); border:1px solid var(--border);
      border-radius:6px; padding:4px 10px;
      color:var(--sub); font-size:11px; cursor:pointer;
      font-family:var(--font-ui); transition:.15s;
    }}
    .theme-btn:hover {{ color:var(--text); border-color:var(--border2); }}
    .ts {{ font-size:11px; color:var(--sub); font-family:var(--font-mono); }}

    /* ── KPI Strip ───────────────────────────────────────────────── */
    .kpi-strip {{
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0 28px;
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      gap: 0;
    }}
    .kpi {{
      padding:14px 16px;
      border-right:1px solid var(--border);
      position:relative;
    }}
    .kpi:last-child {{ border-right:none; }}
    .kpi-label {{
      font-size:10px; font-weight:600; letter-spacing:.8px;
      color:var(--sub); text-transform:uppercase; margin-bottom:4px;
    }}
    .kpi-value {{
      font-family:var(--font-mono); font-size:20px; font-weight:600;
      line-height:1.1; letter-spacing:-0.5px;
    }}
    .kpi-sub   {{ font-size:10px; color:var(--sub); margin-top:3px; }}
    .kpi-bar   {{ position:absolute; bottom:0; left:0; right:0; height:2px; opacity:.6; }}

    /* ── Tab nav ─────────────────────────────────────────────────── */
    .tabnav {{
      background:var(--surface);
      border-bottom:1px solid var(--border);
      padding:0 28px;
      display:flex; gap:0;
      position:sticky; top:54px; z-index:90;
      overflow-x:auto;
    }}
    .tab-btn {{
      padding:12px 18px;
      font-size:12px; font-weight:600; letter-spacing:.3px;
      color:var(--sub); background:none; border:none;
      border-bottom:2px solid transparent;
      cursor:pointer; white-space:nowrap;
      font-family:var(--font-ui);
      transition:color .15s, border-color .15s;
    }}
    .tab-btn:hover  {{ color:var(--text); }}
    .tab-btn.active {{ color:var(--accent); border-bottom-color:var(--accent); }}
    .tab-count {{
      display:inline-flex; align-items:center; justify-content:center;
      min-width:16px; height:16px; border-radius:8px; padding:0 4px;
      background:var(--border2); font-size:9px; font-weight:700;
      margin-left:5px; color:var(--sub);
    }}
    .tab-count.green {{ background:#00d4aa25; color:var(--accent); }}
    .tab-count.red   {{ background:#f8717125; color:var(--danger); }}
    .tab-count.gold  {{ background:#f0b42925; color:var(--warn); }}

    /* ── Content ─────────────────────────────────────────────────── */
    .content {{ padding:24px 28px; }}
    .hidden  {{ display:none !important; }}

    /* ── Panels ──────────────────────────────────────────────────── */
    .panel {{
      background:var(--surface);
      border:1px solid var(--border);
      border-radius:10px;
      overflow:hidden;
    }}
    .panel-header {{
      padding:14px 20px;
      border-bottom:1px solid var(--border);
      display:flex; align-items:center; justify-content:space-between;
    }}
    .panel-title {{
      font-size:11px; font-weight:700; letter-spacing:.8px;
      text-transform:uppercase; color:var(--sub);
    }}
    .panel-body {{ padding:20px; }}

    /* ── Grids ───────────────────────────────────────────────────── */
    .stack   {{ display:flex; flex-direction:column; gap:20px; }}
    .grid-2  {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
    .grid-3  {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; }}
    .grid-4  {{ display:grid; grid-template-columns:repeat(4,1fr); gap:20px; }}
    .grid-32 {{ display:grid; grid-template-columns:2fr 1fr; gap:20px; }}
    .grid-23 {{ display:grid; grid-template-columns:1fr 2fr; gap:20px; }}

    @media(max-width:1200px) {{
      .grid-32,.grid-23 {{ grid-template-columns:1fr; }}
      .kpi-strip {{ grid-template-columns:repeat(4,1fr); }}
    }}
    @media(max-width:768px) {{
      .grid-2,.grid-3,.grid-4 {{ grid-template-columns:1fr; }}
      .kpi-strip {{ grid-template-columns:repeat(2,1fr); }}
    }}

    /* ── Tables ──────────────────────────────────────────────────── */
    .tbl {{ width:100%; border-collapse:collapse; }}
    .tbl thead tr {{
      background:var(--surface2);
      border-bottom:1px solid var(--border2);
    }}
    .tbl thead th {{
      padding:10px 20px;
      font-size:10px; font-weight:700; letter-spacing:.7px;
      text-transform:uppercase; color:var(--sub); text-align:left;
      user-select:none;
    }}
    .tbl thead th.tc {{ text-align:center; }}
    .tbl thead th.sortable {{ cursor:pointer; }}
    .tbl thead th.sortable:hover {{ color:var(--text); }}
    .sort-arrow {{ margin-left:4px; opacity:.4; }}
    .sort-arrow.active {{ opacity:1; color:var(--accent); }}
    .trow {{ border-bottom:1px solid var(--border); transition:background .1s; }}
    .trow:hover {{ background:#ffffff05; }}
    .trow:last-child {{ border-bottom:none; }}
    .row-buy   {{ background:#00d4aa08; }}
    .row-avoid {{ background:#f8717108; }}

    /* ── Badges ──────────────────────────────────────────────────── */
    .badge {{
      display:inline-block; padding:2px 7px; border-radius:4px;
      font-size:9px; font-weight:700; letter-spacing:.6px;
      text-transform:uppercase; border:1px solid;
    }}
    .badge-buy    {{ background:#00d4aa20; color:#00d4aa; border-color:#00d4aa50; }}
    .badge-avoid  {{ background:#f8717120; color:#f87171; border-color:#f8717150; }}
    .badge-caution{{ background:#f0b42920; color:#f0b429; border-color:#f0b42950; }}
    .badge-hold   {{ background:#ffffff10; color:#6b7280; border-color:#ffffff20; }}
    .badge-earn   {{ background:#0ea5e920; color:#0ea5e9; border-color:#0ea5e950; }}
    .badge-veto   {{ background:#a855f720; color:#a855f7; border-color:#a855f750; }}

    /* ── Progress bars ───────────────────────────────────────────── */
    .bar-row  {{ display:flex; align-items:center; gap:8px; }}
    .bar-bg   {{ flex:1; background:var(--border2); border-radius:2px; height:4px; min-width:60px; }}
    .bar-fill {{ height:4px; border-radius:2px; transition:width .4s; }}
    .bar-lbl  {{ font-size:10px; font-weight:600; min-width:28px; text-align:right; }}

    /* ── Filter squares ──────────────────────────────────────────── */
    .squares {{ display:flex; gap:3px; }}
    .sq {{ width:7px; height:7px; border-radius:1.5px; }}

    /* ── Regime tag ──────────────────────────────────────────────── */
    .regime-tag {{
      font-size:10px; font-weight:600; padding:2px 6px;
      border-radius:3px; background:var(--border2); color:var(--sub);
    }}

    /* ── Signal pills ────────────────────────────────────────────── */
    .sig-pills {{ display:flex; gap:10px; flex-wrap:wrap; }}
    .sig-pill {{
      display:flex; align-items:center; gap:8px;
      padding:8px 16px; border-radius:8px; border:1px solid;
      font-size:11px; font-weight:700;
    }}
    .pill-buy    {{ background:#00d4aa15; border-color:#00d4aa40; color:#00d4aa; }}
    .pill-avoid  {{ background:#f8717115; border-color:#f8717140; color:#f87171; }}
    .pill-hold   {{ background:#ffffff08; border-color:#ffffff15; color:#6b7280; }}
    .pill-caution{{ background:#f0b42915; border-color:#f0b42940; color:#f0b429; }}
    .pill-num    {{ font-size:22px; font-weight:800; }}

    /* ── Stat cards ──────────────────────────────────────────────── */
    .stat-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
    .stat-card {{
      background:var(--surface2); border:1px solid var(--border);
      border-radius:8px; padding:14px;
    }}
    .stat-label {{ font-size:10px; letter-spacing:.6px; text-transform:uppercase; color:var(--sub); margin-bottom:4px; }}
    .stat-value {{ font-family:var(--font-mono); font-size:18px; font-weight:600; }}

    /* ── Sector cards ────────────────────────────────────────────── */
    .sectors-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:14px; }}
    .sector-card {{
      border:1px solid; border-radius:8px; padding:14px;
      transition:border-color .2s;
    }}
    .card-inflow  {{ background:#00d4aa0a; border-color:#00d4aa30; }}
    .card-outflow {{ background:#f871710a; border-color:#f8717130; }}
    .card-neutral {{ background:var(--surface2); border-color:var(--border); }}
    .sec-header   {{ display:flex; justify-content:space-between; align-items:center; }}
    .sec-name     {{ font-weight:700; font-size:13px; }}
    .sec-badge {{
      font-size:9px; font-weight:700; letter-spacing:.5px;
      border:1px solid; border-radius:3px; padding:1px 5px;
    }}
    .sec-bar-bg   {{ background:var(--border2); border-radius:2px; height:3px; }}
    .sec-bar-fill {{ height:3px; border-radius:2px; }}
    .sec-footer   {{ display:flex; justify-content:space-between; margin-top:8px; }}

    /* ── Earnings ────────────────────────────────────────────────── */
    .earn-list {{ display:flex; flex-direction:column; gap:8px; }}
    .earn-row {{
      display:flex; justify-content:space-between; align-items:center;
      padding:10px 14px; border-radius:6px; border:1px solid;
    }}
    .earn-today {{ background:#f871710a; border-color:#f8717140; }}
    .earn-soon  {{ background:#f0b4290a; border-color:#f0b42940; }}
    .earn-later {{ background:var(--surface2); border-color:var(--border); }}

    /* ── History ─────────────────────────────────────────────────── */
    .hist-row {{
      display:flex; align-items:flex-start; padding:10px 0;
      border-bottom:1px solid var(--border);
    }}
    .hist-row:last-child {{ border-bottom:none; }}

    /* ── Empty state ─────────────────────────────────────────────── */
    .empty-state {{ text-align:center; padding:48px 24px; color:var(--sub); }}
    .empty-icon  {{ font-size:28px; margin-bottom:10px; opacity:.3; }}

    /* ── Weight results ──────────────────────────────────────────── */
    .best-banner {{
      background: linear-gradient(135deg, #00d4aa12, #0ea5e912);
      border: 1px solid #00d4aa40;
      border-radius: 10px; padding: 18px 24px;
      display: flex; align-items:center; justify-content:space-between;
      gap: 20px; flex-wrap:wrap;
    }}
    .best-left  {{ display:flex; align-items:center; gap:14px; }}
    .best-right {{ display:flex; gap:24px; }}
    .best-crown {{ font-size:24px; }}
    .best-title {{ font-weight:700; font-size:13px; margin-bottom:4px; }}
    .best-formula {{
      font-size:11px; color:var(--accent);
      background:var(--surface); padding:4px 10px;
      border-radius:4px; border:1px solid var(--border2);
    }}
    .best-stat     {{ text-align:center; }}
    .best-stat-val {{ font-family:var(--font-mono); font-size:20px; font-weight:700; }}
    .best-stat-lbl {{ font-size:10px; color:var(--sub); text-transform:uppercase; letter-spacing:.5px; }}

    .wt-best {{ background:linear-gradient(90deg,#00d4aa08,transparent) !important; }}
    .wt-bars  {{ display:flex; flex-direction:column; gap:5px; min-width:200px; }}
    .wt-bar-row {{ display:flex; align-items:center; gap:6px; }}
    .wt-lbl {{
      font-size:9px; font-weight:700; letter-spacing:.5px;
      color:var(--sub); width:28px; text-transform:uppercase;
    }}

    /* ── Search / filter ─────────────────────────────────────────── */
    .filter-bar {{
      display:flex; gap:10px; align-items:center; flex-wrap:wrap;
    }}
    .search-box {{
      background:var(--surface2); border:1px solid var(--border);
      border-radius:6px; padding:6px 12px;
      color:var(--text); font-size:12px; font-family:var(--font-ui);
      outline:none; width:200px;
      transition:border-color .15s;
    }}
    .search-box:focus {{ border-color:var(--border2); }}
    .filter-btn {{
      padding:5px 12px; border-radius:5px; border:1px solid var(--border);
      background:var(--surface2); color:var(--sub);
      font-size:11px; font-weight:600; cursor:pointer;
      font-family:var(--font-ui); transition:.15s;
    }}
    .filter-btn:hover,
    .filter-btn.active {{ background:var(--border2); color:var(--text); border-color:var(--border2); }}
    .filter-btn.f-buy.active    {{ background:#00d4aa20; color:#00d4aa; border-color:#00d4aa50; }}
    .filter-btn.f-avoid.active  {{ background:#f8717120; color:#f87171; border-color:#f8717150; }}
    .filter-btn.f-caution.active{{ background:#f0b42920; color:#f0b429; border-color:#f0b42950; }}

    /* ── Footer ──────────────────────────────────────────────────── */
    .footer {{
      border-top:1px solid var(--border);
      padding:14px 28px;
      display:flex; justify-content:space-between; align-items:center;
      font-size:10px; color:var(--sub); letter-spacing:.3px;
      margin-top:32px;
    }}
  </style>
</head>
<body>

<!-- ════════════════════════════════════════════════════════ HEADER -->
<header class="header">
  <div class="logo">
    <div class="logo-mark">α</div>
    <span class="logo-text">AlphaEdge</span>
    <span class="chip chip-paper">PAPER</span>
    <span class="chip chip-v5">V5</span>
    <span class="regime-chip"
          style="color:{regime_color};border-color:{regime_color}40;background:{regime_color}12">
      {regime_icon}
    </span>
  </div>
  <div class="header-right">
    <div class="pulse-dot"></div>
    <span class="ts">Updated {now_str}</span>
    <span class="ts" style="color:var(--border2)">|</span>
    <span class="ts">Last scan: {saved_at}</span>
    <button class="theme-btn" onclick="toggleTheme()" id="themeBtn">☀ Light</button>
  </div>
</header>

<!-- ══════════════════════════════════════════════════════ KPI STRIP -->
<div class="kpi-strip">
  <div class="kpi">
    <div class="kpi-label">Portfolio Value</div>
    <div class="kpi-value" style="color:var(--text)">${total_value:,.2f}</div>
    <div class="kpi-sub">Started ${starting:,.0f}</div>
    <div class="kpi-bar" style="background:var(--accent2)"></div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Total P&amp;L</div>
    <div class="kpi-value" style="color:{pnl_c}">{pnl_sgn}${total_pnl:,.2f}</div>
    <div class="kpi-sub" style="color:{pnl_c}">{pnl_sgn}{total_pct:.2f}%</div>
    <div class="kpi-bar" style="background:{pnl_c}"></div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Unrealized P&amp;L</div>
    <div class="kpi-value" style="color:{unr_c}">{unr_sgn}${unrealized:,.2f}</div>
    <div class="kpi-sub">Across {len(positions)} positions</div>
    <div class="kpi-bar" style="background:{unr_c}"></div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Win Rate</div>
    <div class="kpi-value" style="color:{wr_c}">{win_rate:.1f}%</div>
    <div class="kpi-sub">{len(wins)}W · {len(losses)}L · {len(sells)} trades</div>
    <div class="kpi-bar" style="background:{wr_c}"></div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Sharpe Ratio</div>
    <div class="kpi-value" style="color:{sh_c}">{sharpe:.2f}</div>
    <div class="kpi-sub">Annualised · RF=0</div>
    <div class="kpi-bar" style="background:{sh_c}"></div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Max Drawdown</div>
    <div class="kpi-value" style="color:{dd_c}">{max_dd*100:.2f}%</div>
    <div class="kpi-sub">Peak-to-trough</div>
    <div class="kpi-bar" style="background:{dd_c}"></div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Profit Factor</div>
    <div class="kpi-value" style="color:{pf_c}">{pf:.2f}x</div>
    <div class="kpi-sub">Expectancy ${expectancy:+.2f}</div>
    <div class="kpi-bar" style="background:{pf_c}"></div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Positions</div>
    <div class="kpi-value" style="color:var(--accent2)">{len(positions)}/5</div>
    <div class="kpi-sub">Cash ${capital:,.0f}</div>
    <div class="kpi-bar" style="background:var(--accent2)"></div>
  </div>
</div>

<!-- ════════════════════════════════════════════════════════ TAB NAV -->
<nav class="tabnav">
  <button class="tab-btn active" onclick="showTab('overview')"  id="t-overview">Overview</button>
  <button class="tab-btn"        onclick="showTab('positions')" id="t-positions">
    Positions
    <span class="tab-count {'green' if positions else ''}">{len(positions)}</span>
  </button>
  <button class="tab-btn"        onclick="showTab('signals')"   id="t-signals">
    Signals
    <span class="tab-count {'green' if buy_sigs else ''}">{len(signals)}</span>
  </button>
  <button class="tab-btn"        onclick="showTab('sectors')"   id="t-sectors">Sectors</button>
  <button class="tab-btn"        onclick="showTab('earnings')"  id="t-earnings">
    Earnings
    <span class="tab-count {'red' if any(e.get('days_until',99)<3 for e in earnings) else ''}">{len(earnings)}</span>
  </button>
  <button class="tab-btn"        onclick="showTab('history')"   id="t-history">
    History
    <span class="tab-count">{len(sells)}</span>
  </button>
  <button class="tab-btn"        onclick="showTab('weights')"   id="t-weights">
    Weights
    <span class="tab-count {'gold' if weight_data else ''}">{len(weight_data) if weight_data else 0}</span>
  </button>
</nav>

<!-- ══════════════════════════════════════════════════════ CONTENT -->
<main class="content">

  <!-- ── OVERVIEW ─────────────────────────────────────────────────── -->
  <div id="c-overview" class="stack">

    <div class="sig-pills">
      <div class="sig-pill pill-buy">
        <span class="pill-num">{len(buy_sigs)}</span> BUY
      </div>
      <div class="sig-pill pill-hold">
        <span class="pill-num">{len(hold_sigs)}</span> HOLD
      </div>
      <div class="sig-pill pill-caution">
        <span class="pill-num">{len(caution_sigs)}</span> CAUTION
      </div>
      <div class="sig-pill pill-avoid">
        <span class="pill-num">{len(avoid_sigs)}</span> AVOID
      </div>
    </div>

    <!-- Row 1: Equity + Allocation -->
    <div class="grid-32">
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">Equity Curve</span>
          <span class="ts">{len(chart_vals)-2} closed trades</span>
        </div>
        <div class="panel-body" style="padding:16px 16px 8px">
          <canvas id="equityChart" height="80"></canvas>
        </div>
      </div>
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">Capital Allocation</span>
        </div>
        <div class="panel-body">
          <canvas id="allocChart" height="170"></canvas>
        </div>
      </div>
    </div>

    <!-- Row 2: Drawdown + Per-symbol P&L -->
    <div class="grid-2">
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">Drawdown Curve</span>
          <span class="ts" style="color:{dd_c}">Max {max_dd*100:.2f}%</span>
        </div>
        <div class="panel-body" style="padding:16px 16px 8px">
          <canvas id="ddChart" height="90"></canvas>
        </div>
      </div>
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">Per-Symbol Realized P&amp;L</span>
          <span class="ts">{len(sym_pnl)} symbols traded</span>
        </div>
        <div class="panel-body" style="padding:16px 16px 8px">
          <canvas id="symPnlChart" height="90"></canvas>
        </div>
      </div>
    </div>

    <!-- Row 3: Daily returns histogram + Performance stats -->
    <div class="grid-32">
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">Daily Returns Distribution</span>
          <span class="ts">{len(daily_ret_vals)} trading days</span>
        </div>
        <div class="panel-body" style="padding:16px 16px 8px">
          <canvas id="histChart" height="80"></canvas>
        </div>
      </div>
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">Performance Metrics</span>
        </div>
        <div class="panel-body">
          <div class="stat-grid">
            <div class="stat-card">
              <div class="stat-label">Avg Win</div>
              <div class="stat-value" style="color:#00d4aa">+${avg_win:.2f}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Avg Loss</div>
              <div class="stat-value" style="color:#f87171">-${avg_loss:.2f}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Expectancy</div>
              <div class="stat-value" style="color:{('#00d4aa' if expectancy>=0 else '#f87171')}">${expectancy:+.2f}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Realized P&amp;L</div>
              <div class="stat-value mono" style="color:{('#00d4aa' if realized>=0 else '#f87171')}">{('+' if realized>=0 else '')}${realized:,.2f}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Unrealized P&amp;L</div>
              <div class="stat-value mono" style="color:{unr_c}">{unr_sgn}${unrealized:,.2f}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Total Trades</div>
              <div class="stat-value">{len(sells)}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Row 4: Top BUY signals -->
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">Top BUY Signals Today</span>
        <span class="badge badge-buy">{len(buy_sigs)} signals</span>
      </div>
      <div style="overflow-x:auto">
        <table class="tbl">
          <thead><tr>
            <th>Symbol</th><th>Sector</th>
            <th>ML Score</th><th class="tc">Regime</th>
            <th class="tc">Sentiment</th><th class="tc">Price</th>
          </tr></thead>
          <tbody>{buy_table_rows}</tbody>
        </table>
      </div>
    </div>

  </div><!-- /overview -->

  <!-- ── POSITIONS ─────────────────────────────────────────────────── -->
  <div id="c-positions" class="hidden">
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">Open Positions — {len(positions)}/5 slots</span>
        <span class="ts">Invested ${pos_value:,.2f} · Cash ${capital:,.2f}</span>
      </div>
      <div style="overflow-x:auto">
        <table class="tbl">
          <thead><tr>
            <th>Symbol / Reason</th>
            <th class="tc">Shares</th>
            <th class="tc">Entry</th>
            <th class="tc">Current</th>
            <th class="tc">P&amp;L</th>
            <th class="tc">Stop / Target</th>
            <th>ML Confidence</th>
            <th class="tc">Held</th>
          </tr></thead>
          <tbody>
            {pos_rows}
            {pos_total_row}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- ── SIGNALS ───────────────────────────────────────────────────── -->
  <div id="c-signals" class="hidden">
    <div class="stack">
      <div class="filter-bar">
        <input class="search-box" type="text" id="sigSearch"
               placeholder="Search symbol..." oninput="filterSignals()">
        <button class="filter-btn f-buy"     onclick="toggleSigFilter('BUY')"     id="fb-BUY">BUY</button>
        <button class="filter-btn f-avoid"   onclick="toggleSigFilter('AVOID')"   id="fb-AVOID">AVOID</button>
        <button class="filter-btn f-caution" onclick="toggleSigFilter('CAUTION')" id="fb-CAUTION">CAUTION</button>
        <button class="filter-btn"           onclick="toggleSigFilter('HOLD')"    id="fb-HOLD">HOLD</button>
        <button class="filter-btn"           onclick="clearSigFilters()">Clear</button>
        <span class="ts" id="sigCount">{len(signals)} shown</span>
      </div>
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">AI Signal Scanner</span>
          <div style="display:flex;gap:8px">
            <span class="badge badge-buy">{len(buy_sigs)} BUY</span>
            <span class="badge badge-hold">{len(hold_sigs)} HOLD</span>
            <span class="badge badge-caution">{len(caution_sigs)} CAUTION</span>
            <span class="badge badge-avoid">{len(avoid_sigs)} AVOID</span>
          </div>
        </div>
        <div style="overflow-x:auto">
          <table class="tbl" id="sigTable">
            <thead>
              <tr>
                <th class="sortable" onclick="sortTable('sym')">
                  Symbol <span class="sort-arrow" id="sa-sym">↕</span>
                </th>
                <th class="sortable tc" onclick="sortTable('sig')">
                  Signal <span class="sort-arrow" id="sa-sig">↕</span>
                </th>
                <th class="sortable" onclick="sortTable('pred')">
                  ML Score <span class="sort-arrow" id="sa-pred">↕</span>
                </th>
                <th class="tc">Regime</th>
                <th class="sortable tc" onclick="sortTable('sent')">
                  Sentiment <span class="sort-arrow" id="sa-sent">↕</span>
                </th>
                <th class="sortable tc" onclick="sortTable('price')">
                  Price <span class="sort-arrow" id="sa-price">↕</span>
                </th>
                <th>Filter Layers (9)</th>
              </tr>
            </thead>
            <tbody id="sigBody">{sig_rows}</tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- ── SECTORS ───────────────────────────────────────────────────── -->
  <div id="c-sectors" class="hidden">
    <div class="stack">
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">Sector Momentum Chart</span>
        </div>
        <div class="panel-body" style="padding:16px">
          <canvas id="sectorChart" height="60"></canvas>
        </div>
      </div>
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">Sector Rotation Detail</span>
          <span class="ts">
            {sum(1 for d in sectors.values() if d.get('flow')=='INFLOW')} inflow ·
            {sum(1 for d in sectors.values() if d.get('flow')=='OUTFLOW')} outflow
          </span>
        </div>
        <div class="panel-body">
          <div class="sectors-grid">{sector_cards}</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── EARNINGS ──────────────────────────────────────────────────── -->
  <div id="c-earnings" class="hidden">
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">Earnings Calendar</span>
        <span class="ts">{len(earnings)} upcoming</span>
      </div>
      <div class="panel-body">
        <div class="earn-list">{earn_rows}</div>
      </div>
    </div>
  </div>

  <!-- ── HISTORY ───────────────────────────────────────────────────── -->
  <div id="c-history" class="hidden">
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">Trade History</span>
        <span class="ts">Last 30 of {len(history)} entries</span>
      </div>
      <div class="panel-body">{hist_rows}</div>
    </div>
  </div>

  <!-- ── WEIGHTS ───────────────────────────────────────────────────── -->
  <div id="c-weights" class="hidden">
    <div class="stack">

      {best_banner}

      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">Signal Weight Optimization Results</span>
          <span class="ts">{len(weight_data) if weight_data else 0} combinations tested · Ranked by Sharpe</span>
        </div>
        <div style="overflow-x:auto">
          <table class="tbl">
            <thead><tr>
              <th class="tc">Rank</th>
              <th>Weights (Pred / Sent / Sect)</th>
              <th class="tc">Sharpe</th>
              <th class="tc">Win Rate</th>
              <th class="tc">Return</th>
              <th class="tc">Trades</th>
              <th class="tc">Max DD</th>
              <th class="tc">Prof Factor</th>
            </tr></thead>
            <tbody>{weight_rows_html}</tbody>
          </table>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">Current Formula in main.py</span>
        </div>
        <div class="panel-body">
          <div style="background:var(--surface2);border:1px solid var(--border2);
                      border-radius:6px;padding:14px 18px;font-family:var(--font-mono);
                      font-size:12px;color:var(--accent);line-height:1.8">
            <span style="color:var(--sub)"># Current (V4 hardcoded)</span><br>
            combined = pred × <span style="color:#f0b429">0.6</span> +
                       (sent + 0.5) × <span style="color:#f0b429">0.2</span> +
                       (sect - 0.5) × <span style="color:#f0b429">0.2</span><br><br>
            <span style="color:var(--sub)"># After optimization — update main.py with winning weights above</span>
          </div>
        </div>
      </div>

    </div>
  </div>

</main>

<!-- ══════════════════════════════════════════════════════ FOOTER -->
<footer class="footer">
  <span>AlphaEdge V5 · XGB + CatBoost + RF + LogReg Ensemble · 9-Layer Signal Filter · AI Veto Agent</span>
  <span>Full-width · Auto-refresh 5 min · Paper trading mode · Dark/Light mode</span>
</footer>

<!-- ═══════════════════════════════════════════════════════ SCRIPTS -->
<script>
// ── Tab system ────────────────────────────────────────────────────────────
const TABS = ['overview','positions','signals','sectors','earnings','history','weights'];
function showTab(name) {{
  TABS.forEach(t => {{
    document.getElementById('c-' + t).classList.add('hidden');
    const b = document.getElementById('t-' + t);
    if (b) b.classList.remove('active');
  }});
  document.getElementById('c-' + name).classList.remove('hidden');
  const b = document.getElementById('t-' + name);
  if (b) b.classList.add('active');
  // Resize charts when tab becomes visible
  if (name === 'sectors') {{
    setTimeout(() => sectorChart && sectorChart.resize(), 50);
  }}
}}

// ── Dark / Light mode ─────────────────────────────────────────────────────
function toggleTheme() {{
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', isDark ? 'light' : 'dark');
  document.getElementById('themeBtn').textContent = isDark ? '☾ Dark' : '☀ Light';
  // Redraw charts with new background
  Object.values(allCharts).forEach(c => c && c.update());
}}

// ── Chart global registry ─────────────────────────────────────────────────
const allCharts = {{}};

Chart.defaults.color = '#5a7090';
Chart.defaults.font.family = "'IBM Plex Mono', monospace";
Chart.defaults.font.size   = 10;

// ── Equity curve ──────────────────────────────────────────────────────────
(function() {{
  const ctx    = document.getElementById('equityChart').getContext('2d');
  const vals   = {_json.dumps(chart_vals)};
  const labels = {_json.dumps(chart_labels)};
  const lineC  = '{line_color}';
  const grad   = ctx.createLinearGradient(0, 0, 0, 260);
  grad.addColorStop(0, lineC + '30');
  grad.addColorStop(1, lineC + '00');
  allCharts.equity = new Chart(ctx, {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{
          data: vals, borderColor: lineC, backgroundColor: grad,
          borderWidth: 2, fill: true, tension: 0.35,
          pointRadius: vals.length < 25 ? 3 : 0, pointHoverRadius: 5,
          pointBackgroundColor: lineC,
        }},
        {{
          data: labels.map(() => {starting}),
          borderColor: '#243040', borderDash: [5,4],
          borderWidth: 1, pointRadius: 0, fill: false,
        }}
      ]
    }},
    options: {{
      responsive: true,
      interaction: {{ mode:'index', intersect:false }},
      plugins: {{
        legend: {{ display:false }},
        tooltip: {{
          backgroundColor:'#0d1117', borderColor:'#1a2433', borderWidth:1,
          callbacks: {{ label: c => ' $' + c.parsed.y.toLocaleString('en-US',{{minimumFractionDigits:2}}) }}
        }}
      }},
      scales: {{
        y: {{ grid:{{color:'#1a2433'}}, ticks:{{ callback: v => '$'+(v>=1000?(v/1000).toFixed(1)+'k':v.toFixed(0)) }} }},
        x: {{ grid:{{display:false}}, ticks:{{maxTicksLimit:8,maxRotation:0}} }}
      }}
    }}
  }});
}})();

// ── Allocation donut ──────────────────────────────────────────────────────
(function() {{
  const ctx    = document.getElementById('allocChart').getContext('2d');
  const labels = {_json.dumps(alloc_labels)};
  const vals   = {_json.dumps(alloc_vals)};
  const COLORS = ['#0ea5e9','#00d4aa','#f0b429','#f87171','#a855f7','#06b6d4','#84cc16'];
  allCharts.alloc = new Chart(ctx, {{
    type: 'doughnut',
    data: {{
      labels,
      datasets: [{{ data:vals, backgroundColor:COLORS.slice(0,vals.length),
                    borderColor:'#070b0f', borderWidth:3, hoverOffset:6 }}]
    }},
    options: {{
      responsive:true, cutout:'65%',
      plugins: {{
        legend: {{ position:'bottom', labels:{{ padding:12, usePointStyle:true,
                   pointStyleWidth:8, color:'#9ca3af', font:{{size:10}} }} }},
        tooltip: {{
          backgroundColor:'#0d1117', borderColor:'#1a2433', borderWidth:1,
          callbacks: {{ label: c => ' $'+c.parsed.toLocaleString('en-US',{{minimumFractionDigits:2}}) }}
        }}
      }}
    }}
  }});
}})();

// ── Drawdown curve ────────────────────────────────────────────────────────
(function() {{
  const ctx    = document.getElementById('ddChart').getContext('2d');
  const vals   = {dd_vals_json};
  const labels = {dd_labels_json};
  const grad   = ctx.createLinearGradient(0, 0, 0, 200);
  grad.addColorStop(0, '#f8717130');
  grad.addColorStop(1, '#f8717100');
  allCharts.dd = new Chart(ctx, {{
    type: 'line',
    data: {{
      labels,
      datasets: [{{
        data: vals, borderColor:'#f87171', backgroundColor: grad,
        borderWidth: 1.5, fill: true, tension: 0.3,
        pointRadius: vals.length < 20 ? 2 : 0, pointHoverRadius: 4,
        pointBackgroundColor:'#f87171',
      }}]
    }},
    options: {{
      responsive:true,
      interaction: {{ mode:'index', intersect:false }},
      plugins: {{
        legend: {{ display:false }},
        tooltip: {{
          backgroundColor:'#0d1117', borderColor:'#1a2433', borderWidth:1,
          callbacks: {{ label: c => ' DD: ' + c.parsed.y.toFixed(2) + '%' }}
        }}
      }},
      scales: {{
        y: {{
          reverse: true,
          grid: {{ color:'#1a2433' }},
          ticks: {{ callback: v => v.toFixed(1)+'%' }}
        }},
        x: {{ grid:{{display:false}}, ticks:{{maxTicksLimit:8,maxRotation:0}} }}
      }}
    }}
  }});
}})();

// ── Per-symbol P&L bar chart ──────────────────────────────────────────────
(function() {{
  const ctx    = document.getElementById('symPnlChart').getContext('2d');
  const labels = {sym_pnl_keys};
  const vals   = {sym_pnl_vals};
  const colors = {sym_pnl_colors};
  allCharts.symPnl = new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        data: vals, backgroundColor: colors,
        borderColor: colors, borderWidth: 1,
        borderRadius: 3,
      }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      plugins: {{
        legend: {{ display:false }},
        tooltip: {{
          backgroundColor:'#0d1117', borderColor:'#1a2433', borderWidth:1,
          callbacks: {{
            label: c => ' $' + c.parsed.x.toFixed(2)
          }}
        }}
      }},
      scales: {{
        x: {{
          grid: {{ color:'#1a2433' }},
          ticks: {{ callback: v => '$'+v.toFixed(0) }}
        }},
        y: {{ grid:{{display:false}} }}
      }}
    }}
  }});
}})();

// ── Daily returns histogram ───────────────────────────────────────────────
(function() {{
  const ctx    = document.getElementById('histChart').getContext('2d');
  const vals   = {daily_vals_json};
  const labels = {daily_labels_json};
  const colors = vals.map(v => v >= 0 ? '#00d4aa88' : '#f8717188');
  const border = vals.map(v => v >= 0 ? '#00d4aa' : '#f87171');
  allCharts.hist = new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        data: vals, backgroundColor: colors,
        borderColor: border, borderWidth: 1,
        borderRadius: 2,
      }}]
    }},
    options: {{
      responsive:true,
      plugins: {{
        legend: {{ display:false }},
        tooltip: {{
          backgroundColor:'#0d1117', borderColor:'#1a2433', borderWidth:1,
          callbacks: {{
            label: c => ' ' + (c.parsed.y >= 0 ? '+' : '') + (c.parsed.y * 100).toFixed(2) + '%'
          }}
        }}
      }},
      scales: {{
        y: {{
          grid: {{ color:'#1a2433' }},
          ticks: {{ callback: v => (v*100).toFixed(1)+'%' }}
        }},
        x: {{ grid:{{display:false}}, ticks:{{maxTicksLimit:10,maxRotation:45}} }}
      }}
    }}
  }});
}})();

// ── Sector momentum chart ─────────────────────────────────────────────────
let sectorChart = null;
(function() {{
  const el = document.getElementById('sectorChart');
  if (!el) return;
  const ctx     = el.getContext('2d');
  const rawData = {_json.dumps([
      {{'name': k, 'mom': v.get('momentum_21d', 0), 'flow': v.get('flow','NEUTRAL')}}
      for k, v in sectors.items()
  ])};
  if (!rawData.length) {{ el.parentElement.innerHTML = '<div class="empty-state">No sector data</div>'; return; }}
  const labels = rawData.map(d => d.name);
  const vals   = rawData.map(d => d.mom);
  const colors = rawData.map(d =>
    d.flow === 'INFLOW'  ? '#00d4aa88' :
    d.flow === 'OUTFLOW' ? '#f8717188' : '#6b728088'
  );
  const border = rawData.map(d =>
    d.flow === 'INFLOW'  ? '#00d4aa' :
    d.flow === 'OUTFLOW' ? '#f87171' : '#6b7280'
  );
  sectorChart = new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{ data:vals, backgroundColor:colors, borderColor:border, borderWidth:1, borderRadius:3 }}]
    }},
    options: {{
      responsive:true,
      plugins: {{
        legend: {{ display:false }},
        tooltip: {{
          backgroundColor:'#0d1117', borderColor:'#1a2433', borderWidth:1,
          callbacks: {{ label: c => ' ' + (c.parsed.y>=0?'+':'') + c.parsed.y.toFixed(2)+'%' }}
        }}
      }},
      scales: {{
        y: {{ grid:{{color:'#1a2433'}}, ticks:{{callback:v=>v.toFixed(1)+'%'}} }},
        x: {{ grid:{{display:false}}, ticks:{{maxRotation:30}} }}
      }}
    }}
  }});
  allCharts.sector = sectorChart;
}})();

// ── Signal sort ───────────────────────────────────────────────────────────
let sortState = {{ col: 'pred', dir: -1 }};
function sortTable(col) {{
  const tbody = document.getElementById('sigBody');
  const rows  = Array.from(tbody.querySelectorAll('tr'));
  // Toggle direction
  if (sortState.col === col) sortState.dir *= -1;
  else {{ sortState.col = col; sortState.dir = -1; }}
  // Update arrows
  ['sym','sig','pred','sent','price'].forEach(c => {{
    const el = document.getElementById('sa-'+c);
    if (el) {{ el.textContent = '↕'; el.classList.remove('active'); }}
  }});
  const activeArrow = document.getElementById('sa-'+col);
  if (activeArrow) {{
    activeArrow.textContent = sortState.dir === -1 ? '↓' : '↑';
    activeArrow.classList.add('active');
  }}
  rows.sort((a, b) => {{
    let av = a.dataset[col] || '';
    let bv = b.dataset[col] || '';
    const an = parseFloat(av);
    const bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return (an - bn) * sortState.dir;
    return av.localeCompare(bv) * sortState.dir;
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

// ── Signal filter ─────────────────────────────────────────────────────────
let activeFilters = new Set();
function toggleSigFilter(sig) {{
  const btn = document.getElementById('fb-'+sig);
  if (activeFilters.has(sig)) {{
    activeFilters.delete(sig);
    btn.classList.remove('active');
  }} else {{
    activeFilters.add(sig);
    btn.classList.add('active');
  }}
  filterSignals();
}}
function clearSigFilters() {{
  activeFilters.clear();
  ['BUY','AVOID','CAUTION','HOLD'].forEach(s => {{
    const b = document.getElementById('fb-'+s);
    if(b) b.classList.remove('active');
  }});
  document.getElementById('sigSearch').value = '';
  filterSignals();
}}
function filterSignals() {{
  const search = document.getElementById('sigSearch').value.toLowerCase();
  const rows   = document.querySelectorAll('#sigBody tr');
  let visible  = 0;
  rows.forEach(r => {{
    const sym  = (r.dataset.sym  || '').toLowerCase();
    const sig  = (r.dataset.sig  || '');
    const matchSearch = !search || sym.includes(search);
    const matchFilter = activeFilters.size === 0 || activeFilters.has(sig);
    const show = matchSearch && matchFilter;
    r.style.display = show ? '' : 'none';
    if (show) visible++;
  }});
  const ct = document.getElementById('sigCount');
  if (ct) ct.textContent = visible + ' shown';
}}
</script>
</body>
</html>'''

    with open(DASHBOARD_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n  ✅ AlphaEdge V5 Dashboard generated: {DASHBOARD_FILE}")
    print(f"  Portfolio : ${total_value:,.2f}  ({pnl_sgn}{total_pct:.2f}%)")
    print(f"  Sharpe    : {sharpe:.2f}  |  Max DD : {max_dd*100:.2f}%  |  Win Rate : {win_rate:.1f}%")
    print(f"  Signals   : {len(buy_sigs)} BUY  |  {len(avoid_sigs)} AVOID  |  {len(hold_sigs)} HOLD")
    print(f"  Weights   : {len(weight_data) if weight_data else 0} combinations in results")
    print(f"\n  Open: docs/index.html\n")
    return True


if __name__ == '__main__':
    print("\nAlphaEdge V5 — Generating Institutional Dashboard...")
    generate_dashboard()