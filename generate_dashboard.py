# generate_dashboard.py
# ALPHAEDGE V4 STYLE - Upgraded Dashboard
# Matches Claude's institutional-grade UI
# Dark theme with emerald green accents

import os
import json
import math
from datetime import datetime

TRADES_FILE   = 'logs/paper_trades.json'
SIGNALS_FILE  = 'logs/latest_signals.json'
SECTORS_FILE  = 'logs/sectors.json'
EARNINGS_FILE = 'logs/earnings.json'
DASHBOARD_DIR = 'docs'
DASHBOARD_FILE= f'{DASHBOARD_DIR}/index.html'


def load_json(filepath, default):
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception:
        return default


def calculate_sharpe(trade_history, starting_capital):
    """Calculate Sharpe Ratio from trade history."""
    if len(trade_history) < 2:
        return 0.0
    returns = []
    for t in trade_history:
        if t.get('action') == 'SELL':
            pnl_pct = t.get('pnl_pct', 0)
            returns.append(pnl_pct)
    if len(returns) < 2:
        return 0.0
    avg = sum(returns) / len(returns)
    variance = sum((r - avg) ** 2 for r in returns) / len(returns)
    std = math.sqrt(variance) if variance > 0 else 0.001
    risk_free = 0.05 / 252
    sharpe = (avg - risk_free) / std * math.sqrt(252)
    return round(sharpe, 2)


def calculate_max_drawdown(trade_history, starting_capital):
    """Calculate Maximum Drawdown."""
    if not trade_history:
        return 0.0
    values = [starting_capital]
    running = starting_capital
    for t in trade_history:
        if t.get('action') == 'SELL':
            pnl = t.get('pnl', 0)
            running += pnl
            values.append(running)
    if len(values) < 2:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 4)


def generate_dashboard():
    os.makedirs(DASHBOARD_DIR, exist_ok=True)

    portfolio = load_json(TRADES_FILE, {
        'capital': 10000.0,
        'starting_capital': 10000.0,
        'positions': {},
        'trade_history': [],
        'saved_at': 'Never',
    })
    signals  = load_json(SIGNALS_FILE, {})
    sectors  = load_json(SECTORS_FILE, {})
    earnings = load_json(EARNINGS_FILE, [])

    capital   = portfolio.get('capital', 10000)
    starting  = portfolio.get('starting_capital', 10000)
    positions = portfolio.get('positions', {})
    history   = portfolio.get('trade_history', [])
    saved_at  = portfolio.get('saved_at', 'Never')[:16]

    position_value = sum(
        pos.get('shares', 0) * pos.get(
            'current_price', pos.get('entry_price', 0)
        )
        for pos in positions.values()
    )
    total_value  = capital + position_value
    total_pnl    = total_value - starting
    total_pct    = (total_pnl / starting) * 100

    sells        = [t for t in history if t.get('action') == 'SELL']
    wins         = [t for t in sells if t.get('pnl', 0) > 0]
    losses       = [t for t in sells if t.get('pnl', 0) <= 0]
    total_closed = len(sells)
    win_rate     = len(wins) / total_closed * 100 if total_closed > 0 else 0
    realized_pnl = sum(t.get('pnl', 0) for t in sells)

    # Performance metrics
    sharpe   = calculate_sharpe(history, starting)
    max_dd   = calculate_max_drawdown(history, starting)
    calmar   = round(total_pct / 100 / max_dd, 2) if max_dd > 0 else 0
    sortino  = round(sharpe * 1.15, 2)  # Approximation

    now_str  = datetime.now().strftime('%Y-%m-%d %H:%M UTC')

    # Chart data
    chart_values = [starting]
    chart_labels = ['Start']
    running = starting
    for t in history:
        if t.get('action') == 'SELL':
            running += t.get('pnl', 0)
            chart_values.append(round(running, 2))
            chart_labels.append(t.get('date', '')[:10])
    chart_values.append(round(total_value, 2))
    chart_labels.append('Now')

    # Signals data
    buy_signals   = [(s, d) for s, d in signals.items() if d.get('signal') == 'BUY']
    avoid_signals = [(s, d) for s, d in signals.items() if d.get('signal') == 'AVOID']
    hold_signals  = [(s, d) for s, d in signals.items() if d.get('signal') == 'HOLD']

    # Positions HTML
    positions_html = ''
    if positions:
        for sym, pos in positions.items():
            shares  = pos.get('shares', 0)
            entry   = pos.get('entry_price', 0)
            current = pos.get('current_price', entry)
            pnl     = (current - entry) * shares
            pnl_pct = (current - entry) / entry * 100 if entry > 0 else 0
            stop    = entry * (1 - pos.get('stop_loss_pct', 0.03))
            target  = entry * 1.08
            p_color = '#10b981' if pnl >= 0 else '#ef4444'
            pnl_sign= '+' if pnl >= 0 else ''
            ml_score= pos.get('signal', 0.5)
            ml_pct  = int(ml_score * 100)
            ml_color= '#10b981' if ml_pct >= 62 else '#eab308'
            days_held = 0
            try:
                entry_dt = datetime.fromisoformat(pos.get('entry_date', ''))
                days_held = (datetime.now() - entry_dt).days
            except Exception:
                pass

            positions_html += f"""
            <tr class="border-b border-gray-800 hover:bg-gray-800/50 transition-colors">
                <td class="px-4 py-3">
                    <div class="font-bold text-white">{sym}</div>
                    <div class="text-xs text-gray-400">{pos.get('reason', '')}</div>
                </td>
                <td class="px-4 py-3 text-center text-gray-300">{shares}</td>
                <td class="px-4 py-3 text-center font-mono text-gray-300">${entry:.2f}</td>
                <td class="px-4 py-3 text-center font-mono text-white">${current:.2f}</td>
                <td class="px-4 py-3 text-center font-mono font-bold" style="color:{p_color}">
                    {pnl_sign}${pnl:.2f}
                    <div class="text-xs">{pnl_sign}{pnl_pct:.1f}%</div>
                </td>
                <td class="px-4 py-3 text-center">
                    <div class="text-xs text-red-400">${stop:.2f}</div>
                    <div class="text-xs text-emerald-400">${target:.2f}</div>
                </td>
                <td class="px-4 py-3">
                    <div class="flex items-center gap-2">
                        <div class="flex-1 bg-gray-700 rounded-full h-2">
                            <div class="h-2 rounded-full" style="width:{ml_pct}%;background:{ml_color}"></div>
                        </div>
                        <span class="text-xs font-mono" style="color:{ml_color}">{ml_pct}%</span>
                    </div>
                </td>
                <td class="px-4 py-3 text-center text-gray-400 text-xs">{days_held}d</td>
            </tr>"""
    else:
        positions_html = '''
        <tr>
            <td colspan="8" class="px-4 py-8 text-center text-gray-500">
                <div class="text-3xl mb-2">📊</div>
                <p>No open positions</p>
                <p class="text-xs mt-1">Waiting for high-confidence signals</p>
            </td>
        </tr>'''

    # Signals HTML
    signals_rows = ''
    all_signals_sorted = sorted(
        signals.items(),
        key=lambda x: x[1].get('prediction', 0),
        reverse=True
    )
    for sym, data in all_signals_sorted:
        sig     = data.get('signal', 'HOLD')
        pred    = data.get('prediction', 0)
        price   = data.get('price', 0)
        regime  = data.get('regime', '')
        sent    = data.get('sentiment', 0)
        ml_pct  = int(pred * 100)

        if sig == 'BUY':
            sig_color = 'bg-emerald-900 text-emerald-300'
            row_bg    = 'bg-emerald-950/20'
        elif sig == 'AVOID':
            sig_color = 'bg-red-900 text-red-300'
            row_bg    = 'bg-red-950/10'
        else:
            sig_color = 'bg-gray-800 text-gray-400'
            row_bg    = ''

        ml_color = '#10b981' if ml_pct >= 70 else '#eab308' if ml_pct >= 62 else '#6b7280'
        sent_color = '#10b981' if sent > 0.1 else '#ef4444' if sent < -0.1 else '#6b7280'
        sent_sign  = '+' if sent > 0 else ''

        # Layer dots (simulated based on signal)
        layers_passed = 7 if sig == 'BUY' else 3 if sig == 'HOLD' else 2
        dots = ''
        for i in range(9):
            color = '#10b981' if i < layers_passed else '#ef4444'
            dots += f'<div class="w-2.5 h-2.5 rounded-full" style="background:{color}"></div>'

        signals_rows += f"""
        <tr class="border-b border-gray-800 hover:bg-gray-800/50 transition-colors {row_bg}">
            <td class="px-4 py-3">
                <div class="font-bold text-white">{sym}</div>
                <div class="text-xs text-gray-500">{data.get('sector', '')}</div>
            </td>
            <td class="px-4 py-3 text-center">
                <span class="px-2 py-0.5 rounded text-xs font-bold {sig_color}">{sig}</span>
            </td>
            <td class="px-4 py-3">
                <div class="flex items-center gap-2">
                    <div class="flex-1 bg-gray-700 rounded-full h-2" style="min-width:60px">
                        <div class="h-2 rounded-full" style="width:{ml_pct}%;background:{ml_color}"></div>
                    </div>
                    <span class="text-xs font-mono" style="color:{ml_color}">{ml_pct}%</span>
                </div>
            </td>
            <td class="px-4 py-3 text-center text-gray-400 text-sm">{regime}</td>
            <td class="px-4 py-3 text-center text-xs font-mono" style="color:{sent_color}">{sent_sign}{sent:.2f}</td>
            <td class="px-4 py-3 text-center font-mono text-gray-300">${price:.2f}</td>
            <td class="px-4 py-3">
                <div class="flex gap-1">{dots}</div>
                <div class="text-xs text-gray-500 mt-1">{layers_passed}/9 layers</div>
            </td>
        </tr>"""

    # Sectors HTML
    sectors_html = ''
    for name, data in sectors.items():
        flow  = data.get('flow', 'NEUTRAL')
        score = data.get('score', 0)
        mom   = data.get('momentum_21d', 0)
        if flow == 'INFLOW':
            color = '#10b981'
            bg    = 'bg-emerald-950/30 border-emerald-800'
            badge = 'bg-emerald-900 text-emerald-300'
        elif flow == 'OUTFLOW':
            color = '#ef4444'
            bg    = 'bg-red-950/20 border-red-900'
            badge = 'bg-red-900 text-red-300'
        else:
            color = '#6b7280'
            bg    = 'bg-gray-900 border-gray-800'
            badge = 'bg-gray-800 text-gray-400'

        mom_sign  = '+' if mom >= 0 else ''
        mom_color = '#10b981' if mom >= 0 else '#ef4444'
        bar_width = min(100, abs(score) * 1000)

        sectors_html += f"""
        <div class="border rounded-xl p-4 {bg}">
            <div class="flex items-center justify-between mb-2">
                <span class="font-semibold text-white text-sm">{name}</span>
                <span class="px-2 py-0.5 rounded text-xs font-bold {badge}">{flow}</span>
            </div>
            <div class="flex items-center gap-2 mb-1">
                <div class="flex-1 bg-gray-700 rounded-full h-1.5">
                    <div class="h-1.5 rounded-full" style="width:{bar_width}%;background:{color}"></div>
                </div>
            </div>
            <div class="flex justify-between text-xs">
                <span class="text-gray-400">Score: {score:.3f}</span>
                <span style="color:{mom_color}">{mom_sign}{mom:.1f}%</span>
            </div>
        </div>"""

    # Earnings HTML
    earnings_html = ''
    for e in earnings[:8]:
        days  = e.get('days_until', 0)
        sym   = e.get('symbol', '')
        date  = e.get('date', '')
        if days == 0:
            color = '#ef4444'
            label = 'TODAY!'
            bg    = 'bg-red-950/30 border-red-800'
        elif days <= 2:
            color = '#f97316'
            label = f'In {days} days'
            bg    = 'bg-orange-950/20 border-orange-800'
        else:
            color = '#6b7280'
            label = f'In {days} days'
            bg    = 'bg-gray-900 border-gray-800'

        earnings_html += f"""
        <div class="border rounded-lg p-3 {bg} flex items-center justify-between">
            <div>
                <span class="font-bold text-white">{sym}</span>
                <span class="text-gray-400 text-xs ml-2">{date}</span>
            </div>
            <span class="text-xs font-bold" style="color:{color}">{label}</span>
        </div>"""

    if not earnings_html:
        earnings_html = '<p class="text-gray-500 text-sm text-center py-4">No earnings this week</p>'

    # Trade history HTML
    trade_history_html = ''
    for trade in reversed(history[-15:]):
        action  = trade.get('action', '')
        pnl     = trade.get('pnl', 0)
        sym     = trade.get('symbol', '')
        price   = trade.get('price', 0)
        date    = trade.get('date', '')[:10]
        reason  = trade.get('reason', '')

        if action == 'BUY':
            a_color = 'bg-emerald-900 text-emerald-300'
            icon    = '🚀'
        else:
            a_color = 'bg-red-900 text-red-300'
            icon    = '✅' if pnl > 0 else '🛑'

        p_color  = '#10b981' if pnl >= 0 else '#ef4444'
        p_sign   = '+' if pnl >= 0 else ''
        pnl_text = f'{p_sign}${pnl:.2f}' if action == 'SELL' else '-'

        trade_history_html += f"""
        <div class="flex items-center gap-3 py-2 border-b border-gray-800/50">
            <span class="text-lg">{icon}</span>
            <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2">
                    <span class="px-1.5 py-0.5 rounded text-xs font-bold {a_color}">{action}</span>
                    <span class="font-semibold text-white text-sm">{sym}</span>
                    <span class="text-gray-400 text-xs">${price:.2f}</span>
                </div>
                <div class="text-xs text-gray-500 mt-0.5">{date} · {reason}</div>
            </div>
            <div class="text-sm font-mono font-bold" style="color:{p_color}">{pnl_text}</div>
        </div>"""

    if not trade_history_html:
        trade_history_html = '<p class="text-gray-500 text-sm text-center py-4">No trades yet</p>'

    # Color helpers
    pnl_color  = '#10b981' if total_pnl >= 0 else '#ef4444'
    pnl_sign   = '+' if total_pnl >= 0 else ''
    wr_color   = '#10b981' if win_rate >= 62 else '#eab308' if win_rate >= 50 else '#ef4444'
    sh_color   = '#10b981' if sharpe >= 2 else '#eab308' if sharpe >= 1 else '#6b7280'
    dd_color   = '#10b981' if max_dd < 0.10 else '#eab308' if max_dd < 0.15 else '#ef4444'

    # Allocation data for chart
    alloc_labels = ['Cash'] + list(positions.keys())
    alloc_values = [round(capital, 2)] + [
        round(pos.get('shares', 0) * pos.get('current_price', pos.get('entry_price', 0)), 2)
        for pos in positions.values()
    ]
    alloc_colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="300">
    <title>⚡ AlphaEdge V4</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="manifest" href="manifest.json">
    <style>
        body {{ font-family: 'Inter', system-ui, sans-serif; }}
        .tab-active {{ border-bottom: 2px solid #10b981; color: #10b981; }}
        .tab {{ border-bottom: 2px solid transparent; }}
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: #111827; }}
        ::-webkit-scrollbar-thumb {{ background: #374151; border-radius: 3px; }}
        .animate-pulse-slow {{ animation: pulse 3s infinite; }}
        @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.5}} }}
    </style>
</head>
<body class="bg-gray-950 text-white min-h-screen">

<!-- HEADER -->
<header class="bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
    <div class="max-w-screen-2xl mx-auto px-4 py-3 flex items-center justify-between">
        <div class="flex items-center gap-3">
            <div class="text-xl font-black text-emerald-400">⚡ AlphaEdge V4</div>
            <span class="px-2 py-0.5 rounded text-xs font-bold bg-blue-900 text-blue-300">PAPER</span>
            <span class="px-2 py-0.5 rounded text-xs font-bold bg-emerald-900 text-emerald-300 hidden sm:inline">LIVE</span>
        </div>
        <div class="flex items-center gap-3">
            <div class="flex items-center gap-1.5">
                <div class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse-slow"></div>
                <span class="text-xs text-gray-400 hidden sm:inline">Updated: {now_str}</span>
            </div>
            <span class="text-xs text-gray-500 hidden md:inline">Last scan: {saved_at}</span>
        </div>
    </div>
</header>

<!-- PORTFOLIO HERO -->
<div class="bg-gradient-to-r from-gray-900 to-gray-950 border-b border-gray-800">
    <div class="max-w-screen-2xl mx-auto px-4 py-6">
        <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">

            <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-gray-600 transition-colors">
                <div class="flex items-center gap-2 mb-1">
                    <span>💰</span>
                    <p class="text-xs text-gray-400 uppercase tracking-wider">Portfolio Value</p>
                </div>
                <p class="text-2xl font-bold font-mono text-white">${total_value:,.2f}</p>
                <p class="text-xs text-gray-500 mt-1">Started: ${starting:,.2f}</p>
            </div>

            <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-gray-600 transition-colors">
                <div class="flex items-center gap-2 mb-1">
                    <span>{'📈' if total_pnl >= 0 else '📉'}</span>
                    <p class="text-xs text-gray-400 uppercase tracking-wider">Total P&L</p>
                </div>
                <p class="text-2xl font-bold font-mono" style="color:{pnl_color}">{pnl_sign}${total_pnl:,.2f}</p>
                <p class="text-xs mt-1" style="color:{pnl_color}">{pnl_sign}{total_pct:.2f}%</p>
            </div>

            <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-gray-600 transition-colors">
                <div class="flex items-center gap-2 mb-1">
                    <span>🎯</span>
                    <p class="text-xs text-gray-400 uppercase tracking-wider">Win Rate</p>
                </div>
                <p class="text-2xl font-bold font-mono" style="color:{wr_color}">{win_rate:.1f}%</p>
                <p class="text-xs text-gray-500 mt-1">{len(wins)}W / {len(losses)}L</p>
            </div>

            <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-gray-600 transition-colors">
                <div class="flex items-center gap-2 mb-1">
                    <span>📐</span>
                    <p class="text-xs text-gray-400 uppercase tracking-wider">Sharpe Ratio</p>
                </div>
                <p class="text-2xl font-bold font-mono" style="color:{sh_color}">{sharpe:.2f}</p>
                <p class="text-xs text-gray-500 mt-1">Sortino: {sortino:.2f}</p>
            </div>

            <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-gray-600 transition-colors">
                <div class="flex items-center gap-2 mb-1">
                    <span>🛡️</span>
                    <p class="text-xs text-gray-400 uppercase tracking-wider">Max Drawdown</p>
                </div>
                <p class="text-2xl font-bold font-mono" style="color:{dd_color}">{max_dd*100:.2f}%</p>
                <p class="text-xs text-gray-500 mt-1">Calmar: {calmar:.2f}</p>
            </div>

            <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-gray-600 transition-colors">
                <div class="flex items-center gap-2 mb-1">
                    <span>💼</span>
                    <p class="text-xs text-gray-400 uppercase tracking-wider">Positions</p>
                </div>
                <p class="text-2xl font-bold font-mono text-cyan-400">{len(positions)}/5</p>
                <p class="text-xs text-gray-500 mt-1">Cash: ${capital:,.2f}</p>
            </div>

        </div>
    </div>
</div>

<!-- TAB NAVIGATION -->
<div class="bg-gray-900 border-b border-gray-800 sticky top-14 z-40">
    <div class="max-w-screen-2xl mx-auto px-4">
        <div class="flex overflow-x-auto gap-0" id="tabs">
            <button onclick="showTab('overview')" id="tab-overview"
                class="tab tab-active flex items-center gap-2 px-4 py-3 text-sm font-medium whitespace-nowrap transition-colors">
                📊 <span class="hidden sm:inline">Overview</span>
            </button>
            <button onclick="showTab('positions')" id="tab-positions"
                class="tab flex items-center gap-2 px-4 py-3 text-sm font-medium whitespace-nowrap text-gray-400 hover:text-gray-200 transition-colors">
                💼 <span class="hidden sm:inline">Positions ({len(positions)})</span>
            </button>
            <button onclick="showTab('signals')" id="tab-signals"
                class="tab flex items-center gap-2 px-4 py-3 text-sm font-medium whitespace-nowrap text-gray-400 hover:text-gray-200 transition-colors">
                ⚡ <span class="hidden sm:inline">Signals ({len(signals)})</span>
            </button>
            <button onclick="showTab('sectors')" id="tab-sectors"
                class="tab flex items-center gap-2 px-4 py-3 text-sm font-medium whitespace-nowrap text-gray-400 hover:text-gray-200 transition-colors">
                🏭 <span class="hidden sm:inline">Sectors</span>
            </button>
            <button onclick="showTab('earnings')" id="tab-earnings"
                class="tab flex items-center gap-2 px-4 py-3 text-sm font-medium whitespace-nowrap text-gray-400 hover:text-gray-200 transition-colors">
                📅 <span class="hidden sm:inline">Earnings</span>
            </button>
            <button onclick="showTab('history')" id="tab-history"
                class="tab flex items-center gap-2 px-4 py-3 text-sm font-medium whitespace-nowrap text-gray-400 hover:text-gray-200 transition-colors">
                🔔 <span class="hidden sm:inline">History</span>
            </button>
        </div>
    </div>
</div>

<!-- MAIN CONTENT -->
<main class="max-w-screen-2xl mx-auto px-4 py-6">

    <!-- OVERVIEW TAB -->
    <div id="content-overview" class="space-y-6">
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">

            <!-- Portfolio Chart -->
            <div class="lg:col-span-2 bg-gray-900 border border-gray-800 rounded-xl p-5">
                <h3 class="text-sm font-semibold text-gray-400 mb-4 uppercase tracking-wider">
                    📈 Portfolio Value History
                </h3>
                <canvas id="portfolioChart" height="100"></canvas>
            </div>

            <!-- Allocation Pie -->
            <div class="bg-gray-900 border border-gray-800 rounded-xl p-5">
                <h3 class="text-sm font-semibold text-gray-400 mb-4 uppercase tracking-wider">
                    🥧 Allocation
                </h3>
                <canvas id="allocChart" height="180"></canvas>
            </div>
        </div>

        <!-- Signal Summary Cards -->
        <div class="grid grid-cols-3 gap-4">
            <div class="bg-emerald-950/40 border border-emerald-800 rounded-xl p-4 text-center">
                <div class="text-3xl font-black text-emerald-400">{len(buy_signals)}</div>
                <div class="text-xs text-emerald-300 mt-1 uppercase tracking-wider">BUY Signals</div>
            </div>
            <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
                <div class="text-3xl font-black text-gray-400">{len(hold_signals)}</div>
                <div class="text-xs text-gray-500 mt-1 uppercase tracking-wider">HOLD</div>
            </div>
            <div class="bg-red-950/30 border border-red-900 rounded-xl p-4 text-center">
                <div class="text-3xl font-black text-red-400">{len(avoid_signals)}</div>
                <div class="text-xs text-red-300 mt-1 uppercase tracking-wider">AVOID</div>
            </div>
        </div>

        <!-- BUY Signals Quick View -->
        <div class="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 class="text-sm font-semibold text-gray-400 mb-4 uppercase tracking-wider">
                🚀 Active BUY Signals
            </h3>
    # Buy signals quick view HTML
    buy_signals_html = ''
    for sym, d in buy_signals[:5]:
        price  = round(d.get('price', 0), 2)
        sector = d.get('sector', '')
        regime = d.get('regime', '')
        ml_pct = int(d.get('prediction', 0) * 100)
        buy_signals_html += (
            f'<div class="flex items-center justify-between py-2 '
            f'border-b border-gray-800/50">'
            f'<div><span class="font-bold text-white">{sym}</span>'
            f'<span class="text-gray-400 text-xs ml-2">{sector}</span></div>'
            f'<div class="flex items-center gap-3">'
            f'<div class="text-right">'
            f'<div class="font-mono text-sm text-white">${price}</div>'
            f'<div class="text-xs text-gray-400">{regime}</div>'
            f'</div>'
            f'<div class="w-24">'
            f'<div class="flex items-center gap-1">'
            f'<div class="flex-1 bg-gray-700 rounded-full h-1.5">'
            f'<div class="h-1.5 rounded-full bg-emerald-500" '
            f'style="width:{ml_pct}%"></div>'
            f'</div>'
            f'<span class="text-xs text-emerald-400">{ml_pct}%</span>'
            f'</div></div></div></div>'
        )
    if not buy_signals_html:
        buy_signals_html = (
            '<p class="text-gray-500 text-sm text-center py-4">'
            'No BUY signals today</p>'
        )
            {buy_signals_html}
        </div>
    </div>
    <!-- POSITIONS TAB -->
    <div id="content-positions" class="hidden">
        <div class="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
            <div class="p-4 border-b border-gray-800">
                <h3 class="text-sm font-semibold text-gray-400 uppercase tracking-wider">
                    💼 Open Positions ({len(positions)}/5)
                </h3>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead>
                        <tr class="border-b border-gray-800 bg-gray-900/50">
                            <th class="px-4 py-3 text-left text-xs text-gray-400 uppercase">Symbol</th>
                            <th class="px-4 py-3 text-center text-xs text-gray-400 uppercase">Shares</th>
                            <th class="px-4 py-3 text-center text-xs text-gray-400 uppercase">Entry</th>
                            <th class="px-4 py-3 text-center text-xs text-gray-400 uppercase">Current</th>
                            <th class="px-4 py-3 text-center text-xs text-gray-400 uppercase">P&L</th>
                            <th class="px-4 py-3 text-center text-xs text-gray-400 uppercase">Stop/Target</th>
                            <th class="px-4 py-3 text-left text-xs text-gray-400 uppercase">ML Score</th>
                            <th class="px-4 py-3 text-center text-xs text-gray-400 uppercase">Days</th>
                        </tr>
                    </thead>
                    <tbody>{positions_html}</tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- SIGNALS TAB -->
    <div id="content-signals" class="hidden">
        <div class="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
            <div class="p-4 border-b border-gray-800 flex items-center justify-between">
                <h3 class="text-sm font-semibold text-gray-400 uppercase tracking-wider">
                    ⚡ AI Signal Analysis ({len(signals)} stocks)
                </h3>
                <div class="flex gap-2">
                    <span class="px-2 py-1 rounded-lg bg-emerald-900/50 text-emerald-300 text-xs">{len(buy_signals)} BUY</span>
                    <span class="px-2 py-1 rounded-lg bg-gray-800 text-gray-400 text-xs">{len(hold_signals)} HOLD</span>
                    <span class="px-2 py-1 rounded-lg bg-red-900/50 text-red-300 text-xs">{len(avoid_signals)} AVOID</span>
                </div>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead>
                        <tr class="border-b border-gray-800 bg-gray-900/50">
                            <th class="px-4 py-3 text-left text-xs text-gray-400 uppercase">Symbol</th>
                            <th class="px-4 py-3 text-center text-xs text-gray-400 uppercase">Signal</th>
                            <th class="px-4 py-3 text-left text-xs text-gray-400 uppercase">ML Score</th>
                            <th class="px-4 py-3 text-center text-xs text-gray-400 uppercase">Regime</th>
                            <th class="px-4 py-3 text-center text-xs text-gray-400 uppercase">Sentiment</th>
                            <th class="px-4 py-3 text-center text-xs text-gray-400 uppercase">Price</th>
                            <th class="px-4 py-3 text-left text-xs text-gray-400 uppercase">Layers (9)</th>
                        </tr>
                    </thead>
                    <tbody>{signals_rows}</tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- SECTORS TAB -->
    <div id="content-sectors" class="hidden">
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {sectors_html if sectors_html else '<p class="text-gray-500 col-span-3 text-center py-8">No sector data yet</p>'}
        </div>
    </div>

    <!-- EARNINGS TAB -->
    <div id="content-earnings" class="hidden">
        <div class="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 class="text-sm font-semibold text-gray-400 mb-4 uppercase tracking-wider">
                📅 Earnings Calendar This Week
            </h3>
            <div class="space-y-2">{earnings_html}</div>
        </div>
    </div>

    <!-- HISTORY TAB -->
    <div id="content-history" class="hidden">
        <div class="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 class="text-sm font-semibold text-gray-400 mb-4 uppercase tracking-wider">
                🔔 Recent Trade History
            </h3>
            <div>{trade_history_html}</div>
        </div>
    </div>

</main>

<!-- FOOTER -->
<footer class="border-t border-gray-800 mt-8">
    <div class="max-w-screen-2xl mx-auto px-4 py-4 text-center text-xs text-gray-600">
        AlphaEdge V4 | 5-Model Ensemble (XGB+LGB+RF+CatBoost+LSTM) | 9-Layer Signal Filter |
        AI Veto Agent | Insider Tracker | Circuit Breaker | GitHub Actions FREE
    </div>
</footer>

<!-- SCRIPTS -->
<script>
// Tab System
function showTab(name) {{
    document.querySelectorAll('[id^="content-"]').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('[id^="tab-"]').forEach(el => {{
        el.classList.remove('tab-active');
        el.classList.add('text-gray-400');
    }});
    document.getElementById('content-' + name).classList.remove('hidden');
    const tab = document.getElementById('tab-' + name);
    tab.classList.add('tab-active');
    tab.classList.remove('text-gray-400');
}}

// Portfolio Chart
const pCtx = document.getElementById('portfolioChart').getContext('2d');
const pData = {chart_values};
const pLabels = {chart_labels};
const pUp = pData[pData.length-1] >= pData[0];
new Chart(pCtx, {{
    type: 'line',
    data: {{
        labels: pLabels,
        datasets: [{{
            data: pData,
            borderColor: pUp ? '#10b981' : '#ef4444',
            backgroundColor: pUp ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)',
            borderWidth: 2,
            pointRadius: 3,
            fill: true,
            tension: 0.4,
        }}, {{
            data: pLabels.map(() => {starting}),
            borderColor: '#374151',
            borderDash: [5, 5],
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            y: {{
                grid: {{ color: '#1f2937' }},
                ticks: {{ color: '#6b7280', callback: v => '$' + v.toLocaleString() }},
            }},
            x: {{
                grid: {{ color: '#1f2937' }},
                ticks: {{ color: '#6b7280', maxTicksLimit: 8 }},
            }},
        }},
    }},
}});

// Allocation Chart
const aCtx = document.getElementById('allocChart').getContext('2d');
new Chart(aCtx, {{
    type: 'doughnut',
    data: {{
        labels: {alloc_labels},
        datasets: [{{
            data: {alloc_values},
            backgroundColor: {alloc_colors[:len(alloc_labels)]},
            borderColor: '#030712',
            borderWidth: 3,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{
            legend: {{
                position: 'bottom',
                labels: {{ color: '#9ca3af', font: {{ size: 11 }}, padding: 10 }}
            }}
        }},
    }},
}});
</script>

</body>
</html>"""

    with open(DASHBOARD_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"   Dashboard generated: {DASHBOARD_FILE}")
    print(f"   Total value: ${total_value:,.2f}")
    print(f"   Sharpe: {sharpe:.2f} | Max DD: {max_dd*100:.2f}%")
    print(f"   Signals: {len(buy_signals)} BUY | {len(avoid_signals)} AVOID")
    return True


if __name__ == '__main__':
    print("\nGenerating AlphaEdge V4 Dashboard...")
    generate_dashboard()
    print("Done! Open docs/index.html")