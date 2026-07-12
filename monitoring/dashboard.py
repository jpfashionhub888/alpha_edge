# monitoring/dashboard.py
"""
AlphaEdge Bloomberg Terminal Dashboard -- V7
Bloomberg-style terminal with:
  - Dark background + orange accents
  - IBM Plex Mono monospace font
  - 7-tab navigation (Overview / Positions / Signals / Sectors / Earnings / History / SysConfig)
  - Live KPI strip (updates every 60s)
"""

import json, os
import pandas as pd
from datetime import datetime
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# ── Settings Load ─────────────────────────────────────────────────────────────
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
    atr_stop_mult   = getattr(settings, 'ATR_STOP_MULT', 1.0)
    atr_target_mult = getattr(settings, 'ATR_TARGET_MULT', 2.5)
    trailing_mult   = getattr(settings, 'TRAILING_STOP_MULTIPLIER', 0.8)
    max_risk_trade  = getattr(settings, 'MAX_RISK_PER_TRADE', 0.02)
    max_port_risk   = getattr(settings, 'MAX_PORTFOLIO_RISK', 0.06)
    vol_spike_min   = getattr(settings, 'VOLUME_SPIKE_MIN', 1.3)
    min_rr          = getattr(settings, 'MIN_RISK_REWARD', 2.0)
    mtf_weight      = getattr(settings, 'MTF_WEIGHT_IN_SIGNAL', 0.15)
    watchlist_len   = len(getattr(settings, 'STOCK_WATCHLIST', []))
except Exception:
    kelly_active = True;  kelly_mult = 0.5;  kelly_rr = 2.5
    max_pos_pct = 0.15;   buy_threshold = 0.63; max_dd_limit = 0.10
    max_daily_loss = 0.02; max_positions = 5;   atr_stop_mult = 1.0
    atr_target_mult = 2.5; trailing_mult = 0.8; max_risk_trade = 0.02
    max_port_risk = 0.06;  vol_spike_min = 1.3; min_rr = 2.0
    mtf_weight = 0.15;     watchlist_len = 41

TRADES_FILE   = 'logs/paper_trades_stocks_only.json'
SIGNALS_FILE  = 'logs/latest_signals.json'
SECTORS_FILE  = 'logs/sectors.json'
EARNINGS_FILE = 'logs/earnings.json'

# ── Bloomberg Colors ──────────────────────────────────────────────────────────
C = {
    'bg'     : '#080B10', 'panel'  : '#0D1117', 'card'   : '#10151E',
    'card2'  : '#151C28', 'border' : '#1A2235', 'border2': '#222D42',
    'text'   : '#CDD5E0', 'mid'    : '#7A8599', 'dim'    : '#3D4A5C',
    'orange' : '#F58220', 'green'  : '#00C896', 'red'    : '#FF4757',
    'yellow' : '#FFD700', 'cyan'   : '#00B4D8', 'purple' : '#B57BFF',
}

MONO = "'IBM Plex Mono', 'JetBrains Mono', 'Courier New', monospace"
SANS = "'Inter', 'Segoe UI', Arial, sans-serif"

TBL_HDR = {
    'backgroundColor': '#0D1117', 'color': '#F58220', 'fontWeight': '700',
    'border': '1px solid #1A2235', 'fontSize': '10px', 'letterSpacing': '1.5px',
    'textTransform': 'uppercase', 'fontFamily': MONO, 'padding': '10px 14px',
}
TBL_CELL = {
    'backgroundColor': '#10151E', 'color': '#CDD5E0',
    'border': '1px solid #1A2235', 'textAlign': 'center',
    'padding': '9px 14px', 'fontSize': '12px', 'fontFamily': MONO,
}

BLOOMBERG_CSS = r"""<!DOCTYPE html>
<html>
<head>
    {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

    <!-- PWA: AlphaEdge logo on mobile home screen & browser tab -->
    <link rel="manifest" href="/assets/manifest.json">
    <meta name="theme-color" content="#F58220">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="AlphaEdge">
    <link rel="apple-touch-icon" href="/assets/icon-192.png">
    <link rel="icon" type="image/png" sizes="192x192" href="/assets/icon-192.png">
    <link rel="icon" type="image/png" sizes="512x512" href="/assets/icon-512.png">

    <style>
        *{box-sizing:border-box;margin:0;padding:0}
        html,body{background:#080B10}
        body{scrollbar-width:thin;scrollbar-color:#1A2235 #0D1117}
        ::-webkit-scrollbar{width:5px;height:5px}
        ::-webkit-scrollbar-track{background:#0D1117}
        ::-webkit-scrollbar-thumb{background:#1A2235;border-radius:3px}
        .dash-tabs{border-bottom:1px solid #1A2235 !important;background:#0D1117 !important}
        .dash-tab{background:#0D1117 !important;color:#3D4A5C !important;border:none !important;
            border-bottom:2px solid transparent !important;border-radius:0 !important;
            padding:11px 22px !important;font-family:'IBM Plex Mono',monospace !important;
            font-size:10px !important;letter-spacing:2px !important;font-weight:700 !important;
            text-transform:uppercase !important;cursor:pointer !important;
            transition:color 0.15s,border-color 0.15s !important}
        .dash-tab:hover{color:#7A8599 !important}
        .dash-tab--selected{color:#F58220 !important;border-bottom:2px solid #F58220 !important;background:#0D1117 !important}
        .dash-tab-content{background:transparent !important;padding:0 !important}
        .dash-filter input{background:#0D1117 !important;color:#CDD5E0 !important;
            border-color:#1A2235 !important;font-family:'IBM Plex Mono',monospace !important;font-size:11px !important}
        @keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
        .live-dot{animation:blink 2s ease-in-out infinite}
    </style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body></html>"""


def _jload(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def load_portfolio(): return _jload(TRADES_FILE,   {'capital': 10000.0, 'starting_capital': 10000.0, 'positions': {}, 'trade_history': []})
def load_signals():   return _jload(SIGNALS_FILE,  {})
def load_sectors():   return _jload(SECTORS_FILE,  {})
def load_earnings():  return _jload(EARNINGS_FILE, [])


def kelly_sizing(pred):
    if not kelly_active:
        return 0.0, 0.0
    p = float(pred)
    f = max(0.0, (p * (kelly_rr + 1) - 1) / kelly_rr)
    return f, min(f * kelly_mult, max_pos_pct)


def bb_panel(title, children, subtitle=''):
    return html.Div(
        style={'backgroundColor': C['card'], 'border': f"1px solid {C['border']}",
               'borderRadius': '3px', 'overflow': 'hidden', 'marginBottom': '12px'},
        children=[
            html.Div(
                style={'backgroundColor': C['panel'], 'borderBottom': f"2px solid {C['orange']}",
                       'padding': '9px 16px', 'display': 'flex',
                       'justifyContent': 'space-between', 'alignItems': 'center'},
                children=[
                    html.Span(title, style={'color': C['orange'], 'fontSize': '11px', 'fontWeight': '700',
                                            'letterSpacing': '2px', 'fontFamily': MONO, 'textTransform': 'uppercase'}),
                    html.Span(subtitle, style={'color': C['dim'], 'fontSize': '10px', 'fontFamily': MONO}),
                ]
            ),
            html.Div(children, style={'padding': '16px'}),
        ]
    )


def stat_row(k, v, vc=None):
    return html.Div(
        style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
               'padding': '7px 0', 'borderBottom': f"1px solid {C['border']}"},
        children=[
            html.Span(k, style={'color': C['mid'], 'fontSize': '11px', 'fontFamily': MONO}),
            html.Span(v, style={'color': vc or C['text'], 'fontSize': '11px', 'fontFamily': MONO, 'fontWeight': '700'}),
        ]
    )


def _no_data(msg):
    return html.Div([html.Span(msg, style={'color': C['mid']})],
                    style={'fontFamily': MONO, 'fontSize': '12px', 'padding': '16px 0'})


def create_app():
    app = Dash(
        __name__,
        title='AlphaEdge | Bloomberg Terminal',
        update_title=None,
        suppress_callback_exceptions=True,
    )
    app.index_string = BLOOMBERG_CSS

    app.layout = html.Div(
        style={'backgroundColor': C['bg'], 'minHeight': '100vh', 'fontFamily': SANS},
        children=[
            dcc.Interval(id='tick', interval=60_000, n_intervals=0),
            # Status bar
            html.Div(id='status-bar', style={
                'backgroundColor': '#050709', 'borderBottom': f"1px solid {C['border']}",
                'height': '28px', 'display': 'flex', 'justifyContent': 'space-between',
                'alignItems': 'center', 'padding': '0 24px',
            }),
            # Header
            html.Div(
                style={'backgroundColor': C['panel'], 'borderBottom': f"3px solid {C['orange']}",
                       'padding': '14px 24px', 'display': 'flex',
                       'justifyContent': 'space-between', 'alignItems': 'center'},
                children=[
                    html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '14px'}, children=[
                        html.Div('AE', style={
                            'backgroundColor': C['orange'], 'color': '#080B10', 'fontFamily': MONO,
                            'fontWeight': '900', 'fontSize': '14px', 'padding': '5px 9px',
                            'borderRadius': '3px', 'letterSpacing': '1px',
                        }),
                        html.Div([
                            html.Div('ALPHAEDGE TRADING TERMINAL', style={
                                'color': C['text'], 'fontFamily': MONO, 'fontWeight': '700',
                                'fontSize': '15px', 'letterSpacing': '4px',
                            }),
                            html.Div('INSTITUTIONAL AI MARKET SCANNING SYSTEM  --  V7', style={
                                'color': C['dim'], 'fontFamily': MONO, 'fontSize': '9px',
                                'letterSpacing': '2px', 'marginTop': '3px',
                            }),
                        ]),
                    ]),
                    html.Div(id='header-status', style={'display': 'flex', 'alignItems': 'center', 'gap': '28px'}),
                ]
            ),
            # KPI strip
            html.Div(id='kpi-strip', style={
                'backgroundColor': '#090C13', 'borderBottom': f"1px solid {C['border']}",
                'display': 'flex', 'overflowX': 'auto',
            }),
            # Tabs
            html.Div(style={'backgroundColor': C['panel'], 'padding': '0 24px'}, children=[
                dcc.Tabs(
                    id='tabs', value='overview',
                    colors={'border': C['border'], 'primary': C['orange'], 'background': C['panel']},
                    children=[
                        dcc.Tab(label='OVERVIEW',   value='overview'),
                        dcc.Tab(label='POSITIONS',  value='positions'),
                        dcc.Tab(label='SIGNALS',    value='signals'),
                        dcc.Tab(label='SECTORS',    value='sectors'),
                        dcc.Tab(label='EARNINGS',   value='earnings'),
                        dcc.Tab(label='HISTORY',    value='history'),
                        dcc.Tab(label='SYS CONFIG', value='sysconfig'),
                    ],
                )
            ]),
            html.Div(id='tab-content', style={'padding': '20px 24px', 'minHeight': '600px'}),
            # Footer
            html.Div(
                style={'backgroundColor': '#050709', 'borderTop': f"1px solid {C['border']}",
                       'padding': '8px 24px', 'display': 'flex', 'justifyContent': 'space-between',
                       'fontSize': '9px', 'fontFamily': MONO, 'color': C['dim'], 'letterSpacing': '0.5px'},
                children=[
                    html.Span('ALPHAEDGE V7  |  XGB + LightGBM + CatBoost + RF + LSTM  |  Kelly Sizing  |  ATR Stops  |  Alpaca Paper'),
                    html.Span('FOR RESEARCH PURPOSES ONLY  |  NOT FINANCIAL ADVICE'),
                ]
            ),
        ]
    )

    # ── HEADER + KPI CALLBACK ─────────────────────────────────────────────────
    @app.callback(
        [Output('status-bar', 'children'), Output('header-status', 'children'), Output('kpi-strip', 'children')],
        [Input('tick', 'n_intervals')]
    )
    def update_header(n):
        p        = load_portfolio()
        capital  = p.get('capital', 10000)
        starting = p.get('starting_capital', 10000)
        positions = p.get('positions', {})
        history  = p.get('trade_history', [])
        pos_val  = sum(pos.get('shares', 0) * pos.get('current_price', pos.get('entry_price', 0)) for pos in positions.values())
        total    = capital + pos_val
        pnl      = total - starting
        pnl_pct  = pnl / starting * 100 if starting > 0 else 0
        sells    = [t for t in history if t.get('action') in {'SELL', 'PARTIAL_SELL'}]
        wins     = sum(1 for t in sells if t.get('pnl', 0) > 0)
        losses   = sum(1 for t in sells if t.get('pnl', 0) <= 0)
        closed   = wins + losses
        wr       = wins / closed * 100 if closed > 0 else 0
        realized = sum(t.get('pnl', 0) for t in sells)
        drawdown = min(0, pnl / starting * 100) if starting > 0 else 0
        pnl_c    = C['green'] if pnl >= 0 else C['red']
        wr_c     = C['green'] if wr >= 50 else C['red']
        dd_c     = C['green'] if drawdown >= -2 else (C['yellow'] if drawdown >= -5 else C['red'])
        now_str  = datetime.now().strftime('%Y-%m-%d  %H:%M:%S ET')

        status_bar = [
            html.Span('PAPER TRADING  |  ALPACA CONNECTED  |  BULL MARKET  |  SCAN: 16:15 ET DAILY',
                      style={'color': C['green'], 'fontSize': '9px', 'fontFamily': MONO, 'letterSpacing': '1px'}),
            html.Span(now_str, style={'color': C['mid'], 'fontSize': '9px', 'fontFamily': MONO}),
        ]

        header_status = [
            html.Div([
                html.Div('NEXT SCAN', style={'color': C['dim'], 'fontSize': '9px', 'fontFamily': MONO, 'letterSpacing': '1px', 'marginBottom': '3px'}),
                html.Div('16:15 ET',  style={'color': C['orange'], 'fontSize': '14px', 'fontFamily': MONO, 'fontWeight': '700'}),
            ]),
            html.Div(style={'width': '1px', 'height': '32px', 'backgroundColor': C['border']}),
            html.Div([
                html.Div(className='live-dot', style={
                    'width': '9px', 'height': '9px', 'borderRadius': '50%',
                    'backgroundColor': C['green'], 'boxShadow': f"0 0 8px {C['green']}", 'margin': '0 auto 4px',
                }),
                html.Div('LIVE', style={'color': C['green'], 'fontSize': '9px', 'fontFamily': MONO, 'fontWeight': '700', 'letterSpacing': '1px'}),
            ], style={'textAlign': 'center'}),
        ]

        def kpi(label, value, color, sub=''):
            return html.Div(
                style={'padding': '10px 22px', 'borderRight': f"1px solid {C['border']}", 'minWidth': '140px', 'flexShrink': '0'},
                children=[
                    html.Div(label, style={'color': C['dim'], 'fontSize': '8px', 'fontFamily': MONO, 'letterSpacing': '2px', 'textTransform': 'uppercase', 'marginBottom': '4px'}),
                    html.Div(value, style={'color': color, 'fontSize': '19px', 'fontFamily': MONO, 'fontWeight': '700', 'letterSpacing': '-0.5px'}),
                    html.Div(sub,   style={'color': C['dim'], 'fontSize': '9px', 'fontFamily': MONO, 'marginTop': '2px'}),
                ]
            )

        kpi_strip = [
            kpi('NET LIQUIDATION', f'${total:,.2f}',     C['text'],    f'Started ${starting:,.0f}'),
            kpi('CASH AVAILABLE',  f'${capital:,.2f}',   C['cyan'],    f'{capital/total*100:.1f}% of port' if total > 0 else ''),
            kpi('TOTAL P&L',       f'${pnl:+,.2f}',      pnl_c,        f'{pnl_pct:+.2f}% overall'),
            kpi('REALIZED P&L',    f'${realized:+,.2f}', pnl_c,        f'{closed} closed trades'),
            kpi('POSITIONS',       f'{len(positions)}/{max_positions}', C['yellow'], f'${pos_val:,.2f} invested'),
            kpi('WIN RATE',        f'{wr:.1f}%',          wr_c,         f'{wins}W / {losses}L'),
            kpi('DRAWDOWN',        f'{drawdown:.2f}%',    dd_c,         'Peak-to-trough'),
        ]
        return status_bar, header_status, kpi_strip

    # ── TAB CONTENT CALLBACK ─────────────────────────────────────────────────
    @app.callback(
        Output('tab-content', 'children'),
        [Input('tabs', 'value'), Input('tick', 'n_intervals')]
    )
    def render_tab(tab, n):
        p         = load_portfolio()
        signals   = load_signals()
        sectors   = load_sectors()
        earnings  = load_earnings()
        capital   = p.get('capital', 10000)
        starting  = p.get('starting_capital', 10000)
        positions = p.get('positions', {})
        history   = p.get('trade_history', [])
        pos_val   = sum(pos.get('shares', 0) * pos.get('current_price', pos.get('entry_price', 0)) for pos in positions.values())
        total     = capital + pos_val
        pnl       = total - starting
        sells     = [t for t in history if t.get('action') in {'SELL', 'PARTIAL_SELL'}]
        wins      = sum(1 for t in sells if t.get('pnl', 0) > 0)
        losses    = sum(1 for t in sells if t.get('pnl', 0) <= 0)
        closed    = wins + losses
        wr        = wins / closed * 100 if closed > 0 else 0
        realized  = sum(t.get('pnl', 0) for t in sells)
        pnl_c     = C['green'] if pnl >= 0 else C['red']

        # ── OVERVIEW ──────────────────────────────────────────────────────
        if tab == 'overview':
            vals = [starting]; dates = ['START']
            for t in history:
                if t.get('action') in {'SELL', 'PARTIAL_SELL'}:
                    vals.append(vals[-1] + t.get('pnl', 0))
                    dates.append(t.get('date', '')[:10])
            vals.append(total); dates.append('NOW')
            lc = C['green'] if vals[-1] >= vals[0] else C['red']
            fc = 'rgba(0,200,150,0.06)' if vals[-1] >= vals[0] else 'rgba(255,71,87,0.06)'
            eq = go.Figure()
            eq.add_trace(go.Scatter(x=dates, y=vals, mode='lines+markers',
                line=dict(color=lc, width=2.5, shape='spline'),
                marker=dict(size=5, color=lc, line=dict(color=C['card'], width=1)),
                fill='tozeroy', fillcolor=fc))
            eq.add_hline(y=starting, line_dash='dash', line_color=C['dim'], opacity=0.5)
            eq.update_layout(
                plot_bgcolor=C['card'], paper_bgcolor=C['card'],
                font=dict(color=C['mid'], size=10, family=MONO),
                xaxis=dict(gridcolor=C['border'], showgrid=True, tickfont=dict(size=9), zeroline=False),
                yaxis=dict(gridcolor=C['border'], showgrid=True, tickprefix='$', tickfont=dict(size=9), zeroline=False),
                margin=dict(l=65, r=16, t=14, b=30), height=270, showlegend=False)

            al_l = ['Cash'] + list(positions.keys())
            al_v = [capital] + [p_.get('shares', 0) * p_.get('current_price', p_.get('entry_price', 0)) for p_ in positions.values()]
            al_c = [C['border2'], C['orange'], C['green'], C['cyan'], C['purple'], C['yellow']]
            alloc = go.Figure(go.Pie(labels=al_l, values=al_v,
                marker=dict(colors=al_c[:len(al_l)], line=dict(color=C['card'], width=2)),
                hole=0.65, textfont=dict(color=C['text'], size=10, family=MONO), textinfo='label+percent'))
            alloc.update_layout(plot_bgcolor=C['card'], paper_bgcolor=C['card'],
                font=dict(color=C['text']), margin=dict(l=10, r=10, t=10, b=10), height=270, showlegend=False)

            gp = sum(t.get('pnl', 0) for t in sells if t.get('pnl', 0) > 0)
            gl = sum(t.get('pnl', 0) for t in sells if t.get('pnl', 0) <= 0)
            pf = abs(gp / gl) if gl != 0 else 0
            aw = gp / wins    if wins > 0 else 0
            al_ = gl / losses if losses > 0 else 0
            exp = (wr / 100 * aw) + ((1 - wr / 100) * al_)
            pf_c = C['green'] if pf >= 1.5 else (C['yellow'] if pf >= 1 else C['red'])

            return html.Div([
                html.Div(style={'display': 'grid', 'gridTemplateColumns': '2fr 1fr', 'gap': '12px', 'marginBottom': '12px'}, children=[
                    bb_panel('EQUITY GROWTH CURVE', dcc.Graph(figure=eq,    config={'displayModeBar': False}), f'NAV: ${total:,.2f}'),
                    bb_panel('CAPITAL ALLOCATION',  dcc.Graph(figure=alloc, config={'displayModeBar': False})),
                ]),
                bb_panel('PERFORMANCE ANALYTICS', html.Div(
                    style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '32px'},
                    children=[
                        html.Div([
                            stat_row('Net Liquidation',  f'${total:,.2f}'),
                            stat_row('Starting Capital', f'${starting:,.2f}'),
                            stat_row('Total P&L',        f'${pnl:+,.2f}',    pnl_c),
                            stat_row('Total Return',     f'{pnl/starting*100:+.2f}%' if starting else 'N/A', pnl_c),
                            stat_row('Cash Available',   f'${capital:,.2f}'),
                        ]),
                        html.Div([
                            stat_row('Realized P&L',  f'${realized:+,.2f}', pnl_c),
                            stat_row('Win Rate',      f'{wr:.1f}%',          C['green'] if wr >= 50 else C['red']),
                            stat_row('Profit Factor', f'{pf:.2f}' if closed > 0 else 'N/A', pf_c),
                            stat_row('Avg Win',       f'${aw:+.2f}',         C['green']),
                            stat_row('Avg Loss',      f'${al_:+.2f}',        C['red']),
                        ]),
                        html.Div([
                            stat_row('Total Trades',     str(len(history))),
                            stat_row('Closed Trades',    str(closed)),
                            stat_row('Open Positions',   f'{len(positions)}/{max_positions}'),
                            stat_row('Expectancy/Trade', f'${exp:+.2f}',      C['green'] if exp >= 0 else C['red']),
                            stat_row('Sharpe Ratio',     'N/A (need 20+ trades)', C['dim']),
                        ]),
                    ]
                )),
            ])

        # ── POSITIONS ─────────────────────────────────────────────────────
        elif tab == 'positions':
            if not positions:
                return bb_panel('ACTIVE POSITIONS', _no_data('NO OPEN POSITIONS -- WATCHING 41 STOCKS'))
            rows = []
            for sym, pos in positions.items():
                shares = pos.get('shares', 0); entry = pos.get('entry_price', 0)
                curr   = pos.get('current_price', entry)
                p_pnl  = (curr - entry) * shares
                p_pct  = (curr - entry) / entry * 100 if entry > 0 else 0
                sl     = entry * (1 - pos.get('stop_loss_pct', 0.08))
                _, af  = kelly_sizing(pos.get('signal', 0.5))
                rows.append({'SYMBOL': sym, 'SHARES': shares, 'ENTRY': f'${entry:.2f}', 'CURRENT': f'${curr:.2f}',
                    'STOP LOSS': f'${sl:.2f}', 'ATR': f'${pos.get("atr",0):.2f}',
                    'COST': f'${pos.get("cost", shares*entry):.2f}', 'KELLY SZ': f'{af*100:.1f}%',
                    'P&L $': f'${p_pnl:+.2f}', 'P&L %': f'{p_pct:+.2f}%',
                    'ENTRY DATE': pos.get('entry_date', '')[:10], 'REASON': pos.get('reason', '').upper()})
            df = pd.DataFrame(rows)
            return bb_panel(f'ACTIVE POSITIONS  --  {len(positions)}/{max_positions} SLOTS  |  ${pos_val:,.2f} INVESTED',
                dash_table.DataTable(data=df.to_dict('records'), columns=[{'name': c, 'id': c} for c in df.columns],
                    style_header=TBL_HDR, style_cell=TBL_CELL,
                    style_data_conditional=[
                        {'if': {'filter_query': '{P&L $} contains "+"', 'column_id': 'P&L $'}, 'color': C['green'], 'fontWeight': 'bold'},
                        {'if': {'filter_query': '{P&L $} contains "-"', 'column_id': 'P&L $'}, 'color': C['red'],   'fontWeight': 'bold'},
                        {'if': {'filter_query': '{P&L %} contains "+"', 'column_id': 'P&L %'}, 'color': C['green']},
                        {'if': {'filter_query': '{P&L %} contains "-"', 'column_id': 'P&L %'}, 'color': C['red']},
                    ], sort_action='native', page_size=20))

        # ── SIGNALS ───────────────────────────────────────────────────────
        elif tab == 'signals':
            if not signals:
                return bb_panel('AI SCAN SIGNALS', _no_data('NO SIGNAL DATA -- SCANNER RUNS AT 16:15 ET DAILY'))
            rows = []
            for sym, d in sorted(signals.items(), key=lambda x: x[1].get('combined', x[1].get('prediction', 0)), reverse=True):
                sig = d.get('signal', 'HOLD'); pred = d.get('prediction', 0)
                comb = d.get('combined', 0);   _, af = kelly_sizing(pred)
                rows.append({'SYMBOL': sym, 'SIGNAL': sig, 'AI SCORE': f'{pred*100:.1f}%',
                    'COMBINED': f'{comb:.3f}', 'REGIME': d.get('regime', '').upper(),
                    'SENTIMENT': f'{d.get("sentiment", 0):+.3f}', 'SECTOR': d.get('sector', ''),
                    'KELLY %': f'{af*100:.1f}%', 'TARGET $': f'${total * af:,.0f}', 'PRICE': f'${d.get("price", 0):.2f}'})
            df = pd.DataFrame(rows)
            buy_ct   = sum(1 for r in rows if r['SIGNAL'] == 'BUY')
            avoid_ct = sum(1 for r in rows if r['SIGNAL'] in ('AVOID', 'VETOED'))
            return bb_panel(f'AI SCAN SIGNALS  --  {len(signals)} STOCKS  |  {buy_ct} BUY  |  {avoid_ct} AVOID',
                dash_table.DataTable(data=df.to_dict('records'), columns=[{'name': c, 'id': c} for c in df.columns],
                    style_header=TBL_HDR, style_cell=TBL_CELL,
                    style_data_conditional=[
                        {'if': {'filter_query': '{SIGNAL} = "BUY"',          'column_id': 'SIGNAL'}, 'color': C['green'],  'fontWeight': 'bold', 'backgroundColor': '#001F14'},
                        {'if': {'filter_query': '{SIGNAL} = "AVOID"',         'column_id': 'SIGNAL'}, 'color': C['red'],    'fontWeight': 'bold'},
                        {'if': {'filter_query': '{SIGNAL} = "VETOED"',        'column_id': 'SIGNAL'}, 'color': C['red'],    'fontWeight': 'bold'},
                        {'if': {'filter_query': '{SIGNAL} = "CAUTION"',       'column_id': 'SIGNAL'}, 'color': C['yellow'], 'fontWeight': 'bold'},
                        {'if': {'filter_query': '{SIGNAL} = "EARNINGS_HOLD"', 'column_id': 'SIGNAL'}, 'color': C['orange'], 'fontWeight': 'bold'},
                        {'if': {'filter_query': '{REGIME} = "UPTREND"',       'column_id': 'REGIME'}, 'color': C['green']},
                        {'if': {'filter_query': '{REGIME} = "DOWNTREND"',     'column_id': 'REGIME'}, 'color': C['red']},
                        {'if': {'filter_query': '{REGIME} = "VOLATILE"',      'column_id': 'REGIME'}, 'color': C['yellow']},
                        {'if': {'filter_query': '{SENTIMENT} contains "+"',   'column_id': 'SENTIMENT'}, 'color': C['green']},
                        {'if': {'filter_query': '{SENTIMENT} contains "-"',   'column_id': 'SENTIMENT'}, 'color': C['red']},
                    ], sort_action='native', filter_action='native', page_size=25))

        # ── SECTORS ───────────────────────────────────────────────────────
        elif tab == 'sectors':
            if not sectors:
                return bb_panel('SECTOR ROTATION', _no_data('NO SECTOR DATA -- RUNS WITH NEXT SCAN'))
            def sector_item(name, d):
                flow = d.get('flow', 'NEUTRAL'); score = d.get('score', 0); mom = d.get('momentum_21d', 0)
                sc   = C['green'] if flow == 'INFLOW' else (C['red'] if flow == 'OUTFLOW' else C['mid'])
                dot  = 'UP' if flow == 'INFLOW' else ('DN' if flow == 'OUTFLOW' else '--')
                bar  = min(abs(score) * 500, 100); bc = C['green'] if score >= 0 else C['red']
                return html.Div(style={'marginBottom': '14px'}, children=[
                    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '5px'}, children=[
                        html.Div([
                            html.Span(dot + '  ', style={'color': sc, 'fontSize': '9px', 'fontFamily': MONO, 'fontWeight': '700'}),
                            html.Span(name.upper(), style={'color': C['text'], 'fontSize': '12px', 'fontFamily': MONO, 'fontWeight': '700', 'letterSpacing': '1px'}),
                        ]),
                        html.Div([
                            html.Span(f'{mom:+.2f}%', style={'color': C['green'] if mom >= 0 else C['red'], 'fontSize': '12px', 'fontFamily': MONO, 'marginRight': '10px'}),
                            html.Span(flow, style={'color': sc, 'fontSize': '9px', 'fontFamily': MONO, 'fontWeight': '700',
                                                    'letterSpacing': '1.5px', 'backgroundColor': f'{sc}18', 'padding': '2px 8px', 'borderRadius': '2px'}),
                        ]),
                    ]),
                    html.Div(style={'height': '3px', 'backgroundColor': C['border'], 'borderRadius': '2px'}, children=[
                        html.Div(style={'height': '3px', 'width': f'{bar:.0f}%', 'backgroundColor': bc, 'borderRadius': '2px'})
                    ]),
                ])
            items = [sector_item(k, v) for k, v in sectors.items()]; mid = len(items) // 2
            return bb_panel('SECTOR ROTATION & FUND FLOWS  --  21-DAY MOMENTUM',
                html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '24px'},
                         children=[html.Div(items[:mid]), html.Div(items[mid:])]))

        # ── EARNINGS ──────────────────────────────────────────────────────
        elif tab == 'earnings':
            if not earnings:
                return bb_panel('EARNINGS SAFETY CALENDAR', _no_data('NO EARNINGS THIS WEEK -- ALL CLEAR TO TRADE'))
            rows = []
            for e in sorted(earnings, key=lambda x: x.get('days_until', 99)):
                days = e.get('days_until', 0)
                risk = 'AVOID NOW' if days <= 1 else ('CAUTION' if days <= 3 else 'MONITOR')
                rows.append({'SYMBOL': e.get('symbol', ''), 'DATE': e.get('date', ''),
                    'DAYS UNTIL': days, 'TIME': e.get('time', 'TBD'),
                    'RISK LEVEL': risk, 'EPS EST': str(e.get('eps_estimate', 'N/A'))})
            df = pd.DataFrame(rows)
            return bb_panel('EARNINGS RISK CALENDAR  --  AVOID POSITIONS WITHIN 2 DAYS',
                dash_table.DataTable(data=df.to_dict('records'), columns=[{'name': c, 'id': c} for c in df.columns],
                    style_header=TBL_HDR, style_cell=TBL_CELL,
                    style_data_conditional=[
                        {'if': {'filter_query': '{DAYS UNTIL} <= 1',                            'column_id': 'RISK LEVEL'}, 'color': C['red'],    'fontWeight': 'bold'},
                        {'if': {'filter_query': '{DAYS UNTIL} > 1 && {DAYS UNTIL} <= 3',        'column_id': 'RISK LEVEL'}, 'color': C['yellow'], 'fontWeight': 'bold'},
                        {'if': {'filter_query': '{DAYS UNTIL} > 3',                             'column_id': 'RISK LEVEL'}, 'color': C['green']},
                    ], sort_action='native', page_size=20))

        # ── HISTORY ───────────────────────────────────────────────────────
        elif tab == 'history':
            if not history:
                return bb_panel('TRADE EXECUTION HISTORY', _no_data('NO TRADES EXECUTED YET -- SYSTEM LIVE & WATCHING'))
            rows = []
            for t in reversed(history[-50:]):
                p_ = t.get('pnl', None)
                rows.append({'DATE': t.get('date', '')[:16], 'ACTION': t.get('action', ''),
                    'SYMBOL': t.get('symbol', ''), 'SHARES': t.get('shares', 0),
                    'FILL PRICE': f'${t.get("fill_price", t.get("price", 0)):.2f}',
                    'P&L': f'${p_:+.2f}' if p_ is not None else '--',
                    'P&L %': f'{t.get("pnl_pct",0)*100:+.2f}%' if 'pnl_pct' in t else '--',
                    'REASON': t.get('reason', '').upper(),
                    'SLIPPAGE': f'{t.get("slippage_pct",0)*100:.3f}%', 'COMM': f'${t.get("commission",1.0):.2f}'})
            df = pd.DataFrame(rows)
            return bb_panel(f'TRADE EXECUTION LOG  --  LAST {len(rows)} TRADES',
                dash_table.DataTable(data=df.to_dict('records'), columns=[{'name': c, 'id': c} for c in df.columns],
                    style_header=TBL_HDR, style_cell=TBL_CELL,
                    style_data_conditional=[
                        {'if': {'filter_query': '{ACTION} = "BUY"',          'column_id': 'ACTION'}, 'color': C['green'],  'fontWeight': 'bold'},
                        {'if': {'filter_query': '{ACTION} = "SELL"',         'column_id': 'ACTION'}, 'color': C['red'],    'fontWeight': 'bold'},
                        {'if': {'filter_query': '{ACTION} = "PARTIAL_SELL"', 'column_id': 'ACTION'}, 'color': C['orange'], 'fontWeight': 'bold'},
                        {'if': {'filter_query': '{P&L} contains "+"', 'column_id': 'P&L'}, 'color': C['green'], 'fontWeight': 'bold'},
                        {'if': {'filter_query': '{P&L} contains "-"', 'column_id': 'P&L'}, 'color': C['red'],   'fontWeight': 'bold'},
                    ], sort_action='native', page_size=20))

        # ── SYS CONFIG ────────────────────────────────────────────────────
        elif tab == 'sysconfig':
            def cfg(k, v, hi=False):
                return stat_row(k, v, C['orange'] if hi else C['text'])
            return html.Div([html.Div(
                style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '12px'},
                children=[
                    bb_panel('SIGNAL GENERATION & SIZING', html.Div([
                        cfg('Watchlist Size',       f'{watchlist_len} Tickers'),
                        cfg('BUY Score Threshold',  str(buy_threshold),         hi=True),
                        cfg('Min Volume Spike',     f'{vol_spike_min}x'),
                        cfg('Min Risk-Reward',      f'{min_rr:.1f} R:R'),
                        cfg('MTF Weight',           f'{mtf_weight*100:.0f}%'),
                        cfg('Kelly Sizing',         'ENABLED' if kelly_active else 'DISABLED', hi=True),
                        cfg('Kelly Multiplier',     f'{kelly_mult} (Half-Kelly)'),
                        cfg('Kelly R/R Ratio',      f'{kelly_rr:.1f}x'),
                        cfg('Max Position Cap',     f'{max_pos_pct*100:.0f}%'),
                    ])),
                    bb_panel('PORTFOLIO RISK & STOPS', html.Div([
                        cfg('Max Positions',    str(max_positions),           hi=True),
                        cfg('Max Risk/Trade',   f'{max_risk_trade*100:.1f}%'),
                        cfg('Max Port Risk',    f'{max_port_risk*100:.1f}%'),
                        cfg('Daily Loss Limit', f'{max_daily_loss*100:.1f}%', hi=True),
                        cfg('Max Drawdown',     f'{max_dd_limit*100:.1f}%',   hi=True),
                        cfg('ATR Stop Mult',    f'{atr_stop_mult}x ATR'),
                        cfg('ATR Target Mult',  f'{atr_target_mult}x ATR'),
                        cfg('Trailing Stop',    f'{trailing_mult}x'),
                        cfg('Scan Schedule',    '16:15 ET Daily'),
                    ])),
                    bb_panel('ML ENGINE', html.Div([
                        cfg('Active Models',   'XGBoost + LightGBM + CatBoost + RF + LSTM', hi=True),
                        cfg('Feature Set',     'Alpha158  (167 features)'),
                        cfg('Regime Model',    '4-Class HMM  (Up / Down / Side / Vol)'),
                        cfg('Sentiment NLP',   'FinBERT  (ProsusAI/finbert)'),
                        cfg('MTF Analysis',    '3-Timeframe Composite Score'),
                        cfg('Veto Agent',      'Groq / Llama3 LLM'),
                        cfg('Sector Model',    '11-Sector ETF Momentum'),
                        cfg('Earnings Guard',  'Calendar-Based Hold Filter'),
                    ])),
                    bb_panel('SYSTEM HEALTH', html.Div([
                        cfg('Service Status',    'ACTIVE  (systemd managed)', hi=True),
                        cfg('Mode',              'PAPER TRADING'),
                        cfg('Broker',            'Alpaca Markets  (paper-api)'),
                        cfg('Data Source',       'Yahoo Finance  (yfinance)'),
                        cfg('News Source',       'Alpaca News API'),
                        cfg('Dashboard Refresh', 'Every 60 seconds'),
                        cfg('Heartbeat',         'Every 60s  ->  logs/heartbeats/'),
                        cfg('Log Directory',     '/root/alpha_edge/logs/'),
                    ])),
                ]
            )])

        return html.Div('Select a tab.', style={'color': C['dim']})

    return app


if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  ALPHAEDGE BLOOMBERG TERMINAL  |  V7')
    print('=' * 60)
    print('  http://localhost:8050\n')
    app = create_app()
    app.run(debug=False, host='0.0.0.0', port=8050)