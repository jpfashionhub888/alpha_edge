# monitoring/dashboard.py

"""
AlphaEdge Web Dashboard - Upgraded V3 (Institutional Grade)
Matches the premium styling, custom dark themes, dynamic Kelly positioning calculations,
and System Risk limits inspection.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# ── Load Live Settings ────────────────────────────────────────────────────────
try:
    from config import settings
    kelly_active = getattr(settings, 'KELLY_POSITION_SIZING', True)
    kelly_mult = getattr(settings, 'KELLY_MULTIPLIER', 0.5)
    kelly_rr = getattr(settings, 'KELLY_REWARD_RISK_RATIO', 2.5)
    max_pos_pct = getattr(settings, 'MAX_POSITION_SIZE', 0.15)
    buy_threshold = getattr(settings, 'BUY_THRESHOLD', 0.63)
    max_dd_limit = getattr(settings, 'MAX_DRAWDOWN', 0.10)
    max_daily_loss = getattr(settings, 'MAX_DAILY_LOSS', 0.02)
    max_positions = getattr(settings, 'MAX_OPEN_POSITIONS', 5)
    atr_stop_mult = getattr(settings, 'ATR_STOP_MULT', 1.0)
    atr_target_mult = getattr(settings, 'ATR_TARGET_MULT', 2.5)
    trailing_mult = getattr(settings, 'TRAILING_STOP_MULTIPLIER', 0.8)
    max_risk_per_trade = getattr(settings, 'MAX_RISK_PER_TRADE', 0.02)
    max_portfolio_risk = getattr(settings, 'MAX_PORTFOLIO_RISK', 0.06)
    volume_spike_min = getattr(settings, 'VOLUME_SPIKE_MIN', 1.3)
    min_risk_reward = getattr(settings, 'MIN_RISK_REWARD', 2.0)
    mtf_weight = getattr(settings, 'MTF_WEIGHT_IN_SIGNAL', 0.15)
    watchlist_len = len(getattr(settings, 'STOCK_WATCHLIST', []))
except Exception as e:
    # Fallback defaults
    kelly_active = True
    kelly_mult = 0.5
    kelly_rr = 2.5
    max_pos_pct = 0.15
    buy_threshold = 0.63
    max_dd_limit = 0.10
    max_daily_loss = 0.02
    max_positions = 5
    atr_stop_mult = 1.0
    atr_target_mult = 2.5
    trailing_mult = 0.8
    max_risk_per_trade = 0.02
    max_portfolio_risk = 0.06
    volume_spike_min = 1.3
    min_risk_reward = 2.0
    mtf_weight = 0.15
    watchlist_len = 41

TRADES_FILE   = 'logs/paper_trades.json'
SIGNALS_FILE  = 'logs/latest_signals.json'
SECTORS_FILE  = 'logs/sectors.json'
EARNINGS_FILE = 'logs/earnings.json'

COLORS = {
    'bg'      : '#060814',
    'card'    : '#0f172a',
    'card2'   : '#1e293b',
    'border'  : 'rgba(255, 255, 255, 0.06)',
    'text'    : '#f8fafc',
    'text_dim': '#94a3b8',
    'green'   : '#10b981',
    'red'     : '#f43f5e',
    'yellow'  : '#f59e0b',
    'blue'    : '#3b82f6',
    'orange'  : '#f97316',
    'accent'  : '#0ea5e9',
}

CARD_STYLE = {
    'backgroundColor': COLORS['card'],
    'border'         : f"1px solid {COLORS['border']}",
    'borderRadius'   : '12px',
    'padding'        : '20px',
    'marginBottom'   : '16px',
}


def load_portfolio():
    if not os.path.exists(TRADES_FILE):
        return {
            'capital'         : 10000.0,
            'starting_capital': 10000.0,
            'positions'       : {},
            'trade_history'   : [],
        }
    with open(TRADES_FILE, 'r') as f:
        return json.load(f)


def load_signals():
    if not os.path.exists(SIGNALS_FILE):
        return {}
    with open(SIGNALS_FILE, 'r') as f:
        return json.load(f)


def load_sectors():
    if not os.path.exists(SECTORS_FILE):
        return {}
    with open(SECTORS_FILE, 'r') as f:
        return json.load(f)


def load_earnings():
    if not os.path.exists(EARNINGS_FILE):
        return []
    with open(EARNINGS_FILE, 'r') as f:
        return json.load(f)


def signal_color(signal):
    return {
        'BUY'          : COLORS['green'],
        'AVOID'        : COLORS['red'],
        'CAUTION'      : COLORS['yellow'],
        'EARNINGS_HOLD': COLORS['orange'],
        'HOLD'         : COLORS['text_dim'],
    }.get(signal, COLORS['text_dim'])


def get_kelly_sizing(prediction):
    if not kelly_active:
        return 0.0, 0.0
    p = float(prediction)
    b = kelly_rr
    if p > 0.0:
        kelly_f = (p * (b + 1.0) - 1.0) / b
        kelly_f = max(0.0, kelly_f)
    else:
        kelly_f = 0.0
    alloc_fraction = kelly_f * kelly_mult
    alloc_fraction = min(alloc_fraction, max_pos_pct)
    return kelly_f, alloc_fraction


def create_app():
    app = Dash(
        __name__,
        title                       = 'AlphaEdge Trading Terminal',
        update_title                = None,
        suppress_callback_exceptions= True,
    )

    app.layout = html.Div(
        style={
            'backgroundColor': COLORS['bg'],
            'minHeight'      : '100vh',
            'padding'        : '24px',
            'fontFamily'     : "'Plus Jakarta Sans', Arial, sans-serif",
            'color'          : COLORS['text'],
        },
        children=[

            # Auto refresh every 60s
            dcc.Interval(
                id      ='refresh',
                interval=60 * 1000,
                n_intervals=0
            ),

            # Header
            html.Div(
                style={
                    'backgroundColor': COLORS['card'],
                    'border'         : f"1px solid {COLORS['border']}",
                    'borderRadius'   : '16px',
                    'padding'        : '24px 32px',
                    'marginBottom'   : '24px',
                    'display'        : 'flex',
                    'justifyContent' : 'space-between',
                    'alignItems'     : 'center',
                },
                children=[
                    html.Div([
                        html.H1(
                            "AlphaEdge Trading Terminal",
                            style={
                                'color'       : COLORS['text'],
                                'fontSize'    : '26px',
                                'margin'      : '0',
                                'fontWeight'  : '800',
                                'letterSpacing': '-0.5px',
                            }
                        ),
                        html.P(
                            "Institutional AIUS Market Scanning System",
                            style={
                                'color'    : COLORS['text_dim'],
                                'margin'   : '4px 0 0 0',
                                'fontSize' : '13px',
                            }
                        ),
                    ]),
                    html.Div([
                        html.P(
                            id='last-updated',
                            style={
                                'color'    : COLORS['text_dim'],
                                'fontSize' : '12px',
                                'margin'   : '0',
                                'fontFamily': "'JetBrains Mono', monospace",
                                'textAlign': 'right',
                            }
                        ),
                        html.Div(
                            style={
                                'display'       : 'flex',
                                'alignItems'    : 'center',
                                'gap'           : '8px',
                                'marginTop'     : '6px',
                                'justifyContent': 'flex-end',
                            },
                            children=[
                                html.Div(
                                    style={
                                        'width'          : '8px',
                                        'height'         : '8px',
                                        'borderRadius'   : '50%',
                                        'backgroundColor': COLORS['green'],
                                        'boxShadow'      : f"0 0 6px {COLORS['green']}",
                                    }
                                ),
                                html.Span(
                                    "LIVE SCAN ACTIVE",
                                    style={
                                        'color'    : COLORS['green'],
                                        'fontSize' : '11px',
                                        'fontWeight': '700',
                                        'letterSpacing': '0.5px',
                                    }
                                ),
                            ]
                        ),
                    ]),
                ]
            ),

            # Summary Cards Row
            html.Div(
                id='summary-cards',
                style={
                    'display'      : 'grid',
                    'gridTemplateColumns': 'repeat(6, 1fr)',
                    'gap'          : '16px',
                    'marginBottom' : '24px',
                }
            ),

            # Charts Row
            html.Div(
                style={
                    'display'             : 'grid',
                    'gridTemplateColumns' : '2fr 1fr',
                    'gap'                 : '16px',
                    'marginBottom'        : '24px',
                },
                children=[
                    # Portfolio Chart
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                "Equity Growth Curve",
                                style={
                                    'color'       : COLORS['text'],
                                    'fontSize'    : '14px',
                                    'fontWeight'  : '800',
                                    'marginTop'   : '0',
                                    'marginBottom': '16px',
                                    'letterSpacing': '0.5px',
                                }
                            ),
                            dcc.Graph(
                                id    ='portfolio-chart',
                                config={'displayModeBar': False},
                            ),
                        ]
                    ),
                    # Allocation Chart
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                "Capital Allocation",
                                style={
                                    'color'       : COLORS['text'],
                                    'fontSize'    : '14px',
                                    'fontWeight'  : '800',
                                    'marginTop'   : '0',
                                    'marginBottom': '16px',
                                    'letterSpacing': '0.5px',
                                }
                            ),
                            dcc.Graph(
                                id    ='allocation-chart',
                                config={'displayModeBar': False},
                            ),
                        ]
                    ),
                ]
            ),

            # Sector Rotation + Earnings Row
            html.Div(
                style={
                    'display'            : 'grid',
                    'gridTemplateColumns': '1fr 1fr',
                    'gap'                : '16px',
                    'marginBottom'       : '24px',
                },
                children=[
                    # Sector Rotation
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                "Sector Strength & Flows",
                                style={
                                    'color'       : COLORS['text'],
                                    'fontSize'    : '14px',
                                    'fontWeight'  : '800',
                                    'marginTop'   : '0',
                                    'marginBottom': '16px',
                                    'letterSpacing': '0.5px',
                                }
                            ),
                            html.Div(id='sector-table'),
                        ]
                    ),
                    # Earnings Calendar
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                "Earnings Safety Calendar",
                                style={
                                    'color'       : COLORS['text'],
                                    'fontSize'    : '14px',
                                    'fontWeight'  : '800',
                                    'marginTop'   : '0',
                                    'marginBottom': '16px',
                                    'letterSpacing': '0.5px',
                                }
                            ),
                            html.Div(id='earnings-table'),
                        ]
                    ),
                ]
            ),

            # Open Positions
            html.Div(
                style=CARD_STYLE,
                children=[
                    html.H3(
                        "Active Open Positions",
                        style={
                            'color'       : COLORS['text'],
                            'fontSize'    : '14px',
                            'fontWeight'  : '800',
                            'marginTop'   : '0',
                            'marginBottom': '16px',
                            'letterSpacing': '0.5px',
                        }
                    ),
                    html.Div(id='positions-table'),
                ]
            ),

            # Live Signals
            html.Div(
                style=CARD_STYLE,
                children=[
                    html.H3(
                        "AI Scan Signals & Kelly Allocations",
                        style={
                            'color'       : COLORS['text'],
                            'fontSize'    : '14px',
                            'fontWeight'  : '800',
                            'marginTop'   : '0',
                            'marginBottom': '16px',
                            'letterSpacing': '0.5px',
                        }
                    ),
                    html.Div(id='signals-table'),
                ]
            ),

            # System Config Panel
            html.Div(
                style=CARD_STYLE,
                children=[
                    html.H3(
                        "Live System Configuration & Risk Limits",
                        style={
                            'color'       : COLORS['text'],
                            'fontSize'    : '14px',
                            'fontWeight'  : '800',
                            'marginTop'   : '0',
                            'marginBottom': '16px',
                            'letterSpacing': '0.5px',
                        }
                    ),
                    html.Div(
                        style={
                            'display': 'grid',
                            'gridTemplateColumns': '1fr 1fr',
                            'gap': '32px',
                            'fontSize': '13px',
                        },
                        children=[
                            html.Div([
                                html.H4("Signal Generation & Sizing", style={'color': COLORS['accent'], 'fontWeight': '700', 'marginBottom': '10px'}),
                                html.P(f"Watchlist Size: {watchlist_len} Tickers", style={'margin': '6px 0'}),
                                html.P(f"Buy Signal Score Threshold: {buy_threshold}", style={'margin': '6px 0'}),
                                html.P(f"Minimum Volume Spike Ratio: {volume_spike_min}x", style={'margin': '6px 0'}),
                                html.P(f"Minimum Risk-Reward Target: {min_risk_reward:.1f} R:R", style={'margin': '6px 0'}),
                                html.P(f"Kelly Position Sizing: {'ENABLED' if kelly_active else 'DISABLED'}", style={'color': COLORS['green'] if kelly_active else COLORS['red'], 'fontWeight': '700', 'margin': '6px 0'}),
                                html.P(f"Fractional Kelly Multiplier: {kelly_mult} (Half-Kelly)", style={'margin': '6px 0'}),
                                html.P(f"Kelly Calibrated R/R Ratio: {kelly_rr:.1f}x", style={'margin': '6px 0'}),
                            ]),
                            html.Div([
                                html.H4("Portfolio Risk Rules & Stops", style={'color': COLORS['accent'], 'fontWeight': '700', 'marginBottom': '10px'}),
                                html.P(f"Max Concurrent Positions Limit: {max_positions}", style={'margin': '6px 0'}),
                                html.P(f"Max Position Cap (Capital %): {max_pos_pct*100:.1f}%", style={'margin': '6px 0'}),
                                html.P(f"Max Volatility Risk Per Trade: {max_risk_per_trade*100:.1f}%", style={'margin': '6px 0'}),
                                html.P(f"Max Portfolio Risk Limit: {max_portfolio_risk*100:.1f}%", style={'margin': '6px 0'}),
                                html.P(f"Max Daily Realized Loss Limit: {max_daily_loss*100:.1f}%", style={'margin': '6px 0'}),
                                html.P(f"Max Portfolio Drawdown Limit: {max_dd_limit*100:.1f}%", style={'margin': '6px 0'}),
                                html.P(f"ATR Stop Loss Multiplier: {atr_stop_mult}x ATR", style={'margin': '6px 0'}),
                                html.P(f"ATR Take Profit Multiplier: {atr_target_mult}x ATR", style={'margin': '6px 0'}),
                            ]),
                        ]
                    ),
                ]
            ),

            # Trade History
            html.Div(
                style=CARD_STYLE,
                children=[
                    html.H3(
                        "Trade Execution History",
                        style={
                            'color'       : COLORS['text'],
                            'fontSize'    : '14px',
                            'fontWeight'  : '800',
                            'marginTop'   : '0',
                            'marginBottom': '16px',
                            'letterSpacing': '0.5px',
                        }
                    ),
                    html.Div(id='history-table'),
                ]
            ),

            # Footer
            html.Div(
                style={
                    'textAlign' : 'center',
                    'padding'   : '24px',
                    'color'     : COLORS['text_dim'],
                    'fontSize'  : '11px',
                    'borderTop' : f"1px solid {COLORS['border']}",
                    'marginTop' : '24px',
                },
                children=[
                    html.P(
                        "AlphaEdge V6 Terminal | "
                        "Ensemble Learning Engine (XGB+LGB+RF+CatBoost+LSTM) | "
                        "ATR Volatility Stops | "
                        "Kelly Sizing Integration | "
                        "Automated Execution"
                    ),
                ]
            ),
        ]
    )

    @app.callback(
        [
            Output('summary-cards',    'children'),
            Output('portfolio-chart',  'figure'),
            Output('allocation-chart', 'figure'),
            Output('sector-table',     'children'),
            Output('earnings-table',   'children'),
            Output('positions-table',  'children'),
            Output('signals-table',    'children'),
            Output('history-table',    'children'),
            Output('last-updated',     'children'),
        ],
        [Input('refresh', 'n_intervals')]
    )
    def update_dashboard(n):

        portfolio = load_portfolio()
        signals   = load_signals()
        sectors   = load_sectors()
        earnings  = load_earnings()

        capital   = portfolio.get('capital', 10000)
        starting  = portfolio.get('starting_capital', 10000)
        positions = portfolio.get('positions', {})
        history   = portfolio.get('trade_history', [])

        # Calculate totals
        position_value = sum(
            pos.get('shares', 0) * pos.get(
                'current_price', pos.get('entry_price', 0)
            )
            for pos in positions.values()
        )
        total_value = capital + position_value
        total_pnl   = total_value - starting
        total_pct   = (total_pnl / starting) * 100 if starting > 0 else 0

        # SELL + PARTIAL_SELL both count as realized exits
        _realized  = {'SELL', 'PARTIAL_SELL'}
        sells      = [t for t in history if t.get('action') in _realized]
        wins       = len([t for t in sells if t.get('pnl', 0) > 0])
        losses     = len([t for t in sells if t.get('pnl', 0) <= 0])
        total_closed = wins + losses
        win_rate   = wins / total_closed * 100 if total_closed > 0 else 0
        total_realized_pnl = sum(t.get('pnl', 0) for t in sells)

        # ── Summary Cards ──────────────────────────────────────
        def make_card(label, value, color, subtitle=None):
            return html.Div(
                style={
                    'backgroundColor': COLORS['card'],
                    'border'         : f"1px solid {COLORS['border']}",
                    'borderRadius'   : '12px',
                    'padding'        : '16px',
                    'borderTop'      : f"3px solid {color}",
                },
                children=[
                    html.P(
                        label,
                        style={
                            'color'        : COLORS['text_dim'],
                            'fontSize'     : '10px',
                            'margin'       : '0 0 6px 0',
                            'letterSpacing': '1px',
                            'fontWeight'   : '700',
                            'textTransform': 'uppercase',
                        }
                    ),
                    html.H2(
                        value,
                        style={
                            'color'     : color,
                            'margin'    : '0',
                            'fontSize'  : '20px',
                            'fontWeight': '800',
                            'fontFamily': "'JetBrains Mono', monospace",
                        }
                    ),
                    html.P(
                        subtitle or '',
                        style={
                            'color'   : COLORS['text_dim'],
                            'fontSize': '11px',
                            'margin'  : '4px 0 0 0',
                        }
                    ),
                ]
            )

        pnl_color = COLORS['green'] if total_pnl >= 0 else COLORS['red']
        wr_color  = COLORS['green'] if win_rate >= 50 else COLORS['red']

        cards = [
            make_card(
                "Total Net Liq",
                f"${total_value:,.2f}",
                COLORS['text'],
                f"Started: ${starting:,.2f}"
            ),
            make_card(
                "Cash Available",
                f"${capital:,.2f}",
                COLORS['blue'],
                f"{capital/total_value*100:.1f}% cash" if total_value > 0 else ""
            ),
            make_card(
                "Total P&L",
                f"${total_pnl:+,.2f}",
                pnl_color,
                f"{total_pct:+.2f}% overall"
            ),
            make_card(
                "Realized P&L",
                f"${total_realized_pnl:+,.2f}",
                pnl_color,
                f"{total_closed} closed trades"
            ),
            make_card(
                "Active Slots",
                f"{len(positions)} / {max_positions}",
                COLORS['yellow'],
                f"Invested: ${position_value:,.2f}"
            ),
            make_card(
                "Trade Win Rate",
                f"{win_rate:.1f}%",
                wr_color,
                f"{wins} Wins / {losses} Losses"
            ),
        ]

        # ── Portfolio Chart ────────────────────────────────────
        values = [starting]
        dates  = ['Start']

        for trade in history:
            if trade.get('action') in {'SELL', 'PARTIAL_SELL'}:
                values.append(values[-1] + trade.get('pnl', 0))
                dates.append(trade.get('date', '')[:10])

        values.append(total_value)
        dates.append('Now')

        line_color = COLORS['green'] if values[-1] >= values[0] else COLORS['red']
        fill_color = 'rgba(16,185,129,0.04)' if values[-1] >= values[0] else 'rgba(244,63,94,0.04)'

        portfolio_fig = go.Figure()
        portfolio_fig.add_trace(go.Scatter(
            x         = dates,
            y         = values,
            mode      = 'lines+markers',
            line      = dict(color=line_color, width=2.5, shape='spline'),
            marker    = dict(size=5, color=line_color),
            fill      = 'tozeroy',
            fillcolor = fill_color,
            name      = 'Net Asset Value',
        ))
        portfolio_fig.add_hline(
            y         = starting,
            line_dash = 'dash',
            line_color= COLORS['text_dim'],
            opacity   = 0.4,
        )
        portfolio_fig.update_layout(
            plot_bgcolor = COLORS['card'],
            paper_bgcolor= COLORS['card'],
            font         = dict(color=COLORS['text'], size=10, family="JetBrains Mono"),
            xaxis        = dict(
                gridcolor = 'rgba(255,255,255,0.03)',
                showgrid  = True,
            ),
            yaxis        = dict(
                gridcolor  = 'rgba(255,255,255,0.03)',
                showgrid   = True,
                tickprefix = '$',
            ),
            margin       = dict(l=55, r=15, t=10, b=35),
            height       = 240,
            showlegend   = False,
        )

        # ── Allocation Chart ───────────────────────────────────
        alloc_labels = ['Cash']
        alloc_values = [capital]
        alloc_colors = ['#1e293b']

        pie_colors = [
            COLORS['accent'], COLORS['green'],
            COLORS['yellow'], COLORS['orange'],
            '#a855f7', '#06b6d4',
        ]

        for i, (sym, pos) in enumerate(positions.items()):
            val = pos.get('shares', 0) * pos.get(
                'current_price', pos.get('entry_price', 0)
            )
            alloc_labels.append(sym)
            alloc_values.append(val)
            alloc_colors.append(pie_colors[i % len(pie_colors)])

        alloc_fig = go.Figure(data=[go.Pie(
            labels      = alloc_labels,
            values      = alloc_values,
            marker      = dict(
                colors = alloc_colors,
                line   = dict(color=COLORS['card'], width=2),
            ),
            hole        = 0.6,
            textfont    = dict(color=COLORS['text'], size=10, family="Plus Jakarta Sans"),
            textinfo    = 'label+percent',
        )])
        alloc_fig.update_layout(
            plot_bgcolor = COLORS['card'],
            paper_bgcolor= COLORS['card'],
            font         = dict(color=COLORS['text']),
            margin       = dict(l=10, r=10, t=10, b=10),
            height       = 240,
            showlegend   = False,
        )

        # ── Sector Rotation Table ──────────────────────────────
        if sectors:
            sector_rows = []
            for name, data in sectors.items():
                flow  = data.get('flow', 'NEUTRAL')
                score = data.get('score', 0)
                mom   = data.get('momentum_21d', 0)
                color = (
                    COLORS['green']    if flow == 'INFLOW'
                    else COLORS['red'] if flow == 'OUTFLOW'
                    else COLORS['text_dim']
                )
                sector_rows.append(
                    html.Div(
                        style={
                            'display'        : 'flex',
                            'justifyContent' : 'space-between',
                            'alignItems'     : 'center',
                            'padding'        : '8px 16px',
                            'marginBottom'   : '6px',
                            'backgroundColor': COLORS['card'],
                            'border'         : f"1px solid {COLORS['border']}",
                            'borderRadius'   : '8px',
                            'borderLeft'     : f"4px solid {color}",
                        },
                        children=[
                            html.Span(
                                name,
                                style={
                                    'color'   : COLORS['text'],
                                    'fontSize': '12px',
                                    'fontWeight': '700',
                                }
                            ),
                            html.Span(
                                f"{mom:+.2f}%",
                                style={
                                    'color'   : COLORS['green'] if mom >= 0 else COLORS['red'],
                                    'fontSize': '12px',
                                    'fontFamily': "'JetBrains Mono', monospace",
                                }
                            ),
                            html.Span(
                                flow,
                                style={
                                    'color'       : color,
                                    'fontSize'    : '10px',
                                    'fontWeight'  : '800',
                                    'letterSpacing': '0.5px',
                                }
                            ),
                        ]
                    )
                )
            sector_display = html.Div(sector_rows)
        else:
            sector_display = html.P(
                "No sector data yet. Run scanner first.",
                style={'color': COLORS['text_dim']}
            )

        # ── Earnings Calendar ──────────────────────────────────
        if earnings:
            earn_rows = []
            for e in earnings:
                days  = e.get('days_until', 0)
                color = (
                    COLORS['red']    if days == 0
                    else COLORS['yellow'] if days <= 2
                    else COLORS['text_dim']
                )
                label = (
                    "TODAY!" if days == 0
                    else "TOMORROW" if days == 1
                    else f"In {days} days"
                )
                earn_rows.append(
                    html.Div(
                        style={
                            'display'        : 'flex',
                            'justifyContent' : 'space-between',
                            'padding'        : '8px 16px',
                            'marginBottom'   : '6px',
                            'backgroundColor': COLORS['card'],
                            'border'         : f"1px solid {COLORS['border']}",
                            'borderRadius'   : '8px',
                            'borderLeft'     : f"4px solid {color}",
                        },
                        children=[
                            html.Span(
                                e.get('symbol', ''),
                                style={
                                    'color'     : COLORS['text'],
                                    'fontWeight': '700',
                                    'fontSize'  : '12px',
                                }
                            ),
                            html.Span(
                                e.get('date', ''),
                                style={
                                    'color'   : COLORS['text_dim'],
                                    'fontSize': '11px',
                                    'fontFamily': "'JetBrains Mono', monospace",
                                }
                            ),
                            html.Span(
                                label,
                                style={
                                    'color'     : color,
                                    'fontSize'  : '11px',
                                    'fontWeight': '700',
                                }
                            ),
                        ]
                    )
                )
            earnings_display = html.Div(earn_rows)
        else:
            earnings_display = html.P(
                "No earnings this week.",
                style={'color': COLORS['text_dim']}
            )

        # ── Open Positions Table ───────────────────────────────
        if positions:
            pos_rows = []
            for sym, pos in positions.items():
                shares  = pos.get('shares', 0)
                entry   = pos.get('entry_price', 0)
                current = pos.get('current_price', entry)
                pnl     = (current - entry) * shares
                pnl_pct = (current - entry) / entry * 100 if entry > 0 else 0
                cost    = pos.get('cost', shares * entry)
                
                # Kelly target size reference
                entry_kelly_alloc = get_kelly_sizing(pos.get('signal', 0.5))[1]
                
                pos_rows.append({
                    'Symbol'       : sym,
                    'Shares'       : shares,
                    'Entry Price'  : f"${entry:.2f}",
                    'Current Price': f"${current:.2f}",
                    'Cost'         : f"${cost:.2f}",
                    'Kelly Size'   : f"{entry_kelly_alloc * 100:.1f}%",
                    'P&L'          : f"${pnl:+.2f}",
                    'P&L %'        : f"{pnl_pct:+.2f}%",
                    'Entry Date'   : pos.get('entry_date', '')[:10],
                    'Reason'       : pos.get('reason', ''),
                })

            pos_df = pd.DataFrame(pos_rows)
            positions_display = dash_table.DataTable(
                data    = pos_df.to_dict('records'),
                columns = [{'name': c, 'id': c} for c in pos_df.columns],
                style_header={
                    'backgroundColor': COLORS['card'],
                    'color'          : COLORS['text_dim'],
                    'fontWeight'     : '800',
                    'border'         : f"1px solid {COLORS['border']}",
                    'fontSize'       : '10px',
                    'letterSpacing'  : '1px',
                },
                style_cell={
                    'backgroundColor': COLORS['card'],
                    'color'          : COLORS['text'],
                    'border'         : f"1px solid {COLORS['border']}",
                    'textAlign'      : 'center',
                    'padding'        : '12px',
                    'fontSize'       : '12px',
                    'fontFamily'     : "'JetBrains Mono', monospace",
                },
                style_data_conditional=[
                    {
                        'if'             : {'filter_query': '{P&L} contains "+"'},
                        'color'          : COLORS['green'],
                        'column_id'      : 'P&L',
                        'fontWeight'     : 'bold',
                    },
                    {
                        'if'             : {'filter_query': '{P&L} contains "-"'},
                        'color'          : COLORS['red'],
                        'column_id'      : 'P&L',
                        'fontWeight'     : 'bold',
                    },
                    {
                        'if'             : {'filter_query': '{P&L %} contains "+"'},
                        'color'          : COLORS['green'],
                        'column_id'      : 'P&L %',
                    },
                    {
                        'if'             : {'filter_query': '{P&L %} contains "-"'},
                        'color'          : COLORS['red'],
                        'column_id'      : 'P&L %',
                    },
                ],
            )
        else:
            positions_display = html.P(
                "No open positions.",
                style={'color': COLORS['text_dim'], 'padding': '12px'}
            )

        # ── Signals Table ──────────────────────────────────────
        if signals:
            sig_rows = []
            for sym, data in sorted(
                signals.items(),
                key    = lambda x: x[1].get('prediction', 0),
                reverse= True
            ):
                sig = data.get('signal', 'HOLD')
                pred = data.get('prediction', 0)
                
                # Calculate Kelly Sizing
                kelly_f, alloc_frac = get_kelly_sizing(pred)
                target_allocation_dollars = total_value * alloc_frac
                
                sig_rows.append({
                    'Symbol'        : sym,
                    'Signal'        : sig,
                    'AI Score'      : f"{pred * 100:.1f}%",
                    'Regime'        : data.get('regime', ''),
                    'Sentiment'     : f"{data.get('sentiment', 0):+.2f}",
                    'Kelly Fraction': f"{kelly_f * 100:.1f}%",
                    'Target Size'   : f"{alloc_frac * 100:.1f}%",
                    'Target Alloc'  : f"${target_allocation_dollars:,.2f}",
                    'Price'         : f"${data.get('price', 0):.2f}",
                })

            sig_df = pd.DataFrame(sig_rows)
            signals_display = dash_table.DataTable(
                data    = sig_df.to_dict('records'),
                columns = [{'name': c, 'id': c} for c in sig_df.columns],
                style_header={
                    'backgroundColor': COLORS['card'],
                    'color'          : COLORS['text_dim'],
                    'fontWeight'     : '800',
                    'border'         : f"1px solid {COLORS['border']}",
                    'fontSize'       : '10px',
                    'letterSpacing'  : '1px',
                },
                style_cell={
                    'backgroundColor': COLORS['card'],
                    'color'          : COLORS['text'],
                    'border'         : f"1px solid {COLORS['border']}",
                    'textAlign'      : 'center',
                    'padding'        : '12px',
                    'fontSize'       : '12px',
                    'fontFamily'     : "'JetBrains Mono', monospace",
                },
                style_data_conditional=[
                    {
                        'if'       : {
                            'filter_query': '{Signal} = "BUY"',
                            'column_id'   : 'Signal',
                        },
                        'color'    : COLORS['green'],
                        'fontWeight': 'bold',
                    },
                    {
                        'if'       : {
                            'filter_query': '{Signal} = "AVOID"',
                            'column_id'   : 'Signal',
                        },
                        'color'    : COLORS['red'],
                        'fontWeight': 'bold',
                    },
                    {
                        'if'       : {
                            'filter_query': '{Signal} = "CAUTION"',
                            'column_id'   : 'Signal',
                        },
                        'color'    : COLORS['yellow'],
                        'fontWeight': 'bold',
                    },
                    {
                        'if'       : {
                            'filter_query': '{Signal} = "EARNINGS_HOLD"',
                            'column_id'   : 'Signal',
                        },
                        'color'    : COLORS['orange'],
                        'fontWeight': 'bold',
                    },
                ],
                page_size   = 20,
                sort_action = 'native',
            )
        else:
            signals_display = html.P(
                "No signals yet. Run scanner first.",
                style={'color': COLORS['text_dim'], 'padding': '12px'}
            )

        # ── Trade History Table ────────────────────────────────
        if history:
            hist_rows = []
            for trade in reversed(history[-30:]):
                pnl = trade.get('pnl', 0)
                hist_rows.append({
                    'Date'  : trade.get('date', '')[:16],
                    'Action': trade.get('action', ''),
                    'Symbol': trade.get('symbol', ''),
                    'Shares': trade.get('shares', 0),
                    'Price' : f"${trade.get('price', 0):.2f}",
                    'P&L'   : (
                        f"${pnl:+.2f}"
                        if trade.get('action') in ['SELL', 'PARTIAL_SELL']
                        else '-'
                    ),
                    'Reason': trade.get('reason', ''),
                })

            hist_df = pd.DataFrame(hist_rows)
            history_display = dash_table.DataTable(
                data    = hist_df.to_dict('records'),
                columns = [{'name': c, 'id': c} for c in hist_df.columns],
                style_header={
                    'backgroundColor': COLORS['card'],
                    'color'          : COLORS['text_dim'],
                    'fontWeight'     : '800',
                    'border'         : f"1px solid {COLORS['border']}",
                    'fontSize'       : '10px',
                    'letterSpacing'  : '1px',
                },
                style_cell={
                    'backgroundColor': COLORS['card'],
                    'color'          : COLORS['text'],
                    'border'         : f"1px solid {COLORS['border']}",
                    'textAlign'      : 'center',
                    'padding'        : '12px',
                    'fontSize'       : '12px',
                    'fontFamily'     : "'JetBrains Mono', monospace",
                },
                style_data_conditional=[
                    {
                        'if'       : {
                            'filter_query': '{Action} = "BUY"',
                            'column_id'   : 'Action',
                        },
                        'color'    : COLORS['green'],
                        'fontWeight': 'bold',
                    },
                    {
                        'if'       : {
                            'filter_query': '{Action} = "SELL"',
                            'column_id'   : 'Action',
                        },
                        'color'    : COLORS['red'],
                        'fontWeight': 'bold',
                    },
                    {
                        'if'       : {
                            'filter_query': '{P&L} contains "+"',
                            'column_id'   : 'P&L',
                        },
                        'color'    : COLORS['green'],
                        'fontWeight': 'bold',
                    },
                    {
                        'if'       : {
                            'filter_query': '{P&L} contains "-"',
                            'column_id'   : 'P&L',
                        },
                        'color'    : COLORS['red'],
                        'fontWeight': 'bold',
                    },
                ],
                page_size   = 15,
                sort_action = 'native',
            )
        else:
            history_display = html.P(
                "No trades yet.",
                style={'color': COLORS['text_dim'], 'padding': '12px'}
            )

        # Timestamp
        now       = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp = f"Last updated: {now} | Auto-refresh: 60s"

        return (
            cards,
            portfolio_fig,
            alloc_fig,
            sector_display,
            earnings_display,
            positions_display,
            signals_display,
            history_display,
            timestamp,
        )

    return app


if __name__ == '__main__':
    print("\nStarting AlphaEdge Dashboard Server...")
    print("Open browser: http://localhost:8050")
    app = create_app()
    app.run(debug=False, host='0.0.0.0', port=8050)