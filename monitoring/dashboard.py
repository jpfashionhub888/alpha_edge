# monitoring/dashboard.py

"""
AlphaEdge Web Dashboard - Upgraded V2
Matches BharatEdge style with enhanced features.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go


TRADES_FILE   = 'logs/paper_trades.json'
SIGNALS_FILE  = 'logs/latest_signals.json'
SECTORS_FILE  = 'logs/sectors.json'
EARNINGS_FILE = 'logs/earnings.json'

COLORS = {
    'bg'      : '#0a0e1a',
    'card'    : '#111827',
    'card2'   : '#1a2235',
    'border'  : '#1e2d45',
    'text'    : '#e2e8f0',
    'text_dim': '#94a3b8',
    'green'   : '#00ff88',
    'red'     : '#ff4444',
    'yellow'  : '#ffd700',
    'blue'    : '#3b82f6',
    'orange'  : '#f97316',
    'accent'  : '#00d4ff',
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


def create_app():
    app = Dash(
        __name__,
        title                       = 'AlphaEdge Dashboard',
        update_title                = None,
        suppress_callback_exceptions= True,
    )

    app.layout = html.Div(
        style={
            'backgroundColor': COLORS['bg'],
            'minHeight'      : '100vh',
            'padding'        : '24px',
            'fontFamily'     : "'Segoe UI', Arial, sans-serif",
            'color'          : COLORS['text'],
        },
        children=[

            # Auto refresh
            dcc.Interval(
                id      ='refresh',
                interval=60 * 1000,
                n_intervals=0
            ),

            # Header
            html.Div(
                style={
                    'backgroundColor': COLORS['card2'],
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
                            "AlphaEdge Trading Dashboard",
                            style={
                                'color'       : COLORS['accent'],
                                'fontSize'    : '28px',
                                'margin'      : '0',
                                'fontWeight'  : '700',
                                'letterSpacing': '1px',
                            }
                        ),
                        html.P(
                            "AI-Powered US Stock Trading System",
                            style={
                                'color'    : COLORS['text_dim'],
                                'margin'   : '4px 0 0 0',
                                'fontSize' : '14px',
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
                                'textAlign': 'right',
                            }
                        ),
                        html.Div(
                            style={
                                'display'       : 'flex',
                                'alignItems'    : 'center',
                                'gap'           : '8px',
                                'marginTop'     : '4px',
                            },
                            children=[
                                html.Div(
                                    style={
                                        'width'          : '8px',
                                        'height'         : '8px',
                                        'borderRadius'   : '50%',
                                        'backgroundColor': COLORS['green'],
                                    }
                                ),
                                html.Span(
                                    "LIVE",
                                    style={
                                        'color'    : COLORS['green'],
                                        'fontSize' : '12px',
                                        'fontWeight': '600',
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
                                "Portfolio Performance",
                                style={
                                    'color'       : COLORS['accent'],
                                    'fontSize'    : '16px',
                                    'marginTop'   : '0',
                                    'marginBottom': '16px',
                                    'letterSpacing': '1px',
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
                                "Portfolio Allocation",
                                style={
                                    'color'       : COLORS['accent'],
                                    'fontSize'    : '16px',
                                    'marginTop'   : '0',
                                    'marginBottom': '16px',
                                    'letterSpacing': '1px',
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
                                "Sector Rotation",
                                style={
                                    'color'       : COLORS['accent'],
                                    'fontSize'    : '16px',
                                    'marginTop'   : '0',
                                    'marginBottom': '16px',
                                    'letterSpacing': '1px',
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
                                "Earnings Calendar",
                                style={
                                    'color'       : COLORS['accent'],
                                    'fontSize'    : '16px',
                                    'marginTop'   : '0',
                                    'marginBottom': '16px',
                                    'letterSpacing': '1px',
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
                        "Open Positions",
                        style={
                            'color'       : COLORS['accent'],
                            'fontSize'    : '16px',
                            'marginTop'   : '0',
                            'marginBottom': '16px',
                            'letterSpacing': '1px',
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
                        "Live AI Signals (41 Stocks)",
                        style={
                            'color'       : COLORS['accent'],
                            'fontSize'    : '16px',
                            'marginTop'   : '0',
                            'marginBottom': '16px',
                            'letterSpacing': '1px',
                        }
                    ),
                    html.Div(id='signals-table'),
                ]
            ),

            # Trade History
            html.Div(
                style=CARD_STYLE,
                children=[
                    html.H3(
                        "Trade History",
                        style={
                            'color'       : COLORS['accent'],
                            'fontSize'    : '16px',
                            'marginTop'   : '0',
                            'marginBottom': '16px',
                            'letterSpacing': '1px',
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
                    'fontSize'  : '12px',
                    'borderTop' : f"1px solid {COLORS['border']}",
                    'marginTop' : '24px',
                },
                children=[
                    html.P(
                        "AlphaEdge V3 | "
                        "4-Model Ensemble (XGB+LGB+RF+LSTM) | "
                        "Sector Rotation | "
                        "News Sentiment | "
                        "Cloud Automated"
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
        total_pct   = (total_pnl / starting) * 100

        sells      = [t for t in history if t.get('action') == 'SELL']
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
                    'border'         : f"1px solid {color}33",
                    'borderRadius'   : '12px',
                    'padding'        : '16px',
                    'borderTop'      : f"3px solid {color}",
                },
                children=[
                    html.P(
                        label,
                        style={
                            'color'        : COLORS['text_dim'],
                            'fontSize'     : '11px',
                            'margin'       : '0 0 8px 0',
                            'letterSpacing': '1px',
                            'textTransform': 'uppercase',
                        }
                    ),
                    html.H2(
                        value,
                        style={
                            'color'     : color,
                            'margin'    : '0',
                            'fontSize'  : '22px',
                            'fontWeight': '700',
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
                "Total Value",
                f"${total_value:,.2f}",
                COLORS['accent'],
                f"Started: ${starting:,.2f}"
            ),
            make_card(
                "Cash Available",
                f"${capital:,.2f}",
                COLORS['blue'],
                f"{capital/total_value*100:.1f}% of portfolio"
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
                f"From {total_closed} closed trades"
            ),
            make_card(
                "Open Positions",
                str(len(positions)),
                COLORS['yellow'],
                f"Max 5 allowed"
            ),
            make_card(
                "Win Rate",
                f"{win_rate:.0f}%",
                wr_color,
                f"{wins}W / {losses}L"
            ),
        ]

        # ── Portfolio Chart ────────────────────────────────────
        values = [starting]
        dates  = ['Start']

        for trade in history:
            if trade.get('action') == 'SELL':
                values.append(values[-1] + trade.get('pnl', 0))
                dates.append(trade.get('date', '')[:10])

        values.append(total_value)
        dates.append('Now')

        line_color = COLORS['green'] if values[-1] >= values[0] else COLORS['red']
        fill_color = 'rgba(0,255,136,0.08)' if values[-1] >= values[0] else 'rgba(255,68,68,0.08)'

        portfolio_fig = go.Figure()
        portfolio_fig.add_trace(go.Scatter(
            x         = dates,
            y         = values,
            mode      = 'lines+markers',
            line      = dict(color=line_color, width=2),
            marker    = dict(size=6, color=line_color),
            fill      = 'tozeroy',
            fillcolor = fill_color,
            name      = 'Portfolio Value',
        ))
        portfolio_fig.add_hline(
            y         = starting,
            line_dash = 'dash',
            line_color= COLORS['text_dim'],
            opacity   = 0.5,
        )
        portfolio_fig.update_layout(
            plot_bgcolor = COLORS['card'],
            paper_bgcolor= COLORS['card'],
            font         = dict(color=COLORS['text'], size=11),
            xaxis        = dict(
                gridcolor = COLORS['border'],
                showgrid  = True,
            ),
            yaxis        = dict(
                gridcolor  = COLORS['border'],
                showgrid   = True,
                tickprefix = '$',
            ),
            margin       = dict(l=60, r=20, t=10, b=40),
            height       = 280,
            showlegend   = False,
        )

        # ── Allocation Chart ───────────────────────────────────
        alloc_labels = ['Cash']
        alloc_values = [capital]
        alloc_colors = [COLORS['blue']]

        pie_colors = [
            COLORS['green'], COLORS['yellow'],
            COLORS['orange'], COLORS['accent'],
            '#aa88ff', '#ff88aa',
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
                line   = dict(color=COLORS['bg'], width=2),
            ),
            hole        = 0.5,
            textfont    = dict(color=COLORS['text'], size=11),
            textinfo    = 'label+percent',
        )])
        alloc_fig.update_layout(
            plot_bgcolor = COLORS['card'],
            paper_bgcolor= COLORS['card'],
            font         = dict(color=COLORS['text']),
            margin       = dict(l=10, r=10, t=10, b=10),
            height       = 280,
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
                            'padding'        : '8px 12px',
                            'marginBottom'   : '4px',
                            'backgroundColor': COLORS['card2'],
                            'borderRadius'   : '8px',
                            'borderLeft'     : f"3px solid {color}",
                        },
                        children=[
                            html.Span(
                                name,
                                style={
                                    'color'   : COLORS['text'],
                                    'fontSize': '13px',
                                    'fontWeight': '600',
                                }
                            ),
                            html.Span(
                                f"{mom:+.1f}%",
                                style={
                                    'color'   : COLORS['green'] if mom >= 0 else COLORS['red'],
                                    'fontSize': '13px',
                                }
                            ),
                            html.Span(
                                flow,
                                style={
                                    'color'       : color,
                                    'fontSize'    : '11px',
                                    'fontWeight'  : '700',
                                    'letterSpacing': '1px',
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
                            'padding'        : '8px 12px',
                            'marginBottom'   : '4px',
                            'backgroundColor': COLORS['card2'],
                            'borderRadius'   : '8px',
                            'borderLeft'     : f"3px solid {color}",
                        },
                        children=[
                            html.Span(
                                e.get('symbol', ''),
                                style={
                                    'color'     : COLORS['text'],
                                    'fontWeight': '600',
                                    'fontSize'  : '13px',
                                }
                            ),
                            html.Span(
                                e.get('date', ''),
                                style={
                                    'color'   : COLORS['text_dim'],
                                    'fontSize': '12px',
                                }
                            ),
                            html.Span(
                                label,
                                style={
                                    'color'     : color,
                                    'fontSize'  : '12px',
                                    'fontWeight': '600',
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
                pos_rows.append({
                    'Symbol'       : sym,
                    'Shares'       : shares,
                    'Entry Price'  : f"${entry:.2f}",
                    'Current Price': f"${current:.2f}",
                    'Cost'         : f"${cost:.2f}",
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
                    'backgroundColor': COLORS['card2'],
                    'color'          : COLORS['yellow'],
                    'fontWeight'     : 'bold',
                    'border'         : f"1px solid {COLORS['border']}",
                    'fontSize'       : '12px',
                    'letterSpacing'  : '1px',
                },
                style_cell={
                    'backgroundColor': COLORS['card'],
                    'color'          : COLORS['text'],
                    'border'         : f"1px solid {COLORS['border']}",
                    'textAlign'      : 'center',
                    'padding'        : '10px',
                    'fontSize'       : '13px',
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
                sig_rows.append({
                    'Symbol'    : sym,
                    'Signal'    : sig,
                    'AI Score'  : f"{data.get('prediction', 0):.3f}",
                    'Regime'    : data.get('regime', ''),
                    'Sentiment' : f"{data.get('sentiment', 0):+.2f}",
                    'Sector'    : data.get('sector', ''),
                    'Price'     : f"${data.get('price', 0):.2f}",
                })

            sig_df = pd.DataFrame(sig_rows)
            signals_display = dash_table.DataTable(
                data    = sig_df.to_dict('records'),
                columns = [{'name': c, 'id': c} for c in sig_df.columns],
                style_header={
                    'backgroundColor': COLORS['card2'],
                    'color'          : COLORS['accent'],
                    'fontWeight'     : 'bold',
                    'border'         : f"1px solid {COLORS['border']}",
                    'fontSize'       : '12px',
                    'letterSpacing'  : '1px',
                },
                style_cell={
                    'backgroundColor': COLORS['card'],
                    'color'          : COLORS['text'],
                    'border'         : f"1px solid {COLORS['border']}",
                    'textAlign'      : 'center',
                    'padding'        : '10px',
                    'fontSize'       : '13px',
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
                        if trade.get('action') == 'SELL'
                        else '-'
                    ),
                    'Reason': trade.get('reason', ''),
                })

            hist_df = pd.DataFrame(hist_rows)
            history_display = dash_table.DataTable(
                data    = hist_df.to_dict('records'),
                columns = [{'name': c, 'id': c} for c in hist_df.columns],
                style_header={
                    'backgroundColor': COLORS['card2'],
                    'color'          : '#aa88ff',
                    'fontWeight'     : 'bold',
                    'border'         : f"1px solid {COLORS['border']}",
                    'fontSize'       : '12px',
                    'letterSpacing'  : '1px',
                },
                style_cell={
                    'backgroundColor': COLORS['card'],
                    'color'          : COLORS['text'],
                    'border'         : f"1px solid {COLORS['border']}",
                    'textAlign'      : 'center',
                    'padding'        : '10px',
                    'fontSize'       : '13px',
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
    print("\nStarting AlphaEdge Dashboard...")
    print("Open browser: http://localhost:8050")
    app = create_app()
    app.run(debug=False, host='0.0.0.0', port=8050)