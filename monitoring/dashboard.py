# monitoring/dashboard.py

"""
AlphaEdge Web Dashboard
Auto-refreshes every 30 seconds with latest data.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go


TRADES_FILE = 'logs/paper_trades.json'
SIGNALS_FILE = 'logs/latest_signals.json'
SCAN_INFO_FILE = 'logs/scan_info.json'


def load_portfolio():
    """Load saved portfolio state."""

    if not os.path.exists(TRADES_FILE):
        return {
            'capital': 10000.0,
            'starting_capital': 10000.0,
            'positions': {},
            'trade_history': [],
        }

    with open(TRADES_FILE, 'r') as f:
        return json.load(f)


def load_signals():
    """Load latest signals."""

    if not os.path.exists(SIGNALS_FILE):
        return {}

    with open(SIGNALS_FILE, 'r') as f:
        return json.load(f)


def load_scan_info():
    """Load scan status info."""

    if not os.path.exists(SCAN_INFO_FILE):
        return {}

    with open(SCAN_INFO_FILE, 'r') as f:
        return json.load(f)


def create_app():
    """Create the Dash web application."""

    app = Dash(
        __name__,
        title='AlphaEdge Trading Dashboard'
    )

    app.layout = html.Div(
        style={
            'backgroundColor': '#0a0a0a',
            'minHeight': '100vh',
            'padding': '20px',
            'fontFamily': 'Courier New, monospace',
            'color': '#ffffff',
        },
        children=[

            # Header
            html.Div(
                style={
                    'textAlign': 'center',
                    'marginBottom': '30px',
                },
                children=[
                    html.H1(
                        "🚀 AlphaEdge Trading Dashboard",
                        style={
                            'color': '#00ff88',
                            'fontSize': '36px',
                            'marginBottom': '5px',
                        }
                    ),
                    html.P(
                        id='last-updated',
                        style={
                            'color': '#888888',
                            'fontSize': '14px',
                        }
                    ),
                    html.P(
                        id='scan-status',
                        style={
                            'color': '#ffaa00',
                            'fontSize': '12px',
                        }
                    ),
                ]
            ),

            # Auto refresh every 30 seconds
            dcc.Interval(
                id='refresh',
                interval=30 * 1000,
                n_intervals=0
            ),

            # Summary Cards
            html.Div(
                id='summary-cards',
                style={
                    'display': 'flex',
                    'justifyContent': 'center',
                    'gap': '20px',
                    'marginBottom': '30px',
                    'flexWrap': 'wrap',
                }
            ),

            # Charts Row
            html.Div(
                style={
                    'display': 'flex',
                    'gap': '20px',
                    'marginBottom': '30px',
                    'flexWrap': 'wrap',
                },
                children=[
                    html.Div(
                        style={
                            'flex': '1',
                            'minWidth': '400px',
                            'backgroundColor': '#1a1a1a',
                            'borderRadius': '10px',
                            'padding': '15px',
                            'border': '1px solid #333',
                        },
                        children=[
                            html.H3(
                                "Portfolio Value",
                                style={'color': '#00ff88'}
                            ),
                            dcc.Graph(id='portfolio-chart'),
                        ]
                    ),
                    html.Div(
                        style={
                            'flex': '1',
                            'minWidth': '400px',
                            'backgroundColor': '#1a1a1a',
                            'borderRadius': '10px',
                            'padding': '15px',
                            'border': '1px solid #333',
                        },
                        children=[
                            html.H3(
                                "Position Allocation",
                                style={'color': '#00ff88'}
                            ),
                            dcc.Graph(id='allocation-chart'),
                        ]
                    ),
                ]
            ),

            # Signals
            html.Div(
                style={
                    'backgroundColor': '#1a1a1a',
                    'borderRadius': '10px',
                    'padding': '15px',
                    'marginBottom': '30px',
                    'border': '1px solid #333',
                },
                children=[
                    html.H3(
                        "📊 Live Signals",
                        style={'color': '#00ff88'}
                    ),
                    html.Div(id='signals-table'),
                ]
            ),

            # Positions
            html.Div(
                style={
                    'backgroundColor': '#1a1a1a',
                    'borderRadius': '10px',
                    'padding': '15px',
                    'marginBottom': '30px',
                    'border': '1px solid #333',
                },
                children=[
                    html.H3(
                        "💰 Open Positions",
                        style={'color': '#00ff88'}
                    ),
                    html.Div(id='positions-table'),
                ]
            ),

            # Trade History
            html.Div(
                style={
                    'backgroundColor': '#1a1a1a',
                    'borderRadius': '10px',
                    'padding': '15px',
                    'marginBottom': '30px',
                    'border': '1px solid #333',
                },
                children=[
                    html.H3(
                        "📋 Trade History",
                        style={'color': '#00ff88'}
                    ),
                    html.Div(id='history-table'),
                ]
            ),
        ]
    )

    @app.callback(
        [
            Output('summary-cards', 'children'),
            Output('portfolio-chart', 'figure'),
            Output('allocation-chart', 'figure'),
            Output('signals-table', 'children'),
            Output('positions-table', 'children'),
            Output('history-table', 'children'),
            Output('last-updated', 'children'),
            Output('scan-status', 'children'),
        ],
        [Input('refresh', 'n_intervals')]
    )
    def update_dashboard(n):
        """Update all dashboard components."""

        portfolio = load_portfolio()
        signals = load_signals()
        scan_info = load_scan_info()

        capital = portfolio.get('capital', 10000)
        starting = portfolio.get('starting_capital', 10000)
        positions = portfolio.get('positions', {})
        history = portfolio.get('trade_history', [])

        # Totals
        position_value = sum(
            pos.get('shares', 0) * pos.get('entry_price', 0)
            for pos in positions.values()
        )
        total_value = capital + position_value
        total_pnl = total_value - starting
        total_pct = (total_pnl / starting) * 100

        sells = [
            t for t in history
            if t.get('action') == 'SELL'
        ]
        wins = len(
            [t for t in sells if t.get('pnl', 0) > 0]
        )
        losses = len(
            [t for t in sells if t.get('pnl', 0) <= 0]
        )
        total_closed = wins + losses
        win_rate = (
            wins / total_closed * 100
            if total_closed > 0
            else 0
        )

        # Cards
        def make_card(title, value, color):
            return html.Div(
                style={
                    'backgroundColor': '#1a1a1a',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'minWidth': '180px',
                    'textAlign': 'center',
                    'border': f'1px solid {color}',
                },
                children=[
                    html.P(
                        title,
                        style={
                            'color': '#888',
                            'fontSize': '12px',
                            'margin': '0',
                        }
                    ),
                    html.H2(
                        value,
                        style={
                            'color': color,
                            'margin': '5px 0',
                            'fontSize': '24px',
                        }
                    ),
                ]
            )

        pnl_color = '#00ff88' if total_pnl >= 0 else '#ff4444'

        cards = [
            make_card(
                "TOTAL VALUE",
                f"${total_value:,.2f}",
                '#00ff88'
            ),
            make_card(
                "CASH",
                f"${capital:,.2f}",
                '#4488ff'
            ),
            make_card(
                "TOTAL P&L",
                f"${total_pnl:+,.2f} ({total_pct:+.1f}%)",
                pnl_color
            ),
            make_card(
                "POSITIONS",
                str(len(positions)),
                '#ffaa00'
            ),
            make_card(
                "TRADES",
                str(len(history)),
                '#aa88ff'
            ),
            make_card(
                "WIN RATE",
                f"{win_rate:.0f}%",
                '#00ff88' if win_rate > 50 else '#ff4444'
            ),
        ]

        # Portfolio Chart
        values = [starting]
        dates = ['Start']

        for trade in history:
            pnl = trade.get('pnl', 0)
            if trade.get('action') == 'SELL':
                values.append(values[-1] + pnl)
                dates.append(
                    trade.get('date', '')[:10]
                )

        values.append(total_value)
        dates.append('Now')

        portfolio_fig = go.Figure()
        portfolio_fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            line=dict(color='#00ff88', width=2),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)',
        ))
        portfolio_fig.update_layout(
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font=dict(color='#ffffff'),
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333'),
            margin=dict(l=40, r=20, t=20, b=40),
            height=300,
        )

        # Allocation Chart
        alloc_labels = ['Cash']
        alloc_values = [capital]
        alloc_colors = ['#4488ff']

        colors = [
            '#00ff88', '#ffaa00', '#ff4444',
            '#aa88ff', '#ff88aa', '#88ffaa',
        ]

        for i, (sym, pos) in enumerate(positions.items()):
            val = (
                pos.get('shares', 0)
                * pos.get('entry_price', 0)
            )
            alloc_labels.append(sym)
            alloc_values.append(val)
            alloc_colors.append(colors[i % len(colors)])

        alloc_fig = go.Figure(data=[go.Pie(
            labels=alloc_labels,
            values=alloc_values,
            marker=dict(colors=alloc_colors),
            hole=0.4,
            textfont=dict(color='#ffffff'),
        )])
        alloc_fig.update_layout(
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font=dict(color='#ffffff'),
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            showlegend=True,
            legend=dict(font=dict(color='#ffffff')),
        )

        # Signals Table
        if signals:
            sig_rows = []
            for sym, data in sorted(
                signals.items(),
                key=lambda x: x[1].get('prediction', 0),
                reverse=True
            ):
                sig_rows.append({
                    'Symbol': sym,
                    'Signal': data.get('signal', 'HOLD'),
                    'Prediction': f"{data.get('prediction', 0):.3f}",
                    'Regime': data.get('regime', ''),
                    'Sentiment': f"{data.get('sentiment', 0):+.2f}",
                    'Price': f"${data.get('price', 0):.2f}",
                })

            sig_df = pd.DataFrame(sig_rows)
            signals_tbl = dash_table.DataTable(
                data=sig_df.to_dict('records'),
                columns=[
                    {'name': c, 'id': c}
                    for c in sig_df.columns
                ],
                style_header={
                    'backgroundColor': '#2a2a2a',
                    'color': '#00ff88',
                    'fontWeight': 'bold',
                    'border': '1px solid #444',
                },
                style_cell={
                    'backgroundColor': '#1a1a1a',
                    'color': '#ffffff',
                    'border': '1px solid #333',
                    'textAlign': 'center',
                    'padding': '10px',
                },
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': '{Signal} = "BUY"',
                            'column_id': 'Signal'
                        },
                        'color': '#00ff88',
                        'fontWeight': 'bold',
                    },
                    {
                        'if': {
                            'filter_query': '{Signal} = "AVOID"',
                            'column_id': 'Signal'
                        },
                        'color': '#ff4444',
                        'fontWeight': 'bold',
                    },
                    {
                        'if': {
                            'filter_query': '{Signal} = "CAUTION"',
                            'column_id': 'Signal'
                        },
                        'color': '#ffaa00',
                        'fontWeight': 'bold',
                    },
                ],
            )
        else:
            signals_tbl = html.P(
                "No signals yet. Run scanner first.",
                style={'color': '#888'}
            )

        # Positions Table
        if positions:
            pos_rows = []
            for sym, pos in positions.items():
                shares = pos.get('shares', 0)
                entry = pos.get('entry_price', 0)
                cost = pos.get('cost', shares * entry)

                pos_rows.append({
                    'Symbol': sym,
                    'Shares': shares,
                    'Entry': f"${entry:.2f}",
                    'Cost': f"${cost:.2f}",
                    'Date': pos.get('entry_date', '')[:10],
                    'Reason': pos.get('reason', ''),
                })

            pos_df = pd.DataFrame(pos_rows)
            positions_tbl = dash_table.DataTable(
                data=pos_df.to_dict('records'),
                columns=[
                    {'name': c, 'id': c}
                    for c in pos_df.columns
                ],
                style_header={
                    'backgroundColor': '#2a2a2a',
                    'color': '#ffaa00',
                    'fontWeight': 'bold',
                    'border': '1px solid #444',
                },
                style_cell={
                    'backgroundColor': '#1a1a1a',
                    'color': '#ffffff',
                    'border': '1px solid #333',
                    'textAlign': 'center',
                    'padding': '10px',
                },
            )
        else:
            positions_tbl = html.P(
                "No open positions.",
                style={'color': '#888'}
            )

        # History Table
        if history:
            hist_rows = []
            for trade in reversed(history[-20:]):
                pnl = trade.get('pnl', 0)

                hist_rows.append({
                    'Date': trade.get('date', '')[:16],
                    'Action': trade.get('action', ''),
                    'Symbol': trade.get('symbol', ''),
                    'Shares': trade.get('shares', 0),
                    'Price': f"${trade.get('price', 0):.2f}",
                    'P&L': (
                        f"${pnl:+.2f}"
                        if trade.get('action') == 'SELL'
                        else ''
                    ),
                    'Reason': trade.get('reason', ''),
                })

            hist_df = pd.DataFrame(hist_rows)
            history_tbl = dash_table.DataTable(
                data=hist_df.to_dict('records'),
                columns=[
                    {'name': c, 'id': c}
                    for c in hist_df.columns
                ],
                style_header={
                    'backgroundColor': '#2a2a2a',
                    'color': '#aa88ff',
                    'fontWeight': 'bold',
                    'border': '1px solid #444',
                },
                style_cell={
                    'backgroundColor': '#1a1a1a',
                    'color': '#ffffff',
                    'border': '1px solid #333',
                    'textAlign': 'center',
                    'padding': '10px',
                },
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': '{Action} = "BUY"',
                            'column_id': 'Action'
                        },
                        'color': '#00ff88',
                    },
                    {
                        'if': {
                            'filter_query': '{Action} = "SELL"',
                            'column_id': 'Action'
                        },
                        'color': '#ff4444',
                    },
                ],
            )
        else:
            history_tbl = html.P(
                "No trades yet.",
                style={'color': '#888'}
            )

        # Timestamps
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp = f"Dashboard refreshed: {now}"

        last_scan = scan_info.get('last_scan', 'Never')
        next_min = scan_info.get('next_scan_minutes', 30)
        n_stocks = scan_info.get('stocks_scanned', 0)
        n_crypto = scan_info.get('crypto_scanned', 0)

        scan_status = (
            f"Last scan: {last_scan[:19]}"
            f" | Stocks: {n_stocks}"
            f" | Crypto: {n_crypto}"
            f" | Auto-scan every {next_min} min"
        )

        return (
            cards,
            portfolio_fig,
            alloc_fig,
            signals_tbl,
            positions_tbl,
            history_tbl,
            timestamp,
            scan_status,
        )

    return app


if __name__ == '__main__':
    print("\n" + "🌐" * 25)
    print("STARTING ALPHAEDGE DASHBOARD")
    print("🌐" * 25)
    print("\nOpen browser: http://localhost:8050")
    print("Press Ctrl+C to stop\n")

    app = create_app()
    app.run(debug=False, host='0.0.0.0', port=8050)