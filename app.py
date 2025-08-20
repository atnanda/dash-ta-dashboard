import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# -------------------------------
# Normalize columns
# -------------------------------
def normalize_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]
    return df

# -------------------------------
# Minimum holding filter
# -------------------------------
def filter_min_hold(signals, min_days):
    signals_filtered = signals.copy()
    for i in range(min_days, len(signals)):
        if signals.iloc[i] and signals.iloc[i-min_days:i].any():
            signals_filtered.iloc[i] = False
    return signals_filtered

# -------------------------------
# Support/Resistance
# -------------------------------
def compute_support_resistance(price_data, window=5, top_n=2, proximity=0.1):
    current_price = price_data['close'].iloc[-1]
    resistance = price_data['high'][(price_data['high'].rolling(window, center=True).max() == price_data['high'])]
    support = price_data['low'][(price_data['low'].rolling(window, center=True).min() == price_data['low'])]

    resistance_levels = [r for r in sorted(resistance, reverse=True) if abs(r - current_price)/current_price <= proximity]
    support_levels = [s for s in sorted(support) if abs(s - current_price)/current_price <= proximity]

    def merge_close(levels, threshold=0.02):
        merged = []
        for l in levels:
            if not merged:
                merged.append(l)
            elif abs(l - merged[-1])/merged[-1] <= threshold:
                merged[-1] = (merged[-1] + l)/2
            else:
                merged.append(l)
        return merged

    return merge_close(support_levels)[:top_n], merge_close(resistance_levels)[:top_n]

# -------------------------------
# Compute trades and profits
# -------------------------------
def compute_trade_profits(buy_signals, sell_signals, close_prices):
    buy_indices = buy_signals[buy_signals].index
    sell_indices = sell_signals[sell_signals].index

    trades = []
    sell_pointer = 0
    for buy_idx in buy_indices:
        while sell_pointer < len(sell_indices) and sell_indices[sell_pointer] <= buy_idx:
            sell_pointer += 1
        if sell_pointer < len(sell_indices):
            sell_idx = sell_indices[sell_pointer]
            entry_price = close_prices.loc[buy_idx]
            exit_price = close_prices.loc[sell_idx]
            profit_pct = (exit_price / entry_price - 1) * 100

            trades.append({
                "buy_idx": buy_idx,
                "sell_idx": sell_idx,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "profit_pct": profit_pct
            })
            sell_pointer += 1
    return trades

# -------------------------------
# Main chart function
# -------------------------------
def create_signal_chart(ticker, min_hold_days=3):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    price_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if price_data.empty:
        raise ValueError(f"No data found for {ticker}")

    price_data = normalize_columns(price_data)
    close_prices = price_data['close']

    # -------------------------------
    # Indicators
    # -------------------------------
    sma20 = close_prices.rolling(20).mean()
    sma50 = close_prices.rolling(50).mean()
    sma200 = close_prices.rolling(200).mean().bfill()  # ensure full span

    volume_ma = price_data['volume'].rolling(20).mean()
    bb_std = close_prices.rolling(20).std()
    bb_upper = sma20 + 1.5*bb_std
    bb_lower = sma20 - 1.5*bb_std

    # RSI
    delta = close_prices.diff()
    gain = delta.where(delta>0,0).rolling(14).mean()
    loss = -delta.where(delta<0,0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - 100/(1+rs)
    rsi = rsi.fillna(50)

    # MACD
    exp1 = close_prices.ewm(span=12, adjust=False).mean()
    exp2 = close_prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal_line

    # -------------------------------
    # Score-based Buy/Sell Signals
    # -------------------------------
    uptrend = close_prices > sma200
    downtrend = close_prices < sma200

    rsi_bull = rsi < 40
    rsi_bear = rsi > 60
    macd_bull = macd > signal_line
    macd_bear = macd < signal_line

    price_near_lower = (close_prices < bb_lower) | (close_prices < sma20)
    price_near_upper = (close_prices > bb_upper) | (close_prices > sma20)

    vol_high = price_data['volume'] > volume_ma

    # Compute scores
    buy_score = uptrend.astype(int) + vol_high.astype(int) + price_near_lower.astype(int) + rsi_bull.astype(int) + macd_bull.astype(int)
    sell_score = downtrend.astype(int) + vol_high.astype(int) + price_near_upper.astype(int) + rsi_bear.astype(int) + macd_bear.astype(int)

    threshold = 3
    buy_signals = buy_score >= threshold
    sell_signals = sell_score >= threshold

    buy_signals = filter_min_hold(buy_signals, min_hold_days)
    sell_signals = filter_min_hold(sell_signals, min_hold_days)

    last_idx = price_data.index[-1]

    # Today's action
    if buy_signals.iloc[-1]:
        today_action = "BUY"
        today_color = "green"
    elif sell_signals.iloc[-1]:
        today_action = "SELL"
        today_color = "red"
    else:
        today_action = "HOLD"
        today_color = "yellow"

    # Support/Resistance
    support_levels, resistance_levels = compute_support_resistance(price_data)

    # Trades
    trades = compute_trade_profits(buy_signals, sell_signals, close_prices)

    # -------------------------------
    # Performance summary
    # -------------------------------
    total_trades = len(trades)
    wins = [t for t in trades if t['profit_pct'] > 0]
    losses = [t for t in trades if t['profit_pct'] <= 0]
    win_rate = (len(wins)/total_trades*100) if total_trades > 0 else 0
    avg_profit = (sum(t['profit_pct'] for t in trades)/total_trades) if total_trades > 0 else 0

    # -------------------------------
    # Plotting
    # -------------------------------
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02,
                        row_heights=[0.5,0.2,0.15,0.15],
                        subplot_titles=(f"{ticker} Price with Signals","Volume","RSI","MACD"))

    # Price + Candlestick + SMA + Bollinger
    fig.add_trace(go.Candlestick(x=price_data.index, open=price_data['open'], high=price_data['high'],
                                 low=price_data['low'], close=close_prices, name=ticker), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_data.index, y=close_prices, line=dict(color='blue', width=2), name='Close Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_data.index, y=bb_upper, line=dict(color='gray', dash='dash'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_data.index, y=bb_lower, line=dict(color='gray', dash='dash'), name='BB Lower'), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_data.index, y=sma20, line=dict(color='#ADD8E6', width=1), name='SMA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_data.index, y=sma50, line=dict(color='#FFB84D', width=1), name='SMA50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_data.index, y=sma200, line=dict(color='black', width=1), name='SMA200'), row=1, col=1)

    # Trades markers and connecting lines
    for idx, t in enumerate(trades):
        fig.add_trace(go.Scatter(
            x=[t['buy_idx']], y=[t['entry_price']*0.995],
            mode='markers', marker=dict(symbol='triangle-up', color='green', size=14),
            hovertext=f"BUY\nPrice: ${t['entry_price']:.2f}\nProfit: {t['profit_pct']:.2f}%", hoverinfo='text',
            name=f"Trade {idx+1} Buy"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[t['sell_idx']], y=[t['exit_price']*1.005],
            mode='markers', marker=dict(symbol='triangle-down', color='red', size=14),
            hovertext=f"SELL\nPrice: ${t['exit_price']:.2f}\nProfit: {t['profit_pct']:.2f}%", hoverinfo='text',
            name=f"Trade {idx+1} Sell"
        ), row=1, col=1)
        # Line connecting buy/sell
        fig.add_trace(go.Scatter(
            x=[t['buy_idx'], t['sell_idx']], y=[t['entry_price'], t['exit_price']],
            mode='lines', line=dict(color='purple', width=2, dash='dot'),
            hovertext=f"Trade {idx+1} Profit: {t['profit_pct']:.2f}%", hoverinfo='text',
            showlegend=False
        ), row=1, col=1)

    # Support/resistance
    for level in support_levels:
        fig.add_trace(go.Scatter(x=price_data.index, y=[level]*len(price_data), mode='lines', 
                                 line=dict(color='green', width=1, dash='dot'), name='Support Zone'), row=1, col=1)
    for level in resistance_levels:
        fig.add_trace(go.Scatter(x=price_data.index, y=[level]*len(price_data), mode='lines', 
                                 line=dict(color='red', width=1, dash='dot'), name='Resistance Zone'), row=1, col=1)

    # Today's action marker
    fig.add_trace(go.Scatter(
        x=[last_idx], y=[close_prices.iloc[-1]],
        mode='markers+text',
        marker=dict(symbol='star', color=today_color, size=22),
        text=[f"{today_action}\n${close_prices.iloc[-1]:.2f}\n{last_idx.strftime('%Y-%m-%d')}"],
        textposition='top center',
        showlegend=True,
        name="Today's Action"
    ), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=price_data.index, y=price_data['volume'], name='Volume'), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=price_data.index, y=rsi, line=dict(color='purple'), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=price_data.index, y=macd, line=dict(color='blue'), name='MACD'), row=4, col=1)
    fig.add_trace(go.Scatter(x=price_data.index, y=signal_line, line=dict(color='orange'), name='Signal'), row=4, col=1)
    fig.add_trace(go.Bar(x=price_data.index, y=macd_hist, 
                         marker_color=['red' if h<0 else 'green' for h in macd_hist], name='MACD Hist'), row=4, col=1)

    # Summary table at top-left
    fig.add_trace(go.Table(
        header=dict(values=["Metric","Value"], fill_color='lightgrey'),
        cells=dict(values=[["Total Trades","Wins","Losses","Win Rate","Avg Profit"], 
                           [total_trades,len(wins),len(losses),f"{win_rate:.1f}%",f"{avg_profit:.2f}%"]]),
        domain=dict(x=[0,0.25], y=[0.8,1])
    ))

    fig.update_layout(title=f"{ticker} Technical Analysis", xaxis_rangeslider_visible=False, height=1400)
    return fig

# -------------------------------
# Dash App
# -------------------------------
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("Technical Analysis Dashboard"),
    html.Div([
        html.Label("Tickers (comma-separated):", style={'margin-right':'5px'}),
        dcc.Input(id='ticker-input', type='text', placeholder='Enter tickers', 
                  style={'width':'200px', 'margin-right':'10px'}),
        html.Label("Min Hold Days:", style={'margin-right':'5px'}),
        dcc.Input(id='min-hold-input', type='number', min=1, value=3, 
                  style={'width':'50px', 'margin-right':'10px'}),
        html.Button('Submit', id='submit-tickers', n_clicks=0)
    ], style={'margin-bottom':'5px', 'display':'flex', 'align-items':'center'}),
    
    dcc.Dropdown(id='ticker-dropdown', options=[], value=None, 
                 style={'width':'150px', 'margin-bottom':'5px'}),
    html.Div(id='selected-ticker-text', style={'fontWeight':'bold', 'margin-bottom':'10px'}),
    
    dcc.Graph(id='ta-graph')
])

# -------------------------------
# Populate dropdown only after Submit
# -------------------------------
@app.callback(
    Output('ticker-dropdown', 'options'),
    Output('ticker-dropdown', 'value'),
    Input('submit-tickers', 'n_clicks'),
    State('ticker-input', 'value')
)
def update_dropdown(n_clicks, value):
    if n_clicks == 0 or not value:
        return [], None
    tickers = [t.strip().upper() for t in value.split(',') if t.strip()]
    options = [{'label': t, 'value': t} for t in tickers]
    default_value = tickers[0] if tickers else None
    return options, default_value

@app.callback(
    Output('ta-graph', 'figure'),
    Output('selected-ticker-text', 'children'),
    Input('ticker-dropdown', 'value'),
    Input('min-hold-input', 'value')
)
def update_graph(selected_ticker, min_hold_days):
    if not selected_ticker or not min_hold_days:
        return go.Figure(), ""
    fig = create_signal_chart(selected_ticker, min_hold_days=min_hold_days)
    return fig, f"Selected Ticker: {selected_ticker}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050)
