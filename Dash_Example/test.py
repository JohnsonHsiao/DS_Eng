# test.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

# 创建 Dash 应用
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 模拟数据生成
def get_latest_data():
    timestamps = pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(minutes=20), periods=20, freq='T')
    ph_values = np.random.normal(loc=7, scale=0.2, size=20)
    iron_concentration = np.random.normal(loc=0.8, scale=0.1, size=20)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'ph_value': ph_values,
        'iron_concentration': iron_concentration
    })
    return df

# 应用布局
app.layout = dbc.Container([
    html.H1("Wastewater Treatment Monitoring Dashboard", style={'text-align': 'center'}),
    dbc.Row([
        dbc.Col([dcc.Graph(id="ph-graph", config={'displayModeBar': False})], width=6),
        dbc.Col([dcc.Graph(id="iron-graph", config={'displayModeBar': False})], width=6),
    ]),
    dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0)
])

# 回调函数
@app.callback(
    [Output("ph-graph", "figure"), Output("iron-graph", "figure")],
    [Input("interval-component", "n_intervals")]
)
def update_graph(n):
    df = get_latest_data()
    ph_fig = {'data': [{'x': df['timestamp'], 'y': df['ph_value'], 'type': 'line', 'name': 'pH Level'}],
              'layout': {'title': 'Real-time pH Level', 'xaxis': {'title': 'Timestamp'}, 'yaxis': {'title': 'pH Value'}}}
    iron_fig = {'data': [{'x': df['timestamp'], 'y': df['iron_concentration'], 'type': 'line', 'name': 'Iron Concentration'}],
                'layout': {'title': 'Real-time Iron Concentration', 'xaxis': {'title': 'Timestamp'}, 'yaxis': {'title': 'Concentration (mg/L)'}}}
    return ph_fig, iron_fig

# 添加这一行，便于 Gunicorn 找到 Flask 实例
server = app.server
