import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

# 创建 Dash 应用
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 模拟生成数据的函数
def get_latest_data():
    # 模拟最近20分钟的数据
    timestamps = pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(minutes=20), periods=20, freq='T')
    ph_values = np.random.normal(loc=7, scale=0.2, size=20)  # 模拟pH值数据，均值为7，标准差为0.2
    iron_concentration = np.random.normal(loc=0.8, scale=0.1, size=20)  # 模拟铁离子浓度数据，均值为0.8，标准差为0.1
    
    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'ph_value': ph_values,
        'iron_concentration': iron_concentration
    })
    
    return df

# 应用布局
app.layout = dbc.Container([
    html.H1("Wastewater Treatment Monitoring Dashboard", style={'text-align': 'center'}),
    
    # 实时图表
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="ph-graph", config={'displayModeBar': False}),
        ], width=6),
        dbc.Col([
            dcc.Graph(id="iron-graph", config={'displayModeBar': False}),
        ], width=6),
    ]),

    # 定时更新
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # 每5秒更新一次（以毫秒为单位）
        n_intervals=0
    )
])

# 回调函数更新图表
@app.callback(
    [Output("ph-graph", "figure"), Output("iron-graph", "figure")],
    [Input("interval-component", "n_intervals")]
)
def update_graph(n):
    # 获取最新数据
    df = get_latest_data()
    
    # 绘制pH值图表
    ph_fig = {
        'data': [{
            'x': df['timestamp'],
            'y': df['ph_value'],
            'type': 'line',
            'name': 'pH Level'
        }],
        'layout': {
            'title': 'Real-time pH Level',
            'xaxis': {'title': 'Timestamp'},
            'yaxis': {'title': 'pH Value'},
            'margin': {'l': 50, 'r': 10, 't': 50, 'b': 40},
        }
    }

    # 绘制铁离子浓度图表
    iron_fig = {
        'data': [{
            'x': df['timestamp'],
            'y': df['iron_concentration'],
            'type': 'line',
            'name': 'Iron Concentration'
        }],
        'layout': {
            'title': 'Real-time Iron Concentration',
            'xaxis': {'title': 'Timestamp'},
            'yaxis': {'title': 'Concentration (mg/L)'},
            'margin': {'l': 50, 'r': 10, 't': 50, 'b': 40},
        }
    }

    return ph_fig, iron_fig

# 在Gunicorn中运行时，不需要 app.run_server()
server = app.server  # 添加这一行，使得Gunicorn
