import dash
from dash import html

app = dash.Dash(__name__)
app.layout = html.Div("Hello, World!")
server = app.server  # 用于 gunicorn

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080)

