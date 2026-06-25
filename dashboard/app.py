import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)

print("Base directory:", BASE_DIR)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
import yfinance as yf

# ── Load all data ──────────────────────────────────────────────
try:
    kpi_df   = pd.read_csv("data/processed/kpi_results.csv",       index_col=0)
    hist_var = pd.read_csv("data/processed/historical_var.csv",     index_col=0)
    para_var = pd.read_csv("data/processed/parametric_var.csv",     index_col=0)
    mc_var   = pd.read_csv("data/processed/monte_carlo_var.csv",    index_col=0)
    returns  = pd.read_csv("data/processed/returns_clean.csv",      index_col="Date", parse_dates=True)
    prices   = pd.read_csv("data/raw/prices.csv",                   index_col="Date", parse_dates=True)
    clusters = pd.read_csv("data/processed/clustering_results.csv", index_col=0)
    backtest = pd.read_csv("data/processed/backtest_results.csv",   index_col=0)
    print("All data loaded! ✅")
except Exception as e:
    print(f"Error loading data: {e}")

# ── Constants ──────────────────────────────────────────────────
TICKERS = [
    "JPM", "GS", "HLI",
    "AAPL", "PLTR", "META",
    "JNJ", "MRK", "UNH",
    "MCD", "KO",
    "EL", "ULTA",
    "XOM", "CVX"
]

SECTORS_MAP = {
    "JPM": "Financials", "GS": "Financials",  "HLI": "Financials",
    "AAPL": "Tech",      "PLTR": "Tech",       "META": "Tech",
    "JNJ": "Healthcare", "MRK": "Healthcare",  "UNH": "Healthcare",
    "MCD": "Staples",    "KO": "Staples",
    "EL": "Beauty",      "ULTA": "Beauty",
    "XOM": "Energy",     "CVX": "Energy"
}

COLORS_MAP = {
    "Tech":        "#E24B4A",
    "Beauty":      "#D4537E",
    "Energy":      "#EF9F27",
    "Financials":  "#378ADD",
    "Healthcare":  "#1D9E75",
    "Staples":     "#639922"
}

# ── App ────────────────────────────────────────────────────────
app = Dash(__name__)

app.layout = html.Div([

    # ── HEADER ──────────────────────────────────────────────
    html.Div([
        html.H1("Real-Time Risk Dashboard",
                style={"color": "white", "margin": "0",
                       "fontSize": "28px", "fontWeight": "600"}),
        html.P("VaR + KPI Integration | 15 Stocks | 6 Sectors | 2022–2025",
               style={"color": "#B8CCE4", "margin": "4px 0 0 0",
                      "fontSize": "13px"}),
    ], style={
        "background": "#1A2B4A",
        "padding": "20px 30px",
        "marginBottom": "20px"
    }),

    # ── CONTROLS ────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Label("Select Stock:",
                       style={"fontWeight": "500", "fontSize": "13px"}),
            dcc.Dropdown(
                id="ticker-dropdown",
                options=[{"label": f"{t} — {SECTORS_MAP[t]}", "value": t}
                         for t in TICKERS],
                value="AAPL",
                clearable=False,
                style={"fontSize": "13px"}
            )
        ], style={"width": "30%", "marginRight": "20px"}),

        html.Div([
            html.Label("Confidence Level:",
                       style={"fontWeight": "500", "fontSize": "13px"}),
            dcc.RadioItems(
                id="confidence-radio",
                options=[
                    {"label": " 95%", "value": 0.95},
                    {"label": " 99%", "value": 0.99}
                ],
                value=0.95,
                inline=True,
                style={"marginTop": "8px", "fontSize": "13px"}
            )
        ], style={"width": "20%", "marginRight": "20px"}),

        html.Div([
            html.Label("VaR Method:",
                       style={"fontWeight": "500", "fontSize": "13px"}),
            dcc.RadioItems(
                id="method-radio",
                options=[
                    {"label": " Historical",  "value": "hist"},
                    {"label": " Parametric",  "value": "para"},
                    {"label": " Monte Carlo", "value": "mc"}
                ],
                value="hist",
                inline=True,
                style={"marginTop": "8px", "fontSize": "13px"}
            )
        ], style={"width": "40%"}),

    ], style={
        "display": "flex",
        "alignItems": "flex-end",
        "padding": "15px 30px",
        "background": "#F8F9FA",
        "borderBottom": "1px solid #dee2e6",
        "marginBottom": "20px"
    }),

    # ── KPI CARDS ───────────────────────────────────────────
    html.Div(id="kpi-cards", style={
        "display": "flex",
        "gap": "15px",
        "padding": "0 30px",
        "marginBottom": "20px"
    }),

    # ── CHARTS ROW 1 ────────────────────────────────────────
    html.Div([
        html.Div([
            dcc.Graph(id="price-chart", style={"height": "320px"})
        ], style={"width": "60%", "marginRight": "15px",
                  "background": "white", "borderRadius": "8px",
                  "border": "1px solid #dee2e6", "padding": "10px"}),

        html.Div([
            dcc.Graph(id="returns-dist", style={"height": "320px"})
        ], style={"width": "40%",
                  "background": "white", "borderRadius": "8px",
                  "border": "1px solid #dee2e6", "padding": "10px"}),
    ], style={"display": "flex", "padding": "0 30px",
              "marginBottom": "15px"}),

    # ── CHARTS ROW 2 ────────────────────────────────────────
    html.Div([
        html.Div([
            dcc.Graph(id="var-comparison", style={"height": "320px"})
        ], style={"width": "50%", "marginRight": "15px",
                  "background": "white", "borderRadius": "8px",
                  "border": "1px solid #dee2e6", "padding": "10px"}),

        html.Div([
            dcc.Graph(id="cluster-chart", style={"height": "320px"})
        ], style={"width": "50%",
                  "background": "white", "borderRadius": "8px",
                  "border": "1px solid #dee2e6", "padding": "10px"}),
    ], style={"display": "flex", "padding": "0 30px",
              "marginBottom": "15px"}),

    # ── CHARTS ROW 3 ────────────────────────────────────────
    html.Div([
        html.Div([
            dcc.Graph(id="kpi-chart", style={"height": "320px"})
        ], style={"width": "100%",
                  "background": "white", "borderRadius": "8px",
                  "border": "1px solid #dee2e6", "padding": "10px"}),
    ], style={"padding": "0 30px", "marginBottom": "15px"}),

    # ── LIVE REFRESH ────────────────────────────────────────
    dcc.Interval(id="interval", interval=60000, n_intervals=0),

    # ── FOOTER ──────────────────────────────────────────────
    html.Div([
        html.P("Real-Time Risk Dashboard | Applied Quantitative Methods | Master's Project",
               style={"color": "#6c757d", "fontSize": "12px", "margin": "0"})
    ], style={"textAlign": "center", "padding": "20px",
              "borderTop": "1px solid #dee2e6"})

], style={"fontFamily": "Arial, sans-serif",
          "background": "#F0F2F5", "minHeight": "100vh"})


# ── CALLBACKS ─────────────────────────────────────────────────

@app.callback(
    Output("kpi-cards", "children"),
    Input("ticker-dropdown", "value")
)
def update_kpi_cards(ticker):
    row   = kpi_df.loc[ticker]
    cards = [
        ("Sharpe Ratio",  f"{row['Sharpe Ratio']:.2f}",      "#378ADD"),
        ("Max Drawdown",  f"{row['Max Drawdown (%)']:.1f}%",  "#E24B4A"),
        ("Beta",          f"{row['Beta']:.2f}",               "#EF9F27"),
        ("Volatility",    f"{row['Volatility (%)']:.1f}%",    "#D4537E"),
        ("CAGR",          f"{row['CAGR (%)']:.1f}%",          "#1D9E75"),
        ("Sector",        SECTORS_MAP[ticker],                 "#534AB7"),
    ]
    return [
        html.Div([
            html.P(label, style={"margin": "0", "fontSize": "11px",
                                  "color": "#6c757d", "fontWeight": "500"}),
            html.H3(value, style={"margin": "4px 0 0 0", "fontSize": "22px",
                                   "color": color, "fontWeight": "600"})
        ], style={
            "background": "white",
            "borderRadius": "8px",
            "border": f"1px solid {color}",
            "borderTop": f"4px solid {color}",
            "padding": "12px 16px",
            "flex": "1"
        }) for label, value, color in cards
    ]


@app.callback(
    Output("price-chart", "figure"),
    Input("ticker-dropdown", "value")
)
def update_price_chart(ticker):
    p   = prices[ticker].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p.index, y=p.values,
        mode="lines",
        name=ticker,
        line=dict(color=COLORS_MAP[SECTORS_MAP[ticker]], width=2),
        fill="tozeroy",
        fillcolor=COLORS_MAP[SECTORS_MAP[ticker]] + "22"
    ))
    fig.update_layout(
        title=f"{ticker} — Price History (2022–2025)",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis_title="Date", yaxis_title="Price (USD)",
        showlegend=False
    )
    return fig


@app.callback(
    Output("returns-dist", "figure"),
    [Input("ticker-dropdown", "value"),
     Input("confidence-radio", "value"),
     Input("method-radio", "value")]
)
def update_returns_dist(ticker, confidence, method):
    r          = returns[ticker].dropna()
    var_source = {"hist": hist_var, "para": para_var, "mc": mc_var}[method]
    col        = "VaR_95%" if confidence == 0.95 else "VaR_99%"
    var_value  = var_source.loc[ticker, col]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r.values, nbinsx=60,
        marker_color=COLORS_MAP[SECTORS_MAP[ticker]],
        opacity=0.75, name="Returns"
    ))
    fig.add_vline(
        x=var_value, line_dash="dash",
        line_color="red", line_width=2,
        annotation_text=f"VaR {int(confidence*100)}%: {var_value*100:.2f}%"
    )
    fig.update_layout(
        title=f"{ticker} — Returns Distribution",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        showlegend=False
    )
    return fig


@app.callback(
    Output("var-comparison", "figure"),
    Input("confidence-radio", "value")
)
def update_var_comparison(confidence):
    col = "VaR_95%" if confidence == 0.95 else "VaR_99%"
    avg = ((hist_var[col] + para_var[col] + mc_var[col]) / 3).sort_values()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=avg.index, y=hist_var.loc[avg.index, col],
        name="Historical", marker_color="#378ADD", opacity=0.8
    ))
    fig.add_trace(go.Bar(
        x=avg.index, y=para_var.loc[avg.index, col],
        name="Parametric", marker_color="#EF9F27", opacity=0.8
    ))
    fig.add_trace(go.Bar(
        x=avg.index, y=mc_var.loc[avg.index, col],
        name="Monte Carlo", marker_color="#E24B4A", opacity=0.8
    ))
    fig.update_layout(
        title=f"VaR {int(confidence*100)}% — All 3 Methods",
        barmode="group",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis_title="Stock", yaxis_title="VaR",
        legend=dict(orientation="h", y=1.1)
    )
    return fig


@app.callback(
    Output("cluster-chart", "figure"),
    Input("ticker-dropdown", "value")
)
def update_cluster(selected_ticker):
    cluster_colors = {
        "🔴 High Risk / High Return": "#E24B4A",
        "🟡 Balanced":                "#EF9F27",
        "🟢 Defensive / Low Risk":    "#1D9E75"
    }

    fig = go.Figure()
    for cluster_name, color in cluster_colors.items():
        if cluster_name not in clusters["Cluster Name"].values:
            continue
        mask   = clusters["Cluster Name"] == cluster_name
        subset = clusters[mask]
        fig.add_trace(go.Scatter(
            x=subset["Volatility"],
            y=subset["Sharpe Ratio"],
            mode="markers+text",
            name=cluster_name,
            text=subset.index.tolist(),
            textposition="top center",
            marker=dict(
                size=[16 if t == selected_ticker else 10
                      for t in subset.index],
                color=color,
                line=dict(width=1, color="white")
            )
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Sector Clustering — Risk vs Return",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis_title="Volatility (%)",
        yaxis_title="Sharpe Ratio",
        legend=dict(orientation="h", y=-0.2)
    )
    return fig


@app.callback(
    Output("kpi-chart", "figure"),
    Input("ticker-dropdown", "value")
)
def update_kpi_chart(selected_ticker):
    bar_colors = [
        "#1A2B4A" if t == selected_ticker
        else COLORS_MAP[SECTORS_MAP[t]]
        for t in kpi_df.index
    ]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Sharpe Ratio", "Volatility (%)", "CAGR (%)")
    )

    fig.add_trace(go.Bar(
        x=kpi_df.index, y=kpi_df["Sharpe Ratio"],
        marker_color=bar_colors, showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=kpi_df.index, y=kpi_df["Volatility (%)"],
        marker_color=bar_colors, showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=kpi_df.index, y=kpi_df["CAGR (%)"],
        marker_color=bar_colors, showlegend=False
    ), row=1, col=3)

    fig.update_layout(
        title=f"KPI Comparison — Selected: {selected_ticker} (dark blue)",
        plot_bgcolor="white", paper_bgcolor="white",
        height=300,
        margin=dict(l=40, r=20, t=60, b=30)
    )
    return fig


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)