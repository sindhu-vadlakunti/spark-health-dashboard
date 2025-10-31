# health_dashboard.py
import pandas as pd
import glob
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# --- Load Data ---
data_path = "/home/sindhu/bigdata/output/health.csv/*.csv"
all_files = glob.glob(data_path)

if not all_files:
    raise FileNotFoundError(f"No CSV files found in {data_path}")

df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

# --- Initialize App ---
app = Dash(__name__, title="Family Health Super Dashboard")
server = app.server

# --- Dropdown Options ---
risk_options = [{"label": r, "value": r} for r in sorted(df["Risk_Level"].unique())]
family_options = [{"label": f"Family {fid}", "value": fid} for fid in sorted(df["Family_ID"].unique())]

# --- Layout ---
app.layout = html.Div(
    style={
        "backgroundColor": "#0B0C10",
        "color": "#C5C6C7",
        "fontFamily": "Arial, sans-serif",
        "padding": "20px"
    },
    children=[
        html.H1(
            "🏥 Family Health Analytics Dashboard",
            style={"textAlign": "center", "color": "#66FCF1", "marginBottom": "40px"}
        ),

        html.Div([
            html.Div([
                html.Label("Select Risk Level:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="risk_filter",
                    options=risk_options,
                    placeholder="Filter by Risk Level",
                    multi=True,
                    style={"backgroundColor": "#1F2833", "color": "#000"}
                ),
            ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),

            html.Div([
                html.Label("Select Family:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="family_filter",
                    options=family_options,
                    placeholder="Filter by Family ID",
                    multi=True,
                    style={"backgroundColor": "#1F2833", "color": "#000"}
                ),
            ], style={"width": "48%", "display": "inline-block", "marginLeft": "2%", "verticalAlign": "top"})
        ], style={"marginBottom": "30px"}),

        html.Div([
            html.Div([dcc.Graph(id="wellness_chart")],
                     style={"width": "48%", "display": "inline-block"}),
            html.Div([dcc.Graph(id="risk_chart")],
                     style={"width": "48%", "display": "inline-block", "marginLeft": "2%"})
        ]),

        html.Div([
            html.Div([dcc.Graph(id="composite_chart")],
                     style={"width": "48%", "display": "inline-block"}),
            html.Div([dcc.Graph(id="family_chart")],
                     style={"width": "48%", "display": "inline-block", "marginLeft": "2%"})
        ])
    ]
)

# --- Callbacks ---
@app.callback(
    [Output("wellness_chart", "figure"),
     Output("risk_chart", "figure"),
     Output("composite_chart", "figure"),
     Output("family_chart", "figure")],
    [Input("risk_filter", "value"),
     Input("family_filter", "value")]
)
def update_charts(selected_risks, selected_families):
    dff = df.copy()

    if selected_risks:
        dff = dff[dff["Risk_Level"].isin(selected_risks)]
    if selected_families:
        dff = dff[dff["Family_ID"].isin(selected_families)]

    wellness_fig = px.histogram(
        dff, x="Wellness_Score", nbins=10,
        title="Wellness Score Distribution",
        color_discrete_sequence=["#66FCF1"]
    )
    wellness_fig.update_layout(
        plot_bgcolor="#1F2833", paper_bgcolor="#1F2833", font_color="#C5C6C7"
    )

    risk_fig = px.histogram(
        dff, x="Risk_Level", color="Risk_Level",
        title="Risk Level Distribution"
    )
    risk_fig.update_layout(
        plot_bgcolor="#1F2833", paper_bgcolor="#1F2833", font_color="#C5C6C7"
    )

    composite_fig = px.scatter(
        dff, x="Wellness_Score", y="Composite_Risk_Score",
        color="Composite_Risk_Level",
        hover_data=["Family_ID", "Age", "Gender"],
        title="Composite Risk vs Wellness Score"
    )
    composite_fig.update_layout(
        plot_bgcolor="#1F2833", paper_bgcolor="#1F2833", font_color="#C5C6C7"
    )

    family_avg = dff.groupby("Family_ID").agg({"Wellness_Score": "mean"}).reset_index()
    family_fig = px.bar(
        family_avg, x="Family_ID", y="Wellness_Score",
        title="Average Wellness per Family",
        color="Wellness_Score", color_continuous_scale="turbo"
    )
    family_fig.update_layout(
        plot_bgcolor="#1F2833", paper_bgcolor="#1F2833", font_color="#C5C6C7"
    )

    return wellness_fig, risk_fig, composite_fig, family_fig


# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True, port=8050)
