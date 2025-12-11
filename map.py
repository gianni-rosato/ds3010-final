import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from plotly.graph_objs import Figure
import main

print("Loading data...")
df: pd.DataFrame = main.load_data_from_kaggle()
df = main.clean_data(df)

# compute climate vulnerability index if missing
if "climate_vulnerability_index" not in df.columns:
    print("Calculating Climate Vulnerability Index...")

    def normalize(series: pd.Series) -> pd.Series:
        return (series - series.min()) / (series.max() - series.min())

    cvi: pd.Series = (
        normalize(df["pm25_ugm3"]) * 0.4
        + normalize(df["temp_anomaly_celsius"]) * 0.3
        + normalize(df["extreme_weather_events"]) * 0.3
    ) * 100
    df["climate_vulnerability_index"] = cvi

# init app
app: dash.Dash = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Global Climate-Health Dashboard", style={"textAlign": "center"}),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Filter by Income Level:"),
                        dcc.Dropdown(
                            id="income-filter",
                            options=[
                                {"label": i, "value": i}
                                for i in sorted(df["income_level"].unique())
                            ],
                            value=None,
                            placeholder="Select Income Level (All)",
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.Label("Lag Health Outcomes (Weeks):"),
                        dcc.Slider(
                            id="lag-slider",
                            min=0,
                            max=4,
                            step=1,
                            value=0,
                            marks={i: f"{i} weeks" for i in range(5)},
                        ),
                    ],
                    style={"width": "48%", "float": "right", "display": "inline-block"},
                ),
            ],
            style={"padding": "20px"},
        ),
        html.Div(
            [
                dcc.Graph(id="choropleth-map", style={"height": "600px"}),
            ]
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="trend-chart")],
                    style={"width": "49%", "display": "inline-block"},
                ),
                html.Div(
                    [dcc.Graph(id="scatter-plot")],
                    style={"width": "49%", "display": "inline-block", "float": "right"},
                ),
            ]
        ),
    ]
)


@app.callback(
    [
        Output("choropleth-map", "figure"),
        Output("trend-chart", "figure"),
        Output("scatter-plot", "figure"),
    ],
    [
        Input("income-filter", "value"),
        Input("lag-slider", "value"),
        Input("choropleth-map", "clickData"),
    ],
)
def update_dashboard(
    income_level: Optional[str],
    lag_weeks: int,
    click_data: Optional[Dict[str, Any]],
) -> Tuple[Figure, Figure, Figure]:
    filtered_df: pd.DataFrame = df.copy()

    # lag health outcomes
    if lag_weeks > 0:
        health_cols: List[str] = [
            "respiratory_disease_rate",
            "cardio_mortality_rate",
            "vector_disease_risk_score",
            "waterborne_disease_incidents",
            "heat_related_admissions",
        ]
        filtered_df[health_cols] = filtered_df.groupby("country_name")[
            health_cols
        ].shift(-lag_weeks)
        filtered_df = filtered_df.dropna(subset=health_cols)

    if income_level:
        filtered_df = filtered_df[filtered_df["income_level"] == income_level]

    # aggregate by country
    country_df: pd.DataFrame = (
        filtered_df.groupby(["country_name", "country_code"])
        .mean(numeric_only=True)
        .reset_index()
    )

    fig_map: Figure = px.choropleth(
        country_df,
        locations="country_code",
        color="climate_vulnerability_index",
        hover_name="country_name",
        title="Global Climate Vulnerability Index",
        color_continuous_scale=px.colors.sequential.Turbo,
        projection="natural earth",
    )
    fig_map.update_layout(clickmode="event+select")

    selected_country_name: str = "Global"
    if click_data:
        selected_country_code: str = click_data["points"][0]["location"]
        country_row = country_df[country_df["country_code"] == selected_country_code]
        if not country_row.empty:
            selected_country_name = country_row["country_name"].iloc[0]
            filtered_df = filtered_df[
                filtered_df["country_name"] == selected_country_name
            ]

    # trend: temp vs respiratory disease
    trend_df: pd.DataFrame = (
        filtered_df.groupby("date").mean(numeric_only=True).reset_index()
    )

    fig_trend: Figure = px.line(
        trend_df,
        x="date",
        y=["temperature_celsius", "respiratory_disease_rate"],
        title=f"Temperature vs Respiratory Disease ({selected_country_name})",
        labels={"value": "Value", "variable": "Metric"},
    )

    # scatter: PM2.5 vs respiratory disease
    fig_scatter: Figure = px.scatter(
        filtered_df,
        x="pm25_ugm3",
        y="respiratory_disease_rate",
        color="income_level"
        if selected_country_name == "Global" and not income_level
        else None,
        title=f"PM2.5 vs Respiratory Disease ({selected_country_name})",
        opacity=0.5,
    )

    return fig_map, fig_trend, fig_scatter


if __name__ == "__main__":
    print("Starting Dash Server...")
    app.run(debug=True, port=8050)
