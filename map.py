import os
import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.express as px
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from plotly.graph_objs import Figure
try:
    from sklearn.linear_model import LinearRegression
except Exception:
    LinearRegression = None
import numpy as np
from datetime import datetime
import main

print("Loading data...")
df: pd.DataFrame = main.load_data_from_kaggle()
df = main.clean_data(df)

# choose the proper temperature column globally (make temp_col available outside the CVI block)
temp_col = (
    "temp_anomaly_celsius"
    if "temp_anomaly_celsius" in df.columns
    else ("temperature_celsius" if "temperature_celsius" in df.columns else None)
)

# compute climate vulnerability index if missing
if "climate_vulnerability_index" not in df.columns:
    print("Calculating Climate Vulnerability Index...")

    def normalize(series: pd.Series) -> pd.Series:
        return (series - series.min()) / (series.max() - series.min())

    # prefer anomaly if it's present; otherwise use base temperature if available

    components = []
    if "pm25_ugm3" in df.columns:
        components.append(normalize(df["pm25_ugm3"]) * 0.4)
    if temp_col is not None:
        components.append(normalize(df[temp_col]) * 0.3)
    if "extreme_weather_events" in df.columns:
        components.append(normalize(df["extreme_weather_events"]) * 0.3)

    if components:
        # Sum weighted components and rescale to 0-100
        combined = sum(components)
        df["climate_vulnerability_index"] = (combined - combined.min()) / (
            combined.max() - combined.min()
        ) * 100

# Train a simple predictive model for respiratory disease if possible
# We use a small linear regression to generate a 'predicted_respiratory_rate' field
model_features = [
    c
    for c in [
        "pm25_ugm3",
        "temperature_celsius",
        "temp_anomaly_celsius",
        "extreme_weather_events",
        "gdp_per_capita_usd",
        "healthcare_access_index",
    ]
    if c in df.columns
]

if "respiratory_disease_rate" in df.columns and model_features and LinearRegression is not None:
    train_df = df.dropna(subset=model_features + ["respiratory_disease_rate"])
    if not train_df.empty:
        try:
            lr = LinearRegression()
            lr.fit(train_df[model_features], train_df["respiratory_disease_rate"])
            medians = df[model_features].median()
            # Fill NA with medians for prediction
            df["predicted_respiratory_rate"] = lr.predict(df[model_features].fillna(medians))
        except Exception:
            # If training fails for any reason, fall back to mean-based prediction
            df["predicted_respiratory_rate"] = df["respiratory_disease_rate"].mean()
else:
    # fallback to mean if training isn't possible or sklearn missing, but ensure the column exists
    if "respiratory_disease_rate" in df.columns:
        df["predicted_respiratory_rate"] = df["respiratory_disease_rate"].mean()

# init app
app: dash.Dash = dash.Dash(__name__)
server = app.server

# Precompute UI option lists and date defaults (used in layout)
available_income_levels = (
    sorted(df["income_level"].dropna().unique()) if "income_level" in df.columns else []
)
# choose region column if present
region_col = None
if "region" in df.columns:
    region_col = "region"
elif "region_name" in df.columns:
    region_col = "region_name"

available_regions = sorted(df[region_col].dropna().unique()) if region_col else []
available_metrics = [
    c
    for c in [
        "climate_vulnerability_index",
        "pm25_ugm3",
        "vector_disease_risk_score",
        "temperature_celsius",
        "predicted_respiratory_rate",
        "respiratory_disease_rate",
    ]
    if c in df.columns
]
start_date_default = df["date"].min().date() if "date" in df.columns else None
end_date_default = df["date"].max().date() if "date" in df.columns else None

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
                            options=[{"label": i, "value": i} for i in available_income_levels],
                            value=None,
                            placeholder="Select Income Level (All)",
                        ),
                        html.Br(),
                        html.Label(f"Filter by {region_col.replace('_', ' ').title() if region_col else 'Region'}:"),
                        dcc.Dropdown(
                            id="region-filter",
                            options=[{"label": r, "value": r} for r in available_regions],
                            value=None,
                            placeholder="Select Region (All)",
                        ),
                        html.Br(),
                        html.Label("Metric to color map:"),
                        dcc.Dropdown(
                            id="map-metric",
                            options=[{"label": m.replace("_", " ").title(), "value": m} for m in available_metrics],
                            value=available_metrics[0] if available_metrics else "climate_vulnerability_index",
                            placeholder="Select Metric",
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block", "verticalAlign": "top"},
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
                        html.Br(),
                        html.Label("Select Date Range:"),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=df["date"].min().date() if "date" in df.columns else None,
                            max_date_allowed=df["date"].max().date() if "date" in df.columns else None,
                            start_date=start_date_default,
                            end_date=end_date_default,
                        ),
                        html.Br(),
                        html.Button("Reset Selection", id="reset-selection", n_clicks=0),
                    ],
                    style={"width": "48%", "float": "right", "display": "inline-block", "verticalAlign": "top"},
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
        Input("region-filter", "value"),
        Input("map-metric", "value"),
        Input("lag-slider", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("choropleth-map", "clickData"),
        Input("reset-selection", "n_clicks"),
    ],
)
def update_dashboard(
    income_level: Optional[str],
    region_value: Optional[str],
    metric_value: Optional[str],
    lag_weeks: int,
    start_date: Optional[str],
    end_date: Optional[str],
    click_data: Optional[Dict[str, Any]],
    reset_clicks: Optional[int],
) -> Tuple[Figure, Figure, Figure]:
    filtered_df: pd.DataFrame = df.copy()

    # apply date range filters
    if start_date:
        try:
            start_dt = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df["date"] >= start_dt]
        except Exception:
            # keep original filtered_df if parsing fails
            pass
    if end_date:
        try:
            end_dt = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df["date"] <= end_dt]
        except Exception:
            pass

    # region filter (optional)
    if region_value and region_col and region_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[region_col] == region_value]

    # clear click_data only when reset-selection actually triggered this callback
    triggered = callback_context.triggered
    if triggered and len(triggered) > 0 and "reset-selection" in triggered[0].get("prop_id", ""):
        click_data = None

    # income level filter (optional)
    if income_level:
        filtered_df = filtered_df[filtered_df["income_level"] == income_level]

    # ensure deterministic order for shifting
    if "country_name" in filtered_df.columns and "date" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(["country_name", "date"])

    # lag health outcomes (shift health metrics into earlier rows, to align with climate data)
    if lag_weeks > 0:
        health_cols: List[str] = [
            "respiratory_disease_rate",
            "cardio_mortality_rate",
            "vector_disease_risk_score",
            "waterborne_disease_incidents",
            "heat_related_admissions",
        ]
        health_cols = [c for c in health_cols if c in filtered_df.columns]
        if health_cols:
            filtered_df.loc[:, health_cols] = filtered_df.groupby("country_name")[
                health_cols
            ].shift(-lag_weeks)
            filtered_df = filtered_df.dropna(subset=health_cols)

    # aggregate by country
    if "country_name" in filtered_df.columns and "country_code" in filtered_df.columns:
        country_df: pd.DataFrame = (
            filtered_df.groupby(["country_name", "country_code"])
            .mean(numeric_only=True)
            .reset_index()
        )
    else:
        # fallback to at least having the original dataframe
        country_df: pd.DataFrame = filtered_df.copy()

    # determine which metric to display on the map
    default_metrics = [
        "climate_vulnerability_index",
        "pm25_ugm3",
        "vector_disease_risk_score",
        "temperature_celsius",
        "predicted_respiratory_rate",
        "respiratory_disease_rate",
    ]
    available_metrics_local: List[str] = [m for m in default_metrics if m in country_df.columns]
    metric_to_display: str = (
        metric_value if (metric_value in available_metrics_local) else (available_metrics_local[0] if available_metrics_local else "climate_vulnerability_index")
    )

    # build hover data list for the maps
    hover_cols = [c for c in ["pm25_ugm3", "respiratory_disease_rate", "predicted_respiratory_rate", "gdp_per_capita_usd", "healthcare_access_index"] if c in country_df.columns]

    fig_map: Figure = px.choropleth(
        country_df,
        locations="country_code",
        color=metric_to_display,
        hover_name="country_name",
        hover_data=hover_cols,
        title=f"Global {metric_to_display.replace('_', ' ').title()}",
        color_continuous_scale=px.colors.sequential.Turbo,
        projection="natural earth",
        labels={metric_to_display: metric_to_display.replace("_", " ").title()},
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

    trend_y_cols = [c for c in ["temperature_celsius", "respiratory_disease_rate", "predicted_respiratory_rate"] if c in trend_df.columns]
    if not trend_y_cols:
        # fallback to numeric columns present
        trend_y_cols = [c for c in trend_df.columns if c not in ["date", "country_name", "country_code"]][:2]

    fig_trend: Figure = px.line(
        trend_df,
        x="date",
        y=trend_y_cols,
        title=f"Trend: {', '.join([c.replace('_', ' ').title() for c in trend_y_cols])} ({selected_country_name})",
        labels={"value": "Value", "variable": "Metric"},
    )

    # scatter: PM2.5 vs respiratory disease
    scatter_color = (
        "income_level"
        if ("income_level" in filtered_df.columns and selected_country_name == "Global" and not income_level)
        else None
    )
    fig_scatter: Figure = px.scatter(
        filtered_df,
        x="pm25_ugm3",
        y="respiratory_disease_rate",
        color=scatter_color,
        title=f"PM2.5 vs Respiratory Disease ({selected_country_name})",
        opacity=0.5,
        hover_data=[c for c in ["country_name", region_col, "predicted_respiratory_rate"] if c and c in filtered_df.columns],
    )

    return fig_map, fig_trend, fig_scatter


if __name__ == "__main__":
    port: int = int(os.environ.get("PORT", 8050))
    print(f"Starting Dash Server on port {port}...")
    app.run(debug=False, host="0.0.0.0", port=port)
