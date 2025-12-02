import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub
from typing import Optional, List, Dict, Any


def load_data_from_kaggle() -> pd.DataFrame:
    """
    Download dataset from Kaggle
    """
    print("Downloading dataset from Kaggle...")

    path: str = kagglehub.dataset_download(
        "sohumgokhale/global-climate-health-impact-tracker-2015-2025"
    )

    print(f"Dataset downloaded to cache: {path}")

    csv_file: Optional[str] = None
    for file in os.listdir(path):
        if file.endswith(".csv"):
            csv_file = os.path.join(path, file)
            break

    if not csv_file:
        print("ERROR: No CSV found in the downloaded files.")
        exit()

    print(f"Loading: {csv_file}")
    df: pd.DataFrame = pd.read_csv(csv_file)
    df["date"] = pd.to_datetime(df["date"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes outliers from *continuous* variables using IQR
    """
    numeric_df: pd.DataFrame = df.select_dtypes(include=[np.number])

    protected_columns: List[str] = [
        "record_id",
        "year",
        "month",
        "week",
        "heat_wave_days",
        "drought_indicator",
        "flood_indicator",
        "extreme_weather_events",
        "vector_disease_risk_score",
    ]

    cols_to_clean: List[str] = [
        col for col in numeric_df.columns if col not in protected_columns
    ]

    print(f"Cleaning outliers from {len(cols_to_clean)} continuous variables...")

    for col in cols_to_clean:
        q1: float = df[col].quantile(0.25)
        q3: float = df[col].quantile(0.75)
        iqr: float = q3 - q1
        lower_bound: float = q1 - 1.5 * iqr
        upper_bound: float = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


def generate_visualizations(df: pd.DataFrame) -> None:
    # 1. Correlation Matrix
    plt.figure(figsize=(12, 10))
    corr_cols: List[str] = [
        "temperature_celsius",
        "precipitation_mm",
        "pm25_ugm3",
        "respiratory_disease_rate",
        "cardio_mortality_rate",
        "heat_related_admissions",
        "mental_health_index",
        "gdp_per_capita_usd",
        "healthcare_access_index",
    ]
    available_cols: List[str] = [c for c in corr_cols if c in df.columns]

    corr_matrix: pd.DataFrame = df[available_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation: Climate Indicators vs Health Outcomes")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    print("Saved 'correlation_matrix.png'")
    plt.close()

    # 2. Scatterplot
    plt.figure(figsize=(10, 6))
    if "pm25_ugm3" in df.columns and "respiratory_disease_rate" in df.columns:
        if "income_level" in df.columns:
            sns.scatterplot(
                data=df,
                x="pm25_ugm3",
                y="respiratory_disease_rate",
                hue="income_level",
                alpha=0.5,
            )
        else:
            sns.scatterplot(
                data=df, x="pm25_ugm3", y="respiratory_disease_rate", alpha=0.5
            )

        plt.title("Impact of Air Quality (PM2.5) on Respiratory Disease")
        plt.tight_layout()
        plt.savefig("pm25_impact.png")
        print("Saved 'pm25_impact.png'")
    plt.close()


def print_summary_stats(df: pd.DataFrame) -> None:
    print("\n" + "=" * 66)
    print("SUMMARY STATS")
    print("=" * 66)
    check_cols: List[str] = [
        "heat_wave_days",
        "extreme_weather_events",
        "temperature_celsius",
    ]
    existing_check_cols: List[str] = [c for c in check_cols if c in df.columns]
    print(df[existing_check_cols].describe().to_string())


if __name__ == "__main__":
    df: pd.DataFrame = load_data_from_kaggle()
    df_clean: pd.DataFrame = clean_data(df)
    print_summary_stats(df_clean)
    generate_visualizations(df_clean)
