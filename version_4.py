# ==============================================================
# Stream Water-Level Prediction Using Daily Precipitation
# Predicts the Next 14 Days
#
# Input CSV structure:
# Column A: date
# Column B: wl in m
# Column C: precip in mm
#
# Author: Babak / ChatGPT
# ==============================================================

from pathlib import Path
import html
import json
import warnings
from urllib.parse import quote, urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================================================
# 1. USER SETTINGS
# ==============================================================

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "input.csv"

OUTPUT_DIR = PROJECT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASIN_IMAGE_FILE = PROJECT_DIR / "huron_river_basin.jpg"


def versioned_output_file(filename):
    """
    Returns the output path for files created by the model.
    """

    return OUTPUT_DIR / filename

FORECAST_DAYS = 14
FORECAST_START = "today"  # Use "today" or "tomorrow"
STAGE_PLOT_START_DATE = "2025-01-01"
OPEN_METEO_PAST_DAYS = 14
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEOUT_SECONDS = 60

BASIN_FORECAST_POINTS = [
    (41.2923, -82.6158),
    (41.3031, -82.6845),
    (41.2628, -82.6769),
    (41.2613, -82.7415),
    (41.2484, -82.7284),
    (41.2277, -82.6934),
    (41.1745, -82.8074),
    (41.1337, -82.8445),
    (41.0881, -82.8060),
    (41.2706, -82.5980),
    (41.2277, -82.6591),
    (41.1781, -82.6556),
    (40.9675, -82.7181),
    (41.1042, -82.5492),
    (41.0602, -82.7339),
    (41.1600, -82.5547),
    (41.0804, -82.6659),
    (41.1668, -82.6344),
]

# Feature settings
MAX_PRECIP_LAG = 14      # precipitation memory, days
MAX_WL_LAG = 7           # water-level memory, days
ROLLING_WINDOWS = [3, 7, 14, 30]

# Chronological split
TEST_FRACTION = 0.20

RANDOM_STATE = 42

# Water-level stage thresholds, meters
ACTION_STAGE_M = 4.2672
MINOR_FLOOD_STAGE_M = 5.1816
MODERATE_FLOOD_STAGE_M = 5.6388
MAJOR_FLOOD_STAGE_M = 6.4008

FORECAST_SCENARIOS = [
    {
        "name": "10% runoff reduction",
        "column_prefix": "runoff_reduction_10pct",
        "kind": "multiply",
        "value": 0.90,
        "plot_color": "#2e7d32",
        "plot_style": "--",
    },
    {
        "name": "20% runoff reduction",
        "column_prefix": "runoff_reduction_20pct",
        "kind": "multiply",
        "value": 0.80,
        "plot_color": "#00695c",
        "plot_style": "-.",
    },
    {
        "name": "2.5 mm storage",
        "column_prefix": "storage_2_5mm",
        "kind": "storage",
        "value": 2.5,
        "plot_color": "#6a1b9a",
        "plot_style": "--",
    },
    {
        "name": "5 mm storage",
        "column_prefix": "storage_5mm",
        "kind": "storage",
        "value": 5.0,
        "plot_color": "#ef6c00",
        "plot_style": "-.",
    },
]


# ==============================================================
# 2. HELPER FUNCTIONS
# ==============================================================

def nash_sutcliffe_efficiency(obs, pred):
    """
    Nash-Sutcliffe Efficiency.
    NSE = 1 is perfect.
    NSE = 0 means model is only as good as the observed mean.
    NSE < 0 means model is worse than using the observed mean.
    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)

    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((obs - pred) ** 2)

    if denominator == 0:
        return np.nan

    return 1 - numerator / denominator


def clean_input_data(input_file):
    """
    Reads the CSV and standardizes columns:
    date, wl_m, precip_mm
    """

    df = pd.read_csv(input_file)

    if df.shape[1] < 3:
        raise ValueError(
            "The input CSV must have at least three columns: date, water level, precipitation."
        )

    # Use first three columns regardless of their original names
    df = df.iloc[:, :3].copy()
    df.columns = ["date", "wl_m", "precip_mm"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["wl_m"] = pd.to_numeric(df["wl_m"], errors="coerce")
    df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce")

    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    # If duplicate dates exist, average water level and sum precipitation
    df = (
        df.groupby("date", as_index=False)
        .agg({
            "wl_m": "mean",
            "precip_mm": "sum"
        })
    )

    # Create continuous daily date range
    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_dates)
    df.index.name = "date"
    df = df.reset_index()

    # Fill missing precipitation with 0.
    # If missing precipitation means "missing data" rather than "no rain",
    # you may want to change this later.
    df["precip_mm"] = df["precip_mm"].fillna(0)

    return df


def add_features(df):
    """
    Creates hydrologically meaningful predictors.

    Target variable:
    wl_m on the current day.

    Predictors include:
    - precipitation on current and previous days
    - rolling precipitation totals
    - previous water levels
    - rolling previous water level means
    - seasonal sine/cosine variables
    """

    df = df.copy()

    # Calendar features
    df["dayofyear"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month

    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    # Precipitation lag features
    # precip_lag_0 = precipitation on the prediction day
    for lag in range(0, MAX_PRECIP_LAG + 1):
        df[f"precip_lag_{lag}"] = df["precip_mm"].shift(lag)

    # Rolling precipitation totals including current day
    for window in ROLLING_WINDOWS:
        df[f"precip_sum_{window}d"] = (
            df["precip_mm"]
            .rolling(window=window, min_periods=window)
            .sum()
        )

    # Water-level lag features
    # wl_lag_1 = yesterday's water level
    # We do NOT use today's water level to predict today's water level.
    for lag in range(1, MAX_WL_LAG + 1):
        df[f"wl_lag_{lag}"] = df["wl_m"].shift(lag)

    # Rolling previous water-level means
    for window in [3, 7, 14]:
        df[f"wl_mean_previous_{window}d"] = (
            df["wl_m"]
            .shift(1)
            .rolling(window=window, min_periods=window)
            .mean()
        )

    # Optional: water-level change from yesterday to day before yesterday
    df["wl_change_1d"] = df["wl_lag_1"] - df["wl_lag_2"]

    return df


def get_feature_columns(df):
    """
    Returns all feature columns used by the model.
    """

    feature_cols = []

    feature_cols += [f"precip_lag_{lag}" for lag in range(0, MAX_PRECIP_LAG + 1)]
    feature_cols += [f"precip_sum_{window}d" for window in ROLLING_WINDOWS]
    feature_cols += [f"wl_lag_{lag}" for lag in range(1, MAX_WL_LAG + 1)]
    feature_cols += [f"wl_mean_previous_{window}d" for window in [3, 7, 14]]

    feature_cols += [
        "wl_change_1d",
        "sin_doy",
        "cos_doy",
        "month",
    ]

    return feature_cols


def train_test_split_time_series(model_df, test_fraction):
    """
    Chronological split.
    No random splitting for time-series hydrology data.
    """

    n = len(model_df)
    test_size = int(np.ceil(n * test_fraction))
    train_size = n - test_size

    train_df = model_df.iloc[:train_size].copy()
    test_df = model_df.iloc[train_size:].copy()

    return train_df, test_df


def evaluate_model(obs, pred):
    """
    Calculates model performance metrics.
    """

    mae = mean_absolute_error(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    r2 = r2_score(obs, pred)
    nse = nash_sutcliffe_efficiency(obs, pred)

    return {
        "MAE_m": mae,
        "RMSE_m": rmse,
        "R2": r2,
        "NSE": nse
    }


def classify_water_level_stage(water_level_m):
    """
    Converts a predicted water level into the requested flood-stage category.
    """

    if water_level_m < ACTION_STAGE_M:
        return "Action stage"
    if water_level_m < MINOR_FLOOD_STAGE_M:
        return "Minor flood stage"
    if water_level_m < MODERATE_FLOOD_STAGE_M:
        return "Moderate flood stage"
    if water_level_m < MAJOR_FLOOD_STAGE_M:
        return "Major flood stage"

    return "Above major flood stage"


def plot_observed_vs_predicted(test_df, output_file):
    """
    Creates time-series plot for observed vs predicted water level.
    """

    fig, ax = plt.subplots(figsize=(14, 6))
    ax2 = ax.twinx()
    ax2.bar(
        test_df["date"],
        test_df["precip_mm"],
        color="lightskyblue",
        alpha=0.45,
        label="Precipitation",
        width=0.8
    )
    ax.plot(test_df["date"], test_df["wl_m"], label="Observed", linewidth=2)
    ax.plot(test_df["date"], test_df["predicted_wl_m"], label="Predicted", linewidth=2)

    ax.set_zorder(2)
    ax2.set_zorder(1)
    ax.patch.set_alpha(0)
    ax.set_xlabel("Date")
    ax.set_ylabel("Water-level in Huron River (m)")
    ax2.set_ylabel("Precipitation (mm)")
    ax2.yaxis.label.set_color("#111111")
    ax2.tick_params(axis="y", colors="#111111")
    ax.set_title("Observed vs Predicted Stream Water Level")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(handles_1 + handles_2, labels_1 + labels_2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close()


def plot_stage_forecast_summary(
    historical_df,
    test_df,
    forecast_df,
    output_file,
    scenario_forecasts=None
):
    """
    Creates a USGS-style stage plot with observed, estimated, and forecast water levels.
    """

    plot_start = pd.Timestamp(STAGE_PLOT_START_DATE)

    observed_df = historical_df[historical_df["date"] >= plot_start].copy()
    estimated_df = test_df[test_df["date"] >= plot_start].copy()
    forecast_plot_df = forecast_df.copy()
    scenario_forecasts = scenario_forecasts or []

    scenario_max_values = [
        scenario["forecast_df"]["predicted_wl_m"].max()
        for scenario in scenario_forecasts
    ]

    series_max = max(
        observed_df["wl_m"].max(),
        estimated_df["predicted_wl_m"].max(),
        forecast_plot_df["predicted_wl_m"].max(),
        *scenario_max_values,
        MAJOR_FLOOD_STAGE_M,
    )
    y_min = max(0, min(observed_df["wl_m"].min(), forecast_plot_df["predicted_wl_m"].min()) - 0.3)
    y_max = series_max + 0.5

    fig, ax = plt.subplots(figsize=(16, 8))

    stage_bands = [
        (y_min, ACTION_STAGE_M, "#ffffff", "Action stage"),
        (ACTION_STAGE_M, MINOR_FLOOD_STAGE_M, "#fff9e8", "Minor flood stage"),
        (MINOR_FLOOD_STAGE_M, MODERATE_FLOOD_STAGE_M, "#fde8e6", "Moderate flood stage"),
        (MODERATE_FLOOD_STAGE_M, MAJOR_FLOOD_STAGE_M, "#f7eafa", "Major flood stage"),
        (MAJOR_FLOOD_STAGE_M, y_max, "#ead3f5", "Above major flood stage"),
    ]

    for lower, upper, color, _ in stage_bands:
        ax.axhspan(lower, upper, color=color, zorder=0)

    stage_lines = [
        (ACTION_STAGE_M, "#f6d84a", "Action stage"),
        (MINOR_FLOOD_STAGE_M, "#f39c12", "Minor flood stage"),
        (MODERATE_FLOOD_STAGE_M, "#ff2f20", "Moderate flood stage"),
        (MAJOR_FLOOD_STAGE_M, "#d05cff", "Major flood stage"),
    ]

    for threshold, color, label in stage_lines:
        ax.axhline(threshold, color=color, linewidth=2, zorder=1)
        ax.text(
            0.01,
            threshold + 0.03,
            f"{threshold:.2f} m - {label}",
            transform=ax.get_yaxis_transform(),
            color="#111111",
            fontsize=10,
            va="bottom",
        )

    ax.plot(
        observed_df["date"],
        observed_df["wl_m"],
        color="#173a8a",
        linewidth=2.0,
        label="Observed",
        zorder=3,
    )
    ax.plot(
        estimated_df["date"],
        estimated_df["predicted_wl_m"],
        color="#8b1e1e",
        linestyle="--",
        linewidth=2.0,
        label="Estimated",
        zorder=4,
    )
    ax.plot(
        forecast_plot_df["date"],
        forecast_plot_df["predicted_wl_m"],
        color="#0b5d1e",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Forecast water-level",
        zorder=5,
    )

    for scenario in scenario_forecasts:
        scenario_df = scenario["forecast_df"]
        ax.plot(
            scenario_df["date"],
            scenario_df["predicted_wl_m"],
            color=scenario["plot_color"],
            linestyle=scenario["plot_style"],
            linewidth=2.0,
            marker="o",
            markersize=3,
            label=scenario["name"],
            zorder=5,
        )

    ax.axvline(
        forecast_plot_df["date"].min(),
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )

    ax.set_title("Observed, Estimated, and Forecast Stream Water Level", fontsize=16, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Water-level in Huron River (m)")
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, color="#d7d7d7", linewidth=0.8, alpha=0.8)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def plot_scatter_observed_vs_predicted(test_df, output_file):
    """
    Creates scatter plot for observed vs predicted water level.
    """

    obs = test_df["wl_m"]
    pred = test_df["predicted_wl_m"]

    min_val = min(obs.min(), pred.min())
    max_val = max(obs.max(), pred.max())

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(obs, pred, alpha=0.6)
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    ax.set_xlabel("Observed water level (m)")
    ax.set_ylabel("Predicted water level (m)")
    ax.set_title("Observed vs Predicted Water Level")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close()


def display_table_column_name(column):
    """
    Converts output dataframe column names into dashboard table labels.
    """

    labels = {
        "date": "Date",
        "forecast_day": "Forecast Day",
        "forecast_precip_mm": "Forecast Precip (mm)",
        "predicted_wl_m": "Predicted WL (m)",
        "water_level_stage": "Water Level Stage",
        "runoff_reduction_10pct_predicted_wl_m": "Runoff Reduction 10% Predicted WL (m)",
        "runoff_reduction_10pct_water_level_stage": "Runoff Reduction 10% Water Level Stage",
        "runoff_reduction_20pct_predicted_wl_m": "Runoff Reduction 20% Predicted WL (m)",
        "runoff_reduction_20pct_water_level_stage": "Runoff Reduction 20% Water Level Stage",
        "storage_2_5mm_predicted_wl_m": "Storage 2.5 mm Predicted WL (m)",
        "storage_2_5mm_water_level_stage": "Storage 2.5 mm Water Level Stage",
        "storage_5mm_predicted_wl_m": "Storage 5 mm Predicted WL (m)",
        "storage_5mm_water_level_stage": "Storage 5 mm Water Level Stage",
    }

    return labels.get(column, column.replace("_", " ").title())


def round_numeric_columns(df, decimals=2):
    """
    Rounds numeric output columns while preserving dates and text categories.
    """

    rounded_df = df.copy()
    numeric_columns = rounded_df.select_dtypes(include=[np.number]).columns
    rounded_df[numeric_columns] = rounded_df[numeric_columns].round(decimals)
    return rounded_df


def dataframe_to_html_table(df, columns):
    """
    Builds an HTML table body from selected dataframe columns.
    """

    header_cells = "".join(
        f"<th>{html.escape(display_table_column_name(column))}</th>"
        for column in columns
    )
    rows = []

    for _, row in df.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                display_value = ""
            elif isinstance(value, pd.Timestamp):
                display_value = value.strftime("%Y-%m-%d")
            elif "date" in column:
                display_value = pd.Timestamp(value).strftime("%Y-%m-%d")
            elif isinstance(value, (float, np.floating)):
                display_value = f"{value:.2f}"
            else:
                display_value = str(value)
            cells.append(f"<td>{html.escape(display_value)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"<thead><tr>{header_cells}</tr></thead><tbody>{''.join(rows)}</tbody>"


def series_for_interactive_chart(name, dates, values, color, dash="solid"):
    """
    Creates a Plotly trace dictionary with consistent hover text.
    """

    return {
        "type": "scatter",
        "mode": "lines+markers",
        "name": name,
        "x": [pd.Timestamp(date).strftime("%Y-%m-%d") for date in dates],
        "y": [None if pd.isna(value) else round(float(value), 2) for value in values],
        "line": {
            "color": color,
            "width": 2.5,
            "dash": dash,
        },
        "marker": {
            "size": 5,
        },
        "hovertemplate": (
            "<b>%{fullData.name}</b><br>"
            "Date: %{x}<br>"
            "Water level: %{y:.2f} m"
            "<extra></extra>"
        ),
    }


def long_date_label(date):
    """
    Formats dates for dashboard axes without a leading zero on the day.
    """

    timestamp = pd.Timestamp(date)
    return f"{timestamp.strftime('%B')} {timestamp.day}, {timestamp.year}"


def write_interactive_index(
    historical_df,
    test_df,
    forecast_df,
    forecast_output_df,
    scenario_forecasts,
    metrics,
    output_file
):
    """
    Writes an interactive HTML dashboard with hoverable chart and forecast table.
    """

    plot_start = pd.Timestamp(STAGE_PLOT_START_DATE)
    observed_df = historical_df[historical_df["date"] >= plot_start].copy()
    estimated_df = test_df[test_df["date"] >= plot_start].copy()

    model_test_traces = [
        {
            "type": "bar",
            "name": "Precipitation",
            "x": [pd.Timestamp(date).strftime("%Y-%m-%d") for date in observed_df["date"]],
            "y": [None if pd.isna(value) else round(float(value), 2) for value in observed_df["precip_mm"]],
            "yaxis": "y2",
            "marker": {"color": "rgba(135, 206, 250, 0.58)"},
            "opacity": 0.58,
            "hovertemplate": (
                "<b>Precipitation</b><br>"
                "Date: %{x}<br>"
                "Precipitation: %{y:.2f} mm"
                "<extra></extra>"
            ),
        },
        series_for_interactive_chart(
            "Observed",
            observed_df["date"],
            observed_df["wl_m"],
            "#173a8a",
            "solid"
        ),
        series_for_interactive_chart(
            "Estimated",
            estimated_df["date"],
            estimated_df["predicted_wl_m"],
            "#8b1e1e",
            "dash"
        ),
    ]

    forecast_traces = [
        {
            "type": "bar",
            "name": "Forecast precipitation",
            "x": [pd.Timestamp(date).strftime("%Y-%m-%d") for date in forecast_df["date"]],
            "y": [None if pd.isna(value) else round(float(value), 2) for value in forecast_df["forecast_precip_mm"]],
            "yaxis": "y2",
            "marker": {"color": "rgba(135, 206, 250, 0.58)"},
            "opacity": 0.58,
            "hovertemplate": (
                "<b>Forecast precipitation</b><br>"
                "Date: %{x}<br>"
                "Precipitation: %{y:.2f} mm"
                "<extra></extra>"
            ),
        },
        series_for_interactive_chart(
            "Forecast water-level",
            forecast_df["date"],
            forecast_df["predicted_wl_m"],
            "#0b5d1e",
            "solid"
        ),
    ]

    for scenario in scenario_forecasts:
        forecast_traces.append(
            series_for_interactive_chart(
                scenario["name"],
                scenario["forecast_df"]["date"],
                scenario["forecast_df"]["predicted_wl_m"],
                scenario["plot_color"],
                "dashdot" if scenario["plot_style"] == "-." else "dash"
            )
        )

    y_values = []
    for trace in forecast_traces + model_test_traces:
        if trace.get("yaxis") == "y2":
            continue
        y_values.extend(value for value in trace["y"] if value is not None)

    y_min = max(0, min(y_values) - 0.3)
    y_max = max(max(y_values), MAJOR_FLOOD_STAGE_M) + 0.5
    precipitation_values = forecast_df["forecast_precip_mm"].dropna()
    precipitation_y_max = max(1.0, float(precipitation_values.max()) * 1.2) if not precipitation_values.empty else 1.0
    model_precipitation_values = observed_df["precip_mm"].dropna()
    model_precipitation_y_max = (
        max(1.0, float(model_precipitation_values.max()) * 1.2)
        if not model_precipitation_values.empty
        else 1.0
    )
    forecast_tick_dates = [
        pd.Timestamp(date).strftime("%Y-%m-%d")
        for date in forecast_df["date"]
    ]
    forecast_tick_labels = [
        long_date_label(date)
        for date in forecast_df["date"]
    ]
    forecast_x_range = [
        pd.Timestamp(forecast_df["date"].min()).strftime("%Y-%m-%d"),
        pd.Timestamp(forecast_df["date"].max()).strftime("%Y-%m-%d"),
    ]
    model_test_x_range = [
        min(observed_df["date"].min(), estimated_df["date"].min()).strftime("%Y-%m-%d"),
        max(observed_df["date"].max(), estimated_df["date"].max()).strftime("%Y-%m-%d"),
    ]
    model_test_months = pd.date_range(
        start=min(observed_df["date"].min(), estimated_df["date"].min()).to_period("M").to_timestamp(),
        end=max(observed_df["date"].max(), estimated_df["date"].max()).to_period("M").to_timestamp(),
        freq="MS",
    )
    model_test_tick_dates = [
        pd.Timestamp(date).strftime("%Y-%m-%d")
        for date in model_test_months
    ]
    model_test_tick_labels = [
        pd.Timestamp(date).strftime("%b %Y")
        for date in model_test_months
    ]

    stage_shapes = [
        {
            "type": "rect",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": y_min,
            "y1": ACTION_STAGE_M,
            "fillcolor": "#ffffff",
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": ACTION_STAGE_M,
            "y1": MINOR_FLOOD_STAGE_M,
            "fillcolor": "#fff9e8",
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": MINOR_FLOOD_STAGE_M,
            "y1": MODERATE_FLOOD_STAGE_M,
            "fillcolor": "#fde8e6",
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": MODERATE_FLOOD_STAGE_M,
            "y1": MAJOR_FLOOD_STAGE_M,
            "fillcolor": "#f7eafa",
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": MAJOR_FLOOD_STAGE_M,
            "y1": y_max,
            "fillcolor": "#ead3f5",
            "line": {"width": 0},
            "layer": "below",
        },
    ]

    stage_lines = [
        (ACTION_STAGE_M, "#f6d84a", "Action stage"),
        (MINOR_FLOOD_STAGE_M, "#f39c12", "Minor flood stage"),
        (MODERATE_FLOOD_STAGE_M, "#ff2f20", "Moderate flood stage"),
        (MAJOR_FLOOD_STAGE_M, "#d05cff", "Major flood stage"),
    ]

    for threshold, color, _ in stage_lines:
        stage_shapes.append({
            "type": "line",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": threshold,
            "y1": threshold,
            "line": {"color": color, "width": 2},
        })

    stage_annotations = [
        {
            "xref": "paper",
            "x": 0.01,
            "yref": "y",
            "y": threshold + 0.03,
            "text": f"{threshold:.2f} m - {label}",
            "showarrow": False,
            "xanchor": "left",
            "font": {"size": 12, "color": "#1f2933"},
            "bgcolor": "rgba(255,255,255,0.55)",
        }
        for threshold, _, label in stage_lines
    ]

    base_chart_layout = {
        "margin": {"l": 92, "r": 92, "t": 72, "b": 96},
        "height": 620,
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "hovermode": "x unified",
        "legend": {
            "orientation": "h",
            "x": 0,
            "y": 1.04,
            "xanchor": "left",
            "yanchor": "bottom",
            "font": {"size": 15},
        },
        "xaxis": {
            "title": "",
            "type": "date",
            "tickformat": "%B %-d, %Y",
            "automargin": True,
            "tickfont": {"size": 15, "color": "#111111"},
            "showgrid": True,
            "gridcolor": "#d7dee8",
            "showline": True,
            "linecolor": "#000000",
            "linewidth": 2,
            "mirror": True,
            "zeroline": False,
        },
        "yaxis": {
            "title": {
                "text": "Water-level in Huron River (m)",
                "font": {"size": 19, "color": "#111111"},
            },
            "range": [y_min, y_max],
            "automargin": True,
            "tickfont": {"size": 15, "color": "#111111"},
            "tickformat": ".2f",
            "showgrid": True,
            "gridcolor": "#d7dee8",
            "showline": True,
            "linecolor": "#000000",
            "linewidth": 2,
            "mirror": True,
            "zeroline": False,
        },
        "yaxis2": {
            "title": {
                "text": "Forecast precipitation (mm)",
                "font": {"size": 17, "color": "#111111"},
            },
            "range": [0, precipitation_y_max],
            "overlaying": "y",
            "side": "right",
            "automargin": True,
            "tickfont": {"size": 14, "color": "#111111"},
            "tickformat": ".2f",
            "showgrid": False,
            "showline": True,
            "linecolor": "#000000",
            "linewidth": 2,
            "zeroline": False,
        },
        "shapes": stage_shapes,
        "annotations": stage_annotations,
    }

    forecast_chart_layout = {
        **base_chart_layout,
        "xaxis": {
            **base_chart_layout["xaxis"],
            "range": forecast_x_range,
            "tickmode": "array",
            "tickvals": forecast_tick_dates,
            "ticktext": forecast_tick_labels,
            "tickangle": -35,
        },
    }

    model_test_chart_layout = {
        **base_chart_layout,
        "yaxis2": {
            **base_chart_layout["yaxis2"],
            "title": {
                "text": "Precipitation (mm)",
                "font": {"size": 17, "color": "#111111"},
            },
            "range": [0, model_precipitation_y_max],
        },
        "xaxis": {
            **base_chart_layout["xaxis"],
            "range": model_test_x_range,
            "tickmode": "array",
            "tickvals": model_test_tick_dates,
            "ticktext": model_test_tick_labels,
            "tickangle": -45,
        },
    }

    table_columns = forecast_output_df.columns.tolist()
    forecast_table_html = dataframe_to_html_table(forecast_output_df, table_columns)

    forecast_start = pd.Timestamp(forecast_df["date"].min()).strftime("%Y-%m-%d")
    forecast_end = pd.Timestamp(forecast_df["date"].max()).strftime("%Y-%m-%d")
    current_timestamp = pd.Timestamp.now(tz="America/New_York")
    today_date = current_timestamp.strftime("%Y-%m-%d")
    updated_at = current_timestamp.strftime("%Y-%m-%d %I:%M %p %Z")
    basin_image_html = ""
    if BASIN_IMAGE_FILE.exists():
        basin_image_src = quote(BASIN_IMAGE_FILE.name)
        basin_image_html = f"""
        <aside class="basin-card" aria-label="Huron River watershed map">
          <h2>Huron River Watershed, Western Lake Erie Basin, Ohio</h2>
          <img src="{basin_image_src}" alt="Huron River Watershed">
        </aside>
        """

    document = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Stream Water-Level Forecast Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
      :root {{
        --ink: #1f2933;
        --muted: #64748b;
        --line: #d7dee8;
        --panel: #ffffff;
        --surface: #f3f7fb;
        --brand: #1f4e79;
        --brand-2: #2d7a78;
        --shadow: 0 18px 45px rgba(31, 78, 121, 0.12);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        min-width: 320px;
        color: var(--ink);
        background: linear-gradient(180deg, rgba(31, 78, 121, 0.10), rgba(45, 122, 120, 0.04) 320px), var(--surface);
        font-family: Arial, Helvetica, sans-serif;
      }}
      .dashboard {{
        width: min(1900px, calc(100% - 32px));
        margin: 0 auto;
        padding: 28px 0 44px;
      }}
      .topbar {{
        display: block;
        margin-bottom: 20px;
      }}
      .overview-block {{
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 20px;
      }}
      h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
      h1 {{ max-width: 980px; font-size: 34px; line-height: 1.12; }}
      h2 {{ font-size: 20px; line-height: 1.25; }}
      h3 {{ margin: 22px 0 10px; color: var(--brand); font-size: 16px; }}
      .eyebrow {{
        margin: 0 0 6px;
        color: var(--brand-2);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
      }}
      .status-strip {{
        display: grid;
        grid-template-columns: repeat(3, minmax(120px, 1fr));
        gap: 8px;
      }}
      .status-strip span {{
        padding: 11px 12px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.82);
        border-radius: 8px;
        color: var(--muted);
        font-size: 13px;
      }}
      .status-strip b {{
        display: block;
        color: var(--ink);
        font-size: 15px;
        margin-bottom: 3px;
      }}
      .basin-card {{
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #fff;
        box-shadow: var(--shadow);
        overflow: hidden;
      }}
      .basin-card h2 {{
        padding: 14px 16px;
        border-bottom: 1px solid var(--line);
        color: var(--brand);
        font-size: 18px;
      }}
      .basin-card img {{
        display: block;
        width: 100%;
        height: 620px;
        object-fit: cover;
        background: #f8fbfd;
      }}
      .preview-panel {{
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel);
        box-shadow: var(--shadow);
        overflow: hidden;
      }}
      .preview-toolbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        padding: 18px 20px;
        border-bottom: 1px solid var(--line);
      }}
      .email-preview {{ padding: 20px; }}
      .forecast-layout {{
        display: grid;
        grid-template-columns: minmax(260px, 340px) minmax(0, 1fr) minmax(280px, 380px);
        align-items: stretch;
        gap: 18px;
      }}
      .model-info-card {{
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #fff;
        padding: 16px;
        box-shadow: var(--shadow);
        color: var(--ink);
        font-size: 14px;
        line-height: 1.45;
      }}
      .model-info-card p {{
        margin: 0 0 12px;
      }}
      .model-info-card h4 {{
        margin: 16px 0 7px;
        color: var(--brand);
        font-size: 14px;
      }}
      .model-info-card ul {{
        margin: 0 0 12px 18px;
        padding: 0;
      }}
      .model-info-card li {{
        margin: 3px 0;
      }}
      .chart-card {{
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #fff;
        padding: 16px;
      }}
      #forecastChart,
      #modelTestChart {{ width: 100%; min-height: 620px; }}
      .forecast-insights {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 16px 0 8px;
      }}
      .insight-card {{
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px;
        background: #f8fbfd;
      }}
      .insight-card h3 {{
        margin: 0;
        color: var(--ink);
        font-size: 18px;
      }}
      .insight-card p:last-child {{
        margin: 6px 0 0;
        color: var(--muted);
        font-size: 13px;
      }}
      .table-scroll {{
        overflow-x: auto;
        border: 1px solid var(--line);
        border-radius: 8px;
      }}
      .data-table {{
        width: 100%;
        border-collapse: collapse;
        min-width: 1500px;
        background: #fff;
      }}
      .data-table th,
      .data-table td {{
        padding: 10px 11px;
        border-bottom: 1px solid var(--line);
        border-right: 1px solid #edf1f6;
        text-align: left;
        white-space: nowrap;
        font-size: 13px;
      }}
      .data-table th {{
        position: sticky;
        top: 0;
        z-index: 1;
        color: #fff;
        background: var(--brand);
      }}
      .data-table tr:nth-child(even) td {{ background: #f8fbfd; }}
      @media (max-width: 900px) {{
        .status-strip {{
          grid-template-columns: 1fr;
          margin-top: 16px;
        }}
        .forecast-layout {{ grid-template-columns: 1fr; }}
        .basin-card img {{ height: auto; }}
        .forecast-insights {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <main class="dashboard">
      <section class="topbar" aria-label="Dashboard overview">
        <div class="overview-block">
          <div>
            <p class="eyebrow">Flood Prediction</p>
            <h1>14-Day Flood Prediction for Huron River, Ohio</h1>
          </div>
          <div class="status-strip" aria-label="Forecast status">
            <span><b>Today's Date</b>{html.escape(today_date)}</span>
            <span><b>Updated</b>{html.escape(updated_at)}</span>
            <span><b>Forecast Window</b>{html.escape(forecast_start)} to {html.escape(forecast_end)}</span>
          </div>
        </div>
      </section>

      <section class="preview-panel" aria-label="Interactive forecast preview">
        <div class="preview-toolbar">
          <div>
            <p class="eyebrow">Interactive Dashboard</p>
          </div>
        </div>

        <article class="email-preview">
          <h3>14-Day Forecast and Runoff/Storage Scenarios</h3>
          <div class="forecast-layout">
            <aside class="model-info-card" aria-label="Model information">
              <p>14-Day flood prediction model predicting daily stream water-level</p>
              <p>Random Forest machine learning algorithm</p>
              <p>Calibrated using ~18.6 years of historical daily data (2007-2026)</p>
              <p>Uses hydrologic lag, rolling precipitation, antecedent wetness, and seasonal features</p>
              <p>Future precipitation obtained from basin-average Open-Meteo forecasts</p>

              <h4>Flood-stage classification</h4>
              <ul>
                <li>Action</li>
                <li>Minor flood</li>
                <li>Moderate flood</li>
                <li>Major flood</li>
              </ul>

              <h4>Preventative-action scenario testing</h4>
              <ul>
                <li>10% runoff reduction</li>
                <li>20% runoff reduction</li>
                <li>2.5 mm storage</li>
                <li>5 mm storage</li>
              </ul>

              <p>Runoff reduction and increased storage could be achieved through detention basins, wetlands, floodplain reconnection, two-stage ditches, controlled drainage, saturated buffers, cover crops, conservation tillage, grassed waterways, and urban green infrastructure.</p>
            </aside>
            <section class="chart-card">
              <div id="forecastChart" aria-label="Interactive water-level forecast chart"></div>
            </section>
            {basin_image_html}
          </div>

          <h3>Forecasted Water-Level (WL) in the Stream and Flood Stage Classification</h3>
          <div class="table-scroll">
            <table class="data-table">
              {forecast_table_html}
            </table>
          </div>

          <h3>Observed and Estimated Water Levels During Model Testing (2025-2026)</h3>
          <section class="chart-card">
            <div id="modelTestChart" aria-label="Interactive model test chart"></div>
          </section>

          <section class="forecast-insights" aria-label="Model summary">
            <article class="insight-card">
              <p class="eyebrow">MAE</p>
              <h3>{metrics["MAE_m"]:.2f} m</h3>
              <p>Testing period error</p>
            </article>
            <article class="insight-card">
              <p class="eyebrow">RMSE</p>
              <h3>{metrics["RMSE_m"]:.2f} m</h3>
              <p>Testing period error</p>
            </article>
            <article class="insight-card">
              <p class="eyebrow">R2</p>
              <h3>{metrics["R2"]:.2f}</h3>
              <p>Testing period score</p>
            </article>
            <article class="insight-card">
              <p class="eyebrow">NSE</p>
              <h3>{metrics["NSE"]:.2f}</h3>
              <p>Hydrologic efficiency</p>
            </article>
          </section>
        </article>
      </section>
    </main>

    <script>
      const forecastChartData = {json.dumps(forecast_traces)};
      const forecastChartLayout = {json.dumps(forecast_chart_layout)};
      const modelTestChartData = {json.dumps(model_test_traces)};
      const modelTestChartLayout = {json.dumps(model_test_chart_layout)};
      Plotly.newPlot("forecastChart", forecastChartData, forecastChartLayout, {{responsive: true, displaylogo: false}});
      Plotly.newPlot("modelTestChart", modelTestChartData, modelTestChartLayout, {{responsive: true, displaylogo: false}});
    </script>
  </body>
</html>
"""

    output_file.write_text(document, encoding="utf-8")


def plot_feature_importance(model, feature_cols, output_file):
    """
    Saves feature importance plot.
    """

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(versioned_output_file("feature_importance.csv"), index=False)

    top_n = min(20, len(importance_df))
    top_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(9, 8))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances")
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def get_forecast_start_date(last_observed_date):
    """
    Returns the first date that should appear in the saved forecast.
    """

    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)

    if FORECAST_START.lower() == "today":
        requested_start = today
    elif FORECAST_START.lower() == "tomorrow":
        requested_start = today + pd.Timedelta(days=1)
    else:
        raise ValueError('FORECAST_START must be either "today" or "tomorrow".')

    first_unobserved_date = pd.Timestamp(last_observed_date) + pd.Timedelta(days=1)

    return max(requested_start, first_unobserved_date)


def fetch_open_meteo_basin_precipitation(forecast_days, last_observed_date, forecast_start_date):
    """
    Downloads daily precipitation from Open-Meteo for the basin points and
    averages the point forecasts by date.
    """

    first_needed_date = pd.Timestamp(last_observed_date) + pd.Timedelta(days=1)
    forecast_end_date = pd.Timestamp(forecast_start_date) + pd.Timedelta(days=forecast_days - 1)
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    request_past_days = max(OPEN_METEO_PAST_DAYS, (today - first_needed_date).days + 1)
    request_forecast_days = max(forecast_days, (forecast_end_date - today).days + 1)

    latitudes = ",".join(f"{lat:.4f}" for lat, _ in BASIN_FORECAST_POINTS)
    longitudes = ",".join(f"{lon:.4f}" for _, lon in BASIN_FORECAST_POINTS)

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "daily": "precipitation_sum",
        "timezone": "America/New_York",
        "past_days": request_past_days,
        "forecast_days": request_forecast_days,
    }

    url = f"{OPEN_METEO_URL}?{urlencode(params)}"

    print("\nDownloading Open-Meteo precipitation forecast for basin points...")
    print(f"Number of forecast points: {len(BASIN_FORECAST_POINTS)}")

    with urlopen(url, timeout=OPEN_METEO_TIMEOUT_SECONDS) as response:
        payload = response.read().decode("utf-8")

    meteo_data = json.loads(payload)

    if isinstance(meteo_data, list):
        point_payloads = [point["daily"] for point in meteo_data]
    elif isinstance(meteo_data, dict) and "daily" in meteo_data:
        point_payloads = [meteo_data["daily"]]
    else:
        raise ValueError("Unexpected Open-Meteo response format.")

    point_frames = []
    for point_index, daily in enumerate(point_payloads, start=1):
        point_df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "point_id": point_index,
            "precip_mm": pd.to_numeric(daily["precipitation_sum"], errors="coerce"),
        })
        point_frames.append(point_df)

    all_points_df = pd.concat(point_frames, ignore_index=True)

    basin_precip_df = (
        all_points_df
        .groupby("date", as_index=False)
        .agg(
            precip_mm=("precip_mm", "mean"),
            point_count=("precip_mm", "count"),
        )
        .sort_values("date")
    )

    round_numeric_columns(basin_precip_df).to_csv(
        versioned_output_file("open_meteo_basin_average_precipitation.csv"),
        index=False,
        float_format="%.2f"
    )

    future_precip_df = basin_precip_df[
        (basin_precip_df["date"] >= first_needed_date)
        & (basin_precip_df["date"] <= forecast_end_date)
    ].copy()

    needed_days = (forecast_end_date - first_needed_date).days + 1

    if len(future_precip_df) < needed_days:
        available_start = basin_precip_df["date"].min().date()
        available_end = basin_precip_df["date"].max().date()
        raise ValueError(
            "Open-Meteo did not return enough precipitation days to bridge from "
            f"the last observed water-level date ({pd.Timestamp(last_observed_date).date()}) "
            f"to the forecast end date ({forecast_end_date.date()}). "
            f"Available precipitation dates: {available_start} to {available_end}."
        )

    if future_precip_df["point_count"].min() < len(BASIN_FORECAST_POINTS):
        warnings.warn(
            "At least one Open-Meteo forecast day has fewer point values than expected. "
            "The basin average was calculated from the available point values."
        )

    future_precip_df = future_precip_df[["date", "precip_mm"]].copy()

    print("Open-Meteo basin-average precipitation selected for recursive forecasting:")
    print(future_precip_df)

    return future_precip_df


def prepare_future_precipitation(df, forecast_days):
    """
    Checks whether the input CSV already contains future rows with precipitation.

    Future rows should look like this:
    date, wl_m, precip_mm
    2026-05-07, , 12.4
    2026-05-08, , 2.1
    ...

    If future precipitation is not found, this function downloads Open-Meteo
    forecasts for the basin points and uses the daily basin-average precipitation.
    """

    historical_df = df.dropna(subset=["wl_m"]).copy()

    if historical_df.empty:
        raise ValueError("No historical water-level data found in Column B.")

    last_observed_date = historical_df["date"].max()
    forecast_start_date = get_forecast_start_date(last_observed_date)
    forecast_end_date = forecast_start_date + pd.Timedelta(days=forecast_days - 1)

    future_rows = df[
        (df["date"] > last_observed_date)
    ].copy()

    needed_future_rows = future_rows[
        (future_rows["date"] >= last_observed_date + pd.Timedelta(days=1))
        & (future_rows["date"] <= forecast_end_date)
    ].copy()

    needed_days = (forecast_end_date - last_observed_date).days

    if len(needed_future_rows) >= needed_days:
        future_precip_df = needed_future_rows.head(needed_days)[["date", "precip_mm"]].copy()
        future_precip_df["precip_mm"] = future_precip_df["precip_mm"].fillna(0)

        print("\nFuture precipitation was found in the input CSV.")
        print("The model will use those values for recursive forecasting.")

    else:
        print(
            "\nNo complete future precipitation was found in the input CSV. "
            "The model will use Open-Meteo basin-average precipitation instead."
        )

        future_precip_df = fetch_open_meteo_basin_precipitation(
            forecast_days=forecast_days,
            last_observed_date=last_observed_date,
            forecast_start_date=forecast_start_date
        )

    return historical_df, future_precip_df, forecast_start_date


def recursive_forecast_next_14_days(
    model,
    historical_df,
    future_precip_df,
    feature_cols,
    forecast_start_date
):
    """
    Predicts water level recursively.

    For each future day:
    - precipitation comes from future_precip_df
    - previous water levels come from observed history and previous predictions
    - dates before forecast_start_date are used only to bridge from the last
      observed water level to the requested forecast window
    """

    working_df = historical_df[["date", "wl_m", "precip_mm"]].copy()
    forecast_records = []
    forecast_start_date = pd.Timestamp(forecast_start_date)
    forecast_day = 0

    for i in range(len(future_precip_df)):

        next_date = future_precip_df.iloc[i]["date"]
        next_precip = future_precip_df.iloc[i]["precip_mm"]

        new_row = pd.DataFrame({
            "date": [next_date],
            "wl_m": [np.nan],
            "precip_mm": [next_precip]
        })

        temp_df = pd.concat([working_df, new_row], ignore_index=True)
        temp_features = add_features(temp_df)

        row_to_predict = temp_features.iloc[[-1]].copy()

        X_future = row_to_predict[feature_cols]

        if X_future.isna().any(axis=None):
            missing_cols = X_future.columns[X_future.isna().any()].tolist()
            raise ValueError(
                f"Missing values found in forecast features for {next_date.date()}. "
                f"Missing columns: {missing_cols}"
            )

        predicted_wl = model.predict(X_future)[0]

        # Replace missing wl with predicted value so it can be used as lag
        working_df = pd.concat([
            working_df,
            pd.DataFrame({
                "date": [next_date],
                "wl_m": [predicted_wl],
                "precip_mm": [next_precip]
            })
        ], ignore_index=True)

        if next_date >= forecast_start_date:
            forecast_day += 1
            forecast_records.append({
                "date": next_date,
                "forecast_day": forecast_day,
                "forecast_precip_mm": next_precip,
                "predicted_wl_m": predicted_wl,
                "water_level_stage": classify_water_level_stage(predicted_wl)
            })

    forecast_df = pd.DataFrame(forecast_records)

    return forecast_df


def apply_forecast_scenario(future_precip_df, forecast_start_date, scenario):
    """
    Applies a runoff-reduction or storage scenario to forecast-window precipitation.
    Bridge days before forecast_start_date are kept unchanged.
    """

    scenario_precip_df = future_precip_df.copy()
    forecast_mask = scenario_precip_df["date"] >= forecast_start_date

    if scenario["kind"] == "multiply":
        scenario_precip_df.loc[forecast_mask, "precip_mm"] = (
            scenario_precip_df.loc[forecast_mask, "precip_mm"] * scenario["value"]
        )
    elif scenario["kind"] == "storage":
        scenario_precip_df.loc[forecast_mask, "precip_mm"] = (
            scenario_precip_df.loc[forecast_mask, "precip_mm"] - scenario["value"]
        ).clip(lower=0)
    else:
        raise ValueError(f"Unknown forecast scenario type: {scenario['kind']}")

    return scenario_precip_df


def build_scenario_forecasts(model, historical_df, future_precip_df, feature_cols, forecast_start_date):
    """
    Runs all forecast scenarios and returns their forecast data frames with metadata.
    """

    scenario_forecasts = []

    for scenario in FORECAST_SCENARIOS:
        scenario_precip_df = apply_forecast_scenario(
            future_precip_df=future_precip_df,
            forecast_start_date=forecast_start_date,
            scenario=scenario
        )
        scenario_forecast_df = recursive_forecast_next_14_days(
            model=model,
            historical_df=historical_df,
            future_precip_df=scenario_precip_df,
            feature_cols=feature_cols,
            forecast_start_date=forecast_start_date
        )

        scenario_forecasts.append({
            **scenario,
            "forecast_df": scenario_forecast_df
        })

    return scenario_forecasts


def add_scenarios_to_forecast_table(forecast_df, scenario_forecasts):
    """
    Adds each scenario water level and stage category as extra columns.
    """

    output_df = forecast_df.copy()

    for scenario in scenario_forecasts:
        scenario_df = scenario["forecast_df"][[
            "date",
            "predicted_wl_m",
            "water_level_stage"
        ]].copy()
        scenario_df = scenario_df.rename(columns={
            "predicted_wl_m": f"{scenario['column_prefix']}_predicted_wl_m",
            "water_level_stage": f"{scenario['column_prefix']}_water_level_stage",
        })

        output_df = output_df.merge(scenario_df, on="date", how="left")

    return output_df


# ==============================================================
# 3. MAIN WORKFLOW
# ==============================================================

def main():

    print("\n====================================================")
    print("Stream Water-Level ML Prediction")
    print("====================================================")

    print(f"\nReading input file:\n{INPUT_FILE}")

    df = clean_input_data(INPUT_FILE)

    print("\nData summary after cleaning:")
    print(df.head())
    print(df.tail())
    print(f"\nDate range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Number of daily records: {len(df)}")

    # Separate historical observed data and future precipitation rows if available
    historical_df, future_precip_df, forecast_start_date = prepare_future_precipitation(
        df,
        FORECAST_DAYS
    )

    print(f"\nLast observed water-level date: {historical_df['date'].max().date()}")
    print(f"Saved forecast starts on: {forecast_start_date.date()}")

    # Feature engineering using only historical observed water-level data
    feature_df = add_features(historical_df)

    feature_cols = get_feature_columns(feature_df)

    # Model-ready data
    model_df = feature_df.dropna(subset=feature_cols + ["wl_m"]).copy()

    if model_df.empty:
        raise ValueError(
            "After feature engineering, no complete rows are available for training. "
            "Check missing water-level or precipitation data."
        )

    print(f"\nNumber of rows available for model training/testing: {len(model_df)}")

    # Chronological train/test split
    train_df, test_df = train_test_split_time_series(model_df, TEST_FRACTION)

    X_train = train_df[feature_cols]
    y_train = train_df["wl_m"]

    X_test = test_df[feature_cols]
    y_test = test_df["wl_m"]

    print("\nTraining period:")
    print(f"{train_df['date'].min().date()} to {train_df['date'].max().date()}")

    print("\nTesting period:")
    print(f"{test_df['date'].min().date()} to {test_df['date'].max().date()}")

    # ==============================================================
    # 4. TRAIN MODEL
    # ==============================================================

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)

    # ==============================================================
    # 5. TEST MODEL
    # ==============================================================

    test_df["predicted_wl_m"] = model.predict(X_test)

    metrics = evaluate_model(test_df["wl_m"], test_df["predicted_wl_m"])

    print("\nModel performance on testing period:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")

    # Save metrics
    metrics_df = round_numeric_columns(pd.DataFrame([metrics]))
    metrics_df.to_csv(
        versioned_output_file("model_performance_metrics.csv"),
        index=False,
        float_format="%.2f"
    )

    # Save test predictions
    test_output = test_df[["date", "wl_m", "predicted_wl_m", "precip_mm"]].copy()
    test_output = round_numeric_columns(test_output)
    test_output.to_csv(
        versioned_output_file("test_observed_vs_predicted.csv"),
        index=False,
        float_format="%.2f"
    )

    # Plots
    plot_observed_vs_predicted(
        test_df,
        versioned_output_file("observed_vs_predicted_timeseries.png")
    )

    plot_scatter_observed_vs_predicted(
        test_df,
        versioned_output_file("observed_vs_predicted_scatter.png")
    )

    plot_feature_importance(
        model,
        feature_cols,
        versioned_output_file("feature_importance.png")
    )

    # ==============================================================
    # 6. RETRAIN MODEL ON ALL HISTORICAL DATA
    # ==============================================================

    print("\nRetraining model on all historical data before forecasting...")

    final_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    X_all = model_df[feature_cols]
    y_all = model_df["wl_m"]

    final_model.fit(X_all, y_all)

    # ==============================================================
    # 7. FORECAST NEXT 14 DAYS
    # ==============================================================

    print("\nForecasting next 14 days...")

    forecast_df = recursive_forecast_next_14_days(
        model=final_model,
        historical_df=historical_df,
        future_precip_df=future_precip_df,
        feature_cols=feature_cols,
        forecast_start_date=forecast_start_date
    )

    scenario_forecasts = build_scenario_forecasts(
        model=final_model,
        historical_df=historical_df,
        future_precip_df=future_precip_df,
        feature_cols=feature_cols,
        forecast_start_date=forecast_start_date
    )

    forecast_output_df = add_scenarios_to_forecast_table(
        forecast_df=forecast_df,
        scenario_forecasts=scenario_forecasts
    )

    forecast_output_df = round_numeric_columns(forecast_output_df)
    forecast_output_df.to_csv(
        versioned_output_file("next_14_day_water_level_forecast.csv"),
        index=False,
        float_format="%.2f"
    )

    print("\nNext 14-day forecast:")
    print(forecast_output_df)

    # Plot forecast
    recent_history = historical_df.tail(60)[["date", "wl_m", "precip_mm"]].copy()

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()
    ax2.bar(
        forecast_df["date"],
        forecast_df["forecast_precip_mm"],
        color="lightskyblue",
        alpha=0.45,
        label="Forecast precipitation",
        width=0.8
    )
    ax1.plot(
        recent_history["date"],
        recent_history["wl_m"],
        label="Observed water level",
        linewidth=2
    )
    ax1.plot(
        forecast_df["date"],
        forecast_df["predicted_wl_m"],
        marker="o",
        label="Forecast water-level",
        linewidth=2
    )
    for scenario in scenario_forecasts:
        scenario_df = scenario["forecast_df"]
        ax1.plot(
            scenario_df["date"],
            scenario_df["predicted_wl_m"],
            marker="o",
            linestyle=scenario["plot_style"],
            color=scenario["plot_color"],
            label=scenario["name"],
            linewidth=2
        )

    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.patch.set_alpha(0)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Water-level in Huron River (m)")
    ax2.set_ylabel("Forecast precipitation (mm)")
    ax2.yaxis.label.set_color("#111111")
    ax2.tick_params(axis="y", colors="#111111")
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax1.set_title("Observed Recent Water Level and 14-Day Forecast")
    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(versioned_output_file("next_14_day_forecast_plot.png"), dpi=300)
    plt.close()

    plot_stage_forecast_summary(
        historical_df=historical_df,
        test_df=test_df,
        forecast_df=forecast_df,
        output_file=versioned_output_file("observed_estimated_forecast_stages.png"),
        scenario_forecasts=scenario_forecasts
    )

    write_interactive_index(
        historical_df=historical_df,
        test_df=test_df,
        forecast_df=forecast_df,
        forecast_output_df=forecast_output_df,
        scenario_forecasts=scenario_forecasts,
        metrics=metrics,
        output_file=versioned_output_file("index.html")
    )

    # Save full feature list
    pd.DataFrame({"feature": feature_cols}).to_csv(
        versioned_output_file("model_feature_columns.csv"),
        index=False
    )

    print("\n====================================================")
    print("Analysis complete.")
    print(f"Outputs saved here:\n{OUTPUT_DIR}")
    print("====================================================")


if __name__ == "__main__":
    main()
