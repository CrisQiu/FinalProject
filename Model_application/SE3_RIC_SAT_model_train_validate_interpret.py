
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SE3: Train, validate, and interpret statistical models for the daily
relationship between river ice concentration (RIC) and surface air temperature (SAT).

This script:
1. Reads daily SAT GeoTIFFs from: FinalProject/raw_data/ERA5_temperature
2. Reads daily RIC GeoTIFFs from: FinalProject/analysis_ready/RIC
3. Extracts daily mean SAT and daily mean RIC
4. Matches SAT and RIC by date
5. Applies a 10-day moving average to SAT
6. Fits a linear model and a logistic model
7. Extracts model coefficients and fit statistics
8. Saves tables and figures to: FinalProject/Model_application

All comments, labels, and outputs are written in standard English.
"""

from __future__ import annotations

import itertools
import math
import re
import sys
import warnings
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import rasterio
    from scipy.optimize import curve_fit
    from scipy.stats import gaussian_kde, linregress
except ImportError as exc:
    missing_pkg = str(exc).replace("No module named ", "").replace("'", "")
    raise SystemExit(
        f"Missing required package: {missing_pkg}. "
        "Please install the missing dependency in your JupyterLab environment first."
    )

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# User settings
# =============================================================================

RIVER_NAME = "Ob River"
SAT_MOVING_AVERAGE_WINDOW = 10
SAT_MOVING_AVERAGE_CENTER = True
MIN_VALID_SAMPLES_FOR_MODEL = 30
FIG_DPI = 300
SHOW_PLOTS = False

# Automatic RIC scaling:
# - "auto": divide by 100 only if values appear to be in 0-100
# - "none": keep original values
# - "percent_to_fraction": always divide by 100
RIC_SCALE_MODE = "auto"

# Plot limits
SCATTER_XLIM = None   # Example: (-35, 25)
SCATTER_YLIM = (-0.05, 1.05)

# =============================================================================
# Path utilities
# =============================================================================

def find_project_root() -> Path:
    """
    Try to locate the FinalProject directory robustly, regardless of where
    the script is launched from.
    """
    search_roots = []

    cwd = Path.cwd().resolve()
    search_roots.extend([cwd, *cwd.parents])

    if "__file__" in globals():
        script_dir = Path(__file__).resolve().parent
        search_roots.extend([script_dir, *script_dir.parents])

    checked = set()
    for base in search_roots:
        if base in checked:
            continue
        checked.add(base)

        if base.name == "FinalProject":
            return base

        candidate = base / "FinalProject"
        if candidate.exists():
            return candidate.resolve()

    # Fallback path if the folder is not found automatically
    return (cwd / "FinalProject").resolve()


PROJECT_ROOT = find_project_root()
SAT_DIR = PROJECT_ROOT / "raw_data" / "ERA5_temperature"
RIC_DIR = PROJECT_ROOT / "analysis_ready" / "RIC"
OUTPUT_DIR = PROJECT_ROOT / "Model_application"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"

for folder in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Helper functions
# =============================================================================

def decimal_year(dt: pd.Timestamp) -> float:
    return dt.year + (dt.timetuple().tm_yday - 1) / 365.25


def hydrological_year(dt: pd.Timestamp) -> int:
    """
    Hydrological year definition:
    October 1 to September 30 belongs to the next calendar year.
    Example:
    - 2020-10-01 -> hydrological year 2021
    - 2021-04-15 -> hydrological year 2021
    """
    return dt.year + 1 if dt.month >= 10 else dt.year


def extract_date_from_filename(path_like: Path) -> pd.Timestamp | None:
    """
    Extract a valid date from file names such as:
    - 2000-10-01_temperature_2m_ERA5_Land.tif
    - ObRiverIce_20001001_RIC.tif
    - 2000_10_01.tif
    """
    stem = Path(path_like).stem

    patterns = [
        r"(?<!\d)(\d{4})[-_.](\d{2})[-_.](\d{2})(?!\d)",
        r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, stem):
            y, m, d = match.groups()
            try:
                return pd.Timestamp(datetime(int(y), int(m), int(d)))
            except ValueError:
                continue
    return None


def read_single_band_mean(tif_path: Path) -> float:
    """
    Read the first band of a GeoTIFF and calculate the mean of valid pixels.
    """
    with rasterio.open(tif_path) as src:
        arr = src.read(1, masked=True).astype("float64")

        if hasattr(arr, "filled"):
            arr = arr.filled(np.nan)

        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return np.nan
        return float(np.mean(valid))


def build_daily_mean_table(input_dir: Path, value_col: str) -> pd.DataFrame:
    """
    Read all tif/tiff files in a directory recursively and create a daily mean table.
    """
    tif_files = sorted(
        list(input_dir.rglob("*.tif")) + list(input_dir.rglob("*.tiff"))
    )

    records = []
    skipped = []

    for fp in tif_files:
        dt = extract_date_from_filename(fp)
        if dt is None:
            skipped.append(fp.name)
            continue

        mean_val = read_single_band_mean(fp)
        records.append(
            {
                "Date": dt.normalize(),
                "DecimalYear": decimal_year(dt),
                value_col: mean_val,
                "FileName": fp.name,
                "FilePath": str(fp.resolve()),
                "Hydrological_Year": hydrological_year(dt),
            }
        )

    if not records:
        raise FileNotFoundError(
            f"No valid GeoTIFF files with recognizable dates were found in: {input_dir}"
        )

    df = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates(subset="Date", keep="first").reset_index(drop=True)

    if skipped:
        skipped_path = LOG_DIR / f"skipped_files_{value_col.lower()}.txt"
        skipped_path.write_text(
            "\n".join(skipped), encoding="utf-8"
        )

    return df


def normalize_ric_values(df_ric: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Normalize RIC values if necessary.
    """
    df = df_ric.copy()
    scale_note = "RIC values kept unchanged."

    valid = df["Mean_RIC"].dropna()
    if valid.empty:
        return df, "RIC values are empty."

    max_val = float(valid.max())

    if RIC_SCALE_MODE == "percent_to_fraction":
        df["Mean_RIC"] = df["Mean_RIC"] / 100.0
        scale_note = "RIC values were divided by 100.0 (forced percent-to-fraction conversion)."
    elif RIC_SCALE_MODE == "auto":
        if max_val > 1.5 and max_val <= 100.5:
            df["Mean_RIC"] = df["Mean_RIC"] / 100.0
            scale_note = (
                "RIC values were automatically divided by 100.0 because the detected "
                "range suggested percentages (0-100)."
            )
        else:
            scale_note = "RIC values were kept unchanged after automatic inspection."
    elif RIC_SCALE_MODE == "none":
        scale_note = "RIC values were kept unchanged (manual setting)."
    else:
        raise ValueError("RIC_SCALE_MODE must be one of: 'auto', 'none', 'percent_to_fraction'.")

    return df, scale_note


def linear_model(x, a, b):
    return a * x + b


def logistic_4p(x, lower, upper, x0, k):
    """
    Four-parameter logistic function.
    """
    x = np.asarray(x, dtype=float)
    k = np.where(np.abs(k) < 1e-8, 1e-8, k)
    exponent = np.clip((x - x0) / k, -60, 60)
    return lower + (upper - lower) / (1.0 + np.exp(exponent))


def logistic_2p(x, x0, k):
    """
    Two-parameter logistic fallback with lower=0 and upper=1.
    """
    x = np.asarray(x, dtype=float)
    k = np.where(np.abs(k) < 1e-8, 1e-8, k)
    exponent = np.clip((x - x0) / k, -60, 60)
    return 1.0 / (1.0 + np.exp(exponent))


def calculate_metrics(y_true, y_pred) -> dict:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    tss = float(np.sum((np.asarray(y_true) - np.mean(y_true)) ** 2))
    rss = float(np.sum(residuals ** 2))
    r2 = float(1.0 - rss / tss) if tss > 0 else np.nan

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "RSS": rss,
        "TSS": tss,
    }


def calculate_point_density(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 5 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        z = np.ones_like(x, dtype=float)
        return x, y, z

    xy = np.vstack([x, y])

    try:
        z = gaussian_kde(xy)(xy)
    except Exception:
        try:
            jitter_x = x + np.random.normal(0, 1e-6, size=len(x))
            jitter_y = y + np.random.normal(0, 1e-6, size=len(y))
            xy_jitter = np.vstack([jitter_x, jitter_y])
            z = gaussian_kde(xy_jitter)(xy_jitter)
        except Exception:
            z = np.ones_like(x, dtype=float)

    idx = np.argsort(z)
    return x[idx], y[idx], z[idx]


def fit_linear_model(x, y) -> dict:
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    yhat = linear_model(x, slope, intercept)
    metrics = calculate_metrics(y, yhat)

    return {
        "Model": "Linear",
        "Equation": "y = a*x + b",
        "Parameters": {
            "a_slope": float(slope),
            "b_intercept": float(intercept),
        },
        "Pearson_r": float(r_value),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "yhat": yhat,
        **metrics,
    }


def fit_logistic_model(x, y) -> dict:
    """
    Fit a logistic model with multiple starting guesses.
    Fallback to a two-parameter sigmoid if the four-parameter model fails.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    x_span = max(x_max - x_min, 1.0)

    best = None
    best_rss = np.inf

    lower_candidates = [max(0.0, float(np.nanpercentile(y, 5))), 0.0]
    upper_candidates = [min(1.0, float(np.nanpercentile(y, 95))), 1.0]
    x0_candidates = [
        float(np.nanmedian(x)),
        float(np.nanpercentile(x, 40)),
        float(np.nanpercentile(x, 60)),
    ]
    k_candidates = [
        max(x_span / 20.0, 0.2),
        max(x_span / 10.0, 0.5),
        max(x_span / 5.0, 1.0),
    ]

    for lower0, upper0, x00, k0 in itertools.product(
        lower_candidates, upper_candidates, x0_candidates, k_candidates
    ):
        p0 = [lower0, upper0, x00, k0]

        try:
            popt, _ = curve_fit(
                logistic_4p,
                x,
                y,
                p0=p0,
                bounds=([-0.05, 0.50, x_min - 20.0, 0.01],
                        [0.20, 1.05, x_max + 20.0, 50.0]),
                maxfev=100000,
            )
            yhat = logistic_4p(x, *popt)
            metrics = calculate_metrics(y, yhat)

            if metrics["RSS"] < best_rss:
                best_rss = metrics["RSS"]
                best = {
                    "Model": "Logistic",
                    "Equation": "y = lower + (upper - lower) / (1 + exp((x - x0) / k))",
                    "Model_Form": "Four-parameter logistic",
                    "Parameters": {
                        "lower": float(popt[0]),
                        "upper": float(popt[1]),
                        "x0": float(popt[2]),
                        "k": float(popt[3]),
                    },
                    "yhat": yhat,
                    **metrics,
                }
        except Exception:
            continue

    if best is not None:
        return best

    # Fallback
    fallback_guesses = [
        [float(np.nanmedian(x)), max(x_span / 10.0, 0.5)],
        [float(np.nanpercentile(x, 40)), max(x_span / 5.0, 1.0)],
        [float(np.nanpercentile(x, 60)), max(x_span / 20.0, 0.2)],
    ]

    for p0 in fallback_guesses:
        try:
            popt, _ = curve_fit(
                logistic_2p,
                x,
                y,
                p0=p0,
                bounds=([x_min - 20.0, 0.01], [x_max + 20.0, 50.0]),
                maxfev=100000,
            )
            yhat = logistic_2p(x, *popt)
            metrics = calculate_metrics(y, yhat)

            return {
                "Model": "Logistic",
                "Equation": "y = 1 / (1 + exp((x - x0) / k))",
                "Model_Form": "Two-parameter logistic fallback",
                "Parameters": {
                    "lower": 0.0,
                    "upper": 1.0,
                    "x0": float(popt[0]),
                    "k": float(popt[1]),
                },
                "yhat": yhat,
                **metrics,
            }
        except Exception:
            continue

    # Final fallback if all logistic fits fail
    nan_array = np.full_like(y, np.nan, dtype=float)
    return {
        "Model": "Logistic",
        "Equation": "Fit failed",
        "Model_Form": "Fit failed",
        "Parameters": {
            "lower": np.nan,
            "upper": np.nan,
            "x0": np.nan,
            "k": np.nan,
        },
        "yhat": nan_array,
        "R2": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "RSS": np.nan,
        "TSS": np.nan,
    }


def prepare_model_dataframe(df_sat: pd.DataFrame, df_ric: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Merge daily SAT and RIC by date and prepare the model input table.
    """
    df = pd.merge(
        df_sat[["Date", "DecimalYear", "Mean_SAT", "Hydrological_Year"]],
        df_ric[["Date", "Mean_RIC"]],
        on="Date",
        how="inner",
    )

    # Recalculate decimal year after merge to guarantee consistency
    df["DecimalYear"] = df["Date"].apply(decimal_year)
    df["Hydrological_Year"] = df["Date"].apply(hydrological_year)

    # SAT moving average
    df = df.sort_values("Date").reset_index(drop=True)
    df["Mean_SAT_MA10"] = df["Mean_SAT"].rolling(
        window=SAT_MOVING_AVERAGE_WINDOW,
        center=SAT_MOVING_AVERAGE_CENTER,
    ).mean()

    # Keep original data count information
    n_before_filter = len(df)

    # Filter valid model rows
    df_model = df.dropna(subset=["Mean_SAT_MA10", "Mean_RIC"]).copy()
    df_model = df_model[(df_model["Mean_RIC"] >= 0) & (df_model["Mean_RIC"] <= 1)].reset_index(drop=True)

    filter_note = (
        f"Rows before filtering: {n_before_filter}. "
        f"Rows after filtering valid Mean_SAT_MA10 and 0<=Mean_RIC<=1: {len(df_model)}."
    )

    return df_model, filter_note


def plot_time_series(df_sat: pd.DataFrame, df_ric: pd.DataFrame) -> Path:
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    ax1.plot(df_sat["Date"], df_sat["Mean_SAT"], linewidth=1.0, color='red', label="Daily mean SAT")
    ax2.plot(df_ric["Date"], df_ric["Mean_RIC"], linewidth=1.0, label="Daily mean RIC")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Daily mean SAT (°C)")
    ax2.set_ylabel("Daily mean RIC")
    ax1.set_title(f"{RIVER_NAME}: Daily mean SAT and daily mean RIC")
    ax1.grid(True, alpha=0.3)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=False)

    out_path = FIGURE_DIR / "daily_mean_sat_and_ric_time_series.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_scatter_with_fits(df_model: pd.DataFrame, linear_res: dict, logistic_res: dict) -> Path:
    x = df_model["Mean_SAT_MA10"].to_numpy()
    y = df_model["Mean_RIC"].to_numpy()

    x_plot, y_plot, z_plot = calculate_point_density(x, y)

    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    sc = ax.scatter(
        x_plot,
        y_plot,
        c=z_plot,
        cmap="Oranges",
        s=18,
        edgecolor="none",
    )

    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 500)
    y_line_linear = linear_model(
        x_line,
        linear_res["Parameters"]["a_slope"],
        linear_res["Parameters"]["b_intercept"],
    )

    if logistic_res["Model_Form"] == "Fit failed":
        y_line_logistic = np.full_like(x_line, np.nan, dtype=float)
    elif logistic_res["Model_Form"] == "Two-parameter logistic fallback":
        y_line_logistic = logistic_2p(
            x_line,
            logistic_res["Parameters"]["x0"],
            logistic_res["Parameters"]["k"],
        )
    else:
        y_line_logistic = logistic_4p(
            x_line,
            logistic_res["Parameters"]["lower"],
            logistic_res["Parameters"]["upper"],
            logistic_res["Parameters"]["x0"],
            logistic_res["Parameters"]["k"],
        )

    ax.plot(x_line, y_line_linear, linestyle="--", linewidth=2.0, label="Linear fit")
    ax.plot(x_line, y_line_logistic, linestyle="-", linewidth=2.2, label="Logistic fit")

    ax.set_xlabel("Mean SAT (10-day moving average, °C)")
    ax.set_ylabel("Mean RIC")
    ax.set_title(f"{RIVER_NAME}: RIC-SAT relationship")
    ax.grid(True, alpha=0.3)

    if SCATTER_XLIM is not None:
        ax.set_xlim(SCATTER_XLIM)
    if SCATTER_YLIM is not None:
        ax.set_ylim(SCATTER_YLIM)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Point density")

    linear_text = (
        f"Linear fit: y = {linear_res['Parameters']['a_slope']:.3f}x + "
        f"{linear_res['Parameters']['b_intercept']:.3f}\n"
        f"R² = {linear_res['R2']:.3f}, RMSE = {linear_res['RMSE']:.3f}, "
        f"p = {linear_res['p_value']:.3e}"
    )

    logistic_text = (
        f"{logistic_res['Model_Form']}\n"
        f"R² = {logistic_res['R2']:.3f}, RMSE = {logistic_res['RMSE']:.3f}"
    )

    ax.text(
        0.03,
        0.17,
        linear_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=3),
    )
    ax.text(
        0.03,
        0.03,
        logistic_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=3),
    )

    ax.legend(loc="upper right", frameon=False)

    out_path = FIGURE_DIR / "ric_sat_density_scatter_linear_logistic_fit.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_observed_vs_predicted(df_model: pd.DataFrame, linear_res: dict, logistic_res: dict) -> Path:
    y_true = df_model["Mean_RIC"].to_numpy()
    y_pred_linear = linear_res["yhat"]
    y_pred_logistic = logistic_res["yhat"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # Linear
    axes[0].scatter(y_true, y_pred_linear, s=18, alpha=0.7)
    lims = [0.0, 1.0]
    axes[0].plot(lims, lims, linestyle="--", linewidth=1.5)
    axes[0].set_title("Observed vs predicted (Linear)")
    axes[0].set_xlabel("Observed mean RIC")
    axes[0].set_ylabel("Predicted mean RIC")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)

    # Logistic
    axes[1].scatter(y_true, y_pred_logistic, s=18, alpha=0.7)
    axes[1].plot(lims, lims, linestyle="--", linewidth=1.5)
    axes[1].set_title("Observed vs predicted (Logistic)")
    axes[1].set_xlabel("Observed mean RIC")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)

    out_path = FIGURE_DIR / "observed_vs_predicted_linear_logistic.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def save_model_summary(linear_res: dict, logistic_res: dict, df_model: pd.DataFrame) -> Path:
    rows = [
        {
            "River": RIVER_NAME,
            "Model": "Linear",
            "Equation": linear_res["Equation"],
            "Sample_Size": len(df_model),
            "Predictor": "Mean_SAT_MA10",
            "Response": "Mean_RIC",
            "Parameter_1_Name": "a_slope",
            "Parameter_1_Value": linear_res["Parameters"]["a_slope"],
            "Parameter_2_Name": "b_intercept",
            "Parameter_2_Value": linear_res["Parameters"]["b_intercept"],
            "Parameter_3_Name": np.nan,
            "Parameter_3_Value": np.nan,
            "Parameter_4_Name": np.nan,
            "Parameter_4_Value": np.nan,
            "Pearson_r": linear_res["Pearson_r"],
            "p_value": linear_res["p_value"],
            "R2": linear_res["R2"],
            "RMSE": linear_res["RMSE"],
            "MAE": linear_res["MAE"],
        },
        {
            "River": RIVER_NAME,
            "Model": "Logistic",
            "Equation": logistic_res["Equation"],
            "Sample_Size": len(df_model),
            "Predictor": "Mean_SAT_MA10",
            "Response": "Mean_RIC",
            "Parameter_1_Name": "lower",
            "Parameter_1_Value": logistic_res["Parameters"]["lower"],
            "Parameter_2_Name": "upper",
            "Parameter_2_Value": logistic_res["Parameters"]["upper"],
            "Parameter_3_Name": "x0",
            "Parameter_3_Value": logistic_res["Parameters"]["x0"],
            "Parameter_4_Name": "k",
            "Parameter_4_Value": logistic_res["Parameters"]["k"],
            "Pearson_r": np.nan,
            "p_value": np.nan,
            "R2": logistic_res["R2"],
            "RMSE": logistic_res["RMSE"],
            "MAE": logistic_res["MAE"],
        },
    ]

    summary_df = pd.DataFrame(rows)
    out_path = TABLE_DIR / "model_summary.csv"
    summary_df.to_csv(out_path, index=False)
    return out_path


def save_prediction_table(df_model: pd.DataFrame, linear_res: dict, logistic_res: dict) -> Path:
    df_out = df_model.copy()
    df_out["Linear_Predicted_RIC"] = linear_res["yhat"]
    df_out["Logistic_Predicted_RIC"] = logistic_res["yhat"]
    df_out["Linear_Residual"] = df_out["Mean_RIC"] - df_out["Linear_Predicted_RIC"]
    df_out["Logistic_Residual"] = df_out["Mean_RIC"] - df_out["Logistic_Predicted_RIC"]

    out_path = TABLE_DIR / "model_predictions.csv"
    df_out.to_csv(out_path, index=False)
    return out_path


def save_run_metadata(metadata_lines: list[str]) -> Path:
    out_path = LOG_DIR / "run_metadata.txt"
    out_path.write_text("\n".join(metadata_lines), encoding="utf-8")
    return out_path


# =============================================================================
# Main workflow
# =============================================================================

def main():
    print("=" * 80)
    print("SE3 workflow started")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"SAT directory: {SAT_DIR}")
    print(f"RIC directory: {RIC_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    if not SAT_DIR.exists():
        raise FileNotFoundError(f"SAT directory does not exist: {SAT_DIR}")
    if not RIC_DIR.exists():
        raise FileNotFoundError(f"RIC directory does not exist: {RIC_DIR}")

    print("\n[1/6] Reading SAT GeoTIFFs and extracting daily mean SAT ...")
    df_sat = build_daily_mean_table(SAT_DIR, "Mean_SAT")
    sat_csv = TABLE_DIR / "daily_mean_sat.csv"
    df_sat.to_csv(sat_csv, index=False)

    print("[2/6] Reading RIC GeoTIFFs and extracting daily mean RIC ...")
    df_ric = build_daily_mean_table(RIC_DIR, "Mean_RIC")
    df_ric, ric_scale_note = normalize_ric_values(df_ric)
    ric_csv = TABLE_DIR / "daily_mean_ric.csv"
    df_ric.to_csv(ric_csv, index=False)

    print("[3/6] Pairing SAT and RIC by date ...")
    df_model, filter_note = prepare_model_dataframe(df_sat, df_ric)
    paired_csv = TABLE_DIR / "paired_daily_sat_ric.csv"
    df_model.to_csv(paired_csv, index=False)

    if len(df_model) < MIN_VALID_SAMPLES_FOR_MODEL:
        raise ValueError(
            f"Too few valid paired samples after filtering: {len(df_model)}. "
            f"At least {MIN_VALID_SAMPLES_FOR_MODEL} are required."
        )

    print("[4/6] Fitting statistical models ...")
    x = df_model["Mean_SAT_MA10"].to_numpy()
    y = df_model["Mean_RIC"].to_numpy()

    linear_res = fit_linear_model(x, y)
    logistic_res = fit_logistic_model(x, y)

    summary_csv = save_model_summary(linear_res, logistic_res, df_model)
    prediction_csv = save_prediction_table(df_model, linear_res, logistic_res)

    print("[5/6] Creating figures ...")
    ts_fig = plot_time_series(df_sat, df_ric)
    scatter_fig = plot_scatter_with_fits(df_model, linear_res, logistic_res)
    pred_fig = plot_observed_vs_predicted(df_model, linear_res, logistic_res)

    print("[6/6] Writing metadata log ...")
    metadata_lines = [
        "SE3 model train, validate, and interpret workflow",
        f"River: {RIVER_NAME}",
        f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Project root: {PROJECT_ROOT}",
        f"SAT directory: {SAT_DIR}",
        f"RIC directory: {RIC_DIR}",
        f"SAT file count: {len(df_sat)}",
        f"RIC file count: {len(df_ric)}",
        ric_scale_note,
        filter_note,
        f"SAT moving average window: {SAT_MOVING_AVERAGE_WINDOW}",
        f"SAT moving average center: {SAT_MOVING_AVERAGE_CENTER}",
        f"Linear R2: {linear_res['R2']:.6f}",
        f"Linear RMSE: {linear_res['RMSE']:.6f}",
        f"Logistic model form: {logistic_res['Model_Form']}",
        f"Logistic R2: {logistic_res['R2']:.6f}" if np.isfinite(logistic_res["R2"]) else "Logistic R2: NaN",
        f"Logistic RMSE: {logistic_res['RMSE']:.6f}" if np.isfinite(logistic_res["RMSE"]) else "Logistic RMSE: NaN",
        f"Primary scatter figure: {scatter_fig}",
        f"Prediction figure: {pred_fig}",
        f"Time series figure: {ts_fig}",
        f"Summary CSV: {summary_csv}",
        f"Predictions CSV: {prediction_csv}",
        f"Paired data CSV: {paired_csv}",
    ]
    metadata_txt = save_run_metadata(metadata_lines)

    print("\nWorkflow finished successfully.")
    print("-" * 80)
    print("Main outputs:")
    print(f"1. {sat_csv}")
    print(f"2. {ric_csv}")
    print(f"3. {paired_csv}")
    print(f"4. {summary_csv}")
    print(f"5. {prediction_csv}")
    print(f"6. {ts_fig}")
    print(f"7. {scatter_fig}")
    print(f"8. {pred_fig}")
    print(f"9. {metadata_txt}")
    print("-" * 80)


if __name__ == "__main__":
    main()
