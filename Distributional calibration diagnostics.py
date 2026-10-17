from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm, kstest


DATA_DIR = Path("path/to/your/data")
INPUT_FILE = DATA_DIR / "predictions.xlsx"
OUTPUT_DIR = DATA_DIR
OUTPUT_PREFIX = "distributional_calibration_diagnostics"

MIN_VARIANCE = 1e-8
PIT_BINS = 19
NOMINAL_LEVELS = np.arange(0.05, 1.00, 0.05)
USE_LOG_SCALE = True
PLOT_FLOOR = 1e-300
FIGURE_DPI_PNG = 300
FIGURE_SIZE = (10, 4.2)


def read_prediction_results(file_path):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Prediction file not found:\n{file_path}")

    df = pd.read_excel(file_path, engine="openpyxl")
    if df.shape[1] < 4:
        raise ValueError(
            "The input Excel file must contain at least four columns: "
            "time index, observed value, predicted mean, and predicted variance."
        )
    if len(df) == 0:
        raise ValueError("The input Excel file contains no prediction samples.")

    time_index = df.iloc[:, 0].to_numpy()
    observed = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=np.float64)
    predicted_mean = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy(dtype=np.float64)
    predicted_variance = pd.to_numeric(df.iloc[:, 3], errors="coerce").to_numpy(dtype=np.float64)

    if pd.isna(time_index).any():
        raise ValueError("The time-index column contains missing values.")

    numeric_values = np.column_stack((observed, predicted_mean, predicted_variance))
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError(
            "Observed values, predicted means, and predicted variances "
            "must contain only finite numeric values."
        )

    invalid_variance = np.flatnonzero(predicted_variance <= 0)
    if invalid_variance.size:
        rows = (invalid_variance[:10] + 2).tolist()
        raise ValueError(
            "Predicted variances must be positive. Invalid Excel row numbers: "
            f"{rows}"
        )

    effective_variance = np.maximum(predicted_variance, MIN_VARIANCE)
    predicted_std = np.sqrt(effective_variance)

    return {
        "time_index": time_index,
        "observed": observed,
        "predicted_mean": predicted_mean,
        "predicted_variance": effective_variance,
        "predicted_std": predicted_std,
    }


def compute_pit(data):
    z = (data["observed"] - data["predicted_mean"]) / data["predicted_std"]
    pit = norm.cdf(z)
    negative_probability = norm.cdf(-data["predicted_mean"] / data["predicted_std"])
    return z, pit, negative_probability


def pit_histogram(pit):
    edges = np.linspace(0.0, 1.0, PIT_BINS + 1)
    counts, _ = np.histogram(pit, bins=edges)
    density, _ = np.histogram(pit, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    return pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "bin_center": centers,
        "pit_count": counts,
        "pit_density": density,
        "expected_density": np.ones_like(centers),
    })


def reliability_analysis(data):
    observed = data["observed"]
    mean = data["predicted_mean"]
    std = data["predicted_std"]
    rows = []

    for nominal in NOMINAL_LEVELS:
        lower_quantile = norm.ppf((1.0 - nominal) / 2.0)
        upper_quantile = norm.ppf((1.0 + nominal) / 2.0)
        lower_bound = mean + lower_quantile * std
        upper_bound = mean + upper_quantile * std
        covered = (observed >= lower_bound) & (observed <= upper_bound)
        empirical = float(np.mean(covered))

        rows.append({
            "nominal_coverage": float(nominal),
            "empirical_coverage": empirical,
            "calibration_error": empirical - float(nominal),
            "absolute_calibration_error": abs(empirical - float(nominal)),
            "mean_interval_width": float(np.mean(upper_bound - lower_bound)),
            "covered_sample_count": int(np.sum(covered)),
            "total_sample_count": int(len(observed)),
        })

    return pd.DataFrame(rows)


def probability_mass_assessment(data):
    probability = norm.cdf(-data["predicted_mean"] / data["predicted_std"])
    return pd.DataFrame({
        "time_index": data["time_index"],
        "observed_wind_speed": data["observed"],
        "predicted_mean": data["predicted_mean"],
        "predicted_variance": data["predicted_variance"],
        "probability_mass_below_zero": probability,
    })


def plot_pit(ax, hist_df):
    width = hist_df["bin_right"] - hist_df["bin_left"]
    ax.bar(
        hist_df["bin_center"],
        hist_df["pit_density"],
        width=width,
        alpha=0.55,
        edgecolor="black",
    )
    ax.axhline(1.0, linestyle="--", linewidth=1.5, label="Uniform density")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title("(a) PIT histogram")
    ax.legend(frameon=False)


def plot_reliability(ax, reliability_df):
    ax.plot(
        reliability_df["nominal_coverage"] * 100.0,
        reliability_df["empirical_coverage"] * 100.0,
        marker="o",
        linewidth=2,
    )
    ax.plot([0, 100], [0, 100], linestyle="--")
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("(b) Reliability diagram")


def plot_mass(ax, mass_df):
    probability = np.maximum(mass_df["probability_mass_below_zero"], PLOT_FLOOR)
    ax.scatter(mass_df["predicted_mean"], probability, s=24, alpha=0.7)
    if USE_LOG_SCALE:
        ax.set_yscale("log")
    ax.set_xlabel("Predicted wind speed (m/s)")
    ax.set_ylabel("Probability mass below zero")
    ax.set_title("(c) Probability mass assessment")


def save_figures(hist_df, reliability_df, mass_df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE)
    plot_pit(axes[0], hist_df)
    plot_reliability(axes[1], reliability_df)
    plot_mass(axes[2], mass_df)
    fig.tight_layout()

    path = output_dir / f"{OUTPUT_PREFIX}_combined.png"
    fig.savefig(path, dpi=FIGURE_DPI_PNG, bbox_inches="tight")
    plt.close(fig)
    return path


def save_excel(hist_df, reliability_df, mass_df, pit, negative_probability, output_dir):
    ks_statistic, ks_p_value = kstest(pit, "uniform")
    stats = pd.DataFrame([{
        "sample_size": int(len(pit)),
        "mean_PIT": float(np.mean(pit)),
        "std_PIT": float(np.std(pit, ddof=1)) if len(pit) > 1 else np.nan,
        "PIT_KS_statistic": float(ks_statistic),
        "PIT_KS_p_value": float(ks_p_value),
        "mean_probability_mass_below_zero": float(np.mean(negative_probability)),
        "max_probability_mass_below_zero": float(np.max(negative_probability)),
        "mean_absolute_calibration_error": float(reliability_df["absolute_calibration_error"].mean()),
        "max_absolute_calibration_error": float(reliability_df["absolute_calibration_error"].max()),
    }])

    path = output_dir / f"{OUTPUT_PREFIX}_results.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        hist_df.to_excel(writer, sheet_name="PIT_histogram", index=False)
        reliability_df.to_excel(writer, sheet_name="reliability", index=False)
        mass_df.to_excel(writer, sheet_name="probability_mass", index=False)
        stats.to_excel(writer, sheet_name="statistics", index=False)
    return path


def main(input_file=INPUT_FILE, output_dir=OUTPUT_DIR):
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = read_prediction_results(input_file)
    _, pit, negative_probability = compute_pit(data)
    hist_df = pit_histogram(pit)
    reliability_df = reliability_analysis(data)
    mass_df = probability_mass_assessment(data)

    figure_path = save_figures(hist_df, reliability_df, mass_df, output_dir)
    excel_path = save_excel(
        hist_df,
        reliability_df,
        mass_df,
        pit,
        negative_probability,
        output_dir,
    )

    print(figure_path)
    print(excel_path)


if __name__ == "__main__":
    main()
