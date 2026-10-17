from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm, gaussian_kde, probplot, skew, kurtosis


DATA_DIR = Path("path/to/your/data")
INPUT_FILE = DATA_DIR / "predictions.xlsx"
OUTPUT_DIR = DATA_DIR
OUTPUT_PREFIX = "standardized_residual_diagnostics"

MIN_VARIANCE = 1e-8
HISTOGRAM_BINS = 25
FIGURE_DPI_PNG = 300
FIGURE_DPI_TIF = 600
COMBINED_FIGURE_SIZE = (15, 4.5)
SINGLE_FIGURE_SIZE = (5.2, 4.2)


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
    if len(df) < 4:
        raise ValueError("At least four prediction samples are required for residual diagnostics.")

    time_index = df.iloc[:, 0].to_numpy()
    observed = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=np.float64)
    predicted_mean = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy(dtype=np.float64)
    predicted_variance = pd.to_numeric(df.iloc[:, 3], errors="coerce").to_numpy(dtype=np.float64)

    if pd.isna(time_index).any():
        raise ValueError("The time-index column contains missing values.")

    numeric_values = np.column_stack((observed, predicted_mean, predicted_variance))
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError("Observed values, predicted means, and predicted variances must be finite numeric values.")

    invalid_variance = np.flatnonzero(predicted_variance <= 0)
    if invalid_variance.size:
        rows = (invalid_variance[:10] + 2).tolist()
        raise ValueError(
            "Predicted variances must be positive. Invalid Excel row numbers: "
            f"{rows}"
        )

    effective_variance = np.maximum(predicted_variance, MIN_VARIANCE)
    predicted_std = np.sqrt(effective_variance)
    lower_95 = predicted_mean - 1.96 * predicted_std
    upper_95 = predicted_mean + 1.96 * predicted_std

    return {
        "time_index": time_index,
        "observed": observed,
        "predicted_mean": predicted_mean,
        "predicted_variance": predicted_variance,
        "effective_variance": effective_variance,
        "predicted_std": predicted_std,
        "lower_95": lower_95,
        "upper_95": upper_95,
    }


def compute_residual_diagnostics(data):
    observed = data["observed"]
    predicted_mean = data["predicted_mean"]
    predicted_std = data["predicted_std"]
    residual = observed - predicted_mean
    standardized_residual = residual / predicted_std
    negative_probability = norm.cdf((0.0 - predicted_mean) / predicted_std)
    picp_95 = np.mean((observed >= data["lower_95"]) & (observed <= data["upper_95"]))

    has_variation = np.ptp(standardized_residual) > 0
    skewness = float(skew(standardized_residual, bias=False)) if has_variation else np.nan
    excess_kurtosis = float(kurtosis(standardized_residual, fisher=True, bias=False)) if has_variation else np.nan

    stats = {
        "Sample_size": int(len(standardized_residual)),
        "Mean_of_standardized_residuals": float(np.mean(standardized_residual)),
        "Std_of_standardized_residuals": float(np.std(standardized_residual, ddof=1)),
        "Skewness": skewness,
        "Excess_kurtosis": excess_kurtosis,
        "Mean_negative_probability": float(np.mean(negative_probability)),
        "Max_negative_probability": float(np.max(negative_probability)),
        "PICP_95": float(picp_95),
    }

    return {
        "predicted_variance": data["predicted_variance"],
        "predicted_std": predicted_std,
        "residual_calculated": residual,
        "standardized_residual": standardized_residual,
        "negative_probability": negative_probability,
        "stats": stats,
    }


def prepare_distribution_plot_data(standardized_residual, hist_bins):
    x_min = min(float(np.min(standardized_residual)), -4.0)
    x_max = max(float(np.max(standardized_residual)), 4.0)
    x_grid = np.linspace(x_min, x_max, 500)

    hist_density, bin_edges = np.histogram(
        standardized_residual,
        bins=hist_bins,
        density=True,
    )
    bin_left = bin_edges[:-1]
    bin_right = bin_edges[1:]
    bin_center = 0.5 * (bin_left + bin_right)

    try:
        kde_density = gaussian_kde(standardized_residual)(x_grid)
    except (ValueError, np.linalg.LinAlgError) as exc:
        print(f"Warning: KDE could not be computed: {exc}")
        kde_density = np.full_like(x_grid, np.nan)

    standard_normal_density = norm.pdf(x_grid)
    hist_df = pd.DataFrame({
        "bin_left": bin_left,
        "bin_right": bin_right,
        "bin_center": bin_center,
        "histogram_density": hist_density,
    })
    curve_df = pd.DataFrame({
        "x_standardized_residual": x_grid,
        "kde_density": kde_density,
        "standard_normal_density": standard_normal_density,
    })
    return hist_df, curve_df


def prepare_qq_plot_data(standardized_residual):
    qq = probplot(standardized_residual, dist="norm")
    theoretical_quantiles = qq[0][0]
    ordered_residuals = qq[0][1]
    slope, intercept, r_value = qq[1]
    fitted_line_y = slope * theoretical_quantiles + intercept

    qq_df = pd.DataFrame({
        "x_theoretical_quantiles": theoretical_quantiles,
        "y_ordered_standardized_residuals": ordered_residuals,
        "y_fitted_reference_line": fitted_line_y,
    })
    qq_info_df = pd.DataFrame([{
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
    }])
    return qq_df, qq_info_df


def prepare_scatter_plot_data(time_index, predicted_mean, standardized_residual):
    return pd.DataFrame({
        "time_index": time_index,
        "x_predicted_wind_speed": predicted_mean,
        "y_standardized_residual": standardized_residual,
        "reference_y_0": np.zeros_like(standardized_residual),
        "reference_y_1_96": np.full_like(standardized_residual, 1.96),
        "reference_y_minus_1_96": np.full_like(standardized_residual, -1.96),
    })


def plot_distribution(ax, hist_df, curve_df):
    ax.bar(
        hist_df["bin_center"],
        hist_df["histogram_density"],
        width=hist_df["bin_right"] - hist_df["bin_left"],
        alpha=0.45,
        edgecolor="black",
        label="Histogram",
    )
    ax.plot(
        curve_df["x_standardized_residual"],
        curve_df["kde_density"],
        linewidth=2.0,
        label="KDE",
    )
    ax.plot(
        curve_df["x_standardized_residual"],
        curve_df["standard_normal_density"],
        linestyle="--",
        linewidth=2.0,
        label="Standard normal",
    )
    ax.axvline(0.0, linestyle=":", linewidth=1.5)
    ax.set_xlabel("Standardized residual")
    ax.set_ylabel("Density")
    ax.set_title("(a) Distribution of standardized residuals")
    ax.legend(frameon=False)


def plot_qq(ax, qq_df, qq_info_df):
    ax.scatter(
        qq_df["x_theoretical_quantiles"],
        qq_df["y_ordered_standardized_residuals"],
        s=18,
        alpha=0.75,
    )
    ax.plot(
        qq_df["x_theoretical_quantiles"],
        qq_df["y_fitted_reference_line"],
        linestyle="--",
        linewidth=2.0,
    )
    r_value = qq_info_df.loc[0, "r_value"]
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Ordered standardized residuals")
    ax.set_title(f"(b) Q-Q plot, R = {r_value:.3f}")


def plot_scatter(ax, scatter_df):
    ax.scatter(
        scatter_df["x_predicted_wind_speed"],
        scatter_df["y_standardized_residual"],
        s=18,
        alpha=0.75,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.5)
    ax.axhline(1.96, linestyle=":", linewidth=1.2)
    ax.axhline(-1.96, linestyle=":", linewidth=1.2)
    ax.set_xlabel("Predicted wind speed")
    ax.set_ylabel("Standardized residual")
    ax.set_title("(c) Residuals versus predicted wind speed")


def save_combined_figure(hist_df, curve_df, qq_df, qq_info_df, scatter_df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=COMBINED_FIGURE_SIZE)
    plot_distribution(axes[0], hist_df, curve_df)
    plot_qq(axes[1], qq_df, qq_info_df)
    plot_scatter(axes[2], scatter_df)
    fig.tight_layout()

    png_path = output_dir / f"{OUTPUT_PREFIX}_combined.png"
    tif_path = output_dir / f"{OUTPUT_PREFIX}_combined_600dpi.tif"
    fig.savefig(png_path, dpi=FIGURE_DPI_PNG, bbox_inches="tight")
    fig.savefig(tif_path, dpi=FIGURE_DPI_TIF, bbox_inches="tight")
    plt.close(fig)
    return png_path, tif_path


def save_single_figures(hist_df, curve_df, qq_df, qq_info_df, scatter_df, output_dir):
    paths = {}

    fig, ax = plt.subplots(figsize=SINGLE_FIGURE_SIZE)
    plot_distribution(ax, hist_df, curve_df)
    fig.tight_layout()
    path = output_dir / f"{OUTPUT_PREFIX}_distribution.png"
    fig.savefig(path, dpi=FIGURE_DPI_PNG, bbox_inches="tight")
    plt.close(fig)
    paths["distribution"] = path

    fig, ax = plt.subplots(figsize=SINGLE_FIGURE_SIZE)
    plot_qq(ax, qq_df, qq_info_df)
    fig.tight_layout()
    path = output_dir / f"{OUTPUT_PREFIX}_qq_plot.png"
    fig.savefig(path, dpi=FIGURE_DPI_PNG, bbox_inches="tight")
    plt.close(fig)
    paths["qq"] = path

    fig, ax = plt.subplots(figsize=SINGLE_FIGURE_SIZE)
    plot_scatter(ax, scatter_df)
    fig.tight_layout()
    path = output_dir / f"{OUTPUT_PREFIX}_residual_vs_predicted_wind_speed.png"
    fig.savefig(path, dpi=FIGURE_DPI_PNG, bbox_inches="tight")
    plt.close(fig)
    paths["scatter"] = path
    return paths


def save_excel_outputs(data, diagnostic_data, hist_df, curve_df, qq_df, qq_info_df, scatter_df, output_dir):
    residual_detail_df = pd.DataFrame({
        "time_index": data["time_index"],
        "observed_wind_speed": data["observed"],
        "predicted_mean": data["predicted_mean"],
        "predicted_variance": diagnostic_data["predicted_variance"],
        "predicted_std_used": diagnostic_data["predicted_std"],
        "residual_calculated_observed_minus_mean": diagnostic_data["residual_calculated"],
        "standardized_residual": diagnostic_data["standardized_residual"],
        "negative_probability": diagnostic_data["negative_probability"],
        "lower_95PI": data["lower_95"],
        "upper_95PI": data["upper_95"],
    })
    stats_df = pd.DataFrame([diagnostic_data["stats"]])

    stats_path = output_dir / f"{OUTPUT_PREFIX}_statistics.xlsx"
    plot_data_path = output_dir / f"{OUTPUT_PREFIX}_plot_data.xlsx"
    stats_df.to_excel(stats_path, index=False)

    with pd.ExcelWriter(plot_data_path, engine="openpyxl") as writer:
        residual_detail_df.to_excel(writer, sheet_name="residual_details", index=False)
        stats_df.to_excel(writer, sheet_name="diagnostic_statistics", index=False)
        hist_df.to_excel(writer, sheet_name="fig_a_histogram_data", index=False)
        curve_df.to_excel(writer, sheet_name="fig_a_curve_data", index=False)
        qq_df.to_excel(writer, sheet_name="fig_b_QQ_data", index=False)
        qq_info_df.to_excel(writer, sheet_name="fig_b_QQ_info", index=False)
        scatter_df.to_excel(writer, sheet_name="fig_c_scatter_data", index=False)
    return stats_path, plot_data_path


def main(input_file=INPUT_FILE, output_dir=OUTPUT_DIR):
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = read_prediction_results(input_file)
    diagnostic_data = compute_residual_diagnostics(data)
    standardized_residual = diagnostic_data["standardized_residual"]

    hist_df, curve_df = prepare_distribution_plot_data(
        standardized_residual,
        HISTOGRAM_BINS,
    )
    qq_df, qq_info_df = prepare_qq_plot_data(standardized_residual)
    scatter_df = prepare_scatter_plot_data(
        data["time_index"],
        data["predicted_mean"],
        standardized_residual,
    )

    combined_png_path, combined_tif_path = save_combined_figure(
        hist_df, curve_df, qq_df, qq_info_df, scatter_df, output_dir
    )
    single_paths = save_single_figures(
        hist_df, curve_df, qq_df, qq_info_df, scatter_df, output_dir
    )
    stats_path, plot_data_path = save_excel_outputs(
        data,
        diagnostic_data,
        hist_df,
        curve_df,
        qq_df,
        qq_info_df,
        scatter_df,
        output_dir,
    )

    print("\nResidual diagnostic statistics:")
    print(pd.DataFrame([diagnostic_data["stats"]]).to_string(index=False))
    print("\nFiles saved:")
    for path in [
        combined_png_path,
        combined_tif_path,
        single_paths["distribution"],
        single_paths["qq"],
        single_paths["scatter"],
        stats_path,
        plot_data_path,
    ]:
        print(path)

    print("\nExcel sheets in plot data file:")
    for sheet in [
        "residual_details",
        "diagnostic_statistics",
        "fig_a_histogram_data",
        "fig_a_curve_data",
        "fig_b_QQ_data",
        "fig_b_QQ_info",
        "fig_c_scatter_data",
    ]:
        print(sheet)


if __name__ == "__main__":
    main()
