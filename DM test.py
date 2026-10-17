import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_DIR = Path("path/to/your/predictions")

MODEL_FILES = {
    "P-GWN": DATA_DIR / "P-GWN_predictions.xlsx",
    "P-TM": DATA_DIR / "P-TM_predictions.xlsx",
    "P-LSTM": DATA_DIR / "P-LSTM_predictions.xlsx",
    "P-GRU": DATA_DIR / "P-GRU_predictions.xlsx",
    "QRF": DATA_DIR / "QRF_predictions.xlsx",
    "SGPR": DATA_DIR / "SGPR_predictions.xlsx",
    "P-PM": DATA_DIR / "P-PM_predictions.xlsx",
    "P-ARIMA": DATA_DIR / "P-ARIMA_predictions.xlsx",
    "P-VAR": DATA_DIR / "P-VAR_predictions.xlsx",
}

OUTPUT_DIR = DATA_DIR
OUTPUT_FILE = OUTPUT_DIR / "DM_test_results.xlsx"

SIGNIFICANCE_LEVEL = 0.05
CRITICAL_VALUE_TWO_SIDED_5PCT = 1.959963984540054
EPS_VAR = 1e-12
OBSERVED_ATOL = 1e-3
OBSERVED_RTOL = 1e-7


def normal_two_sided_p_value(z: float) -> float:
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def gaussian_crps_per_sample(
    y_true: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    var = np.asarray(var, dtype=np.float64).reshape(-1)

    if not (len(y_true) == len(mu) == len(var)):
        raise ValueError("y_true, mu and var must have the same length.")
    if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(mu)) and np.all(np.isfinite(var))):
        raise ValueError("Gaussian CRPS inputs contain NaN or Inf.")
    if np.any(var <= 0):
        raise ValueError("Predicted variances must be positive.")

    var = np.maximum(var, EPS_VAR)
    sigma = np.sqrt(var)
    z = (y_true - mu) / sigma
    phi = np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    crps = sigma * (
        z * (2.0 * Phi - 1.0)
        + 2.0 * phi
        - 1.0 / math.sqrt(math.pi)
    )
    return np.asarray(crps, dtype=np.float64)


def newey_west_long_run_variance(d: np.ndarray, lag: int) -> float:
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    t = len(d)
    if t < 2:
        raise ValueError("The loss-differential sequence must contain at least two samples.")
    if lag < 0:
        raise ValueError("HAC lag must be non-negative.")

    lag = min(int(lag), t - 1)
    d_centered = d - np.mean(d)
    gamma_0 = np.sum(d_centered * d_centered) / t
    long_run_variance = gamma_0

    for k in range(1, lag + 1):
        gamma_k = np.sum(d_centered[k:] * d_centered[:-k]) / t
        bartlett_weight = 1.0 - k / (lag + 1.0)
        long_run_variance += 2.0 * bartlett_weight * gamma_k

    return float(long_run_variance)


def automatic_hac_lag(t: int) -> int:
    if t < 1:
        raise ValueError("T must be positive.")
    return int(math.floor(t ** (1.0 / 3.0)))


def dm_test_from_losses(
    loss_proposed: np.ndarray,
    loss_baseline: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    loss_proposed = np.asarray(loss_proposed, dtype=np.float64).reshape(-1)
    loss_baseline = np.asarray(loss_baseline, dtype=np.float64).reshape(-1)

    if len(loss_proposed) != len(loss_baseline):
        raise ValueError("Proposed and baseline loss sequences must have the same length.")
    if len(loss_proposed) < 2:
        raise ValueError("At least two paired loss observations are required.")
    if not (np.all(np.isfinite(loss_proposed)) and np.all(np.isfinite(loss_baseline))):
        raise ValueError("Loss sequences contain NaN or Inf.")

    d = loss_baseline - loss_proposed
    t = len(d)
    lag = automatic_hac_lag(t)
    lrv = newey_west_long_run_variance(d=d, lag=lag)
    mean_d = float(np.mean(d))

    if lrv <= 0.0:
        if np.allclose(d, 0.0):
            dm_signed = 0.0
            p_two_sided = 1.0
        else:
            raise ValueError(
                "Estimated HAC long-run variance is non-positive. "
                "Check the paired loss-differential sequence."
            )
    else:
        dm_signed = mean_d / math.sqrt(lrv / t)
        p_two_sided = normal_two_sided_p_value(dm_signed)

    reject_h0 = p_two_sided < SIGNIFICANCE_LEVEL
    if reject_h0 and dm_signed > 0:
        conclusion = "P-GWN has significantly lower loss"
    elif reject_h0 and dm_signed < 0:
        conclusion = "Baseline has significantly lower loss"
    else:
        conclusion = "No statistically significant difference"

    result = {
        "T": int(t),
        "HAC_lag_floor_T_1_3": int(lag),
        "P-GWN_mean_loss": float(np.mean(loss_proposed)),
        "Baseline_mean_loss": float(np.mean(loss_baseline)),
        "Mean_loss_difference_baseline_minus_P-GWN": mean_d,
        "Long_run_variance": float(lrv),
        "DM_signed": float(dm_signed),
        "P_value_two_sided": float(p_two_sided),
        "Significant_at_5_percent": "Yes" if reject_h0 else "No",
        "Direction": conclusion,
    }
    return result, d


def _canonical_time(series: pd.Series) -> pd.Series:
    values = series.reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").reset_index(drop=True)

    text_values = values.astype(str).str.strip()
    numeric_values = pd.to_numeric(text_values, errors="coerce")
    if numeric_values.notna().all():
        return numeric_values.reset_index(drop=True)

    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().all():
        return parsed.reset_index(drop=True)
    return text_values.reset_index(drop=True)


def load_gaussian_model(model_name: str, file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"{model_name} file not found:\n{file_path}")

    df = pd.read_excel(file_path, engine="openpyxl")
    if df.shape[1] < 4:
        raise ValueError(f"{model_name}: expected at least four columns A-D, but found {df.shape[1]}.")

    out = pd.DataFrame({
        "Time": _canonical_time(df.iloc[:, 0]),
        "Observed": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
        "Predicted_mean": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
        "Predicted_variance": pd.to_numeric(df.iloc[:, 3], errors="coerce"),
    })
    numeric_values = out[["Observed", "Predicted_mean", "Predicted_variance"]].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError(f"{model_name}: non-finite values were found in required columns A-D.")
    if np.any(out["Predicted_variance"].to_numpy(dtype=np.float64) <= 0):
        raise ValueError(f"{model_name}: predicted variances must be positive.")

    error = out["Observed"].to_numpy(dtype=np.float64) - out["Predicted_mean"].to_numpy(dtype=np.float64)
    out["SE_loss"] = error ** 2
    out["CRPS_loss"] = gaussian_crps_per_sample(
        y_true=out["Observed"].to_numpy(dtype=np.float64),
        mu=out["Predicted_mean"].to_numpy(dtype=np.float64),
        var=out["Predicted_variance"].to_numpy(dtype=np.float64),
    )
    return out


def load_qrf_model(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"QRF file not found:\n{file_path}")

    df = pd.read_excel(file_path, engine="openpyxl")
    if df.shape[1] < 11:
        raise ValueError(f"QRF: expected at least 11 columns A-K, but found {df.shape[1]}.")

    crps_column = "Sample_CRPS" if "Sample_CRPS" in df.columns else df.columns[10]
    out = pd.DataFrame({
        "Time": _canonical_time(df.iloc[:, 0]),
        "Observed": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
        "Predicted_mean": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
        "Sample_CRPS": pd.to_numeric(df[crps_column], errors="coerce"),
    })
    numeric_values = out[["Observed", "Predicted_mean", "Sample_CRPS"]].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError("QRF: non-finite values were found in required columns A, B, C or Sample_CRPS.")

    error = out["Observed"].to_numpy(dtype=np.float64) - out["Predicted_mean"].to_numpy(dtype=np.float64)
    out["SE_loss"] = error ** 2
    out["CRPS_loss"] = out["Sample_CRPS"].to_numpy(dtype=np.float64)
    return out


def validate_paired_predictions(
    proposed_df: pd.DataFrame,
    other_df: pd.DataFrame,
    model_name: str,
) -> None:
    if len(other_df) != len(proposed_df):
        raise ValueError(
            f"{model_name}: row count {len(other_df)} does not match "
            f"P-GWN row count {len(proposed_df)}."
        )

    if not proposed_df["Time"].equals(other_df["Time"]):
        mismatch = np.where(
            proposed_df["Time"].astype(str).to_numpy()
            != other_df["Time"].astype(str).to_numpy()
        )[0]
        first = int(mismatch[0]) if len(mismatch) > 0 else "unknown"
        raise ValueError(
            f"{model_name}: Time column is not aligned with P-GWN. "
            f"First mismatched zero-based row: {first}."
        )

    y_proposed = proposed_df["Observed"].to_numpy(dtype=np.float64)
    y_other = other_df["Observed"].to_numpy(dtype=np.float64)
    if not np.allclose(y_proposed, y_other, rtol=OBSERVED_RTOL, atol=OBSERVED_ATOL):
        diff = np.abs(y_proposed - y_other)
        tolerance = OBSERVED_ATOL + OBSERVED_RTOL * np.abs(y_proposed)
        bad = np.where(diff > tolerance)[0]
        first = int(bad[0]) if len(bad) > 0 else int(np.argmax(diff))
        raise ValueError(
            f"{model_name}: Observed values do not match P-GWN within tolerance. "
            f"First mismatched zero-based row: {first}; "
            f"P-GWN={y_proposed[first]}, {model_name}={y_other[first]}, "
            f"abs_diff={diff[first]:.12g}."
        )

    if not np.array_equal(y_proposed, y_other):
        if model_name == "QRF":
            raise ValueError(
                "QRF: Observed values differ from P-GWN. The supplied native "
                "Sample_CRPS cannot be recalculated from the available columns, "
                "so the QRF Observed column must match P-GWN before testing."
            )
        other_df["Observed"] = y_proposed
        error = y_proposed - other_df["Predicted_mean"].to_numpy(dtype=np.float64)
        other_df["SE_loss"] = error ** 2
        other_df["CRPS_loss"] = gaussian_crps_per_sample(
            y_true=y_proposed,
            mu=other_df["Predicted_mean"].to_numpy(dtype=np.float64),
            var=other_df["Predicted_variance"].to_numpy(dtype=np.float64),
        )


def main() -> None:
    print("=" * 79)
    print("DM test: P-GWN vs. eight baseline models")
    print("Losses: squared error + CRPS")
    print("HAC: Newey-West/Bartlett, lag = floor(T^(1/3))")
    print("Inference: signed DM statistic + two-sided p-value, alpha = 0.05")
    print("=" * 79)

    proposed_df = load_gaussian_model("P-GWN", MODEL_FILES["P-GWN"])
    baseline_names = ["P-TM", "P-LSTM", "P-GRU", "QRF", "SGPR", "P-PM", "P-ARIMA", "P-VAR"]
    baseline_data = {}

    for model_name in baseline_names:
        if model_name == "QRF":
            df_model = load_qrf_model(MODEL_FILES[model_name])
        else:
            df_model = load_gaussian_model(model_name, MODEL_FILES[model_name])
        validate_paired_predictions(proposed_df, df_model, model_name)
        baseline_data[model_name] = df_model
        print(f"Loaded and aligned: {model_name} ({len(df_model)} samples)")

    summary_rows = []
    detail_sheets = {}
    proposed_se = proposed_df["SE_loss"].to_numpy(dtype=np.float64)
    proposed_crps = proposed_df["CRPS_loss"].to_numpy(dtype=np.float64)

    for model_name in baseline_names:
        baseline_df = baseline_data[model_name]
        baseline_se = baseline_df["SE_loss"].to_numpy(dtype=np.float64)
        baseline_crps = baseline_df["CRPS_loss"].to_numpy(dtype=np.float64)
        se_result, se_diff = dm_test_from_losses(proposed_se, baseline_se)
        crps_result, crps_diff = dm_test_from_losses(proposed_crps, baseline_crps)

        summary_rows.append({
            "Baseline_model": model_name,
            "SE_T": se_result["T"],
            "SE_HAC_lag_floor_T_1_3": se_result["HAC_lag_floor_T_1_3"],
            "SE_P-GWN_mean_loss": se_result["P-GWN_mean_loss"],
            "SE_Baseline_mean_loss": se_result["Baseline_mean_loss"],
            "SE_Mean_loss_diff_baseline_minus_P-GWN": se_result["Mean_loss_difference_baseline_minus_P-GWN"],
            "SE_Long_run_variance": se_result["Long_run_variance"],
            "SE_DM_signed": se_result["DM_signed"],
            "SE_P_value_two_sided": se_result["P_value_two_sided"],
            "SE_Significant_at_5_percent": se_result["Significant_at_5_percent"],
            "SE_Direction": se_result["Direction"],
            "CRPS_T": crps_result["T"],
            "CRPS_HAC_lag_floor_T_1_3": crps_result["HAC_lag_floor_T_1_3"],
            "CRPS_P-GWN_mean_loss": crps_result["P-GWN_mean_loss"],
            "CRPS_Baseline_mean_loss": crps_result["Baseline_mean_loss"],
            "CRPS_Mean_loss_diff_baseline_minus_P-GWN": crps_result["Mean_loss_difference_baseline_minus_P-GWN"],
            "CRPS_Long_run_variance": crps_result["Long_run_variance"],
            "CRPS_DM_signed": crps_result["DM_signed"],
            "CRPS_P_value_two_sided": crps_result["P_value_two_sided"],
            "CRPS_Significant_at_5_percent": crps_result["Significant_at_5_percent"],
            "CRPS_Direction": crps_result["Direction"],
        })

        detail_sheets[model_name] = pd.DataFrame({
            "Time": proposed_df["Time"],
            "Observed": proposed_df["Observed"],
            "P-GWN_predicted_mean": proposed_df["Predicted_mean"],
            f"{model_name}_predicted_mean": baseline_df["Predicted_mean"],
            "P-GWN_SE_loss": proposed_se,
            f"{model_name}_SE_loss": baseline_se,
            "SE_loss_diff_baseline_minus_P-GWN": se_diff,
            "P-GWN_CRPS_loss": proposed_crps,
            f"{model_name}_CRPS_loss": baseline_crps,
            "CRPS_loss_diff_baseline_minus_P-GWN": crps_diff,
        })

    summary_df = pd.DataFrame(summary_rows)
    configuration_df = pd.DataFrame({
        "Setting": [
            "Reference model",
            "Baseline models",
            "Point-accuracy loss",
            "Probabilistic loss",
            "QRF CRPS definition",
            "Gaussian-model CRPS definition",
            "Loss differential",
            "Observed-value handling",
            "DM sign interpretation",
            "HAC estimator",
            "HAC kernel",
            "HAC truncation lag",
            "P-value",
            "Significance level",
            "Two-sided critical value at 5%",
        ],
        "Value": [
            "P-GWN",
            ", ".join(baseline_names),
            "Squared error: (Observed - Predicted_mean)^2",
            "Sample-wise CRPS",
            "Native weighted-empirical QRF CRPS read from Sample_CRPS",
            "Gaussian CRPS from Observed, Predicted_mean and Predicted_variance",
            "d_t = Loss_baseline,t - Loss_P-GWN,t",
            "Losses use the P-GWN target series; small tolerated differences in Gaussian-model files are replaced by the common target. QRF native CRPS requires matching observed values.",
            "Positive signed DM favors P-GWN; negative signed DM favors the baseline",
            "Newey-West long-run variance estimator",
            "Bartlett",
            "floor(T^(1/3)), computed separately from the paired sample size T",
            "Two-sided standard-normal p-value",
            "0.05",
            str(CRITICAL_VALUE_TWO_SIDED_5PCT),
        ],
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="DM_summary", index=False)
        configuration_df.to_excel(writer, sheet_name="Configuration", index=False)
        for model_name, detail_df in detail_sheets.items():
            detail_df.to_excel(writer, sheet_name=f"Detail_{model_name}"[:31], index=False)

    compact_cols = [
        "Baseline_model",
        "SE_HAC_lag_floor_T_1_3",
        "SE_DM_signed",
        "SE_P_value_two_sided",
        "SE_Significant_at_5_percent",
        "CRPS_HAC_lag_floor_T_1_3",
        "CRPS_DM_signed",
        "CRPS_P_value_two_sided",
        "CRPS_Significant_at_5_percent",
    ]
    print("\nDM test summary")
    print(summary_df[compact_cols].to_string(index=False))
    print("\nSign convention: d_t = Loss_baseline,t - Loss_P-GWN,t")
    print("Positive signed DM -> P-GWN has lower loss.")
    print("Negative signed DM -> baseline has lower loss.")
    print("Statistical significance is based on the two-sided p-value < 0.05.")
    print(f"\nSaved results to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
