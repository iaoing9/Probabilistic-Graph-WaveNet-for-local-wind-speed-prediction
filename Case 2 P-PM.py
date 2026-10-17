from pathlib import Path

import numpy as np
import pandas as pd


# Configuration

class Config:
    data_dir = Path("path/to/your/data")

    train_file = "train.xlsx"
    val_file = "validation.xlsx"
    test_file = "test.xlsx"
    output_dir = data_dir / "results_pm"

    rolling_stride = 1
    forecast_horizon = 1

    time_column_index = 0
    persistence_column_index = 56
    persistence_column_name = "BE"
    target_column_index = 57
    target_column_name = "BF"

    eps_var = 1e-6
    z_value = 1.959963984540054


cfg = Config()


# Data loading and preprocessing

def read_prediction_data(file_path, cfg):
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_excel(file_path, engine="openpyxl")
    required_columns = max(
        cfg.time_column_index,
        cfg.persistence_column_index,
        cfg.target_column_index,
    ) + 1

    if df.shape[1] < required_columns:
        raise ValueError(
            f"{file_path} contains {df.shape[1]} columns; "
            f"at least {required_columns} are required."
        )

    time = df.iloc[:, cfg.time_column_index].to_numpy()
    persistence_mean = pd.to_numeric(
        df.iloc[:, cfg.persistence_column_index], errors="raise"
    ).to_numpy(dtype=np.float64).reshape(-1)
    observed = pd.to_numeric(
        df.iloc[:, cfg.target_column_index], errors="raise"
    ).to_numpy(dtype=np.float64).reshape(-1)

    if len(time) != len(persistence_mean) or len(time) != len(observed):
        raise ValueError(f"Inconsistent row counts in {file_path}.")
    if len(observed) == 0:
        raise ValueError(f"No samples found in {file_path}.")
    if not np.isfinite(persistence_mean).all() or not np.isfinite(observed).all():
        raise ValueError(
            f"Non-finite persistence inputs or targets found in {file_path}."
        )

    return {
        "time": time,
        "mean": persistence_mean,
        "observed": observed,
        "sample_count": len(observed),
    }


# Probabilistic persistence baseline

class ProbabilisticPersistence:
    def __init__(self, eps_var):
        self.eps_var = eps_var
        self.one_step_var = None

    def fit(self, observed, persistence_mean):
        observed = np.asarray(observed, dtype=np.float64).reshape(-1)
        persistence_mean = np.asarray(
            persistence_mean, dtype=np.float64
        ).reshape(-1)

        if observed.size != persistence_mean.size:
            raise ValueError(
                "Training targets and persistence inputs have different sample counts."
            )
        if observed.size == 0:
            raise ValueError("The training set is empty.")

        residual = observed - persistence_mean
        if residual.size > 1:
            variance = np.var(residual, ddof=1)
        else:
            variance = np.var(residual, ddof=0)

        self.one_step_var = float(max(variance, self.eps_var))
        return self

    def predict(self, observed, persistence_mean):
        if self.one_step_var is None:
            raise RuntimeError(
                "The probabilistic persistence baseline must be fitted before prediction."
            )

        observed = np.asarray(observed, dtype=np.float64).reshape(-1)
        mean = np.asarray(persistence_mean, dtype=np.float64).reshape(-1)

        if observed.size != mean.size:
            raise ValueError(
                "Prediction targets and persistence inputs have different sample counts."
            )

        variance = np.full(
            observed.shape,
            self.one_step_var,
            dtype=np.float64,
        )
        return observed, mean, variance


# Output

def train_model(cfg):
    if cfg.rolling_stride != 1 or cfg.forecast_horizon != 1:
        raise ValueError(
            "The Case II persistence baseline requires hourly rolling "
            "one-step forecasting."
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    split_files = {
        "Train": cfg.data_dir / cfg.train_file,
        "Validation": cfg.data_dir / cfg.val_file,
        "Test": cfg.data_dir / cfg.test_file,
    }
    split_data = {
        name: read_prediction_data(path, cfg)
        for name, path in split_files.items()
    }

    train_data = split_data["Train"]
    model = ProbabilisticPersistence(cfg.eps_var).fit(
        train_data["observed"],
        train_data["mean"],
    )

    variance_summary = pd.DataFrame([{
        "Forecast_horizon_h": cfg.forecast_horizon,
        "Persistence_error_variance": model.one_step_var,
        "Persistence_error_std": np.sqrt(model.one_step_var),
        "Persistence_input_column": cfg.persistence_column_name,
        "Persistence_input_column_index": cfg.persistence_column_index,
        "Target_column": cfg.target_column_name,
        "Target_column_index": cfg.target_column_index,
        "Persistence_mean_definition": "ANE_(t-1)",
        "Variance_estimation_method": "Sample variance of training residuals (ddof=1)",
        "Variance_estimation_data": "Training set only",
        "Training_samples": train_data["sample_count"],
    }])
    variance_summary.to_excel(
        cfg.output_dir / "persistence_one_step_variance.xlsx",
        index=False,
    )

    fit_summary = pd.DataFrame([{
        "Model": "Probabilistic persistence baseline",
        "Training_samples": train_data["sample_count"],
        "Prediction_mean": "ANE_(t-1), read directly from column BE",
        "Prediction_variance": model.one_step_var,
        "Variance_estimation_method": "Sample variance of training residuals (ddof=1)",
        "Variance_floor": cfg.eps_var,
        "Forecast_horizon_h": cfg.forecast_horizon,
        "Rolling_stride_h": cfg.rolling_stride,
    }])
    fit_summary.to_excel(
        cfg.output_dir / "model_fit_summary.xlsx",
        index=False,
    )

    for split_name, data in split_data.items():
        observed, mean, variance = model.predict(
            data["observed"],
            data["mean"],
        )

        std = np.sqrt(np.maximum(variance, cfg.eps_var))
        lower = mean - cfg.z_value * std
        upper = mean + cfg.z_value * std

        prediction_df = pd.DataFrame({
            "Time": data["time"],
            "Observed": observed,
            "Predicted_mean": mean,
            "Predicted_variance": variance,
            "Predicted_std": std,
            "Lower_95PI": lower,
            "Upper_95PI": upper,
            "Residual": observed - mean,
            "Persistence_input_column": cfg.persistence_column_name,
        })
        prediction_path = (
            cfg.output_dir / f"{split_name}_prediction_results.xlsx"
        )
        prediction_df.to_excel(prediction_path, index=False)
        print(
            f"{split_name}: saved {len(prediction_df)} sample-level predictions "
            f"to {prediction_path}"
        )

    protocol_df = pd.DataFrame([{
        "Model": "Probabilistic persistence baseline",
        "Rolling_stride_h": cfg.rolling_stride,
        "Forecast_horizon_h": cfg.forecast_horizon,
        "Prediction_mean": "ANE_(t-1), read from column BE",
        "Prediction_variance": "Fixed one-step variance estimated from training residuals",
        "Variance_estimation_data": "Training set only",
        "Temporal_window": "None; each hourly row is one forecasting origin",
        "Input_columns_used_for_prediction_mean": cfg.persistence_column_name,
        "Target_column": cfg.target_column_name,
    }])
    protocol_df.to_excel(
        cfg.output_dir / "forecast_protocol.xlsx",
        index=False,
    )

    input_info_df = pd.DataFrame({
        "Item": [
            "Model",
            "Input used for predictive mean",
            "Input source column",
            "Target",
            "Target source column",
            "Rolling stride (h)",
            "Forecast horizon (h)",
            "Probabilistic output",
            "Variance estimation",
            "Variance estimation split",
        ],
        "Value": [
            "Probabilistic persistence baseline",
            "Latest available ANE observation, ANE_(t-1)",
            cfg.persistence_column_name,
            "Current-hour ANE observation, ANE_t",
            cfg.target_column_name,
            cfg.rolling_stride,
            cfg.forecast_horizon,
            "Predictive mean and fixed observation variance",
            "Sample variance of one-step training residuals",
            "Training set only",
        ],
    })
    input_info_df.to_excel(
        cfg.output_dir / "model_input_information.xlsx",
        index=False,
    )

    print(f"All outputs saved to: {cfg.output_dir}")
    return model


if __name__ == "__main__":
    train_model(cfg)
