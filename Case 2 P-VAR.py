from pathlib import Path

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.api import VAR
except ImportError as exc:
    raise ImportError(
        "statsmodels is required for the VAR baseline. "
        "Install it with: pip install statsmodels"
    ) from exc


# Configuration

class Config:
    data_dir = Path("path/to/your/data")

    train_file = "train.xlsx"
    val_file = "validation.xlsx"
    test_file = "test.xlsx"
    output_dir = data_dir / "results_var"

    rolling_stride = 1
    forecast_horizon = 1

    time_column_index = 0
    input_start_column_index = 1
    input_end_column_index = 57
    target_column_index = 57
    target_column_name = "BF"

    num_nodes = 9
    num_ecmwf_groups = 6

    var_lag_order = 2
    var_trend = "c"
    update_with_observations = True

    eps_var = 1e-6
    z_value = 1.959963984540054


cfg = Config()


# Data loading and preprocessing

def read_station_data(file_path, cfg):
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_excel(file_path, engine="openpyxl")
    required_columns = max(
        cfg.time_column_index,
        cfg.input_end_column_index - 1,
        cfg.target_column_index,
    ) + 1

    if df.shape[1] < required_columns:
        raise ValueError(
            f"{file_path} contains {df.shape[1]} columns; "
            f"at least {required_columns} are required."
        )

    time = df.iloc[:, cfg.time_column_index].to_numpy()
    x_raw = df.iloc[
        :, cfg.input_start_column_index:cfg.input_end_column_index
    ].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    target = pd.to_numeric(
        df.iloc[:, cfg.target_column_index], errors="raise"
    ).to_numpy(dtype=np.float64).reshape(-1)

    if len(time) != len(target) or len(x_raw) != len(target):
        raise ValueError(f"Inconsistent row counts in {file_path}.")
    if target.size == 0:
        raise ValueError(f"No samples found in {file_path}.")
    if x_raw.shape[1] != 56:
        raise ValueError(
            f"{file_path} must contain 56 input columns between B and BE; "
            f"found {x_raw.shape[1]}."
        )
    if not np.isfinite(x_raw).all() or not np.isfinite(target).all():
        raise ValueError(f"Non-finite inputs or target values found in {file_path}.")

    endog, variable_names = build_var_endog(
        x_raw=x_raw,
        target=target,
        num_nodes=cfg.num_nodes,
        num_ecmwf_groups=cfg.num_ecmwf_groups,
    )

    return {
        "time": time,
        "target": target,
        "inputs": x_raw,
        "endog": endog,
        "sample_count": len(target),
        "variable_names": variable_names,
    }


def build_var_endog(x_raw, target, num_nodes=9, num_ecmwf_groups=6):
    x_raw = np.asarray(x_raw, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    expected_ecmwf_columns = num_nodes * num_ecmwf_groups
    if x_raw.ndim != 2 or x_raw.shape[1] != expected_ecmwf_columns + 2:
        raise ValueError(
            "VAR expects 54 ECMWF columns arranged as 6 groups of 9 grid nodes, "
            "followed by two historical ANE features."
        )
    if x_raw.shape[0] != target.size:
        raise ValueError("Input and target sample counts do not match.")

    ecmwf_group_means = []
    for group_index in range(num_ecmwf_groups):
        start = group_index * num_nodes
        end = start + num_nodes
        group_mean = np.mean(x_raw[:, start:end], axis=1, keepdims=True)
        ecmwf_group_means.append(group_mean)

    ecmwf_means = np.concatenate(ecmwf_group_means, axis=1)
    historical_features = x_raw[:, expected_ecmwf_columns:expected_ecmwf_columns + 2]
    endog = np.column_stack([target, ecmwf_means, historical_features])

    variable_names = [
        "Target_wind_speed",
        "ECMWF_gust_10m_mean_9nodes",
        "ECMWF_mean_10m_mean_9nodes",
        "ECMWF_temp_800hPa_mean_9nodes",
        "ECMWF_temp_850hPa_mean_9nodes",
        "ECMWF_wind_800hPa_mean_9nodes",
        "ECMWF_wind_850hPa_mean_9nodes",
        "ANE_t_minus_2_BD",
        "ANE_t_minus_1_BE",
    ]

    return endog, variable_names


def prepare_data(cfg):
    if cfg.forecast_horizon != 1 or cfg.rolling_stride != 1:
        raise ValueError(
            "The Case II VAR baseline requires hourly rolling "
            "one-step forecasting."
        )

    split_files = {
        "Train": cfg.data_dir / cfg.train_file,
        "Validation": cfg.data_dir / cfg.val_file,
        "Test": cfg.data_dir / cfg.test_file,
    }
    split_data = {
        name: read_station_data(path, cfg)
        for name, path in split_files.items()
    }

    reference_names = split_data["Train"]["variable_names"]
    for split_name, data in split_data.items():
        if data["variable_names"] != reference_names:
            raise ValueError(
                f"Endogenous-variable definitions differ in {split_name} data."
            )

    return split_data


# Probabilistic VAR baseline

class ProbabilisticVAR:
    def __init__(self, cfg):
        self.cfg = cfg
        self.result = None
        self.aic = None
        self.bic = None
        self.hqic = None
        self.fpe = None
        self.neqs = None
        self.k_ar = None
        self.one_step_target_var = None

    def fit(self, endog_train):
        endog_train = np.asarray(endog_train, dtype=np.float64)
        if endog_train.ndim != 2 or endog_train.shape[0] <= self.cfg.var_lag_order:
            raise ValueError(
                "The training series must have more rows than the requested VAR lag order."
            )
        if not np.isfinite(endog_train).all():
            raise ValueError("Non-finite values found in the training endogenous series.")

        model = VAR(endog_train)
        self.result = model.fit(
            maxlags=self.cfg.var_lag_order,
            ic=None,
            trend=self.cfg.var_trend,
        )

        self.aic = float(self.result.aic)
        self.bic = float(self.result.bic)
        self.hqic = float(self.result.hqic)
        self.fpe = float(self.result.fpe)
        self.neqs = int(self.result.neqs)
        self.k_ar = int(self.result.k_ar)

        forecast_covariance = np.asarray(
            self.result.forecast_cov(steps=1),
            dtype=np.float64,
        )
        target_variance = forecast_covariance[0, 0, 0]
        self.one_step_target_var = max(
            float(target_variance),
            self.cfg.eps_var,
        )
        return self

    def _forecast_one_step_from_history(self, history):
        if self.result is None:
            raise RuntimeError("The VAR model must be fitted before prediction.")

        history = np.asarray(history, dtype=np.float64)
        if history.ndim != 2 or history.shape[1] != self.neqs:
            raise ValueError(
                f"VAR history must have shape [time, {self.neqs}]."
            )
        if history.shape[0] < self.k_ar:
            raise ValueError(
                f"At least {self.k_ar} rows are required for VAR forecasting; "
                f"received {history.shape[0]}."
            )

        forecast = self.result.forecast(
            y=history[-self.k_ar:],
            steps=1,
        )
        mean = float(forecast[0, 0])
        variance = float(self.one_step_target_var)
        return mean, variance

    def rolling_one_step_forecast(self, initial_history, eval_endog):
        if self.result is None:
            raise RuntimeError("The VAR model must be fitted before prediction.")

        current_history = np.asarray(initial_history, dtype=np.float64).copy()
        eval_endog = np.asarray(eval_endog, dtype=np.float64)

        if current_history.ndim != 2 or current_history.shape[1] != self.neqs:
            raise ValueError("Initial history has an incompatible shape.")
        if eval_endog.ndim != 2 or eval_endog.shape[1] != self.neqs:
            raise ValueError("Evaluation data have an incompatible shape.")
        if not np.isfinite(current_history).all() or not np.isfinite(eval_endog).all():
            raise ValueError("Non-finite values found in VAR forecasting data.")

        observed_values = []
        means = []
        variances = []

        for observed_row in eval_endog:
            mean, variance = self._forecast_one_step_from_history(current_history)
            observed_values.append(float(observed_row[0]))
            means.append(mean)
            variances.append(variance)

            if self.cfg.update_with_observations:
                current_history = np.vstack([
                    current_history,
                    observed_row.reshape(1, -1),
                ])

        return (
            np.asarray(observed_values, dtype=np.float64),
            np.asarray(means, dtype=np.float64),
            np.asarray(variances, dtype=np.float64),
        )

    def rolling_train_forecast(self, train_endog):
        if self.result is None:
            raise RuntimeError("The VAR model must be fitted before prediction.")

        train_endog = np.asarray(train_endog, dtype=np.float64)
        initial_history = train_endog[:self.k_ar]
        evaluation_rows = train_endog[self.k_ar:]

        return self.rolling_one_step_forecast(
            initial_history=initial_history,
            eval_endog=evaluation_rows,
        )


# Output

def save_predictions(output_dir, split_name, time, observed, mean, variance, cfg):
    time = np.asarray(time).reshape(-1)
    observed = np.asarray(observed, dtype=np.float64).reshape(-1)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    variance = np.maximum(
        np.asarray(variance, dtype=np.float64).reshape(-1),
        cfg.eps_var,
    )

    if not (
        len(time) == len(observed) == len(mean) == len(variance)
    ):
        raise ValueError(
            f"Inconsistent prediction lengths for the {split_name} split."
        )

    std = np.sqrt(variance)
    lower = mean - cfg.z_value * std
    upper = mean + cfg.z_value * std

    prediction_df = pd.DataFrame({
        "Time": time,
        "Observed": observed,
        "Predicted_mean": mean,
        "Predicted_variance": variance,
        "Predicted_std": std,
        "Lower_95PI": lower,
        "Upper_95PI": upper,
        "Residual": observed - mean,
    })
    prediction_path = output_dir / f"{split_name}_prediction_results.xlsx"
    prediction_df.to_excel(prediction_path, index=False)

    print(
        f"{split_name}: saved {len(prediction_df)} sample-level predictions "
        f"to {prediction_path}"
    )
    return variance


def train_model(cfg):
    split_data = prepare_data(cfg)
    train_endog = split_data["Train"]["endog"]
    val_endog = split_data["Validation"]["endog"]
    test_endog = split_data["Test"]["endog"]

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    model = ProbabilisticVAR(cfg).fit(train_endog)

    model_info_df = pd.DataFrame([{
        "Model": "VAR",
        "Lag_order_requested": cfg.var_lag_order,
        "Lag_order_used": model.k_ar,
        "Trend": cfg.var_trend,
        "Num_endogenous_variables": model.neqs,
        "AIC": model.aic,
        "BIC": model.bic,
        "HQIC": model.hqic,
        "FPE": model.fpe,
        "Rolling_stride_h": cfg.rolling_stride,
        "Forecast_horizon_h": cfg.forecast_horizon,
        "Update_with_observations": cfg.update_with_observations,
        "Parameter_fit_data": "Training set only",
        "Refit_after_training": False,
        "Target_one_step_predictive_variance": model.one_step_target_var,
        "Variance_definition": "Native VAR one-step forecast variance",
    }])
    model_info_df.to_excel(
        cfg.output_dir / "var_model_info.xlsx",
        index=False,
    )

    variable_info_df = pd.DataFrame({
        "Variable_index": np.arange(len(split_data["Train"]["variable_names"])),
        "Variable_name": split_data["Train"]["variable_names"],
    })
    variable_info_df.to_excel(
        cfg.output_dir / "var_endogenous_variables.xlsx",
        index=False,
    )

    fit_summary_df = pd.DataFrame([{
        "Model": "VAR",
        "Training_samples": len(train_endog),
        "Num_endogenous_variables": model.neqs,
        "Lag_order": model.k_ar,
        "Parameter_fit_data": "Training set only",
        "Forecast_protocol": "Hourly rolling one-step ahead",
        "State_update": "Append each realized multivariate row after forecasting",
    }])
    fit_summary_df.to_excel(
        cfg.output_dir / "model_fit_summary.xlsx",
        index=False,
    )

    train_observed, train_mean, train_variance = model.rolling_train_forecast(
        train_endog
    )
    train_time = split_data["Train"]["time"][model.k_ar:]

    val_observed, val_mean, val_variance = model.rolling_one_step_forecast(
        initial_history=train_endog,
        eval_endog=val_endog,
    )
    test_initial_history = np.vstack([train_endog, val_endog])
    test_observed, test_mean, test_variance = model.rolling_one_step_forecast(
        initial_history=test_initial_history,
        eval_endog=test_endog,
    )

    split_outputs = [
        (
            "Train",
            train_observed,
            train_mean,
            train_variance,
            train_time,
        ),
        (
            "Validation",
            val_observed,
            val_mean,
            val_variance,
            split_data["Validation"]["time"],
        ),
        (
            "Test",
            test_observed,
            test_mean,
            test_variance,
            split_data["Test"]["time"],
        ),
    ]

    variance_diagnostics = []
    for split_name, observed, mean, variance, time in split_outputs:
        variance = save_predictions(
            cfg.output_dir,
            split_name,
            time,
            observed,
            mean,
            variance,
            cfg,
        )

        if split_name in ("Validation", "Test"):
            variance_diagnostics.append({
                "Split": split_name,
                "Variance_mean": float(np.mean(variance)),
                "Variance_std": float(np.std(variance)),
                "Variance_min": float(np.min(variance)),
                "Variance_max": float(np.max(variance)),
                "Forecast_horizon_h": cfg.forecast_horizon,
            })

    pd.DataFrame(variance_diagnostics).to_excel(
        cfg.output_dir / "var_one_step_variance_diagnostic.xlsx",
        index=False,
    )

    protocol_df = pd.DataFrame([{
        "Model": "Standard VAR",
        "Lag_order": model.k_ar,
        "Rolling_stride_h": cfg.rolling_stride,
        "Forecast_horizon_h": cfg.forecast_horizon,
        "Num_endogenous_variables": model.neqs,
        "Forecast_protocol": "Hourly rolling one-step forecast",
        "History_update": "Append observed multivariate row after its forecast",
        "Parameter_fit_data": "Training set only",
        "Parameter_refit_after_training": False,
        "Predictive_variance": "Native VAR h=1 forecast variance; fixed across origins",
        "Target_series": f"Target column {cfg.target_column_name}",
        "ECMWF_handling": (
            "Six nine-grid-point spatial means are included as endogenous series; "
            "the forecast uses their lagged values"
        ),
        "Historical_ANE_handling": (
            "ANE_(t-2) and ANE_(t-1) columns are included as endogenous series"
        ),
    }])
    protocol_df.to_excel(
        cfg.output_dir / "forecast_protocol.xlsx",
        index=False,
    )

    input_info_df = pd.DataFrame({
        "Item": [
            "Model",
            "Target series",
            "Target source column",
            "Endogenous variables",
            "ECMWF representation",
            "Historical ANE features",
            "Rolling stride (h)",
            "Forecast horizon (h)",
            "Probabilistic output",
            "Parameter estimation",
            "State update",
        ],
        "Value": [
            "VAR",
            "Hourly ANE target series",
            cfg.target_column_name,
            "Target + six ECMWF group means + two historical ANE features",
            "Each of six ECMWF variables is averaged over nine grid points",
            "ANE_(t-2) and ANE_(t-1)",
            cfg.rolling_stride,
            cfg.forecast_horizon,
            "Predictive mean and native one-step predictive variance",
            "Training multivariate series only",
            "Append each realized multivariate row after forecasting, without refitting",
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
