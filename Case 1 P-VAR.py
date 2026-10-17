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
    output_dir_name = "results_var"

    samples_per_hour = 6
    horizon_steps = [1, 6, 12, 24]
    max_horizon_step = max(horizon_steps)
    input_window_hours = 24
    input_len = input_window_hours * samples_per_hour
    max_lead_steps = max(horizon_steps) * samples_per_hour
    stride = 1
    window_alignment = "same_input_window"

    var_lag_order = 48
    var_trend = "c"
    eps_var = 1e-6
    z_value = 1.959963984540054

    @property
    def output_dir(self):
        return self.data_dir / self.output_dir_name


cfg = Config()


# Data loading and preprocessing

def read_input_excel(file_path):
    df = pd.read_excel(file_path, engine="openpyxl")
    if df.shape[1] < 29:
        raise ValueError(
            f"{file_path} has {df.shape[1]} columns, but at least 29 columns A:AC are required."
        )

    time = df.iloc[:, 0].to_numpy()
    ecmwf_raw = df.iloc[:, 1:28].astype(np.float64).to_numpy()
    observed = df.iloc[:, 28].astype(np.float64).to_numpy()

    if not np.isfinite(ecmwf_raw).all():
        raise ValueError(f"The ECMWF columns in {file_path} contain missing or non-finite values.")
    if not np.isfinite(observed).all():
        raise ValueError(f"The observed wind-speed column in {file_path} contains missing or non-finite values.")

    endog, variable_names = build_var_endog(ecmwf_raw, observed)
    return time, endog, variable_names


def build_var_endog(ecmwf_raw, observed):
    if ecmwf_raw.ndim != 2 or ecmwf_raw.shape[1] != 27:
        raise ValueError(
            f"Expected 27 ECMWF columns (B:AB), but received shape {ecmwf_raw.shape}."
        )

    ecmwf_means = np.column_stack([
        ecmwf_raw[:, 0:9].mean(axis=1),
        ecmwf_raw[:, 9:18].mean(axis=1),
        ecmwf_raw[:, 18:27].mean(axis=1),
    ])
    endog = np.column_stack([observed, ecmwf_means]).astype(np.float64)

    variable_names = [
        "Target_observed_wind_speed",
        "ECMWF_10m_mean_wind_speed_spatial_average",
        "ECMWF_100m_mean_wind_speed_spatial_average",
        "ECMWF_10m_gust_wind_speed_spatial_average",
    ]
    return endog, variable_names


def load_raw_splits(cfg):
    paths = {
        "Train": cfg.data_dir / cfg.train_file,
        "Validation": cfg.data_dir / cfg.val_file,
        "Test": cfg.data_dir / cfg.test_file,
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Set Config.data_dir to the folder containing train.xlsx, "
            "validation.xlsx, and test.xlsx. Missing files:\n" + missing_text
        )

    raw_data = {}
    variable_names = None

    for split_name, path in paths.items():
        time, endog, current_names = read_input_excel(path)
        if variable_names is None:
            variable_names = current_names
        elif current_names != variable_names:
            raise ValueError("The endogenous variable definitions differ across input files.")

        raw_data[split_name] = {
            "time": time,
            "endog": endog,
            "y_raw": endog[:, 0],
            "raw_size": len(endog),
        }

    raw_data["Endog_names"] = variable_names
    return raw_data


def build_sample_metadata(
    time,
    sample_count,
    input_len,
    lead_steps,
    max_lead_steps,
    stride=1,
    alignment="same_input_window",
):
    if sample_count != len(time):
        raise ValueError("sample_count and time must have the same length.")
    if input_len <= 0 or lead_steps <= 0 or max_lead_steps <= 0 or stride <= 0:
        raise ValueError("Window lengths, lead times, and stride must be positive.")
    if lead_steps > max_lead_steps:
        raise ValueError("lead_steps must not exceed max_lead_steps.")
    if sample_count < input_len + max_lead_steps:
        raise ValueError(
            f"Split length {sample_count} is too short for input_len={input_len} "
            f"and max_lead_steps={max_lead_steps}."
        )

    input_start_times = []
    input_end_times = []
    target_times = []
    input_start_indices = []
    input_end_indices = []
    target_indices = []

    if alignment == "same_input_window":
        max_start = sample_count - input_len - max_lead_steps
        starts = np.arange(0, max_start + 1, stride, dtype=np.int64)

        for start in starts:
            end = start + input_len - 1
            target_idx = start + input_len + lead_steps - 1
            input_start_times.append(time[start])
            input_end_times.append(time[end])
            target_times.append(time[target_idx])
            input_start_indices.append(start)
            input_end_indices.append(end)
            target_indices.append(target_idx)

    elif alignment == "same_target_time":
        target_start = input_len + max_lead_steps - 1
        targets = np.arange(target_start, sample_count, stride, dtype=np.int64)

        for target_idx in targets:
            end = target_idx - lead_steps
            start = end - input_len + 1
            if start < 0:
                continue

            input_start_times.append(time[start])
            input_end_times.append(time[end])
            target_times.append(time[target_idx])
            input_start_indices.append(start)
            input_end_indices.append(end)
            target_indices.append(target_idx)
    else:
        raise ValueError(
            "window_alignment must be 'same_input_window' or 'same_target_time'."
        )

    if not target_indices:
        raise ValueError("No samples were generated. Check split length and window settings.")

    return {
        "input_start_time": np.asarray(input_start_times),
        "input_end_time": np.asarray(input_end_times),
        "target_time": np.asarray(target_times),
        "input_start_index": np.asarray(input_start_indices, dtype=np.int64),
        "input_end_index": np.asarray(input_end_indices, dtype=np.int64),
        "target_index": np.asarray(target_indices, dtype=np.int64),
    }


def prepare_metadata_for_horizon(cfg, raw_data, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour
    sample_metas = {}

    for split_name in ["Train", "Validation", "Test"]:
        data = raw_data[split_name]
        sample_metas[split_name] = build_sample_metadata(
            time=data["time"],
            sample_count=len(data["endog"]),
            input_len=cfg.input_len,
            lead_steps=lead_steps,
            max_lead_steps=cfg.max_lead_steps,
            stride=cfg.stride,
            alignment=cfg.window_alignment,
        )

    return sample_metas


# Probabilistic VAR baseline

class ProbabilisticVAR:
    def __init__(self, lag_order=48, trend="c", eps_var=1e-6, target_index=0):
        self.lag_order = lag_order
        self.trend = trend
        self.eps_var = eps_var
        self.target_index = target_index
        self.result = None
        self.aic = None
        self.bic = None
        self.hqic = None
        self.fpe = None
        self.neqs = None
        self.k_ar = None
        self.forecast_var_cache = {}

    def fit(self, endog_train):
        endog_train = np.asarray(endog_train, dtype=np.float64)
        if endog_train.ndim != 2:
            raise ValueError("The VAR training data must be a two-dimensional array.")
        if endog_train.shape[0] <= self.lag_order:
            raise ValueError(
                f"Training data must contain more than {self.lag_order} observations."
            )
        if not np.isfinite(endog_train).all():
            raise ValueError("The VAR training data contain missing or non-finite values.")

        model = VAR(endog_train)
        self.result = model.fit(
            maxlags=self.lag_order,
            ic=None,
            trend=self.trend,
        )

        self.aic = float(self.result.aic)
        self.bic = float(self.result.bic)
        self.hqic = float(self.result.hqic)
        self.fpe = float(self.result.fpe)
        self.neqs = int(self.result.neqs)
        self.k_ar = int(self.result.k_ar)
        return self

    def _target_forecast_variance(self, steps):
        if steps not in self.forecast_var_cache:
            covariance = self.result.forecast_cov(steps=steps)
            variance = float(
                covariance[steps - 1, self.target_index, self.target_index]
            )
            self.forecast_var_cache[steps] = max(variance, self.eps_var)
        return self.forecast_var_cache[steps]

    def _forecast_from_history(self, history, steps):
        if self.result is None:
            raise RuntimeError("The VAR model must be fitted before prediction.")

        history = np.asarray(history, dtype=np.float64)
        if history.ndim != 2 or history.shape[1] != self.neqs:
            raise ValueError(
                f"Forecast history must have shape (n, {self.neqs})."
            )
        if history.shape[0] < self.k_ar:
            raise ValueError(
                f"At least {self.k_ar} observations are required for VAR forecasting, "
                f"but only {history.shape[0]} were provided."
            )

        mean_all = self.result.forecast(
            y=history[-self.k_ar:],
            steps=steps,
        )
        mean = float(mean_all[steps - 1, self.target_index])
        variance = self._target_forecast_variance(steps)
        return mean, variance

    def predict_samples(self, split_name, raw_data, sample_meta, lead_steps):
        if split_name == "Train":
            prefix_history = None
            split_endog = raw_data["Train"]["endog"]
        elif split_name == "Validation":
            prefix_history = raw_data["Train"]["endog"]
            split_endog = raw_data["Validation"]["endog"]
        elif split_name == "Test":
            prefix_history = np.vstack([
                raw_data["Train"]["endog"],
                raw_data["Validation"]["endog"],
            ])
            split_endog = raw_data["Test"]["endog"]
        else:
            raise ValueError("split_name must be 'Train', 'Validation', or 'Test'.")

        target_indices = sample_meta["target_index"]
        input_end_indices = sample_meta["input_end_index"]
        observed = split_endog[target_indices, self.target_index].astype(np.float64)

        means = np.zeros(len(target_indices), dtype=np.float64)
        variances = np.zeros(len(target_indices), dtype=np.float64)

        for i, input_end_idx in enumerate(input_end_indices):
            split_history = split_endog[:input_end_idx + 1]
            if prefix_history is None:
                history = split_history
            else:
                history = np.vstack([prefix_history, split_history])

            means[i], variances[i] = self._forecast_from_history(
                history=history,
                steps=lead_steps,
            )

        variances = np.maximum(variances, self.eps_var)
        return observed, means, variances


# Output helpers

def save_static_information(cfg, raw_data):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    split_info = []
    for split_name in ["Train", "Validation", "Test"]:
        data = raw_data[split_name]
        sample_count = len(data["endog"])
        effective_count = sample_count - cfg.input_len - cfg.max_lead_steps + 1
        if cfg.stride > 1:
            effective_count = int(np.floor((effective_count - 1) / cfg.stride) + 1)

        split_info.append({
            "Split": split_name,
            "Raw_sample_size": sample_count,
            "Input_window_samples": cfg.input_len,
            "Input_window_hours": cfg.input_window_hours,
            "Max_lead_samples": cfg.max_lead_steps,
            "Max_lead_hours": cfg.max_horizon_step,
            "Selected_horizons_hours": str(cfg.horizon_steps),
            "Stride_samples": cfg.stride,
            "Effective_sample_size_each_horizon": effective_count,
            "Dropped_samples_due_to_windowing": sample_count - effective_count,
            "Window_alignment": cfg.window_alignment,
        })

    pd.DataFrame(split_info).to_excel(
        cfg.output_dir / "data_split_and_window_sample_size_summary.xlsx",
        index=False,
    )

    variable_info = pd.DataFrame({
        "Variable_index": np.arange(len(raw_data["Endog_names"])),
        "Variable_name": raw_data["Endog_names"],
    })
    variable_info.to_excel(
        cfg.output_dir / "var_endogenous_variables.xlsx",
        index=False,
    )

    feature_info = pd.DataFrame({
        "Feature_group": [
            "Observed_target_wind_speed",
            "ECMWF_10m_mean_wind_speed_spatial_average",
            "ECMWF_100m_mean_wind_speed_spatial_average",
            "ECMWF_10m_gust_wind_speed_spatial_average",
        ],
        "Excel_columns": [
            "AC",
            "B:J averaged over the 9 grid points",
            "K:S averaged over the 9 grid points",
            "T:AB averaged over the 9 grid points",
        ],
        "Role_in_VAR": [
            "Target endogenous variable",
            "Endogenous predictor",
            "Endogenous predictor",
            "Endogenous predictor",
        ],
    })
    feature_info.to_excel(
        cfg.output_dir / "var_feature_definition.xlsx",
        index=False,
    )


def save_combined_outputs(cfg, all_predictions, all_model_info):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    pd.concat(all_model_info, ignore_index=True).to_excel(
        cfg.output_dir / "var_model_information_all_horizons.xlsx",
        index=False,
    )

    for split_name in ["Train", "Validation", "Test"]:
        prediction_frames = [
            all_predictions[horizon][split_name].copy()
            for horizon in cfg.horizon_steps
        ]
        pd.concat(prediction_frames, ignore_index=True).to_excel(
            cfg.output_dir / f"{split_name.lower()}_predictions_all_horizons_long.xlsx",
            index=False,
        )

        first_df = all_predictions[cfg.horizon_steps[0]][split_name]
        wide_df = first_df[["Sample_index"]].copy()

        for horizon_id in cfg.horizon_steps:
            prediction_df = all_predictions[horizon_id][split_name].copy()
            if len(prediction_df) != len(wide_df):
                raise ValueError(
                    f"Sample count mismatch for {split_name}, horizon {horizon_id}."
                )

            prefix = f"H{horizon_id:02d}_"
            columns = [
                "Input_start_time",
                "Input_end_time",
                "Target_time",
                "Observed",
                "Predicted_mean",
                "Predicted_variance",
                "Predicted_std",
                "Lower_95PI",
                "Upper_95PI",
                "Residual",
            ]
            add_columns = prediction_df[columns].rename(
                columns={column: prefix + column for column in columns}
            )
            wide_df = pd.concat([wide_df, add_columns], axis=1)

        wide_df.to_excel(
            cfg.output_dir / f"{split_name.lower()}_predictions_all_horizons_wide.xlsx",
            index=False,
        )


def train_one_horizon(cfg, raw_data, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour
    horizon_dir = cfg.output_dir / f"horizon_{horizon_id:02d}h_ahead"
    horizon_dir.mkdir(parents=True, exist_ok=True)

    sample_metas = prepare_metadata_for_horizon(cfg, raw_data, horizon_id)

    model = ProbabilisticVAR(
        lag_order=cfg.var_lag_order,
        trend=cfg.var_trend,
        eps_var=cfg.eps_var,
        target_index=0,
    ).fit(raw_data["Train"]["endog"])

    model_info = pd.DataFrame([{
        "Model": "VAR",
        "Horizon_hours_ahead": horizon_id,
        "Lead_samples": lead_steps,
        "Lag_order_requested": cfg.var_lag_order,
        "Lag_order_used": model.k_ar,
        "Trend": cfg.var_trend,
        "Number_of_endogenous_variables": model.neqs,
        "AIC": model.aic,
        "BIC": model.bic,
        "HQIC": model.hqic,
        "FPE": model.fpe,
        "Input_window_samples": cfg.input_len,
        "Input_window_hours": cfg.input_window_hours,
        "Window_alignment": cfg.window_alignment,
        "Stride_samples": cfg.stride,
    }])
    model_info.to_excel(
        horizon_dir / f"var_model_info_{horizon_id:02d}h_ahead.xlsx",
        index=False,
    )

    prediction_records = {}

    for split_name in ["Train", "Validation", "Test"]:
        sample_meta = sample_metas[split_name]
        observed, mean, variance = model.predict_samples(
            split_name=split_name,
            raw_data=raw_data,
            sample_meta=sample_meta,
            lead_steps=lead_steps,
        )

        observed = np.asarray(observed, dtype=np.float64).reshape(-1)
        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        variance = np.maximum(
            np.asarray(variance, dtype=np.float64).reshape(-1),
            cfg.eps_var,
        )
        std = np.sqrt(variance)
        lower = mean - cfg.z_value * std
        upper = mean + cfg.z_value * std

        prediction_df = pd.DataFrame({
            "Sample_index": np.arange(1, len(observed) + 1),
            "Horizon_hours_ahead": horizon_id,
            "Lead_samples": lead_steps,
            "Input_start_index_0based": sample_meta["input_start_index"],
            "Input_end_index_0based": sample_meta["input_end_index"],
            "Target_index_0based": sample_meta["target_index"],
            "Input_start_time": sample_meta["input_start_time"],
            "Input_end_time": sample_meta["input_end_time"],
            "Target_time": sample_meta["target_time"],
            "Observed": observed,
            "Predicted_mean": mean,
            "Predicted_variance": variance,
            "Predicted_std": std,
            "Lower_95PI": lower,
            "Upper_95PI": upper,
            "Residual": observed - mean,
        })

        prediction_path = (
            horizon_dir / f"{split_name.lower()}_predictions_{horizon_id:02d}h_ahead.xlsx"
        )
        prediction_df.to_excel(prediction_path, index=False)
        prediction_records[split_name] = prediction_df
        print(f"Saved {split_name.lower()} predictions: {prediction_path}")

    print(
        f"VAR {horizon_id} h ahead: "
        f"train={len(prediction_records['Train'])}, "
        f"validation={len(prediction_records['Validation'])}, "
        f"test={len(prediction_records['Test'])}, "
        f"lag_order={model.k_ar}, "
        f"AIC={model.aic:.3f}, BIC={model.bic:.3f}"
    )

    return model_info, prediction_records


def train_all_horizons(cfg):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print("Using model: Probabilistic VAR")
    print(f"Prediction horizons: {cfg.horizon_steps} h ahead")
    print(f"VAR lag order: {cfg.var_lag_order}")
    print(f"Output directory: {cfg.output_dir}")

    raw_data = load_raw_splits(cfg)
    save_static_information(cfg, raw_data)

    all_predictions = {}
    all_model_info = []

    for horizon_id in cfg.horizon_steps:
        model_info, prediction_records = train_one_horizon(
            cfg=cfg,
            raw_data=raw_data,
            horizon_id=horizon_id,
        )
        all_predictions[horizon_id] = prediction_records
        all_model_info.append(model_info)

    save_combined_outputs(cfg, all_predictions, all_model_info)
    print(f"All VAR outputs saved to: {cfg.output_dir}")
    return all_predictions


if __name__ == "__main__":
    train_all_horizons(cfg)
