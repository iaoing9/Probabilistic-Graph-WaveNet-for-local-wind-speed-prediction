from pathlib import Path
import warnings

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError as exc:
    raise ImportError(
        "statsmodels is required for the ARIMA baseline. "
        "Install it with: pip install statsmodels"
    ) from exc


# Configuration

class Config:
    data_dir = Path("path/to/your/data")
    train_file = "train.xlsx"
    val_file = "validation.xlsx"
    test_file = "test.xlsx"
    output_dir_name = "results_arima"

    samples_per_hour = 6
    horizon_steps = [1, 6, 12, 24]
    max_horizon_step = max(horizon_steps)
    input_window_hours = 24
    input_len = input_window_hours * samples_per_hour
    max_lead_steps = max(horizon_steps) * samples_per_hour
    stride = 1
    window_alignment = "same_input_window"

    arima_order = (2, 0, 1)
    arima_trend = "c"
    update_with_observations = True
    eps_var = 1e-6
    z_value = 1.959963984540054

    @property
    def output_dir(self):
        return self.data_dir / self.output_dir_name


cfg = Config()


# Data loading and sample construction

def read_input_excel(file_path):
    df = pd.read_excel(file_path, engine="openpyxl")
    if df.shape[1] < 29:
        raise ValueError(
            f"{file_path} has {df.shape[1]} columns, but at least 29 columns A:AC are required."
        )

    time = df.iloc[:, 0].to_numpy()
    observed = df.iloc[:, 28].astype(np.float64).to_numpy()
    if not np.isfinite(observed).all():
        raise ValueError(f"The observed wind-speed column in {file_path} contains missing or non-finite values.")
    return time, observed


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
    for split_name, path in paths.items():
        time, observed = read_input_excel(path)
        raw_data[split_name] = {
            "time": time,
            "y_raw": observed,
            "raw_size": len(observed),
        }
    return raw_data


def build_sample_metadata(
    y,
    time,
    input_len,
    lead_steps,
    max_lead_steps,
    stride=1,
    alignment="same_input_window",
):
    n = len(y)
    if n != len(time):
        raise ValueError("y and time must have the same first dimension.")
    if input_len <= 0 or lead_steps <= 0 or max_lead_steps <= 0 or stride <= 0:
        raise ValueError("Window lengths, lead times, and stride must be positive.")
    if lead_steps > max_lead_steps:
        raise ValueError("lead_steps must not exceed max_lead_steps.")
    if n < input_len + max_lead_steps:
        raise ValueError(
            f"Split length {n} is too short for input_len={input_len} "
            f"and max_lead_steps={max_lead_steps}."
        )

    input_start_times = []
    input_end_times = []
    target_times = []
    input_start_indices = []
    input_end_indices = []
    target_indices = []

    if alignment == "same_input_window":
        max_start = n - input_len - max_lead_steps
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
        targets = np.arange(target_start, n, stride, dtype=np.int64)

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


def prepare_samples_for_horizon(cfg, raw_data, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour
    samples = {}

    for split_name, data in raw_data.items():
        sample_meta = build_sample_metadata(
            y=data["y_raw"],
            time=data["time"],
            input_len=cfg.input_len,
            lead_steps=lead_steps,
            max_lead_steps=cfg.max_lead_steps,
            stride=cfg.stride,
            alignment=cfg.window_alignment,
        )
        target_indices = sample_meta["target_index"]
        samples[split_name] = {
            "y": data["y_raw"][target_indices],
            "sample_meta": sample_meta,
            "raw_size": data["raw_size"],
        }

    return samples


# Probabilistic ARIMA baseline

class ProbabilisticARIMA:
    def __init__(
        self,
        order=(2, 0, 1),
        trend="c",
        eps_var=1e-6,
        update_with_observations=True,
    ):
        self.order = order
        self.trend = trend
        self.eps_var = eps_var
        self.update_with_observations = update_with_observations
        self.result = None
        self.aic = None
        self.bic = None
        self.residual_variance = None

    def _fit_series(self, y):
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(
                y,
                order=self.order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            return model.fit()

    def fit(self, y_train):
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        if y_train.size == 0:
            raise ValueError("The ARIMA training series is empty.")

        self.result = self._fit_series(y_train)
        self.aic = float(self.result.aic)
        self.bic = float(self.result.bic)

        residuals = np.asarray(self.result.resid, dtype=np.float64).reshape(-1)
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size > 1:
            self.residual_variance = float(np.var(residuals, ddof=1))
        else:
            self.residual_variance = self.eps_var
        self.residual_variance = max(self.residual_variance, self.eps_var)
        return self

    def _sanitize_variance(self, variance):
        variance = np.asarray(variance, dtype=np.float64).reshape(-1)
        fallback = (
            self.residual_variance
            if self.residual_variance is not None
            else self.eps_var
        )
        variance = np.where(np.isfinite(variance), variance, fallback)
        return np.maximum(variance, self.eps_var)

    def _prediction_variance(self, prediction_result):
        try:
            variance = np.asarray(prediction_result.var_pred_mean, dtype=np.float64)
        except AttributeError:
            frame = prediction_result.summary_frame()
            variance = np.asarray(frame["mean_se"], dtype=np.float64) ** 2
        return self._sanitize_variance(variance)

    def predict_train_by_target_indices(self, y_train, target_indices):
        if self.result is None:
            raise RuntimeError("The ARIMA model must be fitted before prediction.")

        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        target_indices = np.asarray(target_indices, dtype=np.int64).reshape(-1)

        prediction_result = self.result.get_prediction(
            start=0,
            end=len(y_train) - 1,
        )
        mean_all = np.asarray(
            prediction_result.predicted_mean, dtype=np.float64
        ).reshape(-1)
        variance_all = self._prediction_variance(prediction_result)

        return (
            y_train[target_indices].astype(np.float64),
            mean_all[target_indices].astype(np.float64),
            variance_all[target_indices].astype(np.float64),
        )

    def rolling_forecast_for_samples(
        self,
        initial_history,
        eval_series,
        sample_meta,
        lead_steps,
    ):
        if self.result is None:
            raise RuntimeError("The ARIMA model must be fitted before prediction.")

        initial_history = np.asarray(initial_history, dtype=np.float64).reshape(-1)
        eval_series = np.asarray(eval_series, dtype=np.float64).reshape(-1)
        input_end_indices = np.asarray(
            sample_meta["input_end_index"], dtype=np.int64
        ).reshape(-1)
        target_indices = np.asarray(
            sample_meta["target_index"], dtype=np.int64
        ).reshape(-1)

        if input_end_indices.size == 0:
            raise ValueError("No ARIMA samples are available for rolling forecasts.")
        if np.any(input_end_indices >= len(eval_series)):
            raise IndexError("An input-window end index is outside the evaluation series.")
        if np.any(target_indices >= len(eval_series)):
            raise IndexError("A target index is outside the evaluation series.")

        current_result = self._fit_series(initial_history)
        appended_until = -1
        means = []
        variances = []

        for input_end_idx in input_end_indices:
            if input_end_idx > appended_until:
                new_observations = eval_series[appended_until + 1:input_end_idx + 1]
                if self.update_with_observations and new_observations.size > 0:
                    current_result = current_result.append(
                        new_observations,
                        refit=False,
                    )
                appended_until = input_end_idx

            forecast_result = current_result.get_forecast(steps=lead_steps)
            mean_steps = np.asarray(
                forecast_result.predicted_mean, dtype=np.float64
            ).reshape(-1)
            variance_steps = self._prediction_variance(forecast_result)

            means.append(mean_steps[-1])
            variances.append(variance_steps[-1])

        return (
            eval_series[target_indices].astype(np.float64),
            np.asarray(means, dtype=np.float64),
            self._sanitize_variance(np.asarray(variances, dtype=np.float64)),
        )


# Output helpers

def save_static_information(cfg, raw_data):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    split_info = []
    for split_name, data in raw_data.items():
        raw_size = len(data["y_raw"])
        effective_size = raw_size - cfg.input_len - cfg.max_lead_steps + 1
        if cfg.stride > 1:
            effective_size = int(np.floor((effective_size - 1) / cfg.stride) + 1)

        split_info.append({
            "Split": split_name,
            "Raw_sample_size": raw_size,
            "Input_window_samples": cfg.input_len,
            "Input_window_hours": cfg.input_window_hours,
            "Max_lead_samples": cfg.max_lead_steps,
            "Max_lead_hours": cfg.max_horizon_step,
            "Selected_horizons_hours": str(cfg.horizon_steps),
            "Stride_samples": cfg.stride,
            "Effective_sample_size_each_horizon": effective_size,
            "Dropped_samples_due_to_windowing": raw_size - effective_size,
            "Window_alignment": cfg.window_alignment,
        })

    pd.DataFrame(split_info).to_excel(
        cfg.output_dir / "data_split_and_window_sample_size_summary.xlsx",
        index=False,
    )

    feature_info = pd.DataFrame({
        "Feature_group": [
            "Observed_wind_speed",
            "ECMWF_grid_features",
        ],
        "Excel_columns": [
            "AC",
            "B:AB",
        ],
        "Used_as": [
            "Univariate ARIMA series and prediction target",
            "Not used by ARIMA",
        ],
    })
    feature_info.to_excel(
        cfg.output_dir / "feature_definition_arima.xlsx",
        index=False,
    )


def save_combined_outputs(cfg, all_predictions, all_model_info):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

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

    pd.concat(all_model_info, ignore_index=True).to_excel(
        cfg.output_dir / "arima_model_information_all_horizons.xlsx",
        index=False,
    )


def train_one_horizon(cfg, raw_data, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour
    horizon_dir = cfg.output_dir / f"horizon_{horizon_id:02d}h_ahead"
    horizon_dir.mkdir(parents=True, exist_ok=True)

    samples = prepare_samples_for_horizon(cfg, raw_data, horizon_id)
    train_series = raw_data["Train"]["y_raw"]
    validation_series = raw_data["Validation"]["y_raw"]
    test_series = raw_data["Test"]["y_raw"]

    model = ProbabilisticARIMA(
        order=cfg.arima_order,
        trend=cfg.arima_trend,
        eps_var=cfg.eps_var,
        update_with_observations=cfg.update_with_observations,
    ).fit(train_series)

    model_info = pd.DataFrame([{
        "Model": "ARIMA",
        "Horizon_hours_ahead": horizon_id,
        "Lead_samples": lead_steps,
        "Order": str(cfg.arima_order),
        "Trend": cfg.arima_trend,
        "AIC": model.aic,
        "BIC": model.bic,
        "Residual_variance": model.residual_variance,
        "Update_with_observations": cfg.update_with_observations,
        "Input_window_samples": cfg.input_len,
        "Input_window_hours": cfg.input_window_hours,
        "Window_alignment": cfg.window_alignment,
        "Stride_samples": cfg.stride,
    }])
    model_info.to_excel(
        horizon_dir / f"arima_model_info_{horizon_id:02d}h_ahead.xlsx",
        index=False,
    )

    predictions = {}

    train_y, train_mean, train_variance = model.predict_train_by_target_indices(
        y_train=train_series,
        target_indices=samples["Train"]["sample_meta"]["target_index"],
    )
    predictions["Train"] = (train_y, train_mean, train_variance)

    validation_y, validation_mean, validation_variance = model.rolling_forecast_for_samples(
        initial_history=train_series,
        eval_series=validation_series,
        sample_meta=samples["Validation"]["sample_meta"],
        lead_steps=lead_steps,
    )
    predictions["Validation"] = (
        validation_y,
        validation_mean,
        validation_variance,
    )

    test_initial_history = np.concatenate([train_series, validation_series])
    test_y, test_mean, test_variance = model.rolling_forecast_for_samples(
        initial_history=test_initial_history,
        eval_series=test_series,
        sample_meta=samples["Test"]["sample_meta"],
        lead_steps=lead_steps,
    )
    predictions["Test"] = (test_y, test_mean, test_variance)

    prediction_records = {}
    for split_name in ["Train", "Validation", "Test"]:
        observed, mean, variance = predictions[split_name]
        sample_meta = samples[split_name]["sample_meta"]

        observed = np.asarray(observed).reshape(-1)
        mean = np.asarray(mean).reshape(-1)
        variance = np.maximum(np.asarray(variance).reshape(-1), cfg.eps_var)
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
        f"ARIMA {horizon_id} h ahead: "
        f"train={len(samples['Train']['y'])}, "
        f"validation={len(samples['Validation']['y'])}, "
        f"test={len(samples['Test']['y'])}, "
        f"AIC={model.aic:.3f}, BIC={model.bic:.3f}"
    )

    return model_info, prediction_records


def train_all_horizons(cfg):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print("Using model: Probabilistic ARIMA")
    print(f"Prediction horizons: {cfg.horizon_steps} h ahead")
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
    print(f"All ARIMA outputs saved to: {cfg.output_dir}")
    return all_predictions


if __name__ == "__main__":
    train_all_horizons(cfg)
