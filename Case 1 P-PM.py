from pathlib import Path

import numpy as np
import pandas as pd


# Configuration

class Config:
    data_dir = Path("path/to/your/data")
    train_file = "train.xlsx"
    val_file = "validation.xlsx"
    test_file = "test.xlsx"
    output_dir_name = "results_pm"

    samples_per_hour = 6
    horizon_steps = [1, 6, 12, 24]
    max_horizon_step = max(horizon_steps)
    input_window_hours = 24
    input_len = input_window_hours * samples_per_hour
    max_lead_steps = max(horizon_steps) * samples_per_hour
    stride = 1
    window_alignment = "same_input_window"

    eps_var = 1e-6
    z_value = 1.959963984540054

    persistence_reference = "last_observed_wind_speed_in_input_window"

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
    observed = df.iloc[:, 28].astype(np.float64).to_numpy().reshape(-1, 1)
    return time, observed


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


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


def fit_scaler_on_training(raw_data):
    scaler_y = StandardScaler()
    scaler_y.fit(raw_data["Train"]["y_raw"])
    return scaler_y


def transform_raw_splits(raw_data, scaler_y):
    transformed = {}
    for split_name, data in raw_data.items():
        transformed[split_name] = {
            "time": data["time"],
            "y": scaler_y.transform(data["y_raw"]),
            "y_raw": data["y_raw"],
            "raw_size": data["raw_size"],
        }
    return transformed


def build_sliding_windows(
    y,
    time,
    input_len,
    lead_steps,
    max_lead_steps,
    stride,
    alignment,
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

    persistence_values = []
    targets = []
    target_time = []
    input_start_time = []
    input_end_time = []
    input_start_index = []
    input_end_index = []
    target_index = []

    if alignment == "same_input_window":
        max_start = n - input_len - max_lead_steps
        starts = np.arange(0, max_start + 1, stride, dtype=np.int64)

        for start in starts:
            end = start + input_len - 1
            target_idx = start + input_len + lead_steps - 1

            persistence_values.append(y[end])
            targets.append(y[target_idx])
            target_time.append(time[target_idx])
            input_start_time.append(time[start])
            input_end_time.append(time[end])
            input_start_index.append(start)
            input_end_index.append(end)
            target_index.append(target_idx)

    elif alignment == "same_target_time":
        target_start = input_len + max_lead_steps - 1
        targets_idx = np.arange(target_start, n, stride, dtype=np.int64)

        for target_idx in targets_idx:
            end = target_idx - lead_steps
            start = end - input_len + 1
            if start < 0:
                continue

            persistence_values.append(y[end])
            targets.append(y[target_idx])
            target_time.append(time[target_idx])
            input_start_time.append(time[start])
            input_end_time.append(time[end])
            input_start_index.append(start)
            input_end_index.append(end)
            target_index.append(target_idx)
    else:
        raise ValueError(
            "window_alignment must be 'same_input_window' or 'same_target_time'."
        )

    if not targets:
        raise ValueError("No samples were generated. Check split length and window settings.")

    persistence_values = np.stack(persistence_values, axis=0).astype(np.float64)
    targets = np.stack(targets, axis=0).astype(np.float64)

    sample_meta = {
        "target_time": np.asarray(target_time),
        "input_start_time": np.asarray(input_start_time),
        "input_end_time": np.asarray(input_end_time),
        "input_start_index": np.asarray(input_start_index, dtype=np.int64),
        "input_end_index": np.asarray(input_end_index, dtype=np.int64),
        "target_index": np.asarray(target_index, dtype=np.int64),
    }
    return persistence_values, targets, sample_meta


def prepare_data_for_horizon(cfg, transformed_data, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour
    datasets = {}
    sample_metas = {}

    for split_name, data in transformed_data.items():
        persistence_values, targets, sample_meta = build_sliding_windows(
            y=data["y"],
            time=data["time"],
            input_len=cfg.input_len,
            lead_steps=lead_steps,
            max_lead_steps=cfg.max_lead_steps,
            stride=cfg.stride,
            alignment=cfg.window_alignment,
        )
        datasets[split_name] = {
            "persistence_values": persistence_values,
            "y": targets,
        }
        sample_metas[split_name] = sample_meta

    return datasets, sample_metas


# Probabilistic persistence baseline

class ProbabilisticPersistence:
    def __init__(self, eps_var=1e-6):
        self.eps_var = eps_var
        self.error_var = None
        self.error_std = None

    def fit(self, persistence_values, y_train):
        persistence_values = np.asarray(persistence_values, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        if persistence_values.shape != y_train.shape:
            raise ValueError("Persistence values and training targets have different shapes.")
        if y_train.shape[0] == 0:
            raise ValueError("Training data are empty.")

        residual = y_train - persistence_values
        if residual.shape[0] > 1:
            error_var = np.var(residual, axis=0, ddof=1, keepdims=True)
        else:
            error_var = np.var(residual, axis=0, ddof=0, keepdims=True)

        self.error_var = np.maximum(error_var, self.eps_var)
        self.error_std = np.sqrt(self.error_var)
        return self

    def predict(self, persistence_values):
        if self.error_var is None:
            raise RuntimeError(
                "The probabilistic persistence baseline must be fitted before prediction."
            )

        persistence_values = np.asarray(persistence_values, dtype=np.float64)
        variance = np.repeat(self.error_var, persistence_values.shape[0], axis=0)
        return persistence_values.copy(), variance


# Prediction and outputs

def inverse_y_and_variance(y_norm, mean_norm, variance_norm, scaler_y):
    y_std = float(np.squeeze(scaler_y.std))
    y_mean = float(np.squeeze(scaler_y.mean))
    y = y_norm * y_std + y_mean
    mean = mean_norm * y_std + y_mean
    variance = variance_norm * (y_std ** 2)
    return y, mean, variance


def save_static_information(cfg, raw_data, scaler_y):
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
            "Persistence_reference": cfg.persistence_reference,
        })

    pd.DataFrame(split_info).to_excel(
        cfg.output_dir / "data_split_and_window_sample_size_summary.xlsx",
        index=False,
    )

    scaler_info = pd.DataFrame([{
        "Scaler": "target_wind_speed",
        "Shape_mean": str(np.asarray(scaler_y.mean).shape),
        "Shape_std": str(np.asarray(scaler_y.std).shape),
        "Mean_values": np.array2string(
            np.squeeze(scaler_y.mean), precision=6, separator=", "
        ),
        "Std_values": np.array2string(
            np.squeeze(scaler_y.std), precision=6, separator=", "
        ),
    }])
    scaler_info.to_excel(
        cfg.output_dir / "training_only_scaler_information.xlsx",
        index=False,
    )

    feature_info = pd.DataFrame({
        "Feature_group": [
            "Historical_observed_wind_speed",
            "Target_observed_wind_speed",
            "Persistence_prediction_mean",
            "ECMWF_predictors",
        ],
        "Excel_columns": [
            "AC",
            "AC shifted according to prediction horizon",
            "Last AC value within each input window",
            "B:AB",
        ],
        "Used_as": [
            "historical persistence input",
            "prediction target",
            "probabilistic persistence mean",
            "not used by the persistence baseline",
        ],
    })
    feature_info.to_excel(
        cfg.output_dir / "feature_definition.xlsx",
        index=False,
    )

    model_info = pd.DataFrame({
        "Item": [
            "Model",
            "Prediction mean",
            "Predictive variance",
            "Input window samples",
            "Input window hours",
            "Selected prediction horizons",
            "Window alignment",
            "Persistence reference",
        ],
        "Value": [
            "Probabilistic Persistence Model",
            "Last observed wind speed in the input window",
            "Training residual variance estimated separately for each horizon",
            cfg.input_len,
            cfg.input_window_hours,
            str(cfg.horizon_steps),
            cfg.window_alignment,
            cfg.persistence_reference,
        ],
    })
    model_info.to_excel(
        cfg.output_dir / "pm_model_information.xlsx",
        index=False,
    )


def train_one_horizon(cfg, transformed_data, scaler_y, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour
    horizon_dir = cfg.output_dir / f"horizon_{horizon_id:02d}h_ahead"
    horizon_dir.mkdir(parents=True, exist_ok=True)

    datasets, sample_metas = prepare_data_for_horizon(
        cfg, transformed_data, horizon_id
    )

    model = ProbabilisticPersistence(eps_var=cfg.eps_var)
    model.fit(
        datasets["Train"]["persistence_values"],
        datasets["Train"]["y"],
    )

    y_std = float(np.squeeze(scaler_y.std))
    error_var_original_scale = model.error_var.reshape(-1) * (y_std ** 2)

    variance_df = pd.DataFrame([{
        "Horizon_hours_ahead": horizon_id,
        "Lead_samples": lead_steps,
        "Persistence_error_variance_normalized": model.error_var.reshape(-1)[0],
        "Persistence_error_std_normalized": model.error_std.reshape(-1)[0],
        "Persistence_error_variance_original_scale": error_var_original_scale[0],
        "Persistence_error_std_original_scale": np.sqrt(error_var_original_scale[0]),
        "Persistence_reference": cfg.persistence_reference,
    }])
    variance_df.to_excel(
        horizon_dir / f"persistence_error_variance_{horizon_id:02d}h_ahead.xlsx",
        index=False,
    )

    model_info = pd.DataFrame([{
        "Model": "Probabilistic Persistence Model",
        "Horizon_hours_ahead": horizon_id,
        "Lead_samples": lead_steps,
        "Input_window_samples": cfg.input_len,
        "Input_window_hours": cfg.input_window_hours,
        "Window_alignment": cfg.window_alignment,
        "Stride_samples": cfg.stride,
        "Training_samples": len(datasets["Train"]["y"]),
        "Persistence_error_variance_normalized": model.error_var.reshape(-1)[0],
        "Persistence_error_variance_original_scale": error_var_original_scale[0],
        "Persistence_reference": cfg.persistence_reference,
    }])
    model_info.to_excel(
        horizon_dir / f"pm_model_information_{horizon_id:02d}h_ahead.xlsx",
        index=False,
    )

    prediction_records = {}
    for split_name in ["Train", "Validation", "Test"]:
        y_norm = datasets[split_name]["y"]
        mean_norm, variance_norm = model.predict(
            datasets[split_name]["persistence_values"]
        )
        y, mean, variance = inverse_y_and_variance(
            y_norm, mean_norm, variance_norm, scaler_y
        )

        y = y.reshape(-1)
        mean = mean.reshape(-1)
        variance = np.maximum(variance.reshape(-1), 1e-12)
        sigma = np.sqrt(variance)
        lower = mean - cfg.z_value * sigma
        upper = mean + cfg.z_value * sigma
        sample_meta = sample_metas[split_name]

        pred_df = pd.DataFrame({
            "Sample_index": np.arange(1, len(y) + 1),
            "Horizon_hours_ahead": horizon_id,
            "Lead_samples": lead_steps,
            "Input_start_index_0based": sample_meta["input_start_index"],
            "Input_end_index_0based": sample_meta["input_end_index"],
            "Target_index_0based": sample_meta["target_index"],
            "Input_start_time": sample_meta["input_start_time"],
            "Input_end_time": sample_meta["input_end_time"],
            "Target_time": sample_meta["target_time"],
            "Observed": y,
            "Predicted_mean": mean,
            "Predicted_variance": variance,
            "Predicted_std": sigma,
            "Lower_95PI": lower,
            "Upper_95PI": upper,
            "Residual": y - mean,
            "Persistence_reference": cfg.persistence_reference,
        })

        pred_path = horizon_dir / f"{split_name.lower()}_predictions_{horizon_id:02d}h_ahead.xlsx"
        pred_df.to_excel(pred_path, index=False)
        prediction_records[split_name] = pred_df
        print(f"Saved {split_name.lower()} predictions: {pred_path}")

    print(
        f"PM {horizon_id} h ahead: "
        f"train={len(datasets['Train']['y'])}, "
        f"validation={len(datasets['Validation']['y'])}, "
        f"test={len(datasets['Test']['y'])}, "
        f"error variance={error_var_original_scale[0]:.6f}"
    )

    return model_info, prediction_records


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
            pred_df = all_predictions[horizon_id][split_name].copy()
            if len(pred_df) != len(wide_df):
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
            add_cols = pred_df[columns].rename(
                columns={column: prefix + column for column in columns}
            )
            wide_df = pd.concat([wide_df, add_cols], axis=1)

        wide_df.to_excel(
            cfg.output_dir / f"{split_name.lower()}_predictions_all_horizons_wide.xlsx",
            index=False,
        )

    pd.concat(all_model_info, ignore_index=True).to_excel(
        cfg.output_dir / "pm_model_information_all_horizons.xlsx",
        index=False,
    )


def train_all_horizons(cfg):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {cfg.output_dir}")
    print(f"Prediction horizons: {cfg.horizon_steps} h ahead")

    raw_data = load_raw_splits(cfg)
    scaler_y = fit_scaler_on_training(raw_data)
    transformed_data = transform_raw_splits(raw_data, scaler_y)
    save_static_information(cfg, raw_data, scaler_y)

    all_predictions = {}
    all_model_info = []

    for horizon_id in cfg.horizon_steps:
        model_info, predictions = train_one_horizon(
            cfg, transformed_data, scaler_y, horizon_id
        )
        all_predictions[horizon_id] = predictions
        all_model_info.append(model_info)

    save_combined_outputs(cfg, all_predictions, all_model_info)
    print(f"All PM outputs saved to: {cfg.output_dir}")
    return all_predictions


if __name__ == "__main__":
    train_all_horizons(cfg)
