import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# Configuration

class Config:
    data_dir = Path("path/to/your/data")
    train_file = "train.xlsx"
    val_file = "validation.xlsx"
    test_file = "test.xlsx"
    output_dir_name = "results_qrf"

    seed = 2026

    samples_per_hour = 6
    horizon_steps = [1, 6, 12, 24]
    max_horizon_step = max(horizon_steps)
    input_window_hours = 24
    input_len = input_window_hours * samples_per_hour
    max_lead_steps = max_horizon_step * samples_per_hour
    stride = 1
    window_alignment = "same_target_time"

    num_nodes = 9
    num_grid_features = 3
    num_global_features = 1
    input_dim_per_timestep = num_nodes * num_grid_features + num_global_features

    n_estimators = 100
    max_depth = None
    min_samples_split = 2
    min_samples_leaf = 5
    max_features = "sqrt"
    bootstrap = True
    n_jobs = -1

    interval_alpha = 0.05
    lower_quantile = interval_alpha / 2.0
    upper_quantile = 1.0 - interval_alpha / 2.0
    eps_var = 1e-8

    @property
    def output_dir(self):
        return self.data_dir / self.output_dir_name


cfg = Config()


# Reproducibility

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# Data loading and preprocessing

def split_grid_features(ecmwf_raw, num_nodes):
    expected_features = 3 * num_nodes
    if ecmwf_raw.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} ECMWF columns, but got {ecmwf_raw.shape[1]}."
        )

    groups = [
        ecmwf_raw[:, group * num_nodes:(group + 1) * num_nodes]
        for group in range(3)
    ]
    return np.stack(groups, axis=1).transpose(0, 2, 1).astype(np.float32)


def read_input_excel(file_path, cfg):
    df = pd.read_excel(file_path, engine="openpyxl")

    if df.shape[1] < 29:
        raise ValueError(
            f"{file_path} has {df.shape[1]} columns, but at least 29 columns A:AC are required."
        )

    time = df.iloc[:, 0].to_numpy()
    ecmwf_raw = df.iloc[:, 1:28].astype(np.float32).to_numpy()
    ane_raw = df.iloc[:, 28].astype(np.float32).to_numpy().reshape(-1, 1)

    x_grid = split_grid_features(ecmwf_raw, cfg.num_nodes)
    x_global = ane_raw.astype(np.float32)
    y_raw = ane_raw.astype(np.float32)

    return time, x_grid, x_global, y_raw


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data, axis=None):
        self.mean = np.mean(data, axis=axis, keepdims=True)
        self.std = np.std(data, axis=axis, keepdims=True)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


# Sliding-window construction

def build_sliding_windows(
    x_grid,
    x_global,
    y,
    time,
    input_len,
    lead_steps,
    max_lead_steps,
    stride,
    alignment,
):
    n = len(y)

    if n != len(x_grid) or n != len(x_global) or n != len(time):
        raise ValueError("x_grid, x_global, y, and time must have the same first dimension.")
    if input_len <= 0 or lead_steps <= 0 or max_lead_steps <= 0 or stride <= 0:
        raise ValueError("Window length, lead times, and stride must be positive.")
    if lead_steps > max_lead_steps:
        raise ValueError("lead_steps must not be larger than max_lead_steps.")
    if n < input_len + max_lead_steps:
        raise ValueError(
            f"Split length {n} is too short for input_len={input_len} "
            f"and max_lead_steps={max_lead_steps}."
        )

    xg_list = []
    xgl_list = []
    y_list = []
    target_time_list = []
    input_start_time_list = []
    input_end_time_list = []
    input_start_index_list = []
    input_end_index_list = []
    target_index_list = []

    if alignment == "same_input_window":
        max_start = n - input_len - max_lead_steps
        starts = np.arange(0, max_start + 1, stride, dtype=np.int64)

        for start in starts:
            end = start + input_len - 1
            target_idx = start + input_len + lead_steps - 1

            xg_list.append(x_grid[start:start + input_len])
            xgl_list.append(x_global[start:start + input_len])
            y_list.append(y[target_idx])
            target_time_list.append(time[target_idx])
            input_start_time_list.append(time[start])
            input_end_time_list.append(time[end])
            input_start_index_list.append(start)
            input_end_index_list.append(end)
            target_index_list.append(target_idx)

    elif alignment == "same_target_time":
        target_start = input_len + max_lead_steps - 1
        targets = np.arange(target_start, n, stride, dtype=np.int64)

        for target_idx in targets:
            end = target_idx - lead_steps
            start = end - input_len + 1

            if start < 0:
                continue

            xg_list.append(x_grid[start:start + input_len])
            xgl_list.append(x_global[start:start + input_len])
            y_list.append(y[target_idx])
            target_time_list.append(time[target_idx])
            input_start_time_list.append(time[start])
            input_end_time_list.append(time[end])
            input_start_index_list.append(start)
            input_end_index_list.append(end)
            target_index_list.append(target_idx)

    else:
        raise ValueError("alignment must be 'same_input_window' or 'same_target_time'.")

    if not y_list:
        raise ValueError("No sliding-window samples were generated. Check the split length and settings.")

    xg = np.stack(xg_list, axis=0).astype(np.float32)
    xgl = np.stack(xgl_list, axis=0).astype(np.float32)
    yy = np.stack(y_list, axis=0).astype(np.float32)

    sample_meta = {
        "target_time": np.asarray(target_time_list),
        "input_start_time": np.asarray(input_start_time_list),
        "input_end_time": np.asarray(input_end_time_list),
        "input_start_index": np.asarray(input_start_index_list, dtype=np.int64),
        "input_end_index": np.asarray(input_end_index_list, dtype=np.int64),
        "target_index": np.asarray(target_index_list, dtype=np.int64),
    }

    return xg, xgl, yy, sample_meta


# QRF feature construction

def flatten_windows_for_qrf(x_grid, x_global):
    sample_count, time_steps, node_count, grid_feature_count = x_grid.shape
    x_grid_flat = x_grid.reshape(
        sample_count, time_steps, node_count * grid_feature_count
    )
    x = np.concatenate([x_grid_flat, x_global], axis=-1)
    return x.reshape(
        sample_count, time_steps * (node_count * grid_feature_count + x_global.shape[-1])
    ).astype(np.float32)


def load_raw_splits(cfg):
    paths = {
        "Train": cfg.data_dir / cfg.train_file,
        "Validation": cfg.data_dir / cfg.val_file,
        "Test": cfg.data_dir / cfg.test_file,
    }

    missing_files = [path for path in paths.values() if not path.is_file()]
    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Set Config.data_dir to the directory containing the input files. "
            f"The following files could not be found:\n{missing}"
        )

    raw_data = {}
    for split_name, path in paths.items():
        time, x_grid_raw, x_global_raw, y_raw = read_input_excel(path, cfg)
        raw_data[split_name] = {
            "time": time,
            "x_grid_raw": x_grid_raw,
            "x_global_raw": x_global_raw,
            "y_raw": y_raw,
        }

    return raw_data


def fit_scalers_on_training(raw_data):
    train = raw_data["Train"]

    scaler_grid = StandardScaler()
    scaler_global = StandardScaler()
    scaler_y = StandardScaler()

    scaler_grid.fit(train["x_grid_raw"], axis=(0, 1))
    scaler_global.fit(train["x_global_raw"], axis=0)
    scaler_y.fit(train["y_raw"], axis=0)

    return scaler_grid, scaler_global, scaler_y


def transform_raw_splits(raw_data, scaler_grid, scaler_global, scaler_y):
    transformed = {}

    for split_name, data in raw_data.items():
        transformed[split_name] = {
            "time": data["time"],
            "x_grid": scaler_grid.transform(data["x_grid_raw"]),
            "x_global": scaler_global.transform(data["x_global_raw"]),
            "y": scaler_y.transform(data["y_raw"]),
            "y_raw": data["y_raw"],
            "raw_size": len(data["y_raw"]),
        }

    return transformed


def prepare_data_for_horizon(cfg, transformed_data, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour
    qrf_data = {}
    sample_metas = {}

    for split_name, data in transformed_data.items():
        x_grid, x_global, y, sample_meta = build_sliding_windows(
            x_grid=data["x_grid"],
            x_global=data["x_global"],
            y=data["y"],
            time=data["time"],
            input_len=cfg.input_len,
            lead_steps=lead_steps,
            max_lead_steps=cfg.max_lead_steps,
            stride=cfg.stride,
            alignment=cfg.window_alignment,
        )

        qrf_data[split_name] = {
            "x": flatten_windows_for_qrf(x_grid, x_global),
            "y": y.reshape(-1).astype(np.float32),
        }
        sample_metas[split_name] = sample_meta

    return qrf_data, sample_metas


# Standard Quantile Regression Forest

class QuantileRandomForest:
    def __init__(
        self,
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        bootstrap,
        n_jobs,
        random_state,
    ):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.leaf_y_by_tree = None
        self.y_train = None

    def fit(self, x_train, y_train):
        x_train = np.asarray(x_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)

        if x_train.ndim != 2:
            raise ValueError("QRF training features must be a two-dimensional array.")
        if x_train.shape[0] != self.y_train.size:
            raise ValueError("QRF training features and targets have different sample counts.")
        if self.y_train.size == 0:
            raise ValueError("QRF training data are empty.")

        self.rf.fit(x_train, self.y_train)
        train_leaf = self.rf.apply(x_train)
        self.leaf_y_by_tree = []

        for tree_idx in range(train_leaf.shape[1]):
            leaf_map = {}
            for leaf_id, y_value in zip(train_leaf[:, tree_idx], self.y_train):
                leaf_map.setdefault(int(leaf_id), []).append(float(y_value))

            self.leaf_y_by_tree.append({
                leaf_id: np.asarray(values, dtype=np.float64)
                for leaf_id, values in leaf_map.items()
            })

        return self

    def _weighted_distribution_for_one_sample(self, sample_leaf_ids):
        tree_leaf_values = []

        for tree_idx, leaf_id in enumerate(sample_leaf_ids):
            leaf_values = self.leaf_y_by_tree[tree_idx].get(int(leaf_id))
            if leaf_values is not None and leaf_values.size:
                tree_leaf_values.append(leaf_values)

        if not tree_leaf_values:
            values = self.y_train.copy()
            weights = np.full(values.size, 1.0 / values.size, dtype=np.float64)
            return values, weights

        valid_tree_count = len(tree_leaf_values)
        values = np.concatenate(tree_leaf_values)
        weights = np.concatenate([
            np.full(
                leaf_values.size,
                1.0 / (valid_tree_count * leaf_values.size),
                dtype=np.float64,
            )
            for leaf_values in tree_leaf_values
        ])
        weights /= weights.sum()

        return values, weights

    @staticmethod
    def _weighted_quantile(values, weights, quantile):
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be between 0 and 1.")

        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_weights = weights[order]
        cumulative_weights = np.cumsum(sorted_weights)
        index = np.searchsorted(cumulative_weights, quantile, side="left")
        index = min(index, sorted_values.size - 1)

        return float(sorted_values[index])

    def predict(self, x, y_true, lower_q, upper_q, eps_var):
        if self.leaf_y_by_tree is None:
            raise RuntimeError("Call fit before predict.")

        x = np.asarray(x, dtype=np.float32)
        pred_leaf = self.rf.apply(x)
        sample_count = pred_leaf.shape[0]

        mean = np.empty(sample_count, dtype=np.float64)
        median = np.empty(sample_count, dtype=np.float64)
        lower = np.empty(sample_count, dtype=np.float64)
        upper = np.empty(sample_count, dtype=np.float64)
        var = np.empty(sample_count, dtype=np.float64)
        sample_crps = np.empty(sample_count, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)

        if y_true.size != sample_count:
            raise ValueError("y_true and prediction features have different sample counts.")

        for sample_idx, sample_leaf_ids in enumerate(pred_leaf):
            values, weights = self._weighted_distribution_for_one_sample(sample_leaf_ids)
            mean_value = float(np.sum(values * weights))

            mean[sample_idx] = mean_value
            median[sample_idx] = self._weighted_quantile(values, weights, 0.50)
            lower[sample_idx] = self._weighted_quantile(values, weights, lower_q)
            upper[sample_idx] = self._weighted_quantile(values, weights, upper_q)
            var[sample_idx] = float(
                np.sum(weights * (values - mean_value) ** 2) + eps_var
            )

            first_term = np.sum(weights * np.abs(values - y_true[sample_idx]))
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            sorted_weights = weights[order]
            cumulative_weights = np.cumsum(sorted_weights) - sorted_weights
            cumulative_weighted_values = np.cumsum(
                sorted_weights * sorted_values
            ) - sorted_weights * sorted_values
            second_term = np.sum(
                sorted_weights
                * (
                    sorted_values * cumulative_weights
                    - cumulative_weighted_values
                )
            )
            sample_crps[sample_idx] = first_term - second_term

        return mean, median, lower, upper, var, sample_crps


# Prediction and output

def inverse_y(y_norm, scaler_y):
    y_std = float(np.squeeze(scaler_y.std))
    y_mean = float(np.squeeze(scaler_y.mean))
    return y_norm * y_std + y_mean


def inverse_var(var_norm, scaler_y):
    y_std = float(np.squeeze(scaler_y.std))
    return var_norm * (y_std ** 2)


def save_static_information(cfg, raw_data, scaler_grid, scaler_global, scaler_y):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    split_info = []
    for split_name, data in raw_data.items():
        n_raw = len(data["y_raw"])
        n_eff = n_raw - cfg.input_len - cfg.max_lead_steps + 1
        if cfg.stride > 1:
            n_eff = int(np.floor((n_eff - 1) / cfg.stride) + 1)

        split_info.append({
            "Split": split_name,
            "Raw_sample_size": n_raw,
            "Input_window_samples": cfg.input_len,
            "Input_window_hours": cfg.input_window_hours,
            "Max_lead_samples": cfg.max_lead_steps,
            "Max_lead_hours": cfg.max_horizon_step,
            "Selected_horizon_steps_hours": str(cfg.horizon_steps),
            "Stride_samples": cfg.stride,
            "Effective_sample_size_each_horizon": n_eff,
            "Dropped_samples_due_to_windowing": n_raw - n_eff,
            "Window_alignment": cfg.window_alignment,
        })

    pd.DataFrame(split_info).to_excel(
        cfg.output_dir / "data_split_and_window_sample_size_summary.xlsx",
        index=False,
    )

    scaler_info = {
        "Scaler": [],
        "Shape_mean": [],
        "Shape_std": [],
        "Mean_values": [],
        "Std_values": [],
    }
    for name, scaler in [
        ("grid_features", scaler_grid),
        ("global_ANE_feature", scaler_global),
        ("target_ANE_wind_speed", scaler_y),
    ]:
        scaler_info["Scaler"].append(name)
        scaler_info["Shape_mean"].append(str(np.asarray(scaler.mean).shape))
        scaler_info["Shape_std"].append(str(np.asarray(scaler.std).shape))
        scaler_info["Mean_values"].append(
            np.array2string(np.squeeze(scaler.mean), precision=6, separator=", ")
        )
        scaler_info["Std_values"].append(
            np.array2string(np.squeeze(scaler.std), precision=6, separator=", ")
        )

    pd.DataFrame(scaler_info).to_excel(
        cfg.output_dir / "training_only_scaler_information.xlsx",
        index=False,
    )

    feature_info = pd.DataFrame({
        "Feature_group": [
            "ECMWF_10m_mean_wind_speed",
            "ECMWF_100m_mean_wind_speed",
            "ECMWF_10m_gust_wind_speed",
            "ANE_historical_observed_wind_speed",
            "Target_ANE_observed_wind_speed",
        ],
        "Excel_columns": [
            "B:J",
            "K:S",
            "T:AB",
            "AC",
            "AC shifted according to horizon",
        ],
        "Used_as": [
            "grid input",
            "grid input",
            "grid input",
            "historical input",
            "prediction target",
        ],
    })
    feature_info.to_excel(cfg.output_dir / "feature_definition.xlsx", index=False)

    model_info = pd.DataFrame({
        "Item": [
            "Model",
            "Input representation",
            "Number of ECMWF grid nodes",
            "Grid features per node",
            "Global historical features",
            "Input dimension per timestep",
            "Total flattened input dimension",
            "QRF weighting rule",
            "Probabilistic output",
        ],
        "Value": [
            "Quantile Random Forest",
            "Flattened 24 h sequence of grid-node features and historical ANE wind speed",
            cfg.num_nodes,
            cfg.num_grid_features,
            cfg.num_global_features,
            cfg.input_dim_per_timestep,
            cfg.input_len * cfg.input_dim_per_timestep,
            "Normalize observations within each reached leaf, then average leaf distributions equally across trees",
            "Weighted mean, median, variance, and empirical quantiles",
        ],
    })
    model_info.to_excel(cfg.output_dir / "model_input_information.xlsx", index=False)


def train_one_horizon(cfg, transformed_data, scaler_y, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour
    horizon_dir = cfg.output_dir / f"horizon_{horizon_id:02d}h_ahead"
    horizon_dir.mkdir(parents=True, exist_ok=True)

    actual_seed = cfg.seed + horizon_id
    set_seed(actual_seed)

    qrf_data, sample_metas = prepare_data_for_horizon(
        cfg=cfg,
        transformed_data=transformed_data,
        horizon_id=horizon_id,
    )

    train_x = qrf_data["Train"]["x"]
    train_y = qrf_data["Train"]["y"]

    model = QuantileRandomForest(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_split=cfg.min_samples_split,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features=cfg.max_features,
        bootstrap=cfg.bootstrap,
        n_jobs=cfg.n_jobs,
        random_state=actual_seed,
    )

    print("\n" + "=" * 80)
    print(f"Training QRF: {horizon_id} h ahead ({lead_steps} samples)")
    print(f"Input window: {cfg.input_len} samples ({cfg.input_window_hours} h)")
    print(f"Window alignment: {cfg.window_alignment}; stride: {cfg.stride}")
    print(f"Flattened input dimension: {train_x.shape[1]}")
    for split_name in ["Train", "Validation", "Test"]:
        print(f"{split_name} samples: {qrf_data[split_name]['x'].shape[0]}")
    print("=" * 80)

    model.fit(train_x, train_y)
    print("QRF training completed.")

    training_info = pd.DataFrame({
        "Item": [
            "Model",
            "Horizon_hours_ahead",
            "Lead_samples",
            "Input_window_samples",
            "Input_window_hours",
            "Window_alignment",
            "Flattened_input_dimension",
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "random_state",
            "QRF_weighting_rule",
            "Lower_quantile",
            "Upper_quantile",
        ],
        "Value": [
            "Standard Quantile Random Forest",
            horizon_id,
            lead_steps,
            cfg.input_len,
            cfg.input_window_hours,
            cfg.window_alignment,
            train_x.shape[1],
            cfg.n_estimators,
            str(cfg.max_depth),
            cfg.min_samples_split,
            cfg.min_samples_leaf,
            str(cfg.max_features),
            cfg.bootstrap,
            actual_seed,
            "Within-leaf normalization followed by equal-weight averaging across trees",
            cfg.lower_quantile,
            cfg.upper_quantile,
        ],
    })
    training_info.to_excel(
        horizon_dir / f"qrf_training_information_{horizon_id:02d}h_ahead.xlsx",
        index=False,
    )

    feature_importance = pd.DataFrame({
        "Feature_index_1based": np.arange(train_x.shape[1]) + 1,
        "Importance": model.rf.feature_importances_,
    })
    feature_importance.to_excel(
        horizon_dir / f"feature_importance_{horizon_id:02d}h_ahead.xlsx",
        index=False,
    )

    prediction_records = {}

    for split_name in ["Train", "Validation", "Test"]:
        x = qrf_data[split_name]["x"]
        y_norm = qrf_data[split_name]["y"]

        mean_norm, median_norm, lower_norm, upper_norm, var_norm, crps_norm = model.predict(
            x,
            y_true=y_norm,
            lower_q=cfg.lower_quantile,
            upper_q=cfg.upper_quantile,
            eps_var=cfg.eps_var,
        )

        y = inverse_y(y_norm, scaler_y).reshape(-1)
        mean = inverse_y(mean_norm, scaler_y).reshape(-1)
        median = inverse_y(median_norm, scaler_y).reshape(-1)
        lower = inverse_y(lower_norm, scaler_y).reshape(-1)
        upper = inverse_y(upper_norm, scaler_y).reshape(-1)
        var = inverse_var(var_norm, scaler_y).reshape(-1)
        y_std = float(np.squeeze(scaler_y.std))
        sample_crps = (crps_norm * y_std).reshape(-1)
        sigma = np.sqrt(np.maximum(var, 0.0))
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
            "Predicted_median": median,
            "Predicted_variance": var,
            "Predicted_std": sigma,
            "Lower_95PI": lower,
            "Upper_95PI": upper,
            "Residual": y - mean,
            "Residual_median": y - median,
            "Sample_CRPS": sample_crps,
        })

        pred_path = horizon_dir / f"{split_name.lower()}_predictions_{horizon_id:02d}h_ahead.xlsx"
        pred_df.to_excel(pred_path, index=False)
        prediction_records[split_name] = pred_df
        print(f"Saved {split_name.lower()} predictions: {pred_path}")

    return model, prediction_records, training_info


def save_combined_outputs(cfg, all_predictions, all_training_info):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["Train", "Validation", "Test"]:
        prediction_list = [
            all_predictions[horizon_id][split_name].copy()
            for horizon_id in cfg.horizon_steps
        ]
        pd.concat(prediction_list, axis=0, ignore_index=True).to_excel(
            cfg.output_dir / f"{split_name.lower()}_predictions_all_horizons_long.xlsx",
            index=False,
        )

        first_horizon = cfg.horizon_steps[0]
        first_df = all_predictions[first_horizon][split_name]
        wide_df = first_df[["Sample_index", "Target_time", "Observed"]].copy()

        for horizon_id in cfg.horizon_steps:
            pred_df = all_predictions[horizon_id][split_name].copy()

            if len(pred_df) != len(wide_df):
                raise ValueError(
                    f"Sample count mismatch in {split_name} for horizon {horizon_id}."
                )
            if not np.array_equal(
                pred_df["Target_time"].values, wide_df["Target_time"].values
            ):
                raise ValueError(
                    f"Target times are not aligned in {split_name} for horizon {horizon_id}."
                )
            if not np.allclose(
                pred_df["Observed"].astype(float).values,
                wide_df["Observed"].astype(float).values,
                rtol=1e-6,
                atol=1e-6,
            ):
                raise ValueError(
                    f"Observed values are not aligned in {split_name} for horizon {horizon_id}."
                )

            prefix = f"H{horizon_id:02d}_"
            columns = [
                "Input_start_index_0based",
                "Input_end_index_0based",
                "Input_start_time",
                "Input_end_time",
                "Predicted_mean",
                "Predicted_median",
                "Predicted_variance",
                "Predicted_std",
                "Lower_95PI",
                "Upper_95PI",
                "Residual",
                "Residual_median",
                "Sample_CRPS",
            ]
            add_cols = pred_df[columns].rename(
                columns={column: prefix + column for column in columns}
            )
            wide_df = pd.concat([wide_df, add_cols], axis=1)

        wide_df.to_excel(
            cfg.output_dir / f"{split_name.lower()}_predictions_all_horizons_wide.xlsx",
            index=False,
        )

    pd.concat(all_training_info, axis=0, ignore_index=True).to_excel(
        cfg.output_dir / "qrf_training_information_all_horizons.xlsx",
        index=False,
    )


# Main

def train_all_horizons(cfg):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {cfg.output_dir}")
    print(f"Selected horizons: {cfg.horizon_steps} h ahead")
    print("Loading data splits...")
    raw_data = load_raw_splits(cfg)

    print("Fitting scalers on the training split only...")
    scaler_grid, scaler_global, scaler_y = fit_scalers_on_training(raw_data)
    transformed_data = transform_raw_splits(
        raw_data, scaler_grid, scaler_global, scaler_y
    )
    save_static_information(
        cfg, raw_data, scaler_grid, scaler_global, scaler_y
    )

    all_predictions = {}
    all_training_info = []

    for horizon_id in cfg.horizon_steps:
        model, prediction_records, training_info = train_one_horizon(
            cfg=cfg,
            transformed_data=transformed_data,
            scaler_y=scaler_y,
            horizon_id=horizon_id,
        )
        all_predictions[horizon_id] = prediction_records
        all_training_info.append(training_info.assign(Horizon_hours_ahead=horizon_id))
        del model

    save_combined_outputs(cfg, all_predictions, all_training_info)

    print("\nAll prediction files and model information have been saved to:")
    print(cfg.output_dir)

    return all_predictions


if __name__ == "__main__":
    set_seed(cfg.seed)
    train_all_horizons(cfg)
