from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# Configuration

class Config:
    data_dir = Path("path/to/your/data")

    train_file = "train.xlsx"
    val_file = "validation.xlsx"
    test_file = "test.xlsx"
    output_dir = data_dir / "results_qrf"

    seed = 2026

    sequence_length = 24
    rolling_stride = 1
    forecast_horizon = 1

    num_nodes = 9
    num_grid_features = 6
    num_global_features = 2

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


cfg = Config()


# Reproducibility

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# Data loading and preprocessing

def read_station_excel(file_path):
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_excel(file_path, engine="openpyxl")
    if df.shape[1] < 58:
        raise ValueError(
            f"{file_path} must contain at least 58 columns: "
            "time, 56 input features, and target."
        )

    time = df.iloc[:, 0].to_numpy()
    x_raw = df.iloc[:, 1:57].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=np.float32)
    y_raw = pd.to_numeric(
        df.iloc[:, 57], errors="raise"
    ).to_numpy(dtype=np.float32).reshape(-1, 1)

    if len(time) != len(y_raw) or len(x_raw) != len(y_raw):
        raise ValueError(f"Inconsistent row counts in {file_path}.")
    if len(y_raw) == 0:
        raise ValueError(f"No samples found in {file_path}.")
    if not np.isfinite(x_raw).all() or not np.isfinite(y_raw).all():
        raise ValueError(f"Non-finite values found in {file_path}.")

    return time, x_raw, y_raw


def split_raw_features(x_raw, num_nodes):
    if x_raw.ndim != 2 or x_raw.shape[1] != 56:
        raise ValueError("QRF expects 56 input features.")
    if x_raw.shape[1] != num_nodes * 6 + 2:
        raise ValueError(
            "Expected six groups of nine ECMWF grid features followed "
            "by two historical ANE features."
        )

    groups = [
        x_raw[:, group_index * num_nodes:(group_index + 1) * num_nodes]
        for group_index in range(6)
    ]
    x_grid = np.stack(groups, axis=1).transpose(0, 2, 1).astype(np.float32)
    x_global = x_raw[:, 54:56].astype(np.float32)
    return x_grid, x_global


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


def flatten_rows_for_qrf(x_grid, x_global, y):

    sample_count, node_count, feature_count = x_grid.shape
    x_grid_flat = x_grid.reshape(sample_count, node_count * feature_count)
    x = np.concatenate([x_grid_flat, x_global], axis=1).astype(np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    return x, y


def build_rolling_windows(
    x_rows,
    y,
    time,
    sequence_length,
    stride=1,
    history_x=None,
):


    x_rows = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    time = np.asarray(time)

    if len(x_rows) != len(y) or len(time) != len(y):
        raise ValueError("Input features, targets, and timestamps must align.")
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")

    history_length = sequence_length - 1
    if history_x is None or history_length == 0:
        history_x = np.empty((0, x_rows.shape[1]), dtype=np.float32)
    else:
        history_x = np.asarray(history_x, dtype=np.float32)[-history_length:]

    combined_x = np.concatenate([history_x, x_rows], axis=0)
    offset = len(history_x)

    x_windows = []
    y_targets = []
    time_targets = []

    for current_idx in range(0, len(y), stride):
        combined_end = offset + current_idx
        combined_start = combined_end - sequence_length + 1


        if combined_start < 0:
            continue

        window = combined_x[combined_start:combined_end + 1]
        if len(window) != sequence_length:
            continue

        x_windows.append(window.reshape(-1))
        y_targets.append(y[current_idx])
        time_targets.append(time[current_idx])

    if not x_windows:
        raise ValueError(
            f"No rolling windows could be built for sequence_length={sequence_length}."
        )

    return (
        np.asarray(x_windows, dtype=np.float32),
        np.asarray(y_targets, dtype=np.float32),
        np.asarray(time_targets),
    )


def prepare_data(cfg):
    if cfg.forecast_horizon != 1 or cfg.rolling_stride != 1:
        raise ValueError(
            "This Case II QRF implementation requires hourly rolling "
            "one-step forecasting."
        )
    if cfg.sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")

    split_files = {
        "Train": cfg.data_dir / cfg.train_file,
        "Validation": cfg.data_dir / cfg.val_file,
        "Test": cfg.data_dir / cfg.test_file,
    }

    raw_data = {
        name: read_station_excel(path)
        for name, path in split_files.items()
    }


    split_features = {}
    for split_name, (_, x_raw, y_raw) in raw_data.items():
        x_grid, x_global = split_raw_features(x_raw, cfg.num_nodes)
        split_features[split_name] = (x_grid, x_global, y_raw)


    scaler_grid = StandardScaler()
    scaler_global = StandardScaler()
    scaler_y = StandardScaler()

    scaler_grid.fit(split_features["Train"][0], axis=(0, 1))
    scaler_global.fit(split_features["Train"][1], axis=0)
    scaler_y.fit(split_features["Train"][2], axis=0)


    row_data = {}
    for split_name, (x_grid, x_global, y_raw) in split_features.items():
        x_grid_scaled = scaler_grid.transform(x_grid)
        x_global_scaled = scaler_global.transform(x_global)
        y_scaled = scaler_y.transform(y_raw)

        x_rows, y_rows = flatten_rows_for_qrf(
            x_grid_scaled, x_global_scaled, y_scaled
        )
        row_data[split_name] = {
            "x": x_rows,
            "y": y_rows,
            "time": raw_data[split_name][0],
        }


    train_x, train_y, train_time = build_rolling_windows(
        row_data["Train"]["x"],
        row_data["Train"]["y"],
        row_data["Train"]["time"],
        sequence_length=cfg.sequence_length,
        stride=cfg.rolling_stride,
    )

    val_x, val_y, val_time = build_rolling_windows(
        row_data["Validation"]["x"],
        row_data["Validation"]["y"],
        row_data["Validation"]["time"],
        sequence_length=cfg.sequence_length,
        stride=cfg.rolling_stride,
        history_x=row_data["Train"]["x"],
    )

    pre_test_x = np.concatenate(
        [row_data["Train"]["x"], row_data["Validation"]["x"]], axis=0
    )
    test_x, test_y, test_time = build_rolling_windows(
        row_data["Test"]["x"],
        row_data["Test"]["y"],
        row_data["Test"]["time"],
        sequence_length=cfg.sequence_length,
        stride=cfg.rolling_stride,
        history_x=pre_test_x,
    )

    data = {
        "Train": (train_x, train_y, train_time),
        "Validation": (val_x, val_y, val_time),
        "Test": (test_x, test_y, test_time),
    }

    meta = {
        "scaler_y": scaler_y,
        "train_raw_sample_count": len(row_data["Train"]["y"]),
        "validation_raw_sample_count": len(row_data["Validation"]["y"]),
        "test_raw_sample_count": len(row_data["Test"]["y"]),
        "train_sample_count": len(train_y),
        "validation_sample_count": len(val_y),
        "test_sample_count": len(test_y),
        "input_feature_count_per_hour": row_data["Train"]["x"].shape[1],
        "flattened_input_feature_count": train_x.shape[1],
        "sequence_length": cfg.sequence_length,
    }
    return data, meta


# Quantile Regression Forest

class QuantileRandomForest:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rf = RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_split=cfg.min_samples_split,
            min_samples_leaf=cfg.min_samples_leaf,
            max_features=cfg.max_features,
            bootstrap=cfg.bootstrap,
            n_jobs=cfg.n_jobs,
            random_state=cfg.seed,
        )
        self.leaf_y_by_tree = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
        self.rf.fit(x_train, self.y_train)

        train_leaf_ids = self.rf.apply(x_train)
        self.leaf_y_by_tree = []

        for tree_index in range(train_leaf_ids.shape[1]):
            leaf_map = {}
            for leaf_id, y_value in zip(
                train_leaf_ids[:, tree_index], self.y_train
            ):
                leaf_map.setdefault(leaf_id, []).append(float(y_value))

            self.leaf_y_by_tree.append({
                leaf_id: np.asarray(values, dtype=np.float32)
                for leaf_id, values in leaf_map.items()
            })

        return self

    def _weighted_distribution_for_one_sample(self, sample_leaf_ids):
        values = []
        weights = []
        tree_count = len(sample_leaf_ids)

        for tree_index, leaf_id in enumerate(sample_leaf_ids):
            leaf_values = self.leaf_y_by_tree[tree_index].get(leaf_id)

            if leaf_values is not None and leaf_values.size > 0:
                values.extend(leaf_values.tolist())
                weights.extend(
                    [1.0 / tree_count / leaf_values.size] * leaf_values.size
                )

        if not values:
            values = self.y_train.tolist()
            weights = [1.0 / len(self.y_train)] * len(self.y_train)

        values = np.asarray(values, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()
        return values, weights

    def predict_distribution(self, x):
        predicted_leaf_ids = self.rf.apply(x)
        return [
            self._weighted_distribution_for_one_sample(leaf_ids)
            for leaf_ids in predicted_leaf_ids
        ]

    @staticmethod
    def _weighted_quantile(values, weights, quantile):
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be between 0 and 1.")

        values = np.asarray(values, dtype=np.float64).reshape(-1)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)

        if values.size == 0:
            raise ValueError("values must not be empty.")
        if values.size != weights.size:
            raise ValueError("values and weights must have the same size.")
        if not np.isfinite(values).all() or not np.isfinite(weights).all():
            raise ValueError("values and weights must be finite.")
        if np.any(weights < 0.0):
            raise ValueError("weights must be non-negative.")

        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            raise ValueError("The sum of weights must be positive.")

        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_weights = weights[order]

        unique_values, first_indices = np.unique(
            sorted_values,
            return_index=True,
        )
        unique_weights = np.add.reduceat(
            sorted_weights,
            first_indices,
        )

        unique_weights = unique_weights / np.sum(unique_weights)
        cumulative_weights = np.cumsum(unique_weights)
        cumulative_weights[-1] = 1.0

        if unique_values.size == 1:
            return float(unique_values[0])

        if quantile <= cumulative_weights[0]:
            return float(unique_values[0])
        if quantile >= cumulative_weights[-1]:
            return float(unique_values[-1])

        upper_index = int(
            np.searchsorted(
                cumulative_weights,
                quantile,
                side="right",
            )
        )
        lower_index = upper_index - 1

        lower_probability = cumulative_weights[lower_index]
        upper_probability = cumulative_weights[upper_index]
        lower_value = unique_values[lower_index]
        upper_value = unique_values[upper_index]

        if upper_probability <= lower_probability:
            return float(lower_value)

        interpolation_fraction = (
            (quantile - lower_probability)
            / (upper_probability - lower_probability)
        )

        return float(
            lower_value
            + interpolation_fraction * (upper_value - lower_value)
        )
    def predict(self, x, lower_q, upper_q):
        distributions = self.predict_distribution(x)
        sample_count = len(distributions)

        mean = np.empty(sample_count, dtype=np.float64)
        median = np.empty(sample_count, dtype=np.float64)
        lower = np.empty(sample_count, dtype=np.float64)
        upper = np.empty(sample_count, dtype=np.float64)
        variance = np.empty(sample_count, dtype=np.float64)

        for index, (values, weights) in enumerate(distributions):
            mean[index] = np.sum(values * weights)
            median[index] = self._weighted_quantile(values, weights, 0.5)
            lower[index] = self._weighted_quantile(values, weights, lower_q)
            upper[index] = self._weighted_quantile(values, weights, upper_q)
            variance[index] = (
                np.sum(weights * (values - mean[index]) ** 2)
                + self.cfg.eps_var
            )

        return mean, median, lower, upper, variance, distributions


def empirical_crps_one(observed, distribution):
    values, weights = distribution
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)

    if values.size == 0:
        return np.nan

    weights = weights / np.sum(weights)
    term_1 = np.sum(weights * np.abs(values - observed))

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative_weights_before = np.concatenate(
        ([0.0], np.cumsum(sorted_weights)[:-1])
    )
    cumulative_weighted_values_before = np.concatenate(
        ([0.0], np.cumsum(sorted_weights * sorted_values)[:-1])
    )
    term_2 = np.sum(
        sorted_weights
        * (
            sorted_values * cumulative_weights_before
            - cumulative_weighted_values_before
        )
    )
    return float(term_1 - term_2)


def empirical_crps_per_sample(y_true, distributions):
    return np.asarray(
        [
            empirical_crps_one(y_value, distribution)
            for y_value, distribution in zip(
                np.asarray(y_true).reshape(-1), distributions
            )
        ],
        dtype=np.float64,
    )


# Inverse transformation

def inverse_y(y_scaled, scaler_y):
    scale = float(np.squeeze(scaler_y.std))
    offset = float(np.squeeze(scaler_y.mean))
    return np.asarray(y_scaled) * scale + offset


def inverse_variance(variance_scaled, scaler_y):
    scale = float(np.squeeze(scaler_y.std))
    return np.asarray(variance_scaled) * (scale ** 2)


def inverse_distribution_list(distributions_scaled, scaler_y):
    scale = float(np.squeeze(scaler_y.std))
    offset = float(np.squeeze(scaler_y.mean))
    return [
        (values * scale + offset, weights.copy())
        for values, weights in distributions_scaled
    ]


# Training and prediction

def train_model(cfg):
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    data, meta = prepare_data(cfg)
    train_x, train_y, _ = data["Train"]

    model = QuantileRandomForest(cfg)
    print("Training standard Quantile Regression Forest.")
    print(f"Sequence length: {meta['sequence_length']} hourly rows")
    print(f"Input features per hour: {meta['input_feature_count_per_hour']}")
    print(f"Flattened QRF input features: {meta['flattened_input_feature_count']}")
    print(f"Training samples after windowing: {meta['train_sample_count']}")
    print(f"Validation samples: {meta['validation_sample_count']}")
    print(f"Test samples: {meta['test_sample_count']}")

    model.fit(train_x, train_y)

    for split_name in ("Train", "Validation", "Test"):
        x, y_scaled, time_target = data[split_name]

        (
            mean_scaled,
            median_scaled,
            lower_scaled,
            upper_scaled,
            variance_scaled,
            distributions_scaled,
        ) = model.predict(
            x,
            lower_q=cfg.lower_quantile,
            upper_q=cfg.upper_quantile,
        )

        observed = inverse_y(y_scaled, meta["scaler_y"])
        mean = inverse_y(mean_scaled, meta["scaler_y"])
        median = inverse_y(median_scaled, meta["scaler_y"])
        lower = inverse_y(lower_scaled, meta["scaler_y"])
        upper = inverse_y(upper_scaled, meta["scaler_y"])
        variance = inverse_variance(variance_scaled, meta["scaler_y"])
        distributions = inverse_distribution_list(
            distributions_scaled, meta["scaler_y"]
        )

        sample_crps = empirical_crps_per_sample(
            y_true=observed,
            distributions=distributions,
        )

        pred_df = pd.DataFrame({
            "Time": np.asarray(time_target).reshape(-1),
            "Observed": observed.reshape(-1),
            "Predicted_mean": mean.reshape(-1),
            "Predicted_median": median.reshape(-1),
            "Predicted_variance": variance.reshape(-1),
            "Predicted_std": np.sqrt(np.maximum(variance.reshape(-1), 0.0)),
            "Lower_95PI": lower.reshape(-1),
            "Upper_95PI": upper.reshape(-1),
            "Residual_mean": observed.reshape(-1) - mean.reshape(-1),
            "Residual_median": observed.reshape(-1) - median.reshape(-1),
            "Sample_CRPS": sample_crps.reshape(-1),
        })

        prediction_path = (
            cfg.output_dir / f"{split_name}_prediction_results.xlsx"
        )
        pred_df.to_excel(prediction_path, index=False)
        print(
            f"{split_name}: saved {len(pred_df)} sample-level predictions "
            f"to {prediction_path}"
        )

    protocol_df = pd.DataFrame([{
        "Model": "Standard Quantile Regression Forest",
        "Reference": "Meinshausen (2006)",
        "Rolling_stride_h": cfg.rolling_stride,
        "Forecast_horizon_h": cfg.forecast_horizon,
        "Input_protocol": (
            "Each hourly row contains ECMWF predictors valid at that row's "
            "target time plus ANE_(t-2) and ANE_(t-1); the latest 24 rows are "
            "used as the causal input window"
        ),
        "Prediction_protocol": "Hourly rolling one-step probabilistic prediction",
        "Sequence_length_h": cfg.sequence_length,
        "Temporal_window": (
            f"{cfg.sequence_length} consecutive hourly rows, flattened into "
            "a single tabular feature vector for QRF"
        ),
        "Scaler_fit_data": "Training set only",
        "Raw_training_rows": meta["train_raw_sample_count"],
        "Raw_validation_rows": meta["validation_raw_sample_count"],
        "Raw_test_rows": meta["test_raw_sample_count"],
        "Windowed_training_samples": meta["train_sample_count"],
        "Windowed_validation_samples": meta["validation_sample_count"],
        "Windowed_test_samples": meta["test_sample_count"],
        "Conditional_distribution": (
            "Normalize observations within each reached leaf, then average "
            "the leaf distributions equally across trees"
        ),
        "Quantile_extraction": (
            "Linear interpolation of the weighted empirical conditional distribution after aggregating duplicate support values"
        ),
        "Sample_CRPS": (
            "Weighted empirical CRPS computed independently for every sample"
        ),
        "Target_interval": "Central 95% interval from weighted empirical quantiles",
    }])
    protocol_df.to_excel(
        cfg.output_dir / "forecast_protocol.xlsx",
        index=False,
    )

    hourly_feature_names = []
    ecmwf_feature_groups = [
        "ECMWF_gust_10m",
        "ECMWF_mean_10m",
        "ECMWF_temperature_800hPa",
        "ECMWF_temperature_850hPa",
        "ECMWF_wind_speed_800hPa",
        "ECMWF_wind_speed_850hPa",
    ]
    for node_index in range(cfg.num_nodes):
        for group_name in ecmwf_feature_groups:
            hourly_feature_names.append(
                f"{group_name}_node_{node_index + 1}"
            )
    hourly_feature_names.extend(["ANE_t_minus_2_BD", "ANE_t_minus_1_BE"])

    importance_rows = []
    importance_values = model.rf.feature_importances_.reshape(
        cfg.sequence_length, len(hourly_feature_names)
    )
    for window_position in range(cfg.sequence_length):
        lag_hours = cfg.sequence_length - 1 - window_position
        for feature_index, feature_name in enumerate(hourly_feature_names):
            importance_rows.append({
                "Window_position_oldest_to_current": window_position + 1,
                "Lag_before_target_h": lag_hours,
                "Feature_index_within_hour": feature_index + 1,
                "Feature_name": feature_name,
                "Importance": importance_values[window_position, feature_index],
            })

    feature_importance_df = pd.DataFrame(importance_rows)
    feature_importance_df.to_excel(
        cfg.output_dir / "feature_importance.xlsx",
        index=False,
    )

    print(f"All outputs saved to: {cfg.output_dir}")
    return model


if __name__ == "__main__":
    train_model(cfg)
