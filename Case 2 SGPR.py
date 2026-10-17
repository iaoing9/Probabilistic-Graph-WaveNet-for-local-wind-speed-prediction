import math
import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


# Configuration

class Config:
    data_dir = Path("path/to/your/data")

    train_file = "train.xlsx"
    val_file = "validation.xlsx"
    test_file = "test.xlsx"
    output_dir = data_dir / "results_sgpr"

    seed = 2026
    sequence_length = 24
    rolling_stride = 1
    forecast_horizon = 1

    num_nodes = 9
    num_grid_features = 6
    num_global_features = 2
    input_dim = num_nodes * num_grid_features + num_global_features

    num_inducing = 512
    inducing_selection = "random"
    optimize_kernel_on_inducing_subset = True
    n_restarts_optimizer = 0

    kernel_signal_variance = 1.0
    kernel_signal_variance_bounds = (1e-3, 1e3)
    kernel_length_scale = 1.0
    kernel_length_scale_bounds = (1e-2, 1e2)
    kernel_noise_variance = 1e-2
    kernel_noise_variance_bounds = (1e-6, 1e1)
    alpha = 1e-8
    jitter = 1e-6
    length_scale_sample_size = 1000

    eps_var = 1e-6
    z_value = 1.959963984540054
    prediction_batch_size = 4096


cfg = Config()


# Reproducibility

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


set_seed(cfg.seed)


# Data loading and preprocessing

def read_station_excel(file_path, input_dim):
    df = pd.read_excel(file_path, engine="openpyxl")
    expected_columns = input_dim + 2
    if df.shape[1] < expected_columns:
        raise ValueError(
            f"{file_path} contains {df.shape[1]} columns; "
            f"at least {expected_columns} are required."
        )

    time = df.iloc[:, 0].to_numpy()
    x_raw = df.iloc[:, 1:1 + input_dim].to_numpy(dtype=np.float64)
    y_raw = df.iloc[:, 1 + input_dim].to_numpy(dtype=np.float64).reshape(-1, 1)

    if not np.isfinite(x_raw).all() or not np.isfinite(y_raw).all():
        raise ValueError(
            f"Non-finite values found in the input or target columns of {file_path}."
        )

    return time, x_raw, y_raw


def split_raw_features(x_raw, cfg):
    expected_features = cfg.num_nodes * cfg.num_grid_features + cfg.num_global_features
    if x_raw.ndim != 2 or x_raw.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} input features, got {x_raw.shape}."
        )

    grid_groups = [
        x_raw[:, i * cfg.num_nodes:(i + 1) * cfg.num_nodes]
        for i in range(cfg.num_grid_features)
    ]
    x_grid = np.stack(grid_groups, axis=1).transpose(0, 2, 1).astype(np.float64)
    global_start = cfg.num_nodes * cfg.num_grid_features
    x_global = x_raw[
        :, global_start:global_start + cfg.num_global_features
    ].astype(np.float64)

    return x_grid, x_global


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data, axis):
        self.mean = np.mean(data, axis=axis, keepdims=True)
        self.std = np.std(data, axis=axis, keepdims=True)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def transform(self, data):
        return (data - self.mean) / self.std


def prepare_data(cfg):
    if cfg.forecast_horizon != 1 or cfg.rolling_stride != 1:
        raise ValueError(
            "The Case II protocol requires hourly rolling one-step forecasts."
        )
    if cfg.sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")

    train_path = cfg.data_dir / cfg.train_file
    val_path = cfg.data_dir / cfg.val_file
    test_path = cfg.data_dir / cfg.test_file

    for path in (train_path, val_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

    train_time, train_x_raw, train_y_raw = read_station_excel(
        train_path, cfg.input_dim
    )
    val_time, val_x_raw, val_y_raw = read_station_excel(
        val_path, cfg.input_dim
    )
    test_time, test_x_raw, test_y_raw = read_station_excel(
        test_path, cfg.input_dim
    )

    train_x_grid, train_x_global = split_raw_features(train_x_raw, cfg)
    val_x_grid, val_x_global = split_raw_features(val_x_raw, cfg)
    test_x_grid, test_x_global = split_raw_features(test_x_raw, cfg)

    scaler_grid = StandardScaler()
    scaler_global = StandardScaler()
    scaler_y = StandardScaler()

    scaler_grid.fit(train_x_grid, axis=(0, 1))
    scaler_global.fit(train_x_global, axis=0)
    scaler_y.fit(train_y_raw, axis=0)

    train_x_grid = scaler_grid.transform(train_x_grid)
    val_x_grid = scaler_grid.transform(val_x_grid)
    test_x_grid = scaler_grid.transform(test_x_grid)

    train_x_global = scaler_global.transform(train_x_global)
    val_x_global = scaler_global.transform(val_x_global)
    test_x_global = scaler_global.transform(test_x_global)

    train_y = scaler_y.transform(train_y_raw)
    val_y = scaler_y.transform(val_y_raw)
    test_y = scaler_y.transform(test_y_raw)

    train_x_rows = flatten_features_for_sgpr(train_x_grid, train_x_global)
    val_x_rows = flatten_features_for_sgpr(val_x_grid, val_x_global)
    test_x_rows = flatten_features_for_sgpr(test_x_grid, test_x_global)

    train_x, train_y_window, train_time_window = build_rolling_windows(
        train_x_rows,
        train_y,
        train_time,
        sequence_length=cfg.sequence_length,
        stride=cfg.rolling_stride,
    )

    val_x, val_y_window, val_time_window = build_rolling_windows(
        val_x_rows,
        val_y,
        val_time,
        sequence_length=cfg.sequence_length,
        stride=cfg.rolling_stride,
        history_x=train_x_rows,
    )

    prior_test_history = np.concatenate([train_x_rows, val_x_rows], axis=0)
    test_x, test_y_window, test_time_window = build_rolling_windows(
        test_x_rows,
        test_y,
        test_time,
        sequence_length=cfg.sequence_length,
        stride=cfg.rolling_stride,
        history_x=prior_test_history,
    )

    data = {
        "Train": (train_x, train_y_window),
        "Validation": (val_x, val_y_window),
        "Test": (test_x, test_y_window),
    }
    meta = {
        "scaler_y": scaler_y,
        "train_time": train_time_window,
        "val_time": val_time_window,
        "test_time": test_time_window,
        "train_raw_size": len(train_y_raw),
        "val_raw_size": len(val_y_raw),
        "test_raw_size": len(test_y_raw),
        "train_size": len(train_y_window),
        "val_size": len(val_y_window),
        "test_size": len(test_y_window),
        "input_dim_per_hour": cfg.input_dim,
        "flattened_input_dim": cfg.input_dim * cfg.sequence_length,
        "sequence_length": cfg.sequence_length,
    }

    return data, meta


def flatten_features_for_sgpr(x_grid, x_global):
    num_samples, num_nodes, num_grid_features = x_grid.shape
    x_grid_flat = x_grid.reshape(
        num_samples, num_nodes * num_grid_features
    )
    return np.concatenate([x_grid_flat, x_global], axis=1).astype(np.float64)



def build_rolling_windows(
    x_rows,
    y,
    time,
    sequence_length,
    stride=1,
    history_x=None,
):
    x_rows = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    time = np.asarray(time)

    if len(x_rows) != len(y) or len(time) != len(y):
        raise ValueError("Input features, targets, and timestamps must align.")
    if sequence_length < 1 or stride < 1:
        raise ValueError("sequence_length and stride must be positive.")

    history_length = sequence_length - 1
    if history_x is None or history_length == 0:
        history_x = np.empty((0, x_rows.shape[1]), dtype=np.float64)
    else:
        history_x = np.asarray(history_x, dtype=np.float64)[-history_length:]

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
        np.asarray(x_windows, dtype=np.float64),
        np.asarray(y_targets, dtype=np.float64).reshape(-1, 1),
        np.asarray(time_targets),
    )


def flatten_y_for_sgpr(y):
    return y.reshape(-1).astype(np.float64)


# Sparse Gaussian Process Regression

def squared_exponential_kernel(x1, x2, signal_variance, length_scale):
    distances_squared = cdist(
        x1 / length_scale,
        x2 / length_scale,
        metric="sqeuclidean",
    )
    return signal_variance * np.exp(-0.5 * distances_squared)


def estimate_median_length_scale(x, seed, max_points):
    rng = np.random.default_rng(seed)
    if x.shape[0] > max_points:
        indices = rng.choice(x.shape[0], size=max_points, replace=False)
        sample = x[indices]
    else:
        sample = x

    distances = cdist(sample, sample, metric="euclidean")
    distances = distances[np.triu_indices_from(distances, k=1)]
    distances = distances[np.isfinite(distances)]
    distances = distances[distances > 0]

    if distances.size == 0:
        return 1.0
    return float(np.median(distances))


def select_inducing_points(x_train, cfg):
    n_samples = x_train.shape[0]
    n_inducing = min(cfg.num_inducing, n_samples)

    if n_inducing == n_samples:
        return x_train.copy(), np.arange(n_samples, dtype=np.int64)

    rng = np.random.default_rng(cfg.seed)

    if cfg.inducing_selection.lower() == "random":
        indices = np.sort(
            rng.choice(n_samples, size=n_inducing, replace=False)
        ).astype(np.int64)
        return x_train[indices].copy(), indices

    if cfg.inducing_selection.lower() == "kmeans":
        kmeans = MiniBatchKMeans(
            n_clusters=n_inducing,
            random_state=cfg.seed,
            batch_size=min(4096, max(256, n_inducing * 4)),
            n_init=3,
            max_iter=100,
            reassignment_ratio=0.01,
        )
        kmeans.fit(x_train)
        inducing_inputs = kmeans.cluster_centers_.astype(np.float64)
        return inducing_inputs, np.arange(n_inducing, dtype=np.int64)

    raise ValueError(
        "inducing_selection must be either 'random' or 'kmeans'."
    )


def optimize_kernel_on_subset(x_inducing, y_subset, cfg):
    kernel = (
        ConstantKernel(
            constant_value=cfg.kernel_signal_variance,
            constant_value_bounds=cfg.kernel_signal_variance_bounds,
        )
        * RBF(
            length_scale=cfg.kernel_length_scale,
            length_scale_bounds=cfg.kernel_length_scale_bounds,
        )
        + WhiteKernel(
            noise_level=cfg.kernel_noise_variance,
            noise_level_bounds=cfg.kernel_noise_variance_bounds,
        )
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=cfg.alpha,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=cfg.n_restarts_optimizer,
        normalize_y=False,
        random_state=cfg.seed,
    )
    gpr.fit(x_inducing, y_subset)

    learned_kernel = gpr.kernel_
    signal_variance = float(learned_kernel.k1.k1.constant_value)
    length_scale = float(np.squeeze(learned_kernel.k1.k2.length_scale))
    noise_variance = float(learned_kernel.k2.noise_level)

    return signal_variance, length_scale, noise_variance, learned_kernel


class SparseGPRFITC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.signal_variance = cfg.kernel_signal_variance
        self.length_scale = cfg.kernel_length_scale
        self.noise_variance = cfg.kernel_noise_variance
        self.xu = None
        self.inducing_indices = None
        self.learned_kernel = None
        self.l_kmm = None
        self.l_a = None
        self.alpha = None
        self.lambda_diag = None

    def _kernel(self, x1, x2):
        return squared_exponential_kernel(
            x1,
            x2,
            signal_variance=self.signal_variance,
            length_scale=self.length_scale,
        )

    def fit(self, x_train, y_train):
        cfg = self.cfg
        x_train = np.asarray(x_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)

        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Training inputs and targets have different sample counts.")
        if x_train.shape[0] == 0:
            raise ValueError("The training set is empty.")

        self.xu, self.inducing_indices = select_inducing_points(x_train, cfg)

        if cfg.optimize_kernel_on_inducing_subset:
            if cfg.inducing_selection.lower() == "random" or len(self.xu) == len(x_train):
                y_inducing = y_train[self.inducing_indices]
            else:
                nearest_indices = np.argmin(
                    cdist(self.xu, x_train, metric="sqeuclidean"),
                    axis=1,
                )
                y_inducing = y_train[nearest_indices]

            (
                self.signal_variance,
                self.length_scale,
                self.noise_variance,
                self.learned_kernel,
            ) = optimize_kernel_on_subset(self.xu, y_inducing, cfg)
        else:
            if self.length_scale is None or not np.isfinite(self.length_scale):
                self.length_scale = estimate_median_length_scale(
                    self.xu,
                    seed=cfg.seed,
                    max_points=cfg.length_scale_sample_size,
                )

        self.noise_variance = max(float(self.noise_variance), cfg.eps_var)
        self.signal_variance = max(float(self.signal_variance), cfg.eps_var)
        self.length_scale = max(float(self.length_scale), 1e-8)

        n_inducing = self.xu.shape[0]
        kmm = self._kernel(self.xu, self.xu)
        kmm += cfg.jitter * np.eye(n_inducing)
        knm = self._kernel(x_train, self.xu)

        self.l_kmm = np.linalg.cholesky(kmm)
        v = solve_triangular(
            self.l_kmm, knm.T, lower=True, check_finite=False
        )
        qnn_diag = np.sum(v ** 2, axis=0)

        lambda_diag = self.signal_variance - qnn_diag + self.noise_variance
        lambda_diag = np.maximum(lambda_diag, cfg.eps_var)
        self.lambda_diag = lambda_diag
        beta = 1.0 / lambda_diag

        weighted_knm = knm * beta[:, None]
        a = kmm + knm.T @ weighted_knm
        a += cfg.jitter * np.eye(n_inducing)
        self.l_a = np.linalg.cholesky(a)

        b = knm.T @ (beta * y_train)
        tmp = solve_triangular(
            self.l_a, b, lower=True, check_finite=False
        )
        self.alpha = solve_triangular(
            self.l_a.T, tmp, lower=False, check_finite=False
        )
        return self

    def predict(self, x_test, batch_size):
        if self.xu is None or self.alpha is None:
            raise RuntimeError("The SGPR model must be fitted before prediction.")

        x_test = np.asarray(x_test, dtype=np.float64)
        if x_test.ndim != 2:
            raise ValueError("Prediction inputs must be a two-dimensional array.")
        if batch_size < 1:
            raise ValueError("Prediction batch size must be positive.")

        mean_parts = []
        variance_parts = []

        for start in range(0, x_test.shape[0], batch_size):
            batch = x_test[start:start + batch_size]
            ksm = self._kernel(batch, self.xu)
            mean = ksm @ self.alpha

            v_kmm = solve_triangular(
                self.l_kmm, ksm.T, lower=True, check_finite=False
            )
            q_star = np.sum(v_kmm ** 2, axis=0)

            v_a = solve_triangular(
                self.l_a, ksm.T, lower=True, check_finite=False
            )
            a_star = np.sum(v_a ** 2, axis=0)

            latent_variance = self.signal_variance - q_star + a_star
            observation_variance = latent_variance + self.noise_variance
            observation_variance = np.maximum(
                observation_variance, self.cfg.eps_var
            )

            mean_parts.append(mean)
            variance_parts.append(observation_variance)

        return np.concatenate(mean_parts), np.concatenate(variance_parts)


def predict_sgpr(model, x_windowed, cfg):
    mean, variance = model.predict(
        x_windowed, cfg.prediction_batch_size
    )
    variance = np.maximum(variance, cfg.eps_var)
    return mean.reshape(-1, 1), variance.reshape(-1, 1)


def inverse_y_and_var(y_norm, mean_norm, variance_norm, scaler_y):
    y_std = float(np.squeeze(scaler_y.std))
    y_mean = float(np.squeeze(scaler_y.mean))

    y = y_norm * y_std + y_mean
    mean = mean_norm * y_std + y_mean
    variance = variance_norm * (y_std ** 2)

    return y, mean, variance


# Output

def train_model(cfg):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    data, meta = prepare_data(cfg)

    x_train, train_y = data["Train"]
    y_train = flatten_y_for_sgpr(train_y)

    print("Sparse Gaussian Process Regression (FITC)")
    print(f"Sequence length: {cfg.sequence_length} hourly rows")
    print(f"Input features per hour: {meta['input_dim_per_hour']}")
    print(f"Flattened input dimension: {x_train.shape[1]}")
    print(f"Training samples after windowing: {x_train.shape[0]}")
    print(f"Inducing points: {min(cfg.num_inducing, x_train.shape[0])}")
    print(f"Inducing selection: {cfg.inducing_selection}")

    model = SparseGPRFITC(cfg).fit(x_train, y_train)

    model_path = cfg.output_dir / "best_sgpr_fitc_model.pkl"
    with model_path.open("wb") as file:
        pickle.dump(model, file)

    fit_summary = pd.DataFrame([{
        "model": "Sparse Gaussian Process Regression (FITC)",
        "training_samples": x_train.shape[0],
        "sequence_length_h": cfg.sequence_length,
        "input_features_per_hour": meta["input_dim_per_hour"],
        "input_dimension": x_train.shape[1],
        "num_inducing": model.xu.shape[0],
        "inducing_selection": cfg.inducing_selection,
        "seed": cfg.seed,
        "optimize_kernel_on_inducing_subset": cfg.optimize_kernel_on_inducing_subset,
        "signal_variance": model.signal_variance,
        "length_scale": model.length_scale,
        "noise_variance": model.noise_variance,
        "learned_kernel_on_inducing_subset": (
            str(model.learned_kernel) if model.learned_kernel is not None else ""
        ),
        "approximation": "FITC using all training samples",
    }])
    fit_summary.to_excel(cfg.output_dir / "training_history.xlsx", index=False)

    pd.DataFrame({
        "Inducing_point_index_or_cluster_id": model.inducing_indices
    }).to_excel(
        cfg.output_dir / "sgpr_inducing_indices_or_cluster_ids.xlsx",
        index=False,
    )

    prediction_splits = (
        ("Train", "train_time"),
        ("Validation", "val_time"),
        ("Test", "test_time"),
    )

    for split_name, time_key in prediction_splits:
        x_windowed, y_norm_input = data[split_name]
        mean_norm, variance_norm = predict_sgpr(
            model, x_windowed, cfg
        )

        y, mean, variance = inverse_y_and_var(
            y_norm_input,
            mean_norm,
            variance_norm,
            meta["scaler_y"],
        )

        y = y.reshape(-1)
        mean = mean.reshape(-1)
        variance = np.maximum(variance.reshape(-1), 1e-12)
        std = np.sqrt(variance)
        lower = mean - cfg.z_value * std
        upper = mean + cfg.z_value * std

        pred_df = pd.DataFrame({
            "Time": meta[time_key].reshape(-1),
            "Observed": y,
            "Predicted_mean": mean,
            "Predicted_variance": variance,
            "Predicted_std": std,
            "Lower_95PI": lower,
            "Upper_95PI": upper,
            "Residual": y - mean,
        })
        prediction_path = cfg.output_dir / f"{split_name}_prediction_results.xlsx"
        pred_df.to_excel(prediction_path, index=False)
        print(
            f"{split_name}: saved {len(pred_df)} sample-level predictions "
            f"to {prediction_path}"
        )

    protocol_df = pd.DataFrame([{
        "model": "Sparse Gaussian Process Regression (FITC)",
        "rolling_stride_h": cfg.rolling_stride,
        "forecast_horizon_h": cfg.forecast_horizon,
        "input_protocol": (
            "Each hourly row contains ECMWF predictors valid at that target time "
            "and ANE observations at t-2 and t-1; the latest 24 rows form each input window"
        ),
        "prediction_protocol": "Hourly rolling, one target-hour probabilistic forecast per origin",
        "sequence_length_h": cfg.sequence_length,
        "rolling_stride_h": cfg.rolling_stride,
        "temporal_window": (
            f"{cfg.sequence_length} consecutive hourly rows flattened into "
            "one tabular feature vector"
        ),
        "scaler_fit_data": "Training set only",
        "training_data_rule": "All valid training windows contribute through the FITC approximation",
        "num_inducing": model.xu.shape[0],
        "inducing_selection": cfg.inducing_selection,
        "seed": cfg.seed,
    }])
    protocol_df.to_excel(
        cfg.output_dir / "forecast_protocol.xlsx",
        index=False,
    )

    input_info_df = pd.DataFrame({
        "Item": [
            "Model",
            "Input representation",
            "Sequence length (h)",
            "Input features per hour",
            "Rolling stride (h)",
            "Forecast horizon (h)",
            "Input dimension",
            "Number of ECMWF grid nodes",
            "Grid features per node",
            "Historical ANE features per row",
            "Probabilistic output",
            "Sparse approximation",
            "Number of inducing points",
        ],
        "Value": [
            "Sparse Gaussian Process Regression (FITC)",
            "Flattened 24-hour sequence of ECMWF grid predictors and two historical ANE features per hour",
            cfg.sequence_length,
            meta["input_dim_per_hour"],
            cfg.rolling_stride,
            cfg.forecast_horizon,
            meta["flattened_input_dim"],
            cfg.num_nodes,
            cfg.num_grid_features,
            cfg.num_global_features,
            "Gaussian predictive mean and observation variance",
            "FITC inducing-point approximation",
            model.xu.shape[0],
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
