import pickle
import random
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
    output_dir_name = "results_sgpr"

    seed = 2026

    samples_per_hour = 6
    horizon_steps = [1, 6, 12, 24]
    max_horizon_step = max(horizon_steps)
    input_window_hours = 24
    input_len = input_window_hours * samples_per_hour
    max_lead_steps = max(horizon_steps) * samples_per_hour
    stride = 1
    window_alignment = "same_input_window"

    num_nodes = 9
    num_grid_features = 3
    num_global_features = 1
    input_dim_per_timestep = num_nodes * num_grid_features + num_global_features

    num_inducing = 512
    inducing_selection = "random"
    optimize_kernel_on_inducing_subset = True
    n_restarts_optimizer = 0

    signal_variance = 1.0
    length_scale = None
    noise_variance = 1e-2

    constant_value = 1.0
    constant_bounds = (1e-3, 1e3)
    length_scale_initial = 1.0
    length_scale_bounds = (1e-2, 1e2)
    noise_level = 1e-2
    noise_level_bounds = (1e-6, 1e1)
    alpha = 1e-8

    jitter = 1e-6
    eps_var = 1e-6
    interval_alpha = 0.05
    z_value = 1.959963984540054
    prediction_batch_size = 4096
    save_model = True

    @property
    def output_dir(self):
        return self.data_dir / self.output_dir_name


cfg = Config()


# Reproducibility

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# Data preprocessing

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
    observed = df.iloc[:, 28].astype(np.float32).to_numpy().reshape(-1, 1)

    x_grid = split_grid_features(ecmwf_raw, cfg.num_nodes)
    x_global = observed.copy()
    y = observed.copy()

    return time, x_grid, x_global, y


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
        raise ValueError("Window lengths, lead times, and stride must be positive.")
    if lead_steps > max_lead_steps:
        raise ValueError("lead_steps must not exceed max_lead_steps.")
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
        raise ValueError(
            "window_alignment must be 'same_input_window' or 'same_target_time'."
        )

    if not y_list:
        raise ValueError("No samples were generated. Check split length and window settings.")

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


def flatten_windows_for_sgpr(x_grid, x_global):
    sample_count, time_steps, node_count, grid_feature_count = x_grid.shape
    x_grid_flat = x_grid.reshape(
        sample_count, time_steps, node_count * grid_feature_count
    )
    x = np.concatenate([x_grid_flat, x_global], axis=-1)
    return x.reshape(sample_count, -1).astype(np.float64)


def flatten_y_for_sgpr(y):
    return np.asarray(y, dtype=np.float64).reshape(-1)


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
        time, x_grid, x_global, y = read_input_excel(path, cfg)
        raw_data[split_name] = {
            "time": time,
            "x_grid_raw": x_grid,
            "x_global_raw": x_global,
            "y_raw": y,
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
    datasets = {}
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
        datasets[split_name] = {
            "x": flatten_windows_for_sgpr(x_grid, x_global),
            "y": flatten_y_for_sgpr(y),
        }
        sample_metas[split_name] = sample_meta
    return datasets, sample_metas


# SGPR model

def squared_exponential_kernel(x1, x2, signal_variance, length_scale):
    distances_squared = cdist(
        x1 / length_scale, x2 / length_scale, metric="sqeuclidean"
    )
    return signal_variance * np.exp(-0.5 * distances_squared)


def estimate_median_length_scale(x, seed, max_points=1000):
    rng = np.random.default_rng(seed)
    if x.shape[0] > max_points:
        indices = rng.choice(x.shape[0], size=max_points, replace=False)
        subset = x[indices]
    else:
        subset = x

    distances = cdist(subset, subset, metric="euclidean")
    distances = distances[np.triu_indices_from(distances, k=1)]
    distances = distances[np.isfinite(distances) & (distances > 0)]
    return float(np.median(distances)) if distances.size else 1.0


def select_inducing_points(x_train, cfg, seed):
    n_samples = x_train.shape[0]
    n_inducing = min(cfg.num_inducing, n_samples)

    if n_inducing == n_samples:
        return x_train.copy(), np.arange(n_samples, dtype=np.int64)

    rng = np.random.default_rng(seed)
    selection = cfg.inducing_selection.lower()

    if selection == "random":
        indices = np.sort(
            rng.choice(n_samples, size=n_inducing, replace=False)
        ).astype(np.int64)
        return x_train[indices].copy(), indices

    if selection == "kmeans":
        kmeans = MiniBatchKMeans(
            n_clusters=n_inducing,
            random_state=seed,
            batch_size=min(4096, max(256, n_inducing * 4)),
            n_init=3,
            max_iter=100,
            reassignment_ratio=0.01,
        )
        kmeans.fit(x_train)
        return kmeans.cluster_centers_.astype(np.float64), np.arange(n_inducing, dtype=np.int64)

    raise ValueError("inducing_selection must be 'random' or 'kmeans'.")


def optimize_kernel_on_subset(x_inducing, y_subset, cfg, seed):
    kernel = (
        ConstantKernel(
            constant_value=cfg.constant_value,
            constant_value_bounds=cfg.constant_bounds,
        )
        * RBF(
            length_scale=cfg.length_scale_initial,
            length_scale_bounds=cfg.length_scale_bounds,
        )
        + WhiteKernel(
            noise_level=cfg.noise_level,
            noise_level_bounds=cfg.noise_level_bounds,
        )
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=cfg.alpha,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=cfg.n_restarts_optimizer,
        normalize_y=False,
        random_state=seed,
    )
    gpr.fit(x_inducing, y_subset)

    learned_kernel = gpr.kernel_
    signal_variance = float(learned_kernel.k1.k1.constant_value)
    length_scale = float(np.squeeze(learned_kernel.k1.k2.length_scale))
    noise_variance = float(learned_kernel.k2.noise_level)
    return signal_variance, length_scale, noise_variance, learned_kernel


class SparseGPRFITC:
    def __init__(
        self,
        num_inducing,
        inducing_selection,
        signal_variance,
        length_scale,
        noise_variance,
        optimize_kernel_on_inducing_subset,
        eps_var,
        jitter,
        seed,
    ):
        self.num_inducing = num_inducing
        self.inducing_selection = inducing_selection
        self.signal_variance = signal_variance
        self.length_scale = length_scale
        self.noise_variance = noise_variance
        self.optimize_kernel_on_inducing_subset = optimize_kernel_on_inducing_subset
        self.eps_var = eps_var
        self.jitter = jitter
        self.seed = seed

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

    def fit(self, x_train, y_train, cfg):
        x_train = np.asarray(x_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)

        if x_train.ndim != 2:
            raise ValueError("SGPR input features must be a two-dimensional array.")
        if x_train.shape[0] != y_train.size:
            raise ValueError("Training features and targets have different sample counts.")
        if y_train.size == 0:
            raise ValueError("Training data are empty.")

        self.xu, self.inducing_indices = select_inducing_points(
            x_train, cfg, self.seed
        )

        if self.optimize_kernel_on_inducing_subset:
            if self.inducing_selection.lower() == "random":
                y_subset = y_train[self.inducing_indices]
            else:
                nearest_indices = np.argmin(
                    cdist(self.xu, x_train, metric="sqeuclidean"), axis=1
                )
                y_subset = y_train[nearest_indices]

            (
                self.signal_variance,
                self.length_scale,
                self.noise_variance,
                self.learned_kernel,
            ) = optimize_kernel_on_subset(self.xu, y_subset, cfg, self.seed)
        elif self.length_scale is None:
            self.length_scale = estimate_median_length_scale(
                self.xu, seed=self.seed
            )

        self.noise_variance = max(float(self.noise_variance), self.eps_var)
        self.signal_variance = max(float(self.signal_variance), self.eps_var)
        self.length_scale = max(float(self.length_scale), 1e-8)

        n_inducing = self.xu.shape[0]
        kmm = self._kernel(self.xu, self.xu)
        kmm += self.jitter * np.eye(n_inducing)

        knm = self._kernel(x_train, self.xu)
        self.l_kmm = np.linalg.cholesky(kmm)

        v = solve_triangular(
            self.l_kmm, knm.T, lower=True, check_finite=False
        )
        qnn_diag = np.sum(v ** 2, axis=0)

        lambda_diag = (
            self.signal_variance - qnn_diag + self.noise_variance
        )
        self.lambda_diag = np.maximum(lambda_diag, self.eps_var)
        beta = 1.0 / self.lambda_diag

        a = kmm + knm.T @ (knm * beta[:, None])
        a += self.jitter * np.eye(n_inducing)
        self.l_a = np.linalg.cholesky(a)

        b = knm.T @ (beta * y_train)
        intermediate = solve_triangular(
            self.l_a, b, lower=True, check_finite=False
        )
        self.alpha = solve_triangular(
            self.l_a.T, intermediate, lower=False, check_finite=False
        )
        return self

    def predict(self, x_test, batch_size):
        if self.xu is None or self.alpha is None:
            raise RuntimeError("Fit the SGPR model before prediction.")

        x_test = np.asarray(x_test, dtype=np.float64)
        means = []
        variances = []

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
            observation_variance = np.maximum(
                latent_variance + self.noise_variance, self.eps_var
            )

            means.append(mean)
            variances.append(observation_variance)

        return np.concatenate(means), np.concatenate(variances)


# Prediction and outputs

def inverse_y_and_variance(y_norm, mean_norm, variance_norm, scaler_y):
    y_std = float(np.squeeze(scaler_y.std))
    y_mean = float(np.squeeze(scaler_y.mean))

    y = y_norm * y_std + y_mean
    mean = mean_norm * y_std + y_mean
    variance = variance_norm * (y_std ** 2)
    return y, mean, variance



def save_static_information(cfg, raw_data, scaler_grid, scaler_global, scaler_y):
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

    scaler_info = {
        "Scaler": [],
        "Shape_mean": [],
        "Shape_std": [],
        "Mean_values": [],
        "Std_values": [],
    }
    for name, scaler in [
        ("grid_features", scaler_grid),
        ("historical_observed_wind_speed", scaler_global),
        ("target_wind_speed", scaler_y),
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
            "Historical_observed_wind_speed",
            "Target_observed_wind_speed",
        ],
        "Excel_columns": [
            "B:J",
            "K:S",
            "T:AB",
            "AC",
            "AC shifted according to prediction horizon",
        ],
        "Used_as": [
            "grid input",
            "grid input",
            "grid input",
            "historical input",
            "prediction target",
        ],
    })
    feature_info.to_excel(
        cfg.output_dir / "feature_definition.xlsx", index=False
    )

    model_info = pd.DataFrame({
        "Item": [
            "Model",
            "Approximation",
            "Input representation",
            "Number of grid nodes",
            "Grid features per node",
            "Historical features per timestep",
            "Input dimension per timestep",
            "Input window samples",
            "Flattened input dimension",
            "Inducing points",
            "Inducing point selection",
            "Kernel hyperparameter optimization",
            "Prediction variance",
        ],
        "Value": [
            "Sparse Gaussian Process Regression",
            "FITC inducing-point approximation",
            "Flattened sequence of grid-node predictors and historical observed wind speed",
            cfg.num_nodes,
            cfg.num_grid_features,
            cfg.num_global_features,
            cfg.input_dim_per_timestep,
            cfg.input_len,
            cfg.input_len * cfg.input_dim_per_timestep,
            cfg.num_inducing,
            cfg.inducing_selection,
            cfg.optimize_kernel_on_inducing_subset,
            "Observation-level predictive variance including estimated noise",
        ],
    })
    model_info.to_excel(
        cfg.output_dir / "model_input_information.xlsx", index=False
    )


def train_one_horizon(cfg, transformed_data, scaler_y, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour
    horizon_dir = cfg.output_dir / f"horizon_{horizon_id:02d}h_ahead"
    horizon_dir.mkdir(parents=True, exist_ok=True)

    actual_seed = cfg.seed + horizon_id
    set_seed(actual_seed)

    datasets, sample_metas = prepare_data_for_horizon(
        cfg, transformed_data, horizon_id
    )
    x_train = datasets["Train"]["x"]
    y_train = datasets["Train"]["y"]

    model = SparseGPRFITC(
        num_inducing=cfg.num_inducing,
        inducing_selection=cfg.inducing_selection,
        signal_variance=cfg.signal_variance,
        length_scale=cfg.length_scale,
        noise_variance=cfg.noise_variance,
        optimize_kernel_on_inducing_subset=cfg.optimize_kernel_on_inducing_subset,
        eps_var=cfg.eps_var,
        jitter=cfg.jitter,
        seed=actual_seed,
    )

    print("\n" + "=" * 72)
    print(f"SGPR: {horizon_id} h ahead")
    print(f"Input window: {cfg.input_len} samples ({cfg.input_window_hours} h)")
    print(f"Lead time: {lead_steps} samples")
    print(f"Window alignment: {cfg.window_alignment}")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Flattened input dimension: {x_train.shape[1]}")
    print(f"Inducing points: {min(cfg.num_inducing, x_train.shape[0])}")
    print("=" * 72)

    model.fit(x_train, y_train, cfg)

    if cfg.save_model:
        with open(horizon_dir / f"sgpr_fitc_model_{horizon_id:02d}h_ahead.pkl", "wb") as file:
            pickle.dump(model, file)

    model_info = pd.DataFrame([{
        "Model": "Sparse Gaussian Process Regression",
        "Approximation": "FITC",
        "Horizon_hours_ahead": horizon_id,
        "Lead_samples": lead_steps,
        "Input_window_samples": cfg.input_len,
        "Input_window_hours": cfg.input_window_hours,
        "Window_alignment": cfg.window_alignment,
        "Training_samples": x_train.shape[0],
        "Input_dimension": x_train.shape[1],
        "Num_inducing": model.xu.shape[0],
        "Inducing_selection": cfg.inducing_selection,
        "Optimize_kernel_on_inducing_subset": cfg.optimize_kernel_on_inducing_subset,
        "Signal_variance": model.signal_variance,
        "Length_scale": model.length_scale,
        "Noise_variance": model.noise_variance,
        "Learned_kernel_on_inducing_subset": str(model.learned_kernel),
        "Random_seed": actual_seed,
    }])
    model_info.to_excel(
        horizon_dir / f"sgpr_model_information_{horizon_id:02d}h_ahead.xlsx",
        index=False,
    )

    pd.DataFrame({
        "Inducing_point_id_or_training_index": model.inducing_indices
    }).to_excel(
        horizon_dir / f"sgpr_inducing_point_ids_{horizon_id:02d}h_ahead.xlsx",
        index=False,
    )

    prediction_records = {}
    for split_name in ["Train", "Validation", "Test"]:
        x = datasets[split_name]["x"]
        y_norm = datasets[split_name]["y"]

        mean_norm, variance_norm = model.predict(
            x, batch_size=cfg.prediction_batch_size
        )
        y, mean, variance = inverse_y_and_variance(
            y_norm, mean_norm, variance_norm, scaler_y
        )

        y = y.reshape(-1)
        mean = mean.reshape(-1)
        variance = np.maximum(variance.reshape(-1), cfg.eps_var)
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
        })

        pred_path = horizon_dir / f"{split_name.lower()}_predictions_{horizon_id:02d}h_ahead.xlsx"
        pred_df.to_excel(pred_path, index=False)
        prediction_records[split_name] = pred_df
        print(f"Saved {split_name.lower()} predictions: {pred_path}")

    return model, prediction_records, model_info


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

        first_horizon = cfg.horizon_steps[0]
        first_df = all_predictions[first_horizon][split_name]
        wide_df = first_df[
            ["Sample_index", "Input_start_time", "Input_end_time"]
        ].copy()

        for horizon_id in cfg.horizon_steps:
            pred_df = all_predictions[horizon_id][split_name].copy()
            if len(pred_df) != len(wide_df):
                raise ValueError(
                    f"Sample count mismatch for {split_name}, horizon {horizon_id}."
                )

            prefix = f"H{horizon_id:02d}_"
            columns = [
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
        cfg.output_dir / "sgpr_model_information_all_horizons.xlsx",
        index=False,
    )


def train_all_horizons(cfg):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {cfg.output_dir}")
    print(f"Prediction horizons: {cfg.horizon_steps} h ahead")

    raw_data = load_raw_splits(cfg)
    scaler_grid, scaler_global, scaler_y = fit_scalers_on_training(raw_data)
    transformed_data = transform_raw_splits(
        raw_data, scaler_grid, scaler_global, scaler_y
    )
    save_static_information(
        cfg, raw_data, scaler_grid, scaler_global, scaler_y
    )

    all_predictions = {}
    all_model_info = []

    for horizon_id in cfg.horizon_steps:
        model, predictions, model_info = train_one_horizon(
            cfg, transformed_data, scaler_y, horizon_id
        )
        all_predictions[horizon_id] = predictions
        all_model_info.append(model_info)
        del model

    save_combined_outputs(cfg, all_predictions, all_model_info)
    print(f"\nAll SGPR outputs saved to: {cfg.output_dir}")
    return all_predictions


if __name__ == "__main__":
    set_seed(cfg.seed)
    train_all_horizons(cfg)
