import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Configuration

class Config:
    data_dir = Path("path/to/your/data")
    train_file = "train.xlsx"
    val_file = "validation.xlsx"
    test_file = "test.xlsx"
    output_dir = data_dir / "gru_results"

    seed = 2026

    use_cuda_if_available = True
    require_cuda = False
    device = "cuda" if (use_cuda_if_available and torch.cuda.is_available()) else "cpu"

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
    input_dim = num_nodes * num_grid_features + num_global_features

    gru_hidden_size = 64
    gru_num_layers = 2
    dropout = 0.20

    batch_size = 16
    epochs = 150
    learning_rate = 1e-3
    weight_decay = 1e-5
    patience = 9999
    min_delta = 1e-5
    grad_clip = 5.0

    eps_var = 1e-6
    z_value = 1.959963984540054

    save_model = True

    num_workers = 0
    pin_memory = True if device == "cuda" else False
    shuffle_train = True


cfg = Config()


# Reproducibility

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class WindWindowDataset(Dataset):
    def __init__(self, x_grid, x_global, y):
        self.x_grid = torch.tensor(x_grid, dtype=torch.float32)
        self.x_global = torch.tensor(x_global, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x_grid[idx], self.x_global[idx], self.y[idx]


def load_raw_splits(cfg):
    data_dir = Path(cfg.data_dir)
    paths = {
        "Train": data_dir / cfg.train_file,
        "Validation": data_dir / cfg.val_file,
        "Test": data_dir / cfg.test_file,
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
    datasets = {}
    loaders = {}
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

        dataset = WindWindowDataset(x_grid, x_global, y)
        shuffle = cfg.shuffle_train if split_name == "Train" else False
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

        datasets[split_name] = dataset
        loaders[split_name] = loader
        sample_metas[split_name] = sample_meta

    return datasets, loaders, sample_metas


# Probabilistic GRU

class ProbabilisticGRU(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout, eps_var):
        super().__init__()
        self.input_dim = input_dim
        self.eps_var = eps_var

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.mean_head = nn.Linear(hidden_size, 1)
        self.raw_var_head = nn.Linear(hidden_size, 1)

    def forward(self, x_grid, x_global):
        batch_size, time_steps, num_nodes, grid_features = x_grid.shape
        x_grid_flat = x_grid.reshape(
            batch_size, time_steps, num_nodes * grid_features
        )
        x = torch.cat([x_grid_flat, x_global], dim=-1)

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected GRU input dimension {self.input_dim}, got {x.shape[-1]}."
            )

        hidden_sequence, _ = self.gru(x)
        hidden_last = self.output_norm(hidden_sequence[:, -1, :])
        hidden_last = self.dropout_layer(hidden_last)

        mu = self.mean_head(hidden_last)
        raw_var = self.raw_var_head(hidden_last)
        var = F.softplus(raw_var) + self.eps_var

        return mu, var


# Training

def gaussian_nll_loss(y, mu, var):
    return (0.5 * (torch.log(2.0 * math.pi * var) + (y - mu) ** 2 / var)).mean()


def run_one_epoch(model, loader, optimizer, device, train=True, grad_clip=5.0):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_count = 0

    for x_grid, x_global, y in loader:
        x_grid = x_grid.to(device)
        x_global = x_global.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            mu, var = model(x_grid, x_global)
            loss = gaussian_nll_loss(y, mu, var)

            if train:
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_y = []
    all_mu = []
    all_var = []

    for x_grid, x_global, y in loader:
        x_grid = x_grid.to(device)
        x_global = x_global.to(device)
        mu, var = model(x_grid, x_global)

        all_y.append(y.cpu().numpy())
        all_mu.append(mu.cpu().numpy())
        all_var.append(var.cpu().numpy())

    return (
        np.concatenate(all_y, axis=0),
        np.concatenate(all_mu, axis=0),
        np.concatenate(all_var, axis=0),
    )


def inverse_y_and_var(y_norm, mu_norm, var_norm, scaler_y):
    y_std = float(np.squeeze(scaler_y.std))
    y_mean = float(np.squeeze(scaler_y.mean))

    y = y_norm * y_std + y_mean
    mu = mu_norm * y_std + y_mean
    var = var_norm * (y_std ** 2)

    return y, mu, var


def build_model(cfg):
    return ProbabilisticGRU(
        input_dim=cfg.input_dim,
        hidden_size=cfg.gru_hidden_size,
        num_layers=cfg.gru_num_layers,
        dropout=cfg.dropout,
        eps_var=cfg.eps_var,
    ).to(cfg.device)


def train_one_horizon(cfg, transformed_data, scaler_y, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour
    horizon_dir = Path(cfg.output_dir) / f"horizon_{horizon_id:02d}h_ahead"
    horizon_dir.mkdir(parents=True, exist_ok=True)

    actual_seed = cfg.seed + horizon_id
    set_seed(actual_seed)

    datasets, loaders, sample_metas = prepare_data_for_horizon(
        cfg=cfg,
        transformed_data=transformed_data,
        horizon_id=horizon_id,
    )
    model = build_model(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    wait = 0
    history = []
    best_model_path = horizon_dir / f"best_probabilistic_gru_{horizon_id:02d}h_ahead.pt"

    print("\n" + "=" * 80)
    print(f"Training horizon: {horizon_id} h ahead ({lead_steps} samples)")
    print(f"Input length: {cfg.input_len} samples ({cfg.input_window_hours} h)")
    print(f"Window alignment: {cfg.window_alignment}; stride: {cfg.stride}")
    for split_name in ["Train", "Validation", "Test"]:
        print(f"{split_name} samples: {len(datasets[split_name])}")
    print("=" * 80)

    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_one_epoch(
            model,
            loaders["Train"],
            optimizer,
            cfg.device,
            train=True,
            grad_clip=cfg.grad_clip,
        )
        val_loss = run_one_epoch(
            model,
            loaders["Validation"],
            optimizer,
            cfg.device,
            train=False,
            grad_clip=None,
        )

        history.append({
            "Random_seed": actual_seed,
            "Horizon_step": horizon_id,
            "Lead_hours": horizon_id,
            "Lead_samples": lead_steps,
            "Epoch": epoch,
            "Train_GNLL": train_loss,
            "Validation_GNLL": val_loss,
        })

        if val_loss < best_val_loss - cfg.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            wait = 0
            if cfg.save_model:
                torch.save(model.state_dict(), best_model_path)
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Horizon {horizon_id:02d} h | Epoch {epoch:04d} | "
                f"Train GNLL = {train_loss:.6f} | "
                f"Validation GNLL = {val_loss:.6f} | Best epoch = {best_epoch}"
            )

        if wait >= cfg.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_excel(
        horizon_dir / f"training_history_{horizon_id:02d}h_ahead.xlsx",
        index=False,
    )

    if cfg.save_model and best_model_path.is_file():
        model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))

    prediction_records = {}

    for split_name in ["Train", "Validation", "Test"]:
        eval_loader = DataLoader(
            datasets[split_name],
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        y_norm, mu_norm, var_norm = predict(model, eval_loader, cfg.device)
        y, mu, var = inverse_y_and_var(y_norm, mu_norm, var_norm, scaler_y)

        y_flat = y.reshape(-1)
        mu_flat = mu.reshape(-1)
        var_flat = var.reshape(-1)
        sigma_flat = np.sqrt(np.maximum(var_flat, 1e-12))
        lower = mu_flat - cfg.z_value * sigma_flat
        upper = mu_flat + cfg.z_value * sigma_flat
        sample_meta = sample_metas[split_name]

        pred_df = pd.DataFrame({
            "Random_seed": actual_seed,
            "Sample_index": np.arange(1, len(y_flat) + 1),
            "Horizon_step": horizon_id,
            "Lead_hours": horizon_id,
            "Lead_samples": lead_steps,
            "Input_start_index_0based": sample_meta["input_start_index"],
            "Input_end_index_0based": sample_meta["input_end_index"],
            "Target_index_0based": sample_meta["target_index"],
            "Input_start_time": sample_meta["input_start_time"],
            "Input_end_time": sample_meta["input_end_time"],
            "Target_time": sample_meta["target_time"],
            "Observed": y_flat,
            "Predicted_mean": mu_flat,
            "Predicted_variance": var_flat,
            "Predicted_std": sigma_flat,
            "Lower_95PI": lower,
            "Upper_95PI": upper,
            "Residual": y_flat - mu_flat,
        })

        pred_path = horizon_dir / f"{split_name.lower()}_predictions_{horizon_id:02d}h_ahead.xlsx"
        pred_df.to_excel(pred_path, index=False)
        prediction_records[split_name] = pred_df
        print(f"Saved {split_name.lower()} predictions: {pred_path}")

    return model, prediction_records, history_df


# Output

def save_static_information(cfg, raw_data, scaler_grid, scaler_global, scaler_y):
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            "Selected_horizon_steps": str(cfg.horizon_steps),
            "Stride_samples": cfg.stride,
            "Effective_sample_size_each_horizon": n_eff,
            "Dropped_samples_due_to_common_truncation": n_raw - n_eff,
            "Window_alignment": cfg.window_alignment,
        })

    pd.DataFrame(split_info).to_excel(
        output_dir / "data_split_and_window_sample_size_summary.xlsx",
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
        scaler_info["Shape_mean"].append(str(np.array(scaler.mean).shape))
        scaler_info["Shape_std"].append(str(np.array(scaler.std).shape))
        scaler_info["Mean_values"].append(
            np.array2string(np.squeeze(scaler.mean), precision=6, separator=", ")
        )
        scaler_info["Std_values"].append(
            np.array2string(np.squeeze(scaler.std), precision=6, separator=", ")
        )

    pd.DataFrame(scaler_info).to_excel(
        output_dir / "training_only_scaler_information.xlsx",
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
            "grid input flattened for GRU",
            "grid input flattened for GRU",
            "grid input flattened for GRU",
            "global historical input concatenated once at each time step",
            "prediction target",
        ],
    })
    feature_info.to_excel(output_dir / "feature_definition.xlsx", index=False)

    input_info = pd.DataFrame({
        "Item": [
            "Model",
            "Input representation",
            "Number of ECMWF grid nodes",
            "Grid features per node",
            "Global historical features",
            "GRU input dimension",
            "Probabilistic output",
        ],
        "Value": [
            "Probabilistic GRU",
            "Flattened grid-node features plus global historical observation; no graph convolution or spatial adjacency",
            cfg.num_nodes,
            cfg.num_grid_features,
            cfg.num_global_features,
            cfg.input_dim,
            "Gaussian mean and variance",
        ],
    })
    input_info.to_excel(output_dir / "model_input_information.xlsx", index=False)


def save_combined_outputs(cfg, all_predictions):
    output_dir = Path(cfg.output_dir)

    for split_name in ["Train", "Validation", "Test"]:
        prediction_list = [
            all_predictions[horizon_id][split_name].copy()
            for horizon_id in cfg.horizon_steps
        ]
        long_df = pd.concat(prediction_list, axis=0, ignore_index=True)
        long_df.to_excel(
            output_dir / f"{split_name.lower()}_predictions_all_horizons_long.xlsx",
            index=False,
        )

        first_horizon = cfg.horizon_steps[0]
        first_df = all_predictions[first_horizon][split_name]
        wide_df = first_df[["Sample_index", "Target_time", "Observed"]].copy()

        for horizon_id in cfg.horizon_steps:
            pred_df = all_predictions[horizon_id][split_name].copy()

            if len(pred_df) != len(wide_df):
                raise ValueError(
                    f"Sample size mismatch in {split_name} for horizon {horizon_id}."
                )
            if not np.array_equal(
                pred_df["Target_time"].values, wide_df["Target_time"].values
            ):
                raise ValueError(
                    f"Target_time is not aligned in {split_name} for horizon {horizon_id}. "
                    "Set cfg.window_alignment = 'same_target_time'."
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
            add_cols = pred_df[[
                "Input_start_index_0based",
                "Input_end_index_0based",
                "Input_start_time",
                "Input_end_time",
                "Predicted_mean",
                "Predicted_variance",
                "Predicted_std",
                "Lower_95PI",
                "Upper_95PI",
                "Residual",
            ]].copy()
            add_cols = add_cols.rename(columns={c: prefix + c for c in add_cols.columns})
            wide_df = pd.concat([wide_df, add_cols], axis=1)

        wide_df.to_excel(
            output_dir / f"{split_name.lower()}_predictions_all_horizons_wide.xlsx",
            index=False,
        )


def train_all_horizons(cfg):
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.require_cuda and cfg.device != "cuda":
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build and use an NVIDIA GPU, "
            "or set cfg.require_cuda = False."
        )

    print(f"Using device: {cfg.device}")
    if cfg.device == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is unavailable. The script will run on CPU.")

    print(f"Selected horizons (hours ahead): {cfg.horizon_steps}")
    print("Loading input splits...")
    raw_data = load_raw_splits(cfg)

    print("Fitting scalers on the training split only...")
    scaler_grid, scaler_global, scaler_y = fit_scalers_on_training(raw_data)
    transformed_data = transform_raw_splits(
        raw_data, scaler_grid, scaler_global, scaler_y
    )
    save_static_information(cfg, raw_data, scaler_grid, scaler_global, scaler_y)

    all_predictions = {}
    all_histories = []

    for horizon_id in cfg.horizon_steps:
        model, prediction_records, history_df = train_one_horizon(
            cfg=cfg,
            transformed_data=transformed_data,
            scaler_y=scaler_y,
            horizon_id=horizon_id,
        )
        all_predictions[horizon_id] = prediction_records
        all_histories.append(history_df)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_combined_outputs(cfg, all_predictions)
    pd.concat(all_histories, axis=0, ignore_index=True).to_excel(
        output_dir / "training_history_all_horizons.xlsx",
        index=False,
    )

    print("\nPredictions and training histories saved to:")
    print(output_dir)
    return all_predictions


if __name__ == "__main__":
    set_seed(cfg.seed)
    train_all_horizons(cfg)
