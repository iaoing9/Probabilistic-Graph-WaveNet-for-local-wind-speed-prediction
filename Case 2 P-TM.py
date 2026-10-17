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
    output_dir = data_dir / "results"

    seed = 2026
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sequence_length = 24
    rolling_stride = 1

    num_nodes = 9
    num_grid_features = 6
    num_global_features = 2
    input_dim = num_nodes * num_grid_features + num_global_features

    d_model = 64
    nhead = 4
    num_encoder_layers = 2
    dim_feedforward = 128
    dropout = 0.20
    activation = "gelu"

    batch_size = 8
    epochs = 200
    learning_rate = 5e-4
    weight_decay = 1e-5
    min_delta = 1e-5
    grad_clip = 5.0
    shuffle_train = True

    eps_var = 1e-6
    z_value = 1.959963984540054


cfg = Config()


# Reproducibility

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    x_raw = df.iloc[:, 1:1 + input_dim].to_numpy(dtype=np.float32)
    y_raw = df.iloc[:, 1 + input_dim].to_numpy(dtype=np.float32).reshape(-1, 1)

    if not np.isfinite(x_raw).all() or not np.isfinite(y_raw).all():
        raise ValueError(f"Non-finite values found in the input or target columns of {file_path}.")

    return time, x_raw, y_raw


def split_raw_features(x_raw, cfg):
    expected_columns = cfg.num_nodes * cfg.num_grid_features + cfg.num_global_features
    if x_raw.ndim != 2 or x_raw.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} input features, got {x_raw.shape}."
        )

    grid_end = cfg.num_nodes * cfg.num_grid_features
    grid_groups = [
        x_raw[:, i * cfg.num_nodes:(i + 1) * cfg.num_nodes]
        for i in range(cfg.num_grid_features)
    ]
    x_grid = np.stack(grid_groups, axis=1).transpose(0, 2, 1).astype(np.float32)
    x_global = x_raw[:, grid_end:grid_end + cfg.num_global_features].astype(np.float32)

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


def build_rolling_windows(
    x_grid,
    x_global,
    y,
    time,
    sequence_length,
    stride,
    history_x_grid=None,
    history_x_global=None,
):
    n = len(y)
    hist_len = sequence_length - 1

    if history_x_grid is None:
        history_x_grid = np.empty(
            (0, x_grid.shape[1], x_grid.shape[2]),
            dtype=x_grid.dtype,
        )
        history_x_global = np.empty(
            (0, x_global.shape[1]),
            dtype=x_global.dtype,
        )
    else:
        history_x_grid = np.asarray(history_x_grid)[-hist_len:]
        history_x_global = np.asarray(history_x_global)[-hist_len:]

    combined_x_grid = np.concatenate([history_x_grid, x_grid], axis=0)
    combined_x_global = np.concatenate([history_x_global, x_global], axis=0)
    offset = len(history_x_grid)

    xg_windows = []
    xglobal_windows = []
    y_targets = []
    time_targets = []

    for current_idx in range(0, n, stride):
        combined_end = offset + current_idx
        combined_start = combined_end - sequence_length + 1

        if combined_start < 0:
            continue

        xg_windows.append(combined_x_grid[combined_start:combined_end + 1])
        xglobal_windows.append(combined_x_global[combined_start:combined_end + 1])
        y_targets.append(y[current_idx])
        time_targets.append(time[current_idx])

    if not xg_windows:
        raise ValueError(
            f"No rolling windows could be built with sequence_length={sequence_length}."
        )

    return (
        np.asarray(xg_windows, dtype=np.float32),
        np.asarray(xglobal_windows, dtype=np.float32),
        np.asarray(y_targets, dtype=np.float32).reshape(-1, 1),
        np.asarray(time_targets),
    )


class WindRollingDataset(Dataset):
    def __init__(self, x_grid, x_global, y):
        self.x_grid = torch.tensor(x_grid, dtype=torch.float32)
        self.x_global = torch.tensor(x_global, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x_grid[idx], self.x_global[idx], self.y[idx]


def prepare_data(cfg):
    if cfg.rolling_stride != 1:
        raise ValueError("The Case II protocol requires one-hour rolling forecast origins.")

    train_path = cfg.data_dir / cfg.train_file
    val_path = cfg.data_dir / cfg.val_file
    test_path = cfg.data_dir / cfg.test_file

    for path in (train_path, val_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

    train_time, train_x_raw, train_y_raw = read_station_excel(train_path, cfg.input_dim)
    val_time, val_x_raw, val_y_raw = read_station_excel(val_path, cfg.input_dim)
    test_time, test_x_raw, test_y_raw = read_station_excel(test_path, cfg.input_dim)

    train_x_grid, train_x_global = split_raw_features(train_x_raw, cfg)
    val_x_grid, val_x_global = split_raw_features(val_x_raw, cfg)
    test_x_grid, test_x_global = split_raw_features(test_x_raw, cfg)

    scaler_grid = StandardScaler()
    scaler_global = StandardScaler()
    scaler_y = StandardScaler()

    scaler_grid.fit(train_x_grid, axis=(0, 1))
    scaler_global.fit(train_x_global, axis=0)
    scaler_y.fit(train_y_raw, axis=0)

    train_x_grid_scaled = scaler_grid.transform(train_x_grid)
    val_x_grid_scaled = scaler_grid.transform(val_x_grid)
    test_x_grid_scaled = scaler_grid.transform(test_x_grid)

    train_x_global_scaled = scaler_global.transform(train_x_global)
    val_x_global_scaled = scaler_global.transform(val_x_global)
    test_x_global_scaled = scaler_global.transform(test_x_global)

    train_y = scaler_y.transform(train_y_raw)
    val_y = scaler_y.transform(val_y_raw)
    test_y = scaler_y.transform(test_y_raw)

    train_x_grid, train_x_global, train_y, train_time = build_rolling_windows(
        train_x_grid_scaled,
        train_x_global_scaled,
        train_y,
        train_time,
        cfg.sequence_length,
        cfg.rolling_stride,
    )

    val_x_grid, val_x_global, val_y, val_time = build_rolling_windows(
        val_x_grid_scaled,
        val_x_global_scaled,
        val_y,
        val_time,
        cfg.sequence_length,
        cfg.rolling_stride,
        history_x_grid=train_x_grid_scaled,
        history_x_global=train_x_global_scaled,
    )

    prior_test_x_grid = np.concatenate(
        [train_x_grid_scaled, val_x_grid_scaled],
        axis=0,
    )
    prior_test_x_global = np.concatenate(
        [train_x_global_scaled, val_x_global_scaled],
        axis=0,
    )

    test_x_grid, test_x_global, test_y, test_time = build_rolling_windows(
        test_x_grid_scaled,
        test_x_global_scaled,
        test_y,
        test_time,
        cfg.sequence_length,
        cfg.rolling_stride,
        history_x_grid=prior_test_x_grid,
        history_x_global=prior_test_x_global,
    )

    train_loader = DataLoader(
        WindRollingDataset(train_x_grid, train_x_global, train_y),
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train,
        drop_last=False,
    )
    val_loader = DataLoader(
        WindRollingDataset(val_x_grid, val_x_global, val_y),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        WindRollingDataset(test_x_grid, test_x_global, test_y),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    meta = {
        "scaler_y": scaler_y,
        "train_time": train_time,
        "val_time": val_time,
        "test_time": test_time,
    }

    return train_loader, val_loader, test_loader, meta


# Probabilistic Transformer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ProbabilisticTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        sequence_length,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        dropout,
        activation,
        eps_var,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.eps_var = eps_var

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model,
            max_len=sequence_length,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.output_norm = nn.LayerNorm(d_model)
        self.mean_head = nn.Linear(d_model, 1)
        self.raw_var_head = nn.Linear(d_model, 1)

    def forward(self, x_grid, x_global):
        batch_size, time_steps, num_nodes, num_features = x_grid.shape

        if time_steps != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {time_steps}."
            )

        x_grid_flat = x_grid.reshape(batch_size, time_steps, num_nodes * num_features)
        x = torch.cat([x_grid_flat, x_global], dim=-1)

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features per time step, got {x.shape[-1]}."
            )

        h = self.input_projection(x)
        h = self.positional_encoding(h)
        h = self.encoder(h)
        h_last = self.output_norm(h[:, -1, :])

        mu = self.mean_head(h_last)
        raw_var = self.raw_var_head(h_last)
        var = F.softplus(raw_var) + self.eps_var

        return mu, var


# Training loss

def gaussian_nll_loss(y, mu, var):
    loss = 0.5 * (
        torch.log(2.0 * math.pi * var)
        + (y - mu) ** 2 / var
    )
    return loss.mean()


def run_one_epoch(model, loader, optimizer, device, train, grad_clip):
    if train:
        model.train()
    else:
        model.eval()

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


# Output

def train_model(cfg):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader, meta = prepare_data(cfg)

    model = ProbabilisticTransformer(
        input_dim=cfg.input_dim,
        sequence_length=cfg.sequence_length,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        activation=cfg.activation,
        eps_var=cfg.eps_var,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    history = []
    best_model_path = cfg.output_dir / "best_probabilistic_transformer.pt"

    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_one_epoch(
            model,
            train_loader,
            optimizer,
            cfg.device,
            train=True,
            grad_clip=cfg.grad_clip,
        )
        val_loss = run_one_epoch(
            model,
            val_loader,
            optimizer,
            cfg.device,
            train=False,
            grad_clip=None,
        )

        history.append({
            "epoch": epoch,
            "train_GNLL": train_loss,
            "val_GNLL": val_loss,
        })

        if val_loss < best_val_loss - cfg.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"Train GNLL = {train_loss:.6f} | "
                f"Val GNLL = {val_loss:.6f} | "
                f"Best epoch = {best_epoch}"
            )

    pd.DataFrame(history).to_excel(
        cfg.output_dir / "training_history.xlsx",
        index=False,
    )

    model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))

    prediction_splits = (
        ("Train", train_loader, meta["train_time"]),
        ("Validation", val_loader, meta["val_time"]),
        ("Test", test_loader, meta["test_time"]),
    )

    for split_name, loader, time_values in prediction_splits:
        y_norm, mu_norm, var_norm = predict(model, loader, cfg.device)
        y, mu, var = inverse_y_and_var(
            y_norm,
            mu_norm,
            var_norm,
            meta["scaler_y"],
        )

        y = y.reshape(-1)
        mu = mu.reshape(-1)
        var = np.maximum(var.reshape(-1), 1e-12)
        sigma = np.sqrt(var)
        lower = mu - cfg.z_value * sigma
        upper = mu + cfg.z_value * sigma

        pred_df = pd.DataFrame({
            "Time": np.asarray(time_values).reshape(-1),
            "Observed": y,
            "Predicted_mean": mu,
            "Predicted_variance": var,
            "Predicted_std": sigma,
            "Lower_95PI": lower,
            "Upper_95PI": upper,
            "Residual": y - mu,
        })
        output_path = cfg.output_dir / f"{split_name}_prediction_results.xlsx"
        pred_df.to_excel(output_path, index=False)
        print(f"{split_name}: saved {len(pred_df)} sample-level predictions to {output_path}")

    protocol_df = pd.DataFrame([{
        "model": "Probabilistic Transformer",
        "sequence_length_h": cfg.sequence_length,
        "rolling_stride_h": cfg.rolling_stride,
        "input_protocol": "ECMWF forecasts valid at target time and ANE observations at t-2 and t-1",
        "prediction_protocol": "24-h causal rolling context; one target-hour forecast per origin",
        "split_boundary_context": "Previous split inputs may provide causal history; targets remain split-specific",
        "scaler_fit_data": "Training set only",
        "output_token": "Latest token only",
    }])
    protocol_df.to_excel(
        cfg.output_dir / "forecast_protocol.xlsx",
        index=False,
    )

    input_info_df = pd.DataFrame({
        "Item": [
            "Model",
            "Input representation",
            "Temporal context length (h)",
            "Rolling stride (h)",
            "Input dimension per row",
            "Number of ECMWF grid nodes",
            "Grid features per node",
            "Historical ANE features per row",
            "Probabilistic output",
        ],
        "Value": [
            "Probabilistic Transformer",
            "24-h causal rolling sequence of 56-dimensional row-wise inputs",
            cfg.sequence_length,
            cfg.rolling_stride,
            cfg.input_dim,
            cfg.num_nodes,
            cfg.num_grid_features,
            cfg.num_global_features,
            "Gaussian mean and variance for one target hour",
        ],
    })
    input_info_df.to_excel(
        cfg.output_dir / "model_input_information.xlsx",
        index=False,
    )

    print(f"Best validation GNLL: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"All outputs saved to: {cfg.output_dir}")
    return model


if __name__ == "__main__":
    train_model(cfg)
