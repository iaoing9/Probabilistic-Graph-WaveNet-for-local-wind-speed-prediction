import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Configuration

class Config:
    data_dir = Path("path/to/your/data")
    train_file = "train.xlsx"
    val_file = "validation.xlsx"
    test_file = "test.xlsx"
    output_dir = data_dir / "results_gwn_ane_only"

    seed = 2026
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sequence_length = 24
    rolling_stride = 1
    forecast_horizon = 1

    num_nodes = 1
    num_grid_features = 0
    num_global_features = 2
    input_features = num_grid_features + num_global_features

    residual_channels = 32
    dilation_channels = 32
    skip_channels = 64
    end_channels = 128
    gcn_channels = 32
    kernel_size = 3
    blocks = 3
    layers = 4
    gcn_order = 2
    dropout = 0.20
    adaptive_adj = False
    node_emb_dim = 10

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


# Data loading and preprocessing

def read_station_excel(file_path):
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_excel(file_path, engine="openpyxl")
    if df.shape[1] < 58:
        raise ValueError(
            f"{file_path} must contain at least 58 columns: "
            "time, 54 ECMWF features, two lagged ANE features, and target."
        )

    time = df.iloc[:, 0].to_numpy()
    x_raw = df.iloc[:, 55:57].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=np.float32)
    y_raw = pd.to_numeric(
        df.iloc[:, 57], errors="raise"
    ).to_numpy(dtype=np.float32).reshape(-1, 1)

    if len(time) != len(x_raw) or len(x_raw) != len(y_raw):
        raise ValueError(f"Inconsistent row counts in {file_path}.")
    if len(y_raw) == 0:
        raise ValueError(f"No samples found in {file_path}.")
    if not np.isfinite(x_raw).all() or not np.isfinite(y_raw).all():
        raise ValueError(f"Non-finite input or target values found in {file_path}.")

    return time, x_raw, y_raw


def split_raw_features(x_raw, num_nodes, num_global_features):
    if x_raw.ndim != 2 or x_raw.shape[1] != num_global_features:
        raise ValueError(
            f"Expected {num_global_features} ANE input features, got {x_raw.shape}."
        )

    x_grid = np.empty(
        (x_raw.shape[0], num_nodes, 0),
        dtype=np.float32,
    )
    x_global = x_raw[:, :num_global_features].astype(np.float32)
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


def build_rolling_windows(
    x_grid,
    x_global,
    y,
    time,
    sequence_length,
    stride=1,
    history_x_grid=None,
    history_x_global=None,
):
    if len(x_grid) != len(x_global) or len(x_global) != len(y) or len(y) != len(time):
        raise ValueError("Input features, targets, and timestamps must have equal lengths.")
    if sequence_length <= 0 or stride <= 0:
        raise ValueError("sequence_length and stride must be positive.")

    history_length = sequence_length - 1
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
        history_x_grid = history_x_grid[-history_length:]
        history_x_global = history_x_global[-history_length:]

    combined_x_grid = np.concatenate([history_x_grid, x_grid], axis=0)
    combined_x_global = np.concatenate([history_x_global, x_global], axis=0)
    offset = len(history_x_grid)

    grid_windows = []
    global_windows = []
    targets = []
    target_times = []

    for current_idx in range(0, len(y), stride):
        combined_end = offset + current_idx
        combined_start = combined_end - sequence_length + 1
        if combined_start < 0:
            continue

        grid_windows.append(combined_x_grid[combined_start:combined_end + 1])
        global_windows.append(combined_x_global[combined_start:combined_end + 1])
        targets.append(y[current_idx])
        target_times.append(time[current_idx])

    if not grid_windows:
        raise ValueError(
            f"No rolling windows could be built with sequence_length={sequence_length}."
        )

    return (
        np.asarray(grid_windows, dtype=np.float32),
        np.asarray(global_windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32).reshape(-1, 1),
        np.asarray(target_times),
    )


class WindRollingDataset(Dataset):
    def __init__(self, x_grid, x_global, y):
        self.x_grid = torch.as_tensor(x_grid, dtype=torch.float32)
        self.x_global = torch.as_tensor(x_global, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x_grid[idx], self.x_global[idx], self.y[idx]


def prepare_data(config):
    split_files = {
        "Train": config.data_dir / config.train_file,
        "Validation": config.data_dir / config.val_file,
        "Test": config.data_dir / config.test_file,
    }

    raw_data = {
        split_name: read_station_excel(file_path)
        for split_name, file_path in split_files.items()
    }

    features = {}
    for split_name, (time, x_raw, y_raw) in raw_data.items():
        x_grid, x_global = split_raw_features(
            x_raw, config.num_nodes, config.num_global_features
        )
        features[split_name] = {
            "time": time,
            "x_grid": x_grid,
            "x_global": x_global,
            "y_raw": y_raw,
        }

    scaler_global = StandardScaler()
    scaler_y = StandardScaler()
    scaler_global.fit(features["Train"]["x_global"], axis=0)
    scaler_y.fit(features["Train"]["y_raw"], axis=0)

    for split_name, data in features.items():
        data["x_global"] = scaler_global.transform(data["x_global"])
        data["y"] = scaler_y.transform(data["y_raw"])

    train_rows = features["Train"]
    val_rows = features["Validation"]
    test_rows = features["Test"]

    train_x_grid, train_x_global, train_y, train_time = build_rolling_windows(
        train_rows["x_grid"],
        train_rows["x_global"],
        train_rows["y"],
        train_rows["time"],
        config.sequence_length,
        config.rolling_stride,
    )

    val_x_grid, val_x_global, val_y, val_time = build_rolling_windows(
        val_rows["x_grid"],
        val_rows["x_global"],
        val_rows["y"],
        val_rows["time"],
        config.sequence_length,
        config.rolling_stride,
        history_x_grid=train_rows["x_grid"],
        history_x_global=train_rows["x_global"],
    )

    test_x_grid, test_x_global, test_y, test_time = build_rolling_windows(
        test_rows["x_grid"],
        test_rows["x_global"],
        test_rows["y"],
        test_rows["time"],
        config.sequence_length,
        config.rolling_stride,
        history_x_grid=val_rows["x_grid"],
        history_x_global=val_rows["x_global"],
    )

    loaders = {
        "Train": DataLoader(
            WindRollingDataset(train_x_grid, train_x_global, train_y),
            batch_size=config.batch_size,
            shuffle=config.shuffle_train,
            drop_last=False,
        ),
        "Validation": DataLoader(
            WindRollingDataset(val_x_grid, val_x_global, val_y),
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
        ),
        "Test": DataLoader(
            WindRollingDataset(test_x_grid, test_x_global, test_y),
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
        ),
    }

    meta = {
        "scaler_y": scaler_y,
        "times": {
            "Train": train_time,
            "Validation": val_time,
            "Test": test_time,
        },
        "sample_counts": {
            "Train": len(train_y),
            "Validation": len(val_y),
            "Test": len(test_y),
        },
    }
    return loaders, meta


# Probabilistic Graph WaveNet

class NConv(nn.Module):
    def forward(self, x, adj):
        return torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()


class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len, order):
        super().__init__()
        self.nconv = NConv()
        self.order = order
        self.dropout = dropout
        total_channels = c_in * (order * support_len + 1)
        self.mlp = nn.Conv2d(total_channels, c_out, kernel_size=(1, 1))

    def forward(self, x, supports):
        out = [x]
        for adj in supports:
            x1 = self.nconv(x, adj)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, adj)
                out.append(x1)

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class ProbabilisticGWN(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_features,
        predefined_adj,
        station_weights,
        residual_channels,
        dilation_channels,
        skip_channels,
        end_channels,
        gcn_channels,
        kernel_size,
        blocks,
        layers,
        gcn_order,
        dropout,
        adaptive_adj,
        node_emb_dim,
        eps_var,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_features = input_features
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.blocks = blocks
        self.layers = layers
        self.adaptive_adj = adaptive_adj
        self.eps_var = eps_var

        predefined_adj = torch.as_tensor(predefined_adj, dtype=torch.float32)
        station_weights = torch.as_tensor(station_weights, dtype=torch.float32)
        self.register_buffer("predefined_adj", predefined_adj)
        self.register_buffer("predefined_adj_t", predefined_adj.T)
        self.register_buffer("station_weights", station_weights)

        support_len = 2
        if adaptive_adj:
            self.nodevec1 = nn.Parameter(
                torch.randn(num_nodes, node_emb_dim),
                requires_grad=True,
            )
            self.nodevec2 = nn.Parameter(
                torch.randn(node_emb_dim, num_nodes),
                requires_grad=True,
            )
            support_len += 1

        self.start_conv = nn.Conv2d(
            in_channels=input_features,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        receptive_field = 1
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            for layer_idx in range(layers):
                dilation = 2 ** layer_idx
                receptive_field += additional_scope
                additional_scope *= 2

                self.filter_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=(1, dilation),
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=(1, dilation),
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(gcn_channels, residual_channels, kernel_size=(1, 1))
                )
                self.skip_convs.append(
                    nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1))
                )
                self.gconv_layers.append(
                    GraphConv(
                        dilation_channels,
                        gcn_channels,
                        dropout,
                        support_len,
                        gcn_order,
                    )
                )
                self.bn_layers.append(nn.BatchNorm2d(residual_channels))

        self.receptive_field = receptive_field
        self.end_conv_1 = nn.Conv1d(skip_channels, end_channels, kernel_size=1)
        self.end_conv_2 = nn.Conv1d(end_channels, end_channels, kernel_size=1)
        self.mean_head = nn.Conv1d(end_channels, 1, kernel_size=1)
        self.raw_var_head = nn.Conv1d(end_channels, 1, kernel_size=1)

    def build_supports(self):
        supports = [self.predefined_adj, self.predefined_adj_t]
        if self.adaptive_adj:
            adaptive_adj = F.softmax(
                F.relu(torch.mm(self.nodevec1, self.nodevec2)),
                dim=1,
            )
            supports.append(adaptive_adj)
        return supports

    def forward(self, x_grid, x_global):
        if x_grid.ndim != 4 or x_grid.shape[2] != self.num_nodes:
            raise ValueError("x_grid must have shape [batch, time, num_nodes, grid_features].")
        if (
            x_global.ndim != 3
            or x_grid.shape[-1] + x_global.shape[-1] != self.input_features
        ):
            raise ValueError(
                "The combined grid and ANE features must match input_features."
            )

        node_count = x_grid.shape[2]
        x_global_expanded = x_global.unsqueeze(2).expand(-1, -1, node_count, -1)
        x = torch.cat([x_grid, x_global_expanded], dim=-1)
        x = x.permute(0, 3, 2, 1).contiguous()

        if x.size(-1) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(-1), 0, 0, 0))

        x = self.start_conv(x)
        skip = None
        supports = self.build_supports()
        layer_idx = 0

        for _ in range(self.blocks):
            for _ in range(self.layers):
                residual = x
                dilation = self.filter_convs[layer_idx].dilation[1]
                pad_len = dilation * (self.kernel_size - 1)
                x_padded = F.pad(x, (pad_len, 0, 0, 0))

                filter_out = torch.tanh(self.filter_convs[layer_idx](x_padded))
                gate_out = torch.sigmoid(self.gate_convs[layer_idx](x_padded))
                x = filter_out * gate_out

                skip_layer = self.skip_convs[layer_idx](x)
                if skip is None:
                    skip = skip_layer
                else:
                    skip = skip[..., -skip_layer.size(-1):] + skip_layer

                x = self.gconv_layers[layer_idx](x, supports)
                x = self.residual_convs[layer_idx](x)
                x = x + residual[..., -x.size(-1):]
                x = self.bn_layers[layer_idx](x)
                layer_idx += 1

        h = F.relu(skip)
        h = h[..., -1:]
        h = torch.einsum("bcnt,n->bct", h, self.station_weights)
        h = F.relu(self.end_conv_1(h))
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(self.end_conv_2(h))

        mean = self.mean_head(h).squeeze(-1)
        raw_var = self.raw_var_head(h).squeeze(-1)
        variance = F.softplus(raw_var) + self.eps_var
        return mean, variance


# Training and prediction

def gaussian_nll_loss(y, mean, variance):
    loss = 0.5 * (
        torch.log(variance)
        + (y - mean) ** 2 / variance
        + math.log(2.0 * math.pi)
    )
    return loss.mean()


def run_one_epoch(model, loader, optimizer, device, train, grad_clip):
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
            mean, variance = model(x_grid, x_global)
            loss = gaussian_nll_loss(y, mean, variance)
            if train:
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    if total_count == 0:
        raise ValueError("The data loader produced no samples.")
    return total_loss / total_count


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_y = []
    all_mean = []
    all_variance = []

    for x_grid, x_global, y in loader:
        x_grid = x_grid.to(device)
        x_global = x_global.to(device)
        mean, variance = model(x_grid, x_global)
        all_y.append(y.cpu().numpy())
        all_mean.append(mean.cpu().numpy())
        all_variance.append(variance.cpu().numpy())

    if not all_y:
        raise ValueError("The data loader produced no samples.")

    return (
        np.concatenate(all_y, axis=0),
        np.concatenate(all_mean, axis=0),
        np.concatenate(all_variance, axis=0),
    )


def inverse_y_and_variance(y_norm, mean_norm, variance_norm, scaler):
    y_std = float(np.squeeze(scaler.std))
    y_mean = float(np.squeeze(scaler.mean))
    y = y_norm * y_std + y_mean
    mean = mean_norm * y_std + y_mean
    variance = variance_norm * (y_std ** 2)
    return y, mean, variance


# Model construction

def build_model(config):
    predefined_adj = np.eye(config.num_nodes, dtype=np.float32)
    station_weights = np.ones(config.num_nodes, dtype=np.float32)
    station_weights /= station_weights.sum()

    return ProbabilisticGWN(
        num_nodes=config.num_nodes,
        input_features=config.input_features,
        predefined_adj=predefined_adj,
        station_weights=station_weights,
        residual_channels=config.residual_channels,
        dilation_channels=config.dilation_channels,
        skip_channels=config.skip_channels,
        end_channels=config.end_channels,
        gcn_channels=config.gcn_channels,
        kernel_size=config.kernel_size,
        blocks=config.blocks,
        layers=config.layers,
        gcn_order=config.gcn_order,
        dropout=config.dropout,
        adaptive_adj=config.adaptive_adj,
        node_emb_dim=config.node_emb_dim,
        eps_var=config.eps_var,
    ).to(config.device)


# Output

def train_model(config):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    loaders, meta = prepare_data(config)
    model = build_model(config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    history = []
    best_model_path = config.output_dir / "best_probabilistic_gwn_ane_only.pt"

    print(f"Using device: {config.device}")
    print("Input setting: historical ANE observations only.")
    print(
        f"Graph convolution retained with {config.num_nodes} node, "
        f"adaptive adjacency={config.adaptive_adj}."
    )
    for split_name, sample_count in meta["sample_counts"].items():
        print(f"{split_name} samples: {sample_count}")

    for epoch in range(1, config.epochs + 1):
        train_loss = run_one_epoch(
            model,
            loaders["Train"],
            optimizer,
            config.device,
            train=True,
            grad_clip=config.grad_clip,
        )
        val_loss = run_one_epoch(
            model,
            loaders["Validation"],
            optimizer,
            config.device,
            train=False,
            grad_clip=None,
        )

        history.append({
            "Epoch": epoch,
            "Train_GNLL": train_loss,
            "Validation_GNLL": val_loss,
        })

        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:04d} | Train GNLL={train_loss:.6f} | "
                f"Validation GNLL={val_loss:.6f} | Best epoch={best_epoch}"
            )

    pd.DataFrame(history).to_excel(
        config.output_dir / "training_history.xlsx",
        index=False,
    )

    if not best_model_path.exists():
        raise RuntimeError("No model checkpoint was saved during training.")
    model.load_state_dict(torch.load(best_model_path, map_location=config.device))

    for split_name in ("Train", "Validation", "Test"):
        y_norm, mean_norm, variance_norm = predict(
            model,
            loaders[split_name],
            config.device,
        )
        y, mean, variance = inverse_y_and_variance(
            y_norm,
            mean_norm,
            variance_norm,
            meta["scaler_y"],
        )

        observed = y.reshape(-1)
        predicted_mean = mean.reshape(-1)
        predicted_variance = np.maximum(variance.reshape(-1), 1e-12)
        predicted_std = np.sqrt(predicted_variance)
        lower = predicted_mean - config.z_value * predicted_std
        upper = predicted_mean + config.z_value * predicted_std

        prediction_df = pd.DataFrame({
            "Time": meta["times"][split_name].reshape(-1),
            "Observed": observed,
            "Predicted_mean": predicted_mean,
            "Predicted_variance": predicted_variance,
            "Predicted_std": predicted_std,
            "Lower_95PI": lower,
            "Upper_95PI": upper,
            "Residual": observed - predicted_mean,
        })

        output_path = config.output_dir / f"{split_name}_prediction_results.xlsx"
        prediction_df.to_excel(output_path, index=False)
        print(f"{split_name}: saved {len(prediction_df)} predictions to {output_path}")

    protocol_df = pd.DataFrame([{
        "sequence_length_h": config.sequence_length,
        "rolling_stride_h": config.rolling_stride,
        "forecast_horizon_h": config.forecast_horizon,
        "input_protocol": "ANE_(t-2) and ANE_(t-1) only; ECMWF predictors excluded",
        "prediction_protocol": "Hourly rolling, one target-hour probabilistic forecast per origin",
        "ablation_setting": "ANE-only input",
        "graph_structure": "Single-node identity adjacency; graph convolution retained; adaptive adjacency disabled",
        "scaler_fit_data": "Training split only",
    }])
    protocol_df.to_excel(
        config.output_dir / "forecast_protocol.xlsx",
        index=False,
    )

    graph_df = pd.DataFrame([{
        "Number_of_nodes": config.num_nodes,
        "Predefined_adjacency": "identity matrix",
        "Adaptive_adjacency_enabled": config.adaptive_adj,
        "Graph_convolution_module": "retained",
        "Spatial_node_mixing": "none; single-node graph",
        "Readout_weight": 1.0,
    }])
    graph_df.to_excel(
        config.output_dir / "graph_configuration.xlsx",
        index=False,
    )

    print(f"Best validation GNLL: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"All outputs saved to: {config.output_dir}")
    return model


if __name__ == "__main__":
    train_model(cfg)
