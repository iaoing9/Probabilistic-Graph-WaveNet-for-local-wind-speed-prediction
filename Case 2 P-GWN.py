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
    adjacency_sigma_scale = 1.0

    num_nodes = 9
    num_grid_features = 6
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
    adaptive_adj = True
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

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)

# Coordinates and adjacency matrix

station_coord = (40.56, 115.81)

grid_coords = [
    (40.75, 115.50),
    (40.75, 115.75),
    (40.75, 116.00),
    (40.50, 115.50),
    (40.50, 115.75),
    (40.50, 116.00),
    (40.25, 115.50),
    (40.25, 115.75),
    (40.25, 116.00),
]

def haversine_distance_km(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    r = 6371.0
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c

def build_distance_adjacency(coords, sigma_scale):
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            dist[i, j] = haversine_distance_km(coords[i], coords[j])

    positive_dist = dist[dist > 0]
    sigma = positive_dist.mean() * sigma_scale

    adj = np.exp(-dist ** 2 / (sigma ** 2 + 1e-12)).astype(np.float32)
    np.fill_diagonal(adj, 1.0)

    row_sum = adj.sum(axis=1, keepdims=True) + 1e-12
    adj = adj / row_sum

    return adj, dist

def build_station_readout_weights(station, coords):
    d = np.array([haversine_distance_km(station, c) for c in coords], dtype=np.float32)
    inv_d = 1.0 / (d + 1e-6)
    w = inv_d / inv_d.sum()
    return w.astype(np.float32), d

predefined_adj, _ = build_distance_adjacency(
    grid_coords, sigma_scale=cfg.adjacency_sigma_scale
)
station_weights, station_distance = build_station_readout_weights(station_coord, grid_coords)

# Data loading and preprocessing

def read_station_excel(file_path, expected_input_columns):
    df = pd.read_excel(file_path, engine="openpyxl")
    expected_columns = expected_input_columns + 2
    if df.shape[1] < expected_columns:
        raise ValueError(
            f"{file_path} contains {df.shape[1]} columns; "
            f"at least {expected_columns} are required."
        )

    time = df.iloc[:, 0].to_numpy()
    x_raw = df.iloc[:, 1:1 + expected_input_columns].to_numpy(dtype=np.float32)
    y_raw = df.iloc[:, 1 + expected_input_columns:2 + expected_input_columns]
    y_raw = y_raw.to_numpy(dtype=np.float32).reshape(-1, 1)

    if not np.isfinite(x_raw).all() or not np.isfinite(y_raw).all():
        raise ValueError(f"Non-finite values found in the input or target columns of {file_path}.")

    return time, x_raw, y_raw

def split_raw_features(x_raw, num_nodes, num_grid_features, num_global_features):
    expected_columns = num_nodes * num_grid_features + num_global_features
    if x_raw.ndim != 2 or x_raw.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} input features, got {x_raw.shape}."
        )

    grid_end = num_nodes * num_grid_features
    grid_groups = [
        x_raw[:, i * num_nodes:(i + 1) * num_nodes]
        for i in range(num_grid_features)
    ]
    x_grid = np.stack(grid_groups, axis=1).transpose(0, 2, 1).astype(np.float32)
    x_global = x_raw[:, grid_end:grid_end + num_global_features].astype(np.float32)

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
    stride,
    history_x_grid=None,
    history_x_global=None,
):
    n = len(y)
    hist_len = sequence_length - 1

    if history_x_grid is None:
        history_x_grid = np.empty(
            (0, x_grid.shape[1], x_grid.shape[2]), dtype=x_grid.dtype
        )
        history_x_global = np.empty(
            (0, x_global.shape[1]), dtype=x_global.dtype
        )
    else:
        history_x_grid = history_x_grid[-hist_len:]
        history_x_global = history_x_global[-hist_len:]

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
    input_columns = cfg.num_nodes * cfg.num_grid_features + cfg.num_global_features
    train_path = cfg.data_dir / cfg.train_file
    val_path = cfg.data_dir / cfg.val_file
    test_path = cfg.data_dir / cfg.test_file

    for path in (train_path, val_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

    train_time, train_x_raw, train_y_raw = read_station_excel(train_path, input_columns)
    val_time, val_x_raw, val_y_raw = read_station_excel(val_path, input_columns)
    test_time, test_x_raw, test_y_raw = read_station_excel(test_path, input_columns)

    train_x_grid, train_x_global = split_raw_features(
        train_x_raw, cfg.num_nodes, cfg.num_grid_features, cfg.num_global_features
    )
    val_x_grid, val_x_global = split_raw_features(
        val_x_raw, cfg.num_nodes, cfg.num_grid_features, cfg.num_global_features
    )
    test_x_grid, test_x_global = split_raw_features(
        test_x_raw, cfg.num_nodes, cfg.num_grid_features, cfg.num_global_features
    )

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

    train_x_grid_rows, train_x_global_rows = train_x_grid, train_x_global
    val_x_grid_rows, val_x_global_rows = val_x_grid, val_x_global
    test_x_grid_rows, test_x_global_rows = test_x_grid, test_x_global

    train_x_grid, train_x_global, train_y, train_time = build_rolling_windows(
        train_x_grid_rows,
        train_x_global_rows,
        train_y,
        train_time,
        cfg.sequence_length,
        cfg.rolling_stride,
    )

    val_x_grid, val_x_global, val_y, val_time = build_rolling_windows(
        val_x_grid_rows,
        val_x_global_rows,
        val_y,
        val_time,
        cfg.sequence_length,
        cfg.rolling_stride,
        history_x_grid=train_x_grid_rows,
        history_x_global=train_x_global_rows,
    )

    test_x_grid, test_x_global, test_y, test_time = build_rolling_windows(
        test_x_grid_rows,
        test_x_global_rows,
        test_y,
        test_time,
        cfg.sequence_length,
        cfg.rolling_stride,
        history_x_grid=val_x_grid_rows,
        history_x_global=val_x_global_rows,
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
        c_in_total = c_in * (order * support_len + 1)
        self.mlp = nn.Conv2d(c_in_total, c_out, kernel_size=(1, 1))

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
        h = F.dropout(h, self.dropout, training=self.training)
        return h

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

        self.dropout = dropout
        self.kernel_size = kernel_size
        self.blocks = blocks
        self.layers = layers
        self.adaptive_adj = adaptive_adj
        self.eps_var = eps_var

        predefined_adj = torch.tensor(predefined_adj, dtype=torch.float32)
        station_weights = torch.tensor(station_weights, dtype=torch.float32)

        self.register_buffer("predefined_adj", predefined_adj)
        self.register_buffer("predefined_adj_t", predefined_adj.T)
        self.register_buffer("station_weights", station_weights)

        support_len = 2

        if adaptive_adj:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, node_emb_dim), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(node_emb_dim, num_nodes), requires_grad=True)
            support_len += 1

        self.start_conv = nn.Conv2d(
            in_channels=input_features,
            out_channels=residual_channels,
            kernel_size=(1, 1)
        )

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.gconv_layers = nn.ModuleList()

        receptive_field = 1

        for _ in range(blocks):
            additional_scope = kernel_size - 1

            for i in range(layers):
                dilation = 2 ** i
                receptive_field += additional_scope
                additional_scope *= 2

                self.filter_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=(1, dilation)
                    )
                )

                self.gate_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=(1, dilation)
                    )
                )

                self.residual_convs.append(
                    nn.Conv2d(
                        gcn_channels,
                        residual_channels,
                        kernel_size=(1, 1)
                    )
                )

                self.skip_convs.append(
                    nn.Conv2d(
                        dilation_channels,
                        skip_channels,
                        kernel_size=(1, 1)
                    )
                )

                self.gconv_layers.append(
                    GraphConv(
                        dilation_channels,
                        gcn_channels,
                        dropout,
                        support_len,
                        gcn_order
                    )
                )

                self.bn_layers.append(nn.BatchNorm2d(residual_channels))

        self.receptive_field = receptive_field

        self.end_conv_1 = nn.Conv1d(
            skip_channels,
            end_channels,
            kernel_size=1
        )
        self.end_conv_2 = nn.Conv1d(
            end_channels,
            end_channels,
            kernel_size=1
        )

        self.mean_head = nn.Conv1d(end_channels, 1, kernel_size=1)
        self.raw_var_head = nn.Conv1d(end_channels, 1, kernel_size=1)

    def build_supports(self):
        supports = [self.predefined_adj, self.predefined_adj_t]

        if self.adaptive_adj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            supports.append(adp)

        return supports

    def forward(self, x_grid, x_global):

        _, _, n, _ = x_grid.shape

        x_global_expand = x_global.unsqueeze(2).repeat(1, 1, n, 1)

        x = torch.cat([x_grid, x_global_expand], dim=-1)

        x = x.permute(0, 3, 2, 1).contiguous()

        if x.size(-1) < self.receptive_field:
            pad_len = self.receptive_field - x.size(-1)
            x = F.pad(x, (pad_len, 0, 0, 0))

        x = self.start_conv(x)

        skip = 0
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

                s = self.skip_convs[layer_idx](x)

                if isinstance(skip, int):
                    skip = s
                else:
                    skip = skip[..., -s.size(-1):] + s

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

        mu = self.mean_head(h).squeeze(-1)
        raw_var = self.raw_var_head(h).squeeze(-1)

        var = F.softplus(raw_var) + self.eps_var

        return mu, var

# Training loss

def gaussian_nll_loss(y, mu, var):
    loss = 0.5 * (torch.log(2.0 * math.pi * var) + (y - mu) ** 2 / var)
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

    y = np.concatenate(all_y, axis=0)
    mu = np.concatenate(all_mu, axis=0)
    var = np.concatenate(all_var, axis=0)

    return y, mu, var

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

    model = ProbabilisticGWN(
        num_nodes=cfg.num_nodes,
        input_features=cfg.input_features,
        predefined_adj=predefined_adj,
        station_weights=station_weights,
        residual_channels=cfg.residual_channels,
        dilation_channels=cfg.dilation_channels,
        skip_channels=cfg.skip_channels,
        end_channels=cfg.end_channels,
        gcn_channels=cfg.gcn_channels,
        kernel_size=cfg.kernel_size,
        blocks=cfg.blocks,
        layers=cfg.layers,
        gcn_order=cfg.gcn_order,
        dropout=cfg.dropout,
        adaptive_adj=cfg.adaptive_adj,
        node_emb_dim=cfg.node_emb_dim,
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
    best_model_path = cfg.output_dir / "best_probabilistic_gwn.pt"

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

    prediction_loaders = (
        ("Train", train_loader, meta["train_time"]),
        ("Validation", val_loader, meta["val_time"]),
        ("Test", test_loader, meta["test_time"]),
    )

    for split_name, loader, time_values in prediction_loaders:
        y_norm, mu_norm, var_norm = predict(model, loader, cfg.device)
        y, mu, var = inverse_y_and_var(y_norm, mu_norm, var_norm, meta["scaler_y"])

        y = y.reshape(-1)
        mu = mu.reshape(-1)
        var = np.maximum(var.reshape(-1), 1e-12)
        sigma = np.sqrt(var)
        lower = mu - cfg.z_value * sigma
        upper = mu + cfg.z_value * sigma

        pred_df = pd.DataFrame({
            "Time": time_values.reshape(-1),
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
        "sequence_length_h": cfg.sequence_length,
        "rolling_stride_h": cfg.rolling_stride,
        "input_protocol": "ECMWF forecasts valid at target time + ANE observations at t-2 and t-1",
        "prediction_protocol": "Hourly rolling prediction, one target-hour forecast per origin",
    }])
    protocol_df.to_excel(cfg.output_dir / "forecast_protocol.xlsx", index=False)

    coord_df = pd.DataFrame({
        "Node": [f"P{i + 1}" for i in range(cfg.num_nodes)],
        "Latitude": [coord[0] for coord in grid_coords],
        "Longitude": [coord[1] for coord in grid_coords],
        "Distance_to_station_km": station_distance,
        "Station_readout_weight": station_weights,
    })
    coord_df.to_excel(
        cfg.output_dir / "grid_coordinates_and_station_weights.xlsx",
        index=False,
    )

    adj_df = pd.DataFrame(
        predefined_adj,
        index=[f"P{i + 1}" for i in range(cfg.num_nodes)],
        columns=[f"P{i + 1}" for i in range(cfg.num_nodes)],
    )
    adj_df.to_excel(
        cfg.output_dir / "predefined_distance_adjacency.xlsx",
    )

    print(f"Best validation GNLL: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"All outputs saved to: {cfg.output_dir}")
    return model

if __name__ == "__main__":
    train_model(cfg)
