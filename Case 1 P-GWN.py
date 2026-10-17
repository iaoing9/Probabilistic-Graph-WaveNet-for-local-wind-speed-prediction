import os
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path

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

    window_alignment = "same_input_window"

    num_nodes = 9
    num_grid_features = 3
    num_global_features = 1
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


# Coordinates and adjacency matrix

station_coord = (31.77, 120.99)

grid_coords = [
    (32.00, 120.75),
    (32.00, 121.00),
    (32.00, 121.25),
    (31.75, 120.75),
    (31.75, 121.00),
    (31.75, 121.25),
    (31.50, 120.75),
    (31.50, 121.00),
    (31.50, 121.25),
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
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def build_distance_adjacency(coords, sigma_scale=1.0):
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


predefined_adj, grid_distance = build_distance_adjacency(grid_coords)
station_weights, station_distance = build_station_readout_weights(station_coord, grid_coords)


# Data loading and preprocessing

def read_input_excel(file_path):

    df = pd.read_excel(file_path, engine="openpyxl")

    if df.shape[1] < 29:
        raise ValueError(
            f"{file_path} has {df.shape[1]} columns, but at least 29 columns A:AC are required."
        )

    time = df.iloc[:, 0].values

    ecmwf_raw = df.iloc[:, 1:28].astype(np.float32).values
    ane_raw = df.iloc[:, 28].astype(np.float32).values.reshape(-1, 1)

    x_grid = split_grid_features(ecmwf_raw, num_nodes=cfg.num_nodes)
    x_global = ane_raw.astype(np.float32)
    y_raw = ane_raw.astype(np.float32)

    return time, x_grid, x_global, y_raw, df


def split_grid_features(ecmwf_raw, num_nodes):
    if ecmwf_raw.shape[1] != 3 * num_nodes:
        raise ValueError(
            f"Expected {3 * num_nodes} ECMWF columns, but got {ecmwf_raw.shape[1]}."
        )

    groups = []
    for g in range(3):
        groups.append(ecmwf_raw[:, g * num_nodes:(g + 1) * num_nodes])

    x_grid = np.stack(groups, axis=1).transpose(0, 2, 1).astype(np.float32)
    return x_grid


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
    alignment
):

    n = len(y)

    if n != len(x_grid) or n != len(x_global) or n != len(time):
        raise ValueError("x_grid, x_global, y, and time must have the same first dimension.")

    if input_len <= 0 or lead_steps <= 0 or max_lead_steps <= 0:
        raise ValueError("input_len, lead_steps, and max_lead_steps must be positive.")

    if lead_steps > max_lead_steps:
        raise ValueError("lead_steps must not be larger than max_lead_steps.")

    if n < input_len + max_lead_steps:
        raise ValueError(
            f"Split length {n} is too short for input_len={input_len} and max_lead_steps={max_lead_steps}."
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

    xg = np.stack(xg_list, axis=0).astype(np.float32)
    xgl = np.stack(xgl_list, axis=0).astype(np.float32)
    yy = np.stack(y_list, axis=0).astype(np.float32)

    sample_meta = {
        "target_time": np.array(target_time_list),
        "input_start_time": np.array(input_start_time_list),
        "input_end_time": np.array(input_end_time_list),
        "input_start_index": np.array(input_start_index_list, dtype=np.int64),
        "input_end_index": np.array(input_end_index_list, dtype=np.int64),
        "target_index": np.array(target_index_list, dtype=np.int64),
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
    train_path = cfg.data_dir / cfg.train_file
    val_path = cfg.data_dir / cfg.val_file
    test_path = cfg.data_dir / cfg.test_file

    missing_files = [path for path in [train_path, val_path, test_path] if not path.is_file()]
    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Set Config.data_dir to the directory containing the input files. "
            f"The following files could not be found:\n{missing}"
        )

    train_time, train_x_grid_raw, train_x_global_raw, train_y_raw, train_df = read_input_excel(train_path)
    val_time, val_x_grid_raw, val_x_global_raw, val_y_raw, val_df = read_input_excel(val_path)
    test_time, test_x_grid_raw, test_x_global_raw, test_y_raw, test_df = read_input_excel(test_path)

    return {
        "Train": {
            "time": train_time,
            "x_grid_raw": train_x_grid_raw,
            "x_global_raw": train_x_global_raw,
            "y_raw": train_y_raw,
            "df": train_df,
        },
        "Validation": {
            "time": val_time,
            "x_grid_raw": val_x_grid_raw,
            "x_global_raw": val_x_global_raw,
            "y_raw": val_y_raw,
            "df": val_df,
        },
        "Test": {
            "time": test_time,
            "x_grid_raw": test_x_grid_raw,
            "x_global_raw": test_x_global_raw,
            "y_raw": test_y_raw,
            "df": test_df,
        },
    }


def fit_scalers_on_training(raw_data):
    scaler_grid = StandardScaler()
    scaler_global = StandardScaler()
    scaler_y = StandardScaler()

    train = raw_data["Train"]

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

        ds = WindWindowDataset(x_grid, x_global, y)

        shuffle = cfg.shuffle_train if split_name == "Train" else False
        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory
        )

        datasets[split_name] = ds
        loaders[split_name] = loader
        sample_metas[split_name] = sample_meta

    return datasets, loaders, sample_metas


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
        input_len,
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
        self.input_len = input_len
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

        b, t, n, f_grid = x_grid.shape

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


def run_one_epoch(model, loader, optimizer, device, train=True, grad_clip=5.0):
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


def build_model(cfg):
    model = ProbabilisticGWN(
        num_nodes=cfg.num_nodes,
        input_features=cfg.input_features,
        input_len=cfg.input_len,
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

    return model


def train_one_horizon(cfg, transformed_data, scaler_y, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour
    lead_hours = horizon_id

    horizon_dir = os.path.join(cfg.output_dir, f"horizon_{lead_hours:02d}h_ahead")
    os.makedirs(horizon_dir, exist_ok=True)

    actual_seed = cfg.seed + horizon_id
    set_seed(actual_seed)

    datasets, loaders, sample_metas = prepare_data_for_horizon(
        cfg=cfg,
        transformed_data=transformed_data,
        horizon_id=horizon_id
    )

    model = build_model(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    best_val_loss = float("inf")
    best_epoch = -1
    wait = 0
    history = []
    best_model_path = os.path.join(horizon_dir, f"best_probabilistic_gwn_{lead_hours:02d}h_ahead.pt")

    print("\n" + "=" * 80)
    print(f"Training horizon: {lead_hours} h ahead ({lead_steps} samples)")
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
            grad_clip=cfg.grad_clip
        )
        val_loss = run_one_epoch(
            model,
            loaders["Validation"],
            optimizer,
            cfg.device,
            train=False,
            grad_clip=None
        )

        history.append({
            "Random_seed": actual_seed,
            "Horizon_step": horizon_id,
            "Lead_hours": lead_hours,
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
                f"Horizon {lead_hours:02d} h | "
                f"Epoch {epoch:04d} | "
                f"Train GNLL = {train_loss:.6f} | "
                f"Validation GNLL = {val_loss:.6f} | "
                f"Best epoch = {best_epoch}"
            )

        if wait >= cfg.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_excel(
        os.path.join(horizon_dir, f"training_history_{lead_hours:02d}h_ahead.xlsx"),
        index=False
    )

    if cfg.save_model and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))

    prediction_records = {}
    for split_name in ["Train", "Validation", "Test"]:
        eval_loader = DataLoader(
            datasets[split_name],
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory
        )
        y_norm, mu_norm, var_norm = predict(model, eval_loader, cfg.device)
        y, mu, var = inverse_y_and_var(
            y_norm=y_norm,
            mu_norm=mu_norm,
            var_norm=var_norm,
            scaler_y=scaler_y
        )

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
            "Lead_hours": lead_hours,
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

        pred_path = os.path.join(
            horizon_dir,
            f"{split_name.lower()}_predictions_{lead_hours:02d}h_ahead.xlsx"
        )
        pred_df.to_excel(pred_path, index=False)
        prediction_records[split_name] = pred_df
        print(f"Saved {split_name.lower()} predictions: {pred_path}")

    return model, prediction_records, history_df


# Output

def save_static_information(cfg, raw_data, scaler_grid, scaler_global, scaler_y):
    os.makedirs(cfg.output_dir, exist_ok=True)

    coord_df = pd.DataFrame({
        "Node": [f"P{i + 1}" for i in range(cfg.num_nodes)],
        "Latitude": [c[0] for c in grid_coords],
        "Longitude": [c[1] for c in grid_coords],
        "Distance_to_station_km": station_distance,
        "Station_readout_weight": station_weights,
    })
    coord_df.to_excel(
        os.path.join(cfg.output_dir, "grid_coordinates_and_station_weights.xlsx"),
        index=False
    )

    adj_df = pd.DataFrame(
        predefined_adj,
        index=[f"P{i + 1}" for i in range(cfg.num_nodes)],
        columns=[f"P{i + 1}" for i in range(cfg.num_nodes)]
    )
    adj_df.to_excel(
        os.path.join(cfg.output_dir, "predefined_distance_adjacency.xlsx")
    )

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

    split_info_df = pd.DataFrame(split_info)
    split_info_df.to_excel(
        os.path.join(cfg.output_dir, "data_split_and_window_sample_size_summary.xlsx"),
        index=False
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
        scaler_info["Mean_values"].append(np.array2string(np.squeeze(scaler.mean), precision=6, separator=", "))
        scaler_info["Std_values"].append(np.array2string(np.squeeze(scaler.std), precision=6, separator=", "))

    scaler_df = pd.DataFrame(scaler_info)
    scaler_df.to_excel(
        os.path.join(cfg.output_dir, "training_only_scaler_information.xlsx"),
        index=False
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
            "global historical input",
            "prediction target",
        ],
    })
    feature_info.to_excel(
        os.path.join(cfg.output_dir, "feature_definition.xlsx"),
        index=False
    )


def save_combined_outputs(cfg, all_predictions):
    for split_name in ["Train", "Validation", "Test"]:
        prediction_list = [
            all_predictions[horizon_id][split_name].copy()
            for horizon_id in cfg.horizon_steps
        ]
        long_df = pd.concat(prediction_list, axis=0, ignore_index=True)
        long_df.to_excel(
            os.path.join(cfg.output_dir, f"{split_name.lower()}_predictions_all_horizons_long.xlsx"),
            index=False
        )

        first_horizon = cfg.horizon_steps[0]
        base = all_predictions[first_horizon][split_name][[
            "Sample_index",
            "Input_start_time",
            "Input_end_time",
        ]].copy()
        wide_df = base.copy()

        for horizon_id in cfg.horizon_steps:
            pred_df = all_predictions[horizon_id][split_name].copy()
            prefix = f"H{horizon_id:02d}_"
            add_cols = pred_df[[
                "Target_time",
                "Observed",
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
            os.path.join(cfg.output_dir, f"{split_name.lower()}_predictions_all_horizons_by_input_window.xlsx"),
            index=False
        )


def train_all_horizons(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)

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
    print("Transforming all splits using training-only scalers...")
    transformed_data = transform_raw_splits(raw_data, scaler_grid, scaler_global, scaler_y)

    save_static_information(cfg, raw_data, scaler_grid, scaler_global, scaler_y)

    all_predictions = {}
    all_histories = []
    for horizon_id in cfg.horizon_steps:
        model, prediction_records, history_df = train_one_horizon(
            cfg=cfg,
            transformed_data=transformed_data,
            scaler_y=scaler_y,
            horizon_id=horizon_id
        )
        all_predictions[horizon_id] = prediction_records
        all_histories.append(history_df)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_combined_outputs(cfg, all_predictions)

    history_all_df = pd.concat(all_histories, axis=0, ignore_index=True)
    history_all_df.to_excel(
        os.path.join(cfg.output_dir, "training_history_all_horizons.xlsx"),
        index=False
    )

    print("\nPredictions and training histories saved to:")
    print(cfg.output_dir)
    return all_predictions


if __name__ == "__main__":
    set_seed(cfg.seed)
    train_all_horizons(cfg)
