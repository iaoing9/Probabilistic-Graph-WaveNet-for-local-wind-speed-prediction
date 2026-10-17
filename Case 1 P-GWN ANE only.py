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
    use_cuda_if_available = True
    require_cuda = False
    device = "cuda" if use_cuda_if_available and torch.cuda.is_available() else "cpu"

    samples_per_hour = 6
    horizon_steps = [1, 6, 12, 24]
    max_horizon_step = max(horizon_steps)
    input_window_hours = 24
    input_len = input_window_hours * samples_per_hour
    max_lead_steps = max_horizon_step * samples_per_hour
    stride = 1
    window_alignment = "same_input_window"

    num_nodes = 1
    input_features = 1

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


# Data loading

def read_input_excel(file_path):
    df = pd.read_excel(file_path, usecols="A,AC", engine="openpyxl")
    if df.shape[1] != 2:
        raise ValueError(
            f"{file_path} must contain a time column in A and ANE observations in AC."
        )

    time = df.iloc[:, 0].to_numpy()
    ane = df.iloc[:, 1].astype(np.float32).to_numpy().reshape(-1, 1)

    if not np.isfinite(ane).all():
        raise ValueError(f"{file_path} contains missing or non-finite ANE values.")

    return time, ane


# Standardization

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


# Sliding-window construction

def build_sliding_windows(
    x,
    y,
    time,
    input_len,
    lead_steps,
    max_lead_steps,
    stride,
    alignment,
):
    n = len(y)

    if n != len(x) or n != len(time):
        raise ValueError("Input data, targets, and timestamps must have equal lengths.")
    if input_len <= 0 or lead_steps <= 0 or max_lead_steps <= 0:
        raise ValueError("Window length and lead steps must be positive.")
    if lead_steps > max_lead_steps:
        raise ValueError("lead_steps must not exceed max_lead_steps.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if n < input_len + max_lead_steps:
        raise ValueError(
            f"Split length {n} is too short for input_len={input_len} "
            f"and max_lead_steps={max_lead_steps}."
        )

    x_list = []
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
            x_list.append(x[start:start + input_len])
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

            x_list.append(x[start:start + input_len])
            y_list.append(y[target_idx])
            target_time_list.append(time[target_idx])
            input_start_time_list.append(time[start])
            input_end_time_list.append(time[end])
            input_start_index_list.append(start)
            input_end_index_list.append(end)
            target_index_list.append(target_idx)
    else:
        raise ValueError(
            "alignment must be 'same_input_window' or 'same_target_time'."
        )

    if not x_list:
        raise ValueError("No sliding-window samples were generated.")

    x_windows = np.stack(x_list, axis=0).astype(np.float32)
    y_windows = np.stack(y_list, axis=0).astype(np.float32)

    sample_meta = {
        "target_time": np.asarray(target_time_list),
        "input_start_time": np.asarray(input_start_time_list),
        "input_end_time": np.asarray(input_end_time_list),
        "input_start_index": np.asarray(input_start_index_list, dtype=np.int64),
        "input_end_index": np.asarray(input_end_index_list, dtype=np.int64),
        "target_index": np.asarray(target_index_list, dtype=np.int64),
    }

    return x_windows, y_windows, sample_meta


# Dataset and data loaders

class WindWindowDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Data preparation

def load_raw_splits(config):
    split_files = {
        "Train": config.train_file,
        "Validation": config.val_file,
        "Test": config.test_file,
    }
    raw_data = {}

    for split_name, filename in split_files.items():
        file_path = config.data_dir / filename
        time, ane = read_input_excel(file_path)
        raw_data[split_name] = {
            "time": time,
            "ane_raw": ane,
            "raw_size": len(ane),
        }

    return raw_data


def fit_scaler_on_training(raw_data):
    scaler = StandardScaler()
    scaler.fit(raw_data["Train"]["ane_raw"])
    return scaler


def transform_raw_splits(raw_data, scaler):
    transformed = {}

    for split_name, data in raw_data.items():
        ane_scaled = scaler.transform(data["ane_raw"])
        transformed[split_name] = {
            "time": data["time"],
            "x": ane_scaled,
            "y": ane_scaled,
            "y_raw": data["ane_raw"],
            "raw_size": data["raw_size"],
        }

    return transformed


def prepare_data_for_horizon(config, transformed_data, horizon_id):
    lead_steps = horizon_id * config.samples_per_hour
    datasets = {}
    loaders = {}
    sample_metas = {}

    for split_name, data in transformed_data.items():
        x_windows, y_windows, sample_meta = build_sliding_windows(
            x=data["x"],
            y=data["y"],
            time=data["time"],
            input_len=config.input_len,
            lead_steps=lead_steps,
            max_lead_steps=config.max_lead_steps,
            stride=config.stride,
            alignment=config.window_alignment,
        )

        dataset = WindWindowDataset(x_windows, y_windows)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle_train if split_name == "Train" else False,
            drop_last=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

        datasets[split_name] = dataset
        loaders[split_name] = loader
        sample_metas[split_name] = sample_meta

    return datasets, loaders, sample_metas


# GWN architecture

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
                torch.randn(num_nodes, node_emb_dim), requires_grad=True
            )
            self.nodevec2 = nn.Parameter(
                torch.randn(node_emb_dim, num_nodes), requires_grad=True
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
                F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1
            )
            supports.append(adaptive_adj)
        return supports

    def forward(self, x_ane):
        if x_ane.ndim != 3 or x_ane.shape[-1] != 1:
            raise ValueError("ANE input must have shape [batch, time, 1].")

        x = x_ane.permute(0, 2, 1).unsqueeze(2).contiguous()
        if x.size(2) != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} node(s), received {x.size(2)}."
            )
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

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            mean, variance = model(x)
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

    for x, y in loader:
        x = x.to(device)
        mean, variance = model(x)
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


# Prediction post-processing

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


def train_one_horizon(config, transformed_data, scaler, horizon_id):
    lead_steps = horizon_id * config.samples_per_hour
    horizon_dir = (
        config.output_dir / f"Horizon_{horizon_id:02d}_step_{horizon_id:02d}h_ahead"
    )
    horizon_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed + horizon_id)
    datasets, loaders, sample_metas = prepare_data_for_horizon(
        config, transformed_data, horizon_id
    )
    model = build_model(config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    wait = 0
    history = []
    best_model_path = (
        horizon_dir / f"best_ane_only_temporal_wavenet_horizon_{horizon_id:02d}.pt"
    )

    print("\n" + "=" * 72)
    print(
        f"Training {horizon_id} h ahead; lead_steps={lead_steps}; "
        f"input_window={config.input_len} samples"
    )
    for split_name in ("Train", "Validation", "Test"):
        print(f"{split_name} samples: {len(datasets[split_name])}")
    print("=" * 72)

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

        history.append(
            {
                "Horizon_step": horizon_id,
                "Lead_hours": horizon_id,
                "Lead_samples": lead_steps,
                "Epoch": epoch,
                "Train_GNLL": train_loss,
                "Validation_GNLL": val_loss,
            }
        )

        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            wait = 0
            if config.save_model:
                torch.save(model.state_dict(), best_model_path)
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Horizon {horizon_id:02d} | Epoch {epoch:04d} | "
                f"Train GNLL={train_loss:.6f} | Val GNLL={val_loss:.6f} | "
                f"Best epoch={best_epoch}"
            )

        if wait >= config.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_excel(
        horizon_dir / f"training_history_horizon_{horizon_id:02d}.xlsx",
        index=False,
    )

    if config.save_model and best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=config.device))

    prediction_records = {}

    for split_name in ("Train", "Validation", "Test"):
        eval_loader = DataLoader(
            datasets[split_name],
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

        y_norm, mean_norm, variance_norm = predict(model, eval_loader, config.device)
        y, mean, variance = inverse_y_and_variance(
            y_norm, mean_norm, variance_norm, scaler
        )

        y_flat = y.reshape(-1)
        mean_flat = mean.reshape(-1)
        variance_flat = variance.reshape(-1)
        std_flat = np.sqrt(np.maximum(variance_flat, 1e-12))
        lower = mean_flat - config.z_value * std_flat
        upper = mean_flat + config.z_value * std_flat
        sample_meta = sample_metas[split_name]

        pred_df = pd.DataFrame(
            {
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
                "Predicted_mean": mean_flat,
                "Predicted_variance": variance_flat,
                "Predicted_std": std_flat,
                "Lower_95PI": lower,
                "Upper_95PI": upper,
                "Residual": y_flat - mean_flat,
            }
        )

        pred_path = (
            horizon_dir
            / f"{split_name}_prediction_results_horizon_{horizon_id:02d}.xlsx"
        )
        pred_df.to_excel(pred_path, index=False)
        prediction_records[split_name] = pred_df

    print(
        f"Horizon {horizon_id:02d}: best epoch={best_epoch}, "
        f"best validation GNLL={best_val_loss:.6f}"
    )

    return prediction_records, history_df


# Output files

def save_static_information(config, raw_data, scaler):
    config.output_dir.mkdir(parents=True, exist_ok=True)

    split_info = []
    for split_name, data in raw_data.items():
        raw_size = len(data["ane_raw"])
        effective_size = raw_size - config.input_len - config.max_lead_steps + 1
        if config.stride > 1:
            effective_size = int(np.floor((effective_size - 1) / config.stride) + 1)

        split_info.append(
            {
                "Split": split_name,
                "Raw_sample_size": raw_size,
                "Input_window_samples": config.input_len,
                "Input_window_hours": config.input_window_hours,
                "Max_lead_samples": config.max_lead_steps,
                "Max_lead_hours": config.max_horizon_step,
                "Selected_horizon_steps": str(config.horizon_steps),
                "Stride_samples": config.stride,
                "Effective_sample_size_each_horizon": effective_size,
                "Dropped_samples_due_to_common_truncation": raw_size - effective_size,
                "Window_alignment": config.window_alignment,
            }
        )

    pd.DataFrame(split_info).to_excel(
        config.output_dir / "data_split_and_window_sample_size_summary.xlsx",
        index=False,
    )

    scaler_info = pd.DataFrame(
        [
            {
                "Scaler": "historical_ANE_and_target_ANE",
                "Shape_mean": str(np.asarray(scaler.mean).shape),
                "Shape_std": str(np.asarray(scaler.std).shape),
                "Mean_values": np.array2string(
                    np.squeeze(scaler.mean), precision=6, separator=", "
                ),
                "Std_values": np.array2string(
                    np.squeeze(scaler.std), precision=6, separator=", "
                ),
            }
        ]
    )
    scaler_info.to_excel(
        config.output_dir / "training_only_scaler_information.xlsx",
        index=False,
    )

    feature_info = pd.DataFrame(
        {
            "Feature_group": [
                "Historical_ANE_observed_wind_speed",
                "Target_ANE_observed_wind_speed",
                "ECMWF_features",
            ],
            "Excel_columns": ["AC", "AC shifted according to horizon", "B:AB"],
            "Used_as": [
                "temporal input",
                "prediction target",
                "not used",
            ],
        }
    )
    feature_info.to_excel(
        config.output_dir / "feature_definition.xlsx",
        index=False,
    )

    graph_info = pd.DataFrame(
        [
            {
                "Number_of_nodes": config.num_nodes,
                "Predefined_adjacency": "identity matrix",
                "Adaptive_adjacency_enabled": config.adaptive_adj,
                "Graph_convolution_module": "retained",
                "Spatial_node_mixing": "none; single-node graph",
                "Readout_weight": 1.0,
            }
        ]
    )
    graph_info.to_excel(
        config.output_dir / "graph_configuration.xlsx",
        index=False,
    )


def save_combined_outputs(config, all_predictions):
    for split_name in ("Train", "Validation", "Test"):
        prediction_list = [
            all_predictions[horizon_id][split_name].copy()
            for horizon_id in config.horizon_steps
        ]
        long_df = pd.concat(prediction_list, axis=0, ignore_index=True)
        long_df.to_excel(
            config.output_dir
            / f"{split_name}_prediction_results_all_horizons_long.xlsx",
            index=False,
        )

        first_horizon = config.horizon_steps[0]
        base = all_predictions[first_horizon][split_name][
            ["Sample_index", "Input_start_time", "Input_end_time"]
        ].copy()
        wide_df = base.copy()

        output_columns = [
            "Target_time",
            "Observed",
            "Predicted_mean",
            "Predicted_std",
            "Lower_95PI",
            "Upper_95PI",
            "Residual",
        ]

        for horizon_id in config.horizon_steps:
            pred_df = all_predictions[horizon_id][split_name].copy()
            prefix = f"H{horizon_id:02d}_"
            add_cols = pred_df[output_columns].rename(
                columns={column: prefix + column for column in output_columns}
            )
            wide_df = pd.concat([wide_df, add_cols], axis=1)

        wide_df.to_excel(
            config.output_dir
            / f"{split_name}_prediction_results_all_horizons_wide.xlsx",
            index=False,
        )


# Main workflow

def train_all_horizons(config):
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.require_cuda and config.device != "cuda":
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build "
            "or set require_cuda = False."
        )

    print(f"Using device: {config.device}")
    if config.device == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is unavailable. The script will run on CPU.")

    print("Input setting: historical ANE observations only.")
    print(f"Graph convolution retained; num_nodes={config.num_nodes}, adaptive_adj={config.adaptive_adj}.")
    print(f"Selected horizon steps: {config.horizon_steps}")

    raw_data = load_raw_splits(config)
    scaler = fit_scaler_on_training(raw_data)
    transformed_data = transform_raw_splits(raw_data, scaler)
    save_static_information(config, raw_data, scaler)

    all_predictions = {}
    all_histories = []

    for horizon_id in config.horizon_steps:
        prediction_records, history_df = train_one_horizon(
            config=config,
            transformed_data=transformed_data,
            scaler=scaler,
            horizon_id=horizon_id,
        )
        all_predictions[horizon_id] = prediction_records
        all_histories.append(history_df)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_combined_outputs(config, all_predictions)

    history_all_df = pd.concat(all_histories, axis=0, ignore_index=True)
    history_all_df.to_excel(
        config.output_dir / "training_history_all_horizons.xlsx",
        index=False,
    )

    print("\nAll predictions and training histories saved to:")
    print(config.output_dir)


if __name__ == "__main__":
    set_seed(cfg.seed)
    train_all_horizons(cfg)
