from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


TensorOrTuple = Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]


class MLPModel(nn.Module):
    def __init__(self, input_dim: int, train_window: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * train_window, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        channels = max(32, hidden_dim // 2)
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        return self.head(self.features(x))


class RNNModel(nn.Module):
    def __init__(
        self,
        cell: str,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
        bidirectional: bool = False,
        num_layers: int = 2,
    ):
        super().__init__()
        rnn_cls = nn.LSTM if cell.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_mul = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * direction_mul, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float, num_layers: int = 2):
        super().__init__()
        conv_channels = max(32, hidden_dim // 2)
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class ProposedModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        num_layers: int = 2,
        multi_scale: bool = True,
        use_attention: bool = True,
        use_residual: bool = True,
        use_aux: bool = True,
    ):
        super().__init__()
        self.multi_scale = multi_scale
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_aux = use_aux

        if multi_scale:
            self.conv3 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
            self.conv5 = nn.Conv1d(input_dim, 32, kernel_size=5, padding=2)
            self.conv7 = nn.Conv1d(input_dim, 32, kernel_size=7, padding=3)
            conv_dim = 96
        else:
            self.conv_single = nn.Conv1d(input_dim, 96, kernel_size=3, padding=1)
            conv_dim = 96

        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=conv_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.layernorm_lstm = nn.LayerNorm(hidden_dim * 2)

        if use_attention:
            self.attn = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=4, batch_first=True)
            self.layernorm_attn = nn.LayerNorm(hidden_dim * 2)

        if use_residual:
            self.res_proj = nn.Linear(conv_dim, hidden_dim * 2)

        self.dropout = nn.Dropout(dropout)
        self.fc_main = nn.Linear(hidden_dim * 2, output_dim)
        self.fc_aux = nn.Linear(hidden_dim * 2, 1) if use_aux else None

    def _conv_features(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = x.permute(0, 2, 1)
        if self.multi_scale:
            x3 = self.relu(self.conv3(x_conv))
            x5 = self.relu(self.conv5(x_conv))
            x7 = self.relu(self.conv7(x_conv))
            return torch.cat([x3, x5, x7], dim=1)
        return self.relu(self.conv_single(x_conv))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_cat = self._conv_features(x)
        x_cat_t = x_cat.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_cat_t)
        lstm_out = self.layernorm_lstm(lstm_out)

        if self.use_attention:
            feat_seq, _ = self.attn(lstm_out, lstm_out, lstm_out)
            feat_seq = self.layernorm_attn(feat_seq)
        else:
            feat_seq = lstm_out

        feat = feat_seq.mean(dim=1)
        if self.use_residual:
            feat = feat + self.res_proj(x_cat_t.mean(dim=1))

        feat = self.dropout(feat)
        out_main = self.fc_main(feat)
        out_aux = self.fc_aux(feat) if self.fc_aux is not None else None
        return out_main, out_aux


@dataclass(frozen=True)
class ModelSpec:
    name: str
    group: str
    aux_weight: float = 0.0


def build_model(
    name: str,
    input_dim: int,
    train_window: int,
    output_dim: int,
    hidden_dim: int,
    dropout: float,
) -> Tuple[nn.Module, ModelSpec]:
    key = name.lower()
    if key == "mlp":
        return MLPModel(input_dim, train_window, output_dim, hidden_dim, dropout), ModelSpec("MLP", "comparison")
    if key == "cnn":
        return CNNModel(input_dim, output_dim, hidden_dim, dropout), ModelSpec("CNN", "comparison")
    if key == "lstm":
        return RNNModel("lstm", input_dim, output_dim, hidden_dim, dropout), ModelSpec("LSTM", "comparison")
    if key == "gru":
        return RNNModel("gru", input_dim, output_dim, hidden_dim, dropout), ModelSpec("GRU", "comparison")
    if key == "bilstm":
        return RNNModel("lstm", input_dim, output_dim, hidden_dim, dropout, bidirectional=True), ModelSpec("BiLSTM", "comparison")
    if key == "cnn_lstm":
        return CNNLSTMModel(input_dim, output_dim, hidden_dim, dropout), ModelSpec("CNN-LSTM", "comparison")
    if key == "proposed":
        model = ProposedModel(input_dim, hidden_dim, output_dim, dropout)
        return model, ModelSpec("Proposed", "comparison", aux_weight=0.5)
    if key == "full":
        model = ProposedModel(input_dim, hidden_dim, output_dim, dropout)
        return model, ModelSpec("Full Model", "ablation", aux_weight=0.5)
    if key == "single_scale":
        model = ProposedModel(input_dim, hidden_dim, output_dim, dropout, multi_scale=False)
        return model, ModelSpec("w/o Multi-scale CNN", "ablation", aux_weight=0.5)
    if key == "no_attention":
        model = ProposedModel(input_dim, hidden_dim, output_dim, dropout, use_attention=False)
        return model, ModelSpec("w/o Attention", "ablation", aux_weight=0.5)
    if key == "no_residual":
        model = ProposedModel(input_dim, hidden_dim, output_dim, dropout, use_residual=False)
        return model, ModelSpec("w/o Residual", "ablation", aux_weight=0.5)
    if key == "no_aux":
        model = ProposedModel(input_dim, hidden_dim, output_dim, dropout, use_aux=False)
        return model, ModelSpec("w/o Auxiliary Loss", "ablation", aux_weight=0.0)
    raise ValueError(f"Unknown model name: {name}")
