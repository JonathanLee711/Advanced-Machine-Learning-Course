from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from models import ModelSpec, build_model


PAPER_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PAPER_DIR / "data" / "combined_result2.0.xlsx"
RESULT_DIR = PAPER_DIR / "results"
FIGURE_DIR = PAPER_DIR / "figures"
CHECKPOINT_DIR = PAPER_DIR / "checkpoints"

TARGET_COL = "叶丝膨胀干燥_HT入口叶丝含水率"
DEFAULT_MODELS = ["mlp", "cnn", "lstm", "gru", "bilstm", "cnn_lstm", "proposed"]
DEFAULT_ABLATIONS = ["full", "single_scale", "no_attention", "no_residual", "no_aux"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper comparison and ablation experiments.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to source Excel data.")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum epochs for each model.")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--window", type=int, default=16, help="Historical input window.")
    parser.add_argument("--horizon", type=int, default=4, help="Prediction horizon.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Training device.")
    parser.add_argument("--skip-ablation", action="store_true", help="Run comparison models only.")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save best checkpoints.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_excel(path)
    numeric = df.select_dtypes(include=[np.number]).copy()
    if TARGET_COL not in numeric.columns:
        raise ValueError(f"Target column not found: {TARGET_COL}")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").dropna()
    return numeric.reset_index(drop=True)


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    loose_cols = [c for c in df.columns if c.startswith("松散回潮_")]
    if len(loose_cols) >= 10:
        return loose_cols[:10]
    fallback = [c for c in df.columns if c != TARGET_COL]
    if len(fallback) < 10:
        raise ValueError("Not enough numeric feature columns.")
    return fallback[:10]


def make_sequences(values: np.ndarray, target: np.ndarray, window: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    max_start = len(values) - window - horizon + 1
    for start in range(max_start):
        end = start + window
        xs.append(values[start:end])
        ys.append(target[end : end + horizon, 0])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def prepare_data(df: pd.DataFrame, window: int, horizon: int):
    feature_cols = select_feature_columns(df)
    input_cols = feature_cols + [TARGET_COL]
    train_row_end = int(len(df) * 0.70)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(df.loc[: train_row_end - 1, input_cols])
    y_scaler.fit(df.loc[: train_row_end - 1, [TARGET_COL]])

    x_scaled = x_scaler.transform(df[input_cols])
    y_scaled = y_scaler.transform(df[[TARGET_COL]])
    x, y = make_sequences(x_scaled, y_scaled, window, horizon)

    n = len(x)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    splits = {
        "train": (x[:train_end], y[:train_end]),
        "val": (x[train_end:val_end], y[train_end:val_end]),
        "test": (x[val_end:], y[val_end:]),
    }
    return splits, x_scaler, y_scaler, feature_cols, input_cols


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def unpack_output(output):
    if isinstance(output, tuple):
        return output
    return output, None


def train_one_model(
    model: nn.Module,
    spec: ModelSpec,
    loaders: Dict[str, DataLoader],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, float]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.to(device)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loaders["train"]:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, aux = unpack_output(model(xb))
            loss = criterion(pred, yb)
            if aux is not None and spec.aux_weight > 0:
                loss = loss + spec.aux_weight * criterion(aux.squeeze(-1), yb[:, 0])
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = evaluate_loss(model, loaders["val"], criterion, spec, device)
        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - start_time
    info = {"best_val_loss": best_val, "elapsed_sec": elapsed, "epochs_ran": len(history["val_loss"])}
    return model, history, info


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, spec: ModelSpec, device: torch.device) -> float:
    model.eval()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred, aux = unpack_output(model(xb))
        loss = criterion(pred, yb)
        if aux is not None and spec.aux_weight > 0:
            loss = loss + spec.aux_weight * criterion(aux.squeeze(-1), yb[:, 0])
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pred, _ = unpack_output(model(xb))
        preds.append(pred.detach().cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def inverse_target(values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    shape = values.shape
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(shape)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, group: str, info: Dict[str, float]) -> Dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    row: Dict[str, float] = {
        "model": model_name,
        "group": group,
        "MAE": mean_absolute_error(yt, yp),
        "RMSE": mean_squared_error(yt, yp, squared=False),
        "R2": r2_score(yt, yp),
        "Bias": float(np.mean(yp - yt)),
    }
    for i in range(y_true.shape[1]):
        step_true = y_true[:, i]
        step_pred = y_pred[:, i]
        step = i + 1
        row[f"Step{step}_MAE"] = mean_absolute_error(step_true, step_pred)
        row[f"Step{step}_RMSE"] = mean_squared_error(step_true, step_pred, squared=False)
        row[f"Step{step}_R2"] = r2_score(step_true, step_pred)
    row.update(info)
    return row


def build_step_metrics(metrics_rows: Iterable[Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for row in metrics_rows:
        for step in range(1, 5):
            rows.append(
                {
                    "model": row["model"],
                    "group": row["group"],
                    "step": step,
                    "MAE": row[f"Step{step}_MAE"],
                    "RMSE": row[f"Step{step}_RMSE"],
                    "R2": row[f"Step{step}_R2"],
                }
            )
    return pd.DataFrame(rows)


def save_checkpoint_if_needed(model: nn.Module, spec: ModelSpec, args: argparse.Namespace) -> None:
    if not args.save_checkpoints:
        return
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    filename = spec.name.replace("/", "_").replace(" ", "_") + ".pt"
    torch.save(model.state_dict(), CHECKPOINT_DIR / filename)


def run_experiment(model_key: str, splits, y_scaler, args, device):
    input_dim = splits["train"][0].shape[-1]
    model, spec = build_model(model_key, input_dim, args.window, args.horizon, args.hidden_dim, args.dropout)
    loaders = {
        "train": make_loader(*splits["train"], batch_size=args.batch_size, shuffle=True),
        "val": make_loader(*splits["val"], batch_size=args.batch_size, shuffle=False),
        "test": make_loader(*splits["test"], batch_size=args.batch_size, shuffle=False),
    }
    print(f"Training {spec.name} on {device} ...")
    model, history, info = train_one_model(model, spec, loaders, args, device)
    pred_scaled, true_scaled = predict(model, loaders["test"], device)
    pred = inverse_target(pred_scaled, y_scaler)
    true = inverse_target(true_scaled, y_scaler)
    metrics = compute_metrics(true, pred, spec.name, spec.group, info)
    save_checkpoint_if_needed(model, spec, args)
    return metrics, history, true, pred


def plot_comparison(df: pd.DataFrame) -> None:
    comp = df[df["group"] == "comparison"].copy()
    x = np.arange(len(comp))
    width = 0.36
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, comp["MAE"], width, label="MAE")
    plt.bar(x + width / 2, comp["RMSE"], width, label="RMSE")
    plt.xticks(x, comp["model"], rotation=25, ha="right")
    plt.ylabel("Error")
    plt.title("Comparison Experiment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "comparison_mae_rmse.png", dpi=300)
    plt.savefig(FIGURE_DIR / "paper_ready_comparison_mae_rmse.png", dpi=300)
    plt.close()


def plot_step(step_df: pd.DataFrame) -> None:
    comp = step_df[step_df["group"] == "comparison"].copy()
    plt.figure(figsize=(9, 5))
    for model, group in comp.groupby("model"):
        plt.plot(group["step"], group["MAE"], marker="o", label=model)
    plt.xticks([1, 2, 3, 4])
    plt.xlabel("Prediction Step")
    plt.ylabel("MAE")
    plt.title("Step-wise MAE")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "step_mae.png", dpi=300)
    plt.savefig(FIGURE_DIR / "paper_ready_step_mae.png", dpi=300)
    plt.close()


def plot_ablation(df: pd.DataFrame) -> None:
    abl = df[df["group"] == "ablation"].copy()
    if abl.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(abl["model"], abl["MAE"], color="#4C78A8")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("MAE")
    plt.title("Ablation Experiment")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ablation_mae.png", dpi=300)
    plt.savefig(FIGURE_DIR / "paper_ready_ablation_mae.png", dpi=300)
    plt.close()


def plot_prediction_curve(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    n = min(300, len(y_true))
    plt.figure(figsize=(10, 4.8))
    plt.plot(y_true[:n, 0], label="True", linewidth=1.4)
    plt.plot(y_pred[:n, 0], label="Predicted", linewidth=1.2)
    plt.xlabel("Test Sample")
    plt.ylabel("Moisture")
    plt.title("Prediction Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "prediction_curve.png", dpi=300)
    plt.savefig(FIGURE_DIR / "project_checkpoint_prediction_curve.png", dpi=300)
    plt.close()

    residual = y_pred[:, 0] - y_true[:, 0]
    plt.figure(figsize=(7, 4.5))
    plt.hist(residual, bins=45, color="#4C78A8", alpha=0.85)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "residual_distribution.png", dpi=300)
    plt.close()


def plot_loss_curves(histories: Dict[str, Dict[str, List[float]]]) -> None:
    selected = [k for k in ["MLP", "BiLSTM", "Proposed", "Full Model"] if k in histories]
    if not selected:
        selected = list(histories)[:4]
    plt.figure(figsize=(9, 5))
    for name in selected:
        plt.plot(histories[name]["val_loss"], label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "loss_curves.png", dpi=300)
    plt.close()


def save_outputs(metrics_df: pd.DataFrame, step_df: pd.DataFrame, histories, config) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    comp = metrics_df[metrics_df["group"] == "comparison"].copy()
    abl = metrics_df[metrics_df["group"] == "ablation"].copy()

    comp.to_excel(RESULT_DIR / "paper_ready_comparison_metrics.xlsx", index=False)
    abl.to_excel(RESULT_DIR / "paper_ready_ablation_metrics.xlsx", index=False)
    step_df.to_excel(RESULT_DIR / "paper_ready_step_metrics.xlsx", index=False)
    metrics_df.to_excel(RESULT_DIR / "all_metrics.xlsx", index=False)

    summary = pd.DataFrame(
        [
            {"item": "best_comparison_model", "value": comp.sort_values("MAE").iloc[0]["model"] if not comp.empty else ""},
            {"item": "best_comparison_MAE", "value": comp["MAE"].min() if not comp.empty else np.nan},
            {"item": "data_file", "value": str(config["data"])},
            {"item": "window", "value": config["window"]},
            {"item": "horizon", "value": config["horizon"]},
        ]
    )
    with pd.ExcelWriter(RESULT_DIR / "paper_experiment_tables.xlsx") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        comp.to_excel(writer, sheet_name="Comparison", index=False)
        abl.to_excel(writer, sheet_name="Ablation", index=False)
        step_df.to_excel(writer, sheet_name="Step Metrics", index=False)

    history_rows = []
    for model_name, hist in histories.items():
        for epoch, (train_loss, val_loss) in enumerate(zip(hist["train_loss"], hist["val_loss"]), start=1):
            history_rows.append({"model": model_name, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    pd.DataFrame(history_rows).to_excel(RESULT_DIR / "training_history.xlsx", index=False)

    with open(RESULT_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    df = load_dataframe(args.data)
    splits, _, y_scaler, feature_cols, input_cols = prepare_data(df, args.window, args.horizon)

    metrics_rows = []
    histories = {}
    proposed_true = None
    proposed_pred = None

    model_keys = list(DEFAULT_MODELS)
    if not args.skip_ablation:
        model_keys.extend(DEFAULT_ABLATIONS)

    for model_key in model_keys:
        metrics, history, y_true, y_pred = run_experiment(model_key, splits, y_scaler, args, device)
        metrics_rows.append(metrics)
        histories[metrics["model"]] = history
        if metrics["model"] in {"Proposed", "Full Model"} and proposed_true is None:
            proposed_true = y_true
            proposed_pred = y_pred
        print(
            f"{metrics['model']}: MAE={metrics['MAE']:.6f}, "
            f"RMSE={metrics['RMSE']:.6f}, R2={metrics['R2']:.6f}"
        )

    metrics_df = pd.DataFrame(metrics_rows)
    step_df = build_step_metrics(metrics_rows)

    config = {
        **vars(args),
        "device_used": str(device),
        "feature_cols": feature_cols,
        "input_cols": input_cols,
        "target_col": TARGET_COL,
        "train_size": len(splits["train"][0]),
        "val_size": len(splits["val"][0]),
        "test_size": len(splits["test"][0]),
    }
    save_outputs(metrics_df, step_df, histories, config)
    plot_comparison(metrics_df)
    plot_step(step_df)
    plot_ablation(metrics_df)
    plot_loss_curves(histories)
    if proposed_true is not None and proposed_pred is not None:
        plot_prediction_curve(proposed_true, proposed_pred)

    print(f"Done. Results saved to: {RESULT_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
