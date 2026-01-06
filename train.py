from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from dataset import CmapssRandomCropDataset
from model import RulLstm


INDEX_NAMES = ["unit_nr", "time_cycles"]
SETTING_NAMES = ["setting_1", "setting_2", "setting_3"]
SENSOR_NAMES = [f"s_{i}" for i in range(1, 22)]
COL_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES

FEATURES_TO_DROP = [
    "unit_nr",
    "time_cycles",
    "RUL",
    "max_cycle",
    "s_1",
    "s_5",
    "s_10",
    "s_16",
    "s_18",
    "s_19",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_pad(
    batch: List[Tuple[torch.Tensor, float, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs, targets, lengths = zip(*batch)
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)
    targets_t = torch.tensor(targets, dtype=torch.float32)
    return padded.float(), lengths_t, targets_t


def make_loader(
    sequences_by_unit: Dict[int, torch.Tensor],
    rul_by_unit: Dict[int, torch.Tensor],
    samples_per_epoch: int,
    batch_size: int,
    l_min: int,
    l_max: int,
    num_workers: int = 0,
) -> DataLoader:
    ds = CmapssRandomCropDataset(
        sequences_by_unit=sequences_by_unit,
        rul_by_unit=rul_by_unit,
        samples_per_epoch=samples_per_epoch,
        l_min=l_min,
        l_max=l_max,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pad,
        drop_last=True,
        pin_memory=True,
    )


def load_fd(fd_tag: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = f"data/train_{fd_tag}.txt"
    test_path = f"data/test_{fd_tag}.txt"
    rul_path = f"data/RUL_{fd_tag}.txt"

    raw_train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COL_NAMES)
    raw_test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COL_NAMES)
    raw_rul_labels_df = pd.read_csv(rul_path, header=None, names=["RUL_truth"])

    max_cycle = raw_train_df.groupby("unit_nr")["time_cycles"].max().rename("max_cycle")
    raw_train_df = raw_train_df.merge(max_cycle, left_on="unit_nr", right_index=True)
    raw_train_df["RUL"] = raw_train_df["max_cycle"] - raw_train_df["time_cycles"]

    return raw_train_df, raw_test_df, raw_rul_labels_df


def merge_fds(fd_tags: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    next_unit = 1
    train_dfs = []
    test_dfs = []
    test_rul_labels = []

    for fd_tag in fd_tags:
        train_df_chunk, test_df_chunk, rul_labels_chunk = load_fd(fd_tag)
        train_df_chunk["fd"] = fd_tag
        test_df_chunk["fd"] = fd_tag

        test_df_chunk = test_df_chunk.assign(
            unit_nr_orig=test_df_chunk["unit_nr"],
            unit_nr=test_df_chunk["unit_nr"] + next_unit - 1,
        )
        test_dfs.append(test_df_chunk)
        test_rul_labels.append(rul_labels_chunk)

        uniq_units = sorted(train_df_chunk["unit_nr"].unique())
        mapping = {u: next_unit + i for i, u in enumerate(uniq_units)}
        next_unit += len(uniq_units)

        train_df_chunk = train_df_chunk.assign(
            unit_nr_orig=train_df_chunk["unit_nr"],
            unit_nr=train_df_chunk["unit_nr"].map(mapping),
            fd=fd_tag,
        )
        train_dfs.append(train_df_chunk)

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    rul_labels_df = pd.concat(test_rul_labels, ignore_index=True)
    return train_df, test_df, rul_labels_df


def split_train_val(
    train_df: pd.DataFrame,
    train_units: int | None,
    train_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    engine_ids = train_df["unit_nr"].unique()
    if train_units is None:
        split_idx = max(1, int(len(engine_ids) * train_frac))
    else:
        split_idx = min(len(engine_ids), train_units)
        if split_idx >= len(engine_ids):
            split_idx = max(1, int(len(engine_ids) * train_frac))

    train_ids = engine_ids[:split_idx]
    val_ids = engine_ids[split_idx:]
    if len(val_ids) == 0:
        raise ValueError("Validation split is empty. Reduce --train-units or --train-frac.")

    train_split_df = train_df[train_df["unit_nr"].isin(train_ids)]
    val_split_df = train_df[train_df["unit_nr"].isin(val_ids)]
    return train_split_df, val_split_df


def scale_splits(
    train_split_df: pd.DataFrame,
    val_split_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler, List[str]]:
    not_scaled_cols = ["unit_nr", "RUL", "max_cycle", "time_cycles"]
    columns_to_scale = [col for col in COL_NAMES if col not in not_scaled_cols]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_split = scaler.fit_transform(train_split_df[columns_to_scale])
    scaled_val_split = scaler.transform(val_split_df[columns_to_scale])
    scaled_test = scaler.transform(test_df[columns_to_scale])

    scaled_train_split_df = pd.DataFrame(
        scaled_train_split, columns=columns_to_scale, index=train_split_df.index
    )
    scaled_val_split_df = pd.DataFrame(
        scaled_val_split, columns=columns_to_scale, index=val_split_df.index
    )
    scaled_test_df = pd.DataFrame(
        scaled_test, columns=columns_to_scale, index=test_df.index
    )

    scaled_train_split_df.insert(0, "unit_nr", train_split_df["unit_nr"])
    scaled_train_split_df.insert(1, "time_cycles", train_split_df["time_cycles"])
    scaled_train_split_df.insert(
        len(scaled_train_split_df.columns), "RUL", train_split_df["RUL"]
    )
    scaled_train_split_df.insert(
        len(scaled_train_split_df.columns), "max_cycle", train_split_df["max_cycle"]
    )

    scaled_val_split_df.insert(0, "unit_nr", val_split_df["unit_nr"])
    scaled_val_split_df.insert(1, "time_cycles", val_split_df["time_cycles"])
    scaled_val_split_df.insert(
        len(scaled_val_split_df.columns), "RUL", val_split_df["RUL"]
    )
    scaled_val_split_df.insert(
        len(scaled_val_split_df.columns), "max_cycle", val_split_df["max_cycle"]
    )

    scaled_test_df.insert(0, "unit_nr", test_df["unit_nr"])
    scaled_test_df.insert(1, "time_cycles", test_df["time_cycles"])

    return scaled_train_split_df, scaled_val_split_df, scaled_test_df, scaler, columns_to_scale


def build_unit_dicts(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    sequences_by_unit: Dict[int, torch.Tensor] = {}
    rul_by_unit: Dict[int, torch.Tensor] = {}

    df = df.sort_values(["unit_nr", "time_cycles"])
    for unit_id, g in df.groupby("unit_nr", sort=False):
        x = torch.tensor(g[feature_cols].to_numpy(), dtype=torch.float32)
        y = torch.tensor(g["RUL"].to_numpy(), dtype=torch.float32)
        sequences_by_unit[int(unit_id)] = x
        rul_by_unit[int(unit_id)] = y

    return sequences_by_unit, rul_by_unit


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for padded, lengths, targets in loader:
            padded = padded.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)

            preds = model(padded, lengths)
            loss = loss_fn(preds, targets)

            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

    return total_loss / max(total_samples, 1)


def train_full(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    output_dir: Path,
) -> float:
    loss_fn = nn.SmoothL1Loss(reduction="mean")
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            train=True,
            optimizer=optimizer,
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            train=False,
            optimizer=None,
        )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, output_dir / "lstm_model.pth")
        print(f"Model weights saved to {output_dir / 'lstm_model.pth'}")

    return best_val


def export_onnx(
    model: nn.Module,
    feature_count: int,
    output_path: Path,
    seq_len: int,
    batch_size: int,
    opset: int,
) -> None:
    class OnnxExportWrapper(nn.Module):
        def __init__(self, base_model: nn.Module):
            super().__init__()
            self.lstm = base_model.lstm
            self.head = base_model.head

        def forward(self, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(padded)
            lengths = torch.clamp(lengths - 1, min=0)
            idx = lengths.view(-1, 1, 1).expand(-1, 1, out.size(2))
            last = out.gather(1, idx).squeeze(1)
            return self.head(last).squeeze(1)

    export_model = OnnxExportWrapper(model).to("cpu")
    export_model.eval()

    x = torch.randn(batch_size, seq_len, feature_count)
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

    torch.onnx.export(
        export_model,
        (x, lengths),
        output_path.as_posix(),
        input_names=["input", "lengths"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes={
            "input": {0: "batch_size", 1: "sequence_length"},
            "lengths": {0: "batch_size"},
        },
        dynamo=False,
    )
    print(f"ONNX model exported to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM on CMAPSS data.")
    parser.add_argument(
        "--data-tags",
        type=str,
        default="FD001,FD003",
        help="Comma-separated FD tags to use (e.g., FD001,FD003).",
    )
    parser.add_argument(
        "--train-units",
        type=int,
        default=150,
        help="Number of engines to use for training (rest for validation).",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fallback train fraction if --train-units exceeds data size.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-epoch", type=int, default=60000)
    parser.add_argument("--val-samples-per-epoch", type=int, default=14000)
    parser.add_argument("--l-min", type=int, default=30)
    parser.add_argument("--l-max", type=int, default=100)
    parser.add_argument("--val-l-max", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-rul",
        type=str,
        default="auto",
        help="RUL scaling. Use 'auto' to scale by max train RUL, or 'none' to disable.",
    )
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--onnx-opset", type=int, default=18)
    parser.add_argument("--onnx-seq-len", type=int, default=217)
    parser.add_argument("--onnx-batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    fd_tags = [tag.strip() for tag in args.data_tags.split(",") if tag.strip()]
    if not fd_tags:
        raise ValueError("No FD tags provided.")

    train_df, test_df, rul_labels_df = merge_fds(fd_tags)
    train_split_df, val_split_df = split_train_val(
        train_df, train_units=args.train_units, train_frac=args.train_frac
    )

    max_rul_value = None
    if args.max_rul.lower() == "auto":
        max_rul_value = float(train_df["RUL"].max())
    elif args.max_rul.lower() == "none":
        max_rul_value = None
    else:
        max_rul_value = float(args.max_rul)

    if max_rul_value:
        train_split_df = train_split_df.copy()
        val_split_df = val_split_df.copy()
        train_split_df["RUL"] = train_split_df["RUL"] / max_rul_value
        val_split_df["RUL"] = val_split_df["RUL"] / max_rul_value

    (
        scaled_train_split_df,
        scaled_val_split_df,
        scaled_test_df,
        scaler,
        _,
    ) = scale_splits(train_split_df, val_split_df, test_df)

    feature_cols = [c for c in COL_NAMES if c not in FEATURES_TO_DROP]

    train_sequences_by_unit, train_rul_by_unit = build_unit_dicts(
        scaled_train_split_df, feature_cols
    )
    val_sequences_by_unit, val_rul_by_unit = build_unit_dicts(
        scaled_val_split_df, feature_cols
    )

    train_loader = make_loader(
        train_sequences_by_unit,
        train_rul_by_unit,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
        l_min=args.l_min,
        l_max=args.l_max,
    )

    val_loader = make_loader(
        val_sequences_by_unit,
        val_rul_by_unit,
        samples_per_epoch=args.val_samples_per_epoch,
        batch_size=args.batch_size,
        l_min=args.l_min,
        l_max=args.val_l_max,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = RulLstm(
        n_features=len(feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_full(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        output_dir=output_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_dir / "scaler.pkl")
    if max_rul_value:
        (output_dir / "max_rul.txt").write_text(f"{max_rul_value}\n")
    print(f"Scaler saved to {output_dir / 'scaler.pkl'}")

    export_onnx(
        model=model,
        feature_count=len(feature_cols),
        output_path=output_dir / "lstm_model.onnx",
        seq_len=args.onnx_seq_len,
        batch_size=args.onnx_batch_size,
        opset=args.onnx_opset,
    )

    if len(rul_labels_df) > 0:
        print(f"Test RUL labels shape: {rul_labels_df.shape}")
    print("Training complete.")


if __name__ == "__main__":
    main()
