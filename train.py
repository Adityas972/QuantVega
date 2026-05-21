import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

from data_generator import generate_dataset, load_dataset
from neural_pricer import NeuralPricer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples",   type=int,   default=50_000)
    p.add_argument("--width",       type=int,   default=256)
    p.add_argument("--depth",       type=int,   default=4)
    p.add_argument("--dropout",     type=float, default=0.05)
    p.add_argument("--epochs",      type=int,   default=150)
    p.add_argument("--batch_size",  type=int,   default=2048)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-5)
    p.add_argument("--patience",    type=int,   default=20)
    p.add_argument("--data_path",   type=str,   default="data/heston_dataset.parquet")
    p.add_argument("--ckpt_dir",    type=str,   default="checkpoints")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def mse_loss(pred, target):
    return nn.functional.mse_loss(pred, target)


def relative_mse_loss(pred, target, eps=1e-6):
    return ((pred - target) / (target + eps)).pow(2).mean()


def combined_loss(pred, target, alpha=0.5):
    return alpha * mse_loss(pred, target) + (1 - alpha) * relative_mse_loss(pred, target)


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_path = Path(args.data_path)
    if not data_path.exists():
        print("Dataset not found, generating...")
        generate_dataset(n_samples=args.n_samples, save_path=str(data_path))

    splits, _ = load_dataset(str(data_path))
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    x_mean = X_train.mean(axis=0).astype(np.float32)
    x_std  = X_train.std(axis=0).astype(np.float32)

    def to_tensor(X, y):
        return TensorDataset(torch.from_numpy(X).to(device), torch.from_numpy(y).to(device))

    train_ds = to_tensor(X_train, y_train)
    val_ds   = to_tensor(X_val,   y_val)
    test_ds  = to_tensor(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=4096, shuffle=False)

    model = NeuralPricer(
        n_features=X_train.shape[1],
        width=args.width,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    model.set_normalisation(x_mean, x_std)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")

    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=30, T_mult=2, eta_min=1e-5)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss  = float("inf")
    patience_count = 0
    history        = {"train": [], "val": [], "lr": []}

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'LR':>10}  Time")
    print("─" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimiser.zero_grad()
            pred = model(X_batch)
            loss = combined_loss(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * len(X_batch)

        train_loss /= len(train_ds)
        scheduler.step()
        current_lr = optimiser.param_groups[0]["lr"]

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                val_loss += combined_loss(pred, y_batch).item() * len(X_batch)
        val_loss /= len(val_ds)

        elapsed = time.time() - t0
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["lr"].append(current_lr)

        print(f"{epoch:>6} {train_loss:>12.6f} {val_loss:>12.6f} {current_lr:>10.2e}  {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optim_state" : optimiser.state_dict(),
                "val_loss"    : val_loss,
                "x_mean"      : x_mean.tolist(),
                "x_std"       : x_std.tolist(),
                "args"        : vars(args),
            }, ckpt_dir / "best_model.pt")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print("\nLoading best checkpoint...")
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    with torch.no_grad():
        pred_test = model(X_test_t)

    mse   = nn.functional.mse_loss(pred_test, y_test_t).item()
    mae   = (pred_test - y_test_t).abs().mean().item()
    rmse  = mse**0.5
    r_mse = ((pred_test - y_test_t) / (y_test_t + 1e-6)).pow(2).mean().sqrt().item()

    print(f"\n{'='*45}")
    print(f"TEST RESULTS (best val epoch: {ckpt['epoch']})")
    print(f"  RMSE         : {rmse:.6f}")
    print(f"  MAE          : {mae:.6f}")
    print(f"  Relative RMSE: {r_mse*100:.4f}%")
    print(f"  Best val loss: {ckpt['val_loss']:.6f}")
    print(f"{'='*45}")

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f)

    return model, history


if __name__ == "__main__":
    args = parse_args()
    train(args)
