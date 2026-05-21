import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from pricing_models import bs_call_price, bs_greeks
from neural_pricer import NeuralPricer
from data_generator import load_dataset


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]

    model = NeuralPricer(
        n_features=8,
        width=args["width"],
        depth=args["depth"],
        dropout=0.0,
    ).to(device)

    model.set_normalisation(
        np.array(ckpt["x_mean"], dtype=np.float32),
        np.array(ckpt["x_std"],  dtype=np.float32),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}  (val loss: {ckpt['val_loss']:.6f})")
    return model


def bucket_by_moneyness(log_m):
    return np.where(log_m < -0.05, "OTM", np.where(log_m > 0.05, "ITM", "ATM"))


def evaluate_price_accuracy(model, X_test, y_test, df_test, device):
    X_t = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        pred_norm = model(X_t).cpu().numpy()

    S      = df_test["S"].values
    C_true = y_test    * S
    C_pred = pred_norm * S

    rmse   = np.sqrt(np.mean((C_pred - C_true)**2))
    mae    = np.mean(np.abs(C_pred - C_true))
    r_rmse = np.sqrt(np.mean(((C_pred - C_true) / (C_true + 1e-6))**2))

    print(f"\nPRICE ACCURACY  (n={len(y_test):,})")
    print(f"  RMSE          : {rmse:.4f}")
    print(f"  MAE           : {mae:.4f}")
    print(f"  Relative RMSE : {r_rmse*100:.3f}%")

    labels = bucket_by_moneyness(X_test[:, 0])
    for bucket in ["OTM", "ATM", "ITM"]:
        mask = labels == bucket
        if mask.sum() == 0:
            continue
        b_rmse = np.sqrt(np.mean((C_pred[mask] - C_true[mask])**2))
        b_mae  = np.mean(np.abs(C_pred[mask] - C_true[mask]))
        print(f"  {bucket} ({mask.sum():>5,}) — RMSE {b_rmse:.4f}  MAE {b_mae:.4f}")

    return C_true, C_pred, labels


def evaluate_greeks(model, device, n_points=200):
    K       = 100.0
    T       = 1.0
    r       = 0.04
    sigma   = 0.20
    v0      = sigma**2
    kappa   = 2.0
    theta   = sigma**2
    sigma_v = 0.30
    rho     = -0.60

    S_range  = np.linspace(70, 130, n_points)
    bs_delta = np.array([bs_greeks(s, K, T, r, sigma)["delta"] for s in S_range])
    bs_gamma = np.array([bs_greeks(s, K, T, r, sigma)["gamma"] for s in S_range])
    bs_price = np.array([bs_call_price(s, K, T, r, sigma) for s in S_range])

    S_t = torch.tensor(S_range, dtype=torch.float32, device=device)
    X_t = torch.stack([
        torch.log(S_t / K),
        torch.full_like(S_t, T),
        torch.full_like(S_t, r),
        torch.full_like(S_t, v0),
        torch.full_like(S_t, kappa),
        torch.full_like(S_t, theta),
        torch.full_like(S_t, sigma_v),
        torch.full_like(S_t, rho),
    ], dim=1)

    nn_out   = model.predict_with_greeks(X_t, S_t)
    nn_delta = nn_out["delta"].cpu().numpy()
    nn_gamma = nn_out["gamma"].cpu().numpy()
    nn_price = nn_out["price"].cpu().numpy()

    return {
        "S"        : S_range,
        "bs_price" : bs_price,
        "nn_price" : nn_price,
        "bs_delta" : bs_delta,
        "nn_delta" : nn_delta,
        "bs_gamma" : bs_gamma,
        "nn_gamma" : nn_gamma,
    }


def plot_all(C_true, C_pred, labels, greeks, history_path, save_dir):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    colours = {"OTM": "#E07B54", "ATM": "#4C72B0", "ITM": "#55A868"}

    ax1 = fig.add_subplot(gs[0, :2])
    for bucket in ["OTM", "ATM", "ITM"]:
        mask = labels == bucket
        ax1.scatter(C_true[mask], C_pred[mask], s=4, alpha=0.4, label=bucket, color=colours[bucket])
    lim = [C_true.min(), C_true.max()]
    ax1.plot(lim, lim, "k--", lw=1.5, label="Perfect")
    ax1.set_xlabel("Heston Price (True)")
    ax1.set_ylabel("Neural Pricer (Predicted)")
    ax1.set_title("Price Parity: Neural Pricer vs Heston Ground Truth")
    ax1.legend(markerscale=3)

    ax2 = fig.add_subplot(gs[0, 2])
    rel_err = (C_pred - C_true) / (C_true + 1e-6) * 100
    ax2.hist(np.clip(rel_err, -20, 20), bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color="red", lw=1.5, linestyle="--")
    ax2.set_xlabel("Relative Error (%)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Error Distribution  (mean={rel_err.mean():.2f}%, std={rel_err.std():.2f}%)")

    g = greeks

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(g["S"], g["bs_price"], "k-",  lw=2, label="Black-Scholes")
    ax3.plot(g["S"], g["nn_price"], "--",  lw=2, color="#E07B54", label="Neural Pricer")
    ax3.set_xlabel("Spot Price S")
    ax3.set_ylabel("Call Price")
    ax3.set_title("Option Price Curve  (K=100, T=1yr)")
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(g["S"], g["bs_delta"], "k-",  lw=2, label="BS Analytical")
    ax4.plot(g["S"], g["nn_delta"], "--",  lw=2, color="#E07B54", label="NN Autograd")
    ax4.axvline(100, color="grey", lw=1, linestyle=":")
    ax4.set_xlabel("Spot Price S")
    ax4.set_ylabel("Delta")
    ax4.set_title("Delta: Autograd vs Analytical")
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(g["S"], g["bs_gamma"], "k-",  lw=2, label="BS Analytical")
    ax5.plot(g["S"], g["nn_gamma"], "--",  lw=2, color="#E07B54", label="NN Autograd")
    ax5.axvline(100, color="grey", lw=1, linestyle=":")
    ax5.set_xlabel("Spot Price S")
    ax5.set_ylabel("Gamma")
    ax5.set_title("Gamma: Autograd vs Analytical")
    ax5.legend()

    ax6 = fig.add_subplot(gs[2, :2])
    if history_path.exists():
        with open(history_path) as f:
            hist = json.load(f)
        epochs = range(1, len(hist["train"]) + 1)
        ax6.plot(epochs, hist["train"], label="Train", lw=1.5)
        ax6.plot(epochs, hist["val"],   label="Val",   lw=1.5, linestyle="--")
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("Loss")
        ax6.set_title("Training History")
        ax6.set_yscale("log")
        ax6.legend()

    ax7 = fig.add_subplot(gs[2, 2])
    ax7.plot(g["S"], np.abs(g["nn_delta"] - g["bs_delta"]),        lw=2, label="|Delta error|")
    ax7.plot(g["S"], np.abs(g["nn_gamma"] - g["bs_gamma"]) * 10,   lw=2, label="|Gamma error| x10", linestyle="--")
    ax7.set_xlabel("Spot Price S")
    ax7.set_ylabel("Absolute Error")
    ax7.set_title("Greek Errors vs Black-Scholes")
    ax7.legend()

    plt.suptitle("NeuralPricer — Evaluation Dashboard", fontsize=15, fontweight="bold", y=1.01)

    out_path = Path(save_dir) / "evaluation_dashboard.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to: {out_path}")
    plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      default="checkpoints/best_model.pt")
    p.add_argument("--data_path", default="data/heston_dataset.parquet")
    p.add_argument("--save_dir",  default="results")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, device)

    splits, _ = load_dataset(args.data_path)
    X_test, y_test = splits["test"]

    import pandas as pd
    df_full = pd.read_parquet(args.data_path)
    df_test = df_full.iloc[int(0.90 * len(df_full)):].reset_index(drop=True)

    C_true, C_pred, labels = evaluate_price_accuracy(model, X_test, y_test, df_test, device)

    print("\nComputing Greeks...")
    greeks = evaluate_greeks(model, device)

    print(f"\nMean |delta error| : {np.abs(greeks['nn_delta'] - greeks['bs_delta']).mean():.4f}")
    print(f"Mean |gamma error| : {np.abs(greeks['nn_gamma'] - greeks['bs_gamma']).mean():.6f}")

    plot_all(C_true, C_pred, labels, greeks, Path("checkpoints/history.json"), args.save_dir)
