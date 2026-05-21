import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pricing_models import heston_call_price_batch


PARAM_RANGES = {
    "S"       : (80.0,  120.0),
    "K"       : (70.0,  130.0),
    "T"       : (0.05,  2.0),
    "r"       : (0.01,  0.08),
    "v0"      : (0.01,  0.16),
    "kappa"   : (0.5,   5.0),
    "theta"   : (0.01,  0.16),
    "sigma_v" : (0.10,  0.80),
    "rho"     : (-0.90, -0.10),
}


def _sample_valid_params(n, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    while len(rows) < n:
        batch = 2 * n
        S       = rng.uniform(*PARAM_RANGES["S"],       size=batch)
        K       = rng.uniform(*PARAM_RANGES["K"],       size=batch)
        T       = rng.uniform(*PARAM_RANGES["T"],       size=batch)
        r       = rng.uniform(*PARAM_RANGES["r"],       size=batch)
        v0      = rng.uniform(*PARAM_RANGES["v0"],      size=batch)
        kappa   = rng.uniform(*PARAM_RANGES["kappa"],   size=batch)
        theta   = rng.uniform(*PARAM_RANGES["theta"],   size=batch)
        sigma_v = rng.uniform(*PARAM_RANGES["sigma_v"], size=batch)
        rho     = rng.uniform(*PARAM_RANGES["rho"],     size=batch)

        feller_ok    = 2 * kappa * theta > sigma_v**2
        moneyness    = S / K
        moneyness_ok = (0.7 < moneyness) & (moneyness < 1.3)

        mask  = feller_ok & moneyness_ok
        valid = np.column_stack([S, K, T, r, v0, kappa, theta, sigma_v, rho])[mask]
        rows.append(valid)

    all_rows = np.vstack(rows)
    return all_rows[:n]


def generate_dataset(n_samples=50_000, save_path="data/heston_dataset.parquet", chunk_size=500, seed=42):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {n_samples:,} valid Heston parameter sets...")
    params = _sample_valid_params(n_samples, seed=seed)

    print(f"Pricing {n_samples:,} options...")
    prices = np.zeros(n_samples)

    for i in tqdm(range(0, n_samples, chunk_size)):
        chunk = params[i : i + chunk_size]
        prices[i : i + chunk_size] = heston_call_price_batch(chunk)

    S, K = params[:, 0], params[:, 1]

    df = pd.DataFrame({
        "S"             : S,
        "K"             : K,
        "T"             : params[:, 2],
        "r"             : params[:, 3],
        "v0"            : params[:, 4],
        "kappa"         : params[:, 5],
        "theta"         : params[:, 6],
        "sigma_v"       : params[:, 7],
        "rho"           : params[:, 8],
        "log_moneyness" : np.log(S / K),
        "price_norm"    : prices / S,
        "price_raw"     : prices,
    })

    n_before = len(df)
    df = df.dropna().query("price_raw > 0").reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} invalid rows -> {len(df):,} clean samples")

    df.to_parquet(save_path, index=False)
    print(f"Saved to {save_path}")
    return df


def load_dataset(path="data/heston_dataset.parquet"):
    df = pd.read_parquet(path)

    feature_cols = ["log_moneyness", "T", "r", "v0", "kappa", "theta", "sigma_v", "rho"]
    target_col   = "price_norm"

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    n = len(X)
    idx = np.random.default_rng(42).permutation(n)
    train_end = int(0.80 * n)
    val_end   = int(0.90 * n)

    splits = {}
    for name, sl in [("train", slice(0, train_end)),
                     ("val",   slice(train_end, val_end)),
                     ("test",  slice(val_end, None))]:
        splits[name] = (X[idx[sl]], y[idx[sl]])

    print(f"train: {len(splits['train'][0]):,}  val: {len(splits['val'][0]):,}  test: {len(splits['test'][0]):,}")
    return splits, df[feature_cols].describe()


if __name__ == "__main__":
    df = generate_dataset(n_samples=50_000)
    print(df[["log_moneyness", "T", "v0", "rho", "price_norm"]].describe().round(4))
