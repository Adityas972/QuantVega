# QuantVega
# Options Neural Pricer

Neural network option pricing using the Heston stochastic volatility model, with Greeks via automatic differentiation.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Sanity check the pricers
python pricing_models.py

# Generate dataset (~20-40 min for 50k samples)
python data_generator.py

# Train
python train.py

# Evaluate
python evaluate.py
```

Quick test run:
```bash
python train.py --n_samples 5000 --epochs 30 --width 128 --depth 2
```

## File Structure

```
pricing_models.py    Black-Scholes and Heston analytical pricers
data_generator.py    Synthetic dataset generation
neural_pricer.py     Residual MLP + autograd Greeks
train.py             Training loop
evaluate.py          Price accuracy + Greek curves + plots
```

## Model

**Input:** `[log(S/K), T, r, v0, kappa, theta, sigma_v, rho]`

**Output:** `C/S` (normalised call price)

**Architecture:** Input projection → 4x ResBlock(256) → Softplus output

Greeks (delta, gamma, vega) computed via `torch.autograd.grad` — no finite differences.

## Args

| Flag | Default | Description |
|---|---|---|
| `--n_samples` | 50000 | Dataset size |
| `--width` | 256 | Hidden layer width |
| `--depth` | 4 | Number of residual blocks |
| `--epochs` | 150 | Max training epochs |
| `--batch_size` | 2048 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--patience` | 20 | Early stopping patience |
