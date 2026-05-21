import torch
import torch.nn as nn
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, width, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class NeuralPricer(nn.Module):
    def __init__(self, n_features=8, width=256, depth=4, dropout=0.05):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, width),
            nn.BatchNorm1d(width),
            nn.SiLU(),
        )

        self.blocks = nn.Sequential(
            *[ResBlock(width, dropout) for _ in range(depth)]
        )

        self.output_head = nn.Sequential(
            nn.Linear(width, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

        self.register_buffer("x_mean", torch.zeros(n_features))
        self.register_buffer("x_std",  torch.ones(n_features))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def set_normalisation(self, x_mean, x_std):
        self.x_mean.copy_(torch.tensor(x_mean, dtype=torch.float32))
        self.x_std.copy_(torch.tensor(x_std,   dtype=torch.float32))

    def normalise(self, x):
        return (x - self.x_mean) / (self.x_std + 1e-8)

    def forward(self, x):
        x   = self.normalise(x)
        h   = self.input_proj(x)
        h   = self.blocks(h)
        out = self.output_head(h).squeeze(-1)
        return out

    def predict_with_greeks(self, x, S):
        S = S.requires_grad_(True)

        K     = S * torch.exp(-x[:, 0])
        log_m = torch.log(S / K)

        x_diff = torch.cat([log_m.unsqueeze(1), x[:, 1:]], dim=1)

        price_norm = self(x_diff)
        price      = price_norm * S

        delta, = torch.autograd.grad(price.sum(), S, create_graph=True, retain_graph=True)
        gamma, = torch.autograd.grad(delta.sum(), S, retain_graph=False)

        return {
            "price" : price.detach(),
            "delta" : delta.detach(),
            "gamma" : gamma.detach(),
        }

    def predict_vega(self, x, S, v0_idx=3):
        x = x.clone()
        x[:, v0_idx] = x[:, v0_idx].requires_grad_(True)
        v0 = x[:, v0_idx]

        price_norm = self(x)
        price      = price_norm * S

        vega, = torch.autograd.grad(price.sum(), v0)
        return vega.detach()


if __name__ == "__main__":
    model = NeuralPricer(n_features=8, width=256, depth=4)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {total:,}")

    x_dummy = torch.randn(32, 8)
    out = model(x_dummy)
    print(f"Output shape : {out.shape}")
    print(f"Output range : [{out.min():.4f}, {out.max():.4f}]")
