import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class WindowDataset(Dataset):
    def __init__(self, past, future):
        self.past = torch.tensor(past, dtype=torch.float32)     # [N,L,F]
        self.future = torch.tensor(future, dtype=torch.float32)  # [N,H,4]

    def __len__(self):
        return self.past.shape[0]

    def __getitem__(self, idx):
        return self.past[idx], self.future[idx]


class Encoder(nn.Module):
    def __init__(self, in_feat, hidden, z_dim):
        super().__init__()
        self.past_rnn = nn.GRU(
            input_size=in_feat, hidden_size=hidden, batch_first=True)
        self.fut_rnn = nn.GRU(
            input_size=4,       hidden_size=hidden, batch_first=True)  # <-- OHLC
        self.fc = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, past, future):
        _, hp = self.past_rnn(past)
        _, hf = self.fut_rnn(future)
        h = torch.cat([hp[-1], hf[-1]], dim=-1)
        h = self.fc(h)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, in_feat, hidden, z_dim, horizon):
        super().__init__()
        self.horizon = horizon
        self.past_rnn = nn.GRU(
            input_size=in_feat, hidden_size=hidden, batch_first=True)
        self.fc_init = nn.Sequential(
            nn.Linear(hidden + z_dim, hidden), nn.ReLU())

        self.dec_rnn = nn.GRU(
            input_size=4, hidden_size=hidden, batch_first=True)  # <-- OHLC
        # <-- OHLC
        self.out = nn.Linear(hidden, 4)

    def forward(self, past, z, teacher_future=None):
        B = past.size(0)
        _, hp = self.past_rnn(past)
        h = self.fc_init(torch.cat([hp[-1], z], dim=-1)).unsqueeze(0)

        x_t = torch.zeros(B, 1, 4, device=past.device)  # <-- OHLC
        outs = []
        for t in range(self.horizon):
            y, h = self.dec_rnn(x_t, h)
            pred = self.out(y)               # [B,1,4]
            outs.append(pred)
            x_t = teacher_future[:, t:t+1,
                                 :] if teacher_future is not None else pred.detach()
        return torch.cat(outs, dim=1)        # [B,H,4]


class CVAE(nn.Module):
    def __init__(self, in_feat, hidden=64, z_dim=16, horizon=20):
        super().__init__()
        self.enc = Encoder(in_feat, hidden, z_dim)
        self.dec = Decoder(in_feat, hidden, z_dim, horizon)

    @staticmethod
    def reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, past, future):
        mu, logvar = self.enc(past, future)
        z = self.reparam(mu, logvar)
        recon = self.dec(past, z, teacher_future=future)
        return recon, mu, logvar


def kld(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


def train_cvae(
    past_train, future_train,
    past_val, future_val,
    horizon,
    hidden=64, z_dim=16,
    epochs=10, batch=256,
    lr=1e-3, beta=0.1, seed=7
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CVAE(
        in_feat=past_train.shape[-1], hidden=hidden, z_dim=z_dim, horizon=horizon).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    tr_ds = WindowDataset(past_train, future_train)
    va_ds = WindowDataset(past_val, future_val)
    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=False)
    va_dl = DataLoader(va_ds, batch_size=batch, shuffle=False)

    train_losses, val_losses = [], []

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for past, fut in tr_dl:
            past, fut = past.to(device), fut.to(device)
            recon, mu, logvar = model(past, fut)
            loss = mse(recon, fut) + beta * kld(mu, logvar)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += loss.item() * past.size(0)
        tr_loss /= len(tr_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for past, fut in va_dl:
                past, fut = past.to(device), fut.to(device)
                recon, mu, logvar = model(past, fut)
                loss = mse(recon, fut) + beta * kld(mu, logvar)
                va_loss += loss.item() * past.size(0)
        va_loss /= len(va_ds)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        print(f"epoch {ep:02d} | train={tr_loss:.6f} | val={va_loss:.6f}")

    return model, train_losses, val_losses


def sample_future_ohlc_deltas(model, past_window_batch, z_dim, n_samples=200):
    """
    Returns: [S,B,H,4] in SCALED delta-log OHLC space.
    """
    device = next(model.parameters()).device
    model.eval()
    past_window_batch = past_window_batch.to(device)
    B = past_window_batch.size(0)

    outs = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(B, z_dim, device=device)
            gen = model.dec(past_window_batch, z,
                            teacher_future=None)  # [B,H,4]
            outs.append(gen.unsqueeze(0))
    return torch.cat(outs, dim=0)
