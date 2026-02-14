import os
import numpy as np
import torch

from dataset import load_stooq_ohlcv_csv, compute_features, make_windows, walk_forward_split, scale_past_train_only
from train_cvae import train_cvae, sample_future_returns
from forecast_chart import returns_to_price_paths, plot_history_and_future

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    data_path = os.path.join(BASE_DIR, "data", "us100_ohlcv.csv")
    out_img = os.path.join(BASE_DIR, "outputs", "us100_forecast.png")
    model_out = os.path.join(BASE_DIR, "models", "cvae_us100.pt")

    lookback = 60
    horizon = 20
    train_ratio = 0.85
    epochs = 8
    hidden = 64
    z_dim = 16
    beta = 0.1
    n_scenarios = 200
    hist_plot_len = 120

    df = load_stooq_ohlcv_csv(data_path)
    feats = compute_features(df)
    X = feats.values.astype(np.float32)

    past, future = make_windows(X, lookback=lookback, horizon=horizon, target_col=0)
    (past_tr, fut_tr), (past_te, fut_te) = walk_forward_split(past, future, train_ratio=train_ratio)
    past_tr_s, past_te_s, _ = scale_past_train_only(past_tr, past_te)

    model = train_cvae(
        past_train=past_tr_s, future_train=fut_tr,
        past_val=past_te_s, future_val=fut_te,
        horizon=horizon, hidden=hidden, z_dim=z_dim,
        epochs=epochs, batch=128, lr=1e-3, beta=beta
    )

    torch.save({
        "state_dict": model.state_dict(),
        "in_feat": past_tr_s.shape[-1],
        "hidden": hidden,
        "z_dim": z_dim,
        "horizon": horizon,
        "lookback": lookback
    }, model_out)

    latest_past = past_te_s[-1:]
    latest_past_t = torch.tensor(latest_past, dtype=torch.float32)

    samples = sample_future_returns(model, latest_past_t, z_dim=z_dim, n_samples=n_scenarios)
    future_r = samples[:, 0, :, 0].cpu().numpy()

    last_close = float(df["Close"].iloc[-1])
    future_prices = returns_to_price_paths(last_close, future_r)

    plot_history_and_future(
        df_ohlcv=df,
        hist_len=hist_plot_len,
        future_prices=future_prices,
        out_path=out_img,
        title="US100 (Nasdaq-100 ^NDX) Candlestick + Generative AI Future Scenarios (Probabilistic)"
    )

    print(f"Saved chart: {out_img}")
    print(f"Saved model: {model_out}")

if __name__ == "__main__":
    main()
