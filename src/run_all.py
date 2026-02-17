import os
import numpy as np
import torch

from dataset import load_stooq_ohlcv_csv, compute_features, make_windows, walk_forward_split, scale_train_only
from train_cvae import train_cvae, sample_future_ohlc_deltas
from forecast_chart import deltas_to_price_ohlc, plot_history_plus_future_candles, plot_loss_curve

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dirs():
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)


def main():
    ensure_dirs()

    data_path = os.path.join(BASE_DIR, "data", "us100_ohlcv.csv")
    out_img = os.path.join(BASE_DIR, "outputs", "us100_future_candles.png")
    out_loss = os.path.join(BASE_DIR, "outputs", "loss_curve.png")
    model_out = os.path.join(BASE_DIR, "models", "cvae_us100_ohlc.pt")

    # Hyperparams
    lookback = 60
    horizon = 20
    train_ratio = 0.85
    epochs = 12
    hidden = 64
    z_dim = 16
    beta = 0.1
    n_scenarios = 200
    hist_plot_len = 120

    # Load data
    df = load_stooq_ohlcv_csv(data_path)

    feats_df = compute_features(df)
    X = feats_df.values.astype(np.float32)  # past features
    ohlc = df[["Open", "High", "Low", "Close"]].values.astype(np.float32)

    past, future = make_windows(X, ohlc, lookback=lookback, horizon=horizon)
    (past_tr, fut_tr), (past_te, fut_te) = walk_forward_split(
        past, future, train_ratio=train_ratio)

    past_tr_s, past_te_s, fut_tr_s, fut_te_s, scaler = scale_train_only(
        past_tr, past_te, fut_tr, fut_te)

    # Train
    model, tr_losses, va_losses = train_cvae(
        past_train=past_tr_s, future_train=fut_tr_s,
        past_val=past_te_s, future_val=fut_te_s,
        horizon=horizon, hidden=hidden, z_dim=z_dim,
        epochs=epochs, batch=256, lr=1e-3, beta=beta
    )

    torch.save({
        "state_dict": model.state_dict(),
        "in_feat": past_tr_s.shape[-1],
        "hidden": hidden,
        "z_dim": z_dim,
        "horizon": horizon,
        "lookback": lookback,
        "scaler": scaler
    }, model_out)

    # Sample future OHLC deltas from latest window
    anchor_close = float(df["Close"].iloc[-1])
    latest_past = past_te_s[-1]  # [L,F]
    past_t = torch.tensor(latest_past[None, ...], dtype=torch.float32)

    samples = sample_future_ohlc_deltas(
        model, past_t, z_dim=z_dim, n_samples=n_scenarios)  # [S,1,H,4]
    deltas_scaled = samples[:, 0, :, :].cpu().numpy(
    )                                        # [S,H,4]

    # Convert to price OHLC
    future_ohlc_paths = deltas_to_price_ohlc(
        deltas_scaled, anchor_close=anchor_close, scaler=scaler)  # [S,H,4]

    # Plot history + FUTURE CANDLES (mean candles)
    plot_history_plus_future_candles(
        df_ohlcv=df,
        hist_len=hist_plot_len,
        future_ohlc_paths=future_ohlc_paths,
        out_path=out_img,
        title="US100 (Nasdaq-100) Candlesticks + Generative AI Future Candlesticks (OHLC)"
    )

    plot_loss_curve(tr_losses, va_losses, out_loss)

    print("Saved:")
    print(" -", out_img)
    print(" -", out_loss)
    print("Saved model:", model_out)


if __name__ == "__main__":
    main()
