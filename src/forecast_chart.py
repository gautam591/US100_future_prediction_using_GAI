import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from make_candles import plot_candlesticks


def enforce_ohlc_constraints(ohlc: np.ndarray) -> np.ndarray:
    """
    ohlc: [H,4] or [S,H,4] price space.
    Enforce: High >= max(Open,Close), Low <= min(Open,Close)
    """
    x = ohlc.copy()
    O = x[..., 0]
    H = x[..., 1]
    L = x[..., 2]
    C = x[..., 3]
    H = np.maximum(H, np.maximum(O, C))
    L = np.minimum(L, np.minimum(O, C))
    x[..., 1] = H
    x[..., 2] = L
    return x


def deltas_to_price_ohlc(deltas_scaled: np.ndarray, anchor_close: float, scaler: dict) -> np.ndarray:
    """
    deltas_scaled: [S,H,4] scaled delta-log OHLC
    anchor_close:  last historical close (price)
    scaler: contains mu_y, sd_y used to unscale deltas
    Returns: [S,H,4] OHLC in price space
    """
    mu_y = scaler["mu_y"]
    sd_y = scaler["sd_y"]

    deltas = deltas_scaled * sd_y + mu_y                 # unscale to delta-log
    anchor_log = np.log(max(anchor_close, 1e-9))
    log_ohlc = deltas + anchor_log                        # [S,H,4]
    ohlc = np.exp(log_ohlc)                               # price space
    ohlc = enforce_ohlc_constraints(ohlc)
    return ohlc


def _build_all_dates(history_dates: pd.Series, horizon: int) -> pd.Series:
    history_dates = pd.to_datetime(history_dates)
    last_day = history_dates.iloc[-1]
    future_dates = pd.bdate_range(
        last_day, periods=horizon + 1, freq="B")  # includes last_day
    all_dates = pd.concat(
        [history_dates, pd.Series(future_dates[1:])], ignore_index=True)
    return all_dates


def _set_date_ticks(ax, all_dates: pd.Series, max_ticks: int = 9):
    n = len(all_dates)
    idx = np.linspace(0, n - 1, max_ticks).astype(int)
    ax.set_xticks(idx)
    ax.set_xticklabels([all_dates.iloc[i].strftime("%Y-%m-%d")
                       for i in idx], rotation=30, ha="right")


def plot_history_plus_future_candles(df_ohlcv: pd.DataFrame, hist_len: int, future_ohlc_paths: np.ndarray, out_path: str, title: str):
    """
    future_ohlc_paths: [S,H,4] in price space
    We plot:
      - history candles
      - future MEAN candles
      - optional: a few scenario close lines faintly
    """
    d = df_ohlcv.tail(hist_len).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 6))

    # history candles
    plot_candlesticks(ax, d["Open"].values, d["High"].values,
                      d["Low"].values, d["Close"].values, width=0.9)

    S, H, _ = future_ohlc_paths.shape

    # future mean candle path
    # plot several generated future candle scenarios
    n_show = 5   # try 3–10
    for i in range(n_show):
        plot_candlesticks(
            ax,
            future_ohlc_paths[i, :, 0],
            future_ohlc_paths[i, :, 1],
            future_ohlc_paths[i, :, 2],
            future_ohlc_paths[i, :, 3],
            width=0.9,
            x_offset=hist_len
        )

    # OPTIONAL: show a few scenario CLOSE lines lightly (not required)

    # date labels
    all_dates = _build_all_dates(d["Date"], H)
    _set_date_ticks(ax, all_dates)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Index level")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-1, hist_len + H + 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_loss_curve(train_losses, val_losses, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_title("Training / Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
