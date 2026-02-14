import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from make_candles import plot_candlesticks

def returns_to_price_paths(last_close: float, future_returns: np.ndarray):
    S, H = future_returns.shape
    prices = np.zeros((S, H + 1), dtype=np.float64)
    prices[:, 0] = last_close
    for t in range(H):
        prices[:, t + 1] = prices[:, t] * np.exp(future_returns[:, t])
    return prices

def plot_history_and_future(df_ohlcv: pd.DataFrame,
                            hist_len: int,
                            future_prices: np.ndarray,
                            out_path: str,
                            title: str):
    d = df_ohlcv.tail(hist_len).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    plot_candlesticks(ax, d["Open"].values, d["High"].values, d["Low"].values, d["Close"].values)

    S, H1 = future_prices.shape
    future_x = np.arange(len(d) - 1, len(d) - 1 + H1)

    for i in range(min(S, 30)):
        ax.plot(future_x, future_prices[i], linewidth=1)

    q05 = np.quantile(future_prices, 0.05, axis=0)
    q95 = np.quantile(future_prices, 0.95, axis=0)
    mean = future_prices.mean(axis=0)

    ax.fill_between(future_x, q05, q95, alpha=0.2)
    ax.plot(future_x, mean, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Time (history → future horizon)")
    ax.set_ylabel("Index level")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
