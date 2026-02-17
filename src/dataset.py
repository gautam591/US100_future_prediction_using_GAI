import numpy as np
import pandas as pd

REQ_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def load_stooq_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip().capitalize() for c in df.columns]

    # Some datasets use lowercase or different naming; handle common cases
    ren = {}
    for c in df.columns:
        if c.lower() == "date":
            ren[c] = "Date"
        if c.lower() == "open":
            ren[c] = "Open"
        if c.lower() == "high":
            ren[c] = "High"
        if c.lower() == "low":
            ren[c] = "Low"
        if c.lower() == "close":
            ren[c] = "Close"
        if c.lower() == "volume":
            ren[c] = "Volume"
    df = df.rename(columns=ren)

    # If volume missing, create dummy
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # keep only needed cols
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

    # numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Past input features (you can extend these later):
    - log returns of Close
    - range (High-Low) normalized by Close
    - body (Close-Open) normalized by Close
    - log volume (stable)
    """
    out = pd.DataFrame()
    out["r_close"] = np.log(df["Close"]).diff().fillna(0.0)

    out["range_rel"] = ((df["High"] - df["Low"]) / df["Close"]
                        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["body_rel"] = ((df["Close"] - df["Open"]) / df["Close"]
                       ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    out["log_vol"] = np.log(df["Volume"].astype(float) + 1.0)
    out["log_vol"] = (
        out["log_vol"] - out["log_vol"].rolling(20, min_periods=1).mean()).fillna(0.0)

    return out


def make_windows(features: np.ndarray, ohlc: np.ndarray, lookback: int, horizon: int):
    """
    features: [T, F]
    ohlc:     [T, 4] in PRICE space (Open, High, Low, Close)
    Return:
      past:   [N, lookback, F]
      future: [N, horizon, 4] in DELTA-LOG space relative to last past close
    """
    T = features.shape[0]
    F = features.shape[1]
    assert ohlc.shape[1] == 4

    past_list, fut_list = [], []

    log_ohlc = np.log(np.maximum(ohlc, 1e-9))  # [T,4]
    log_close = np.log(np.maximum(ohlc[:, 3], 1e-9))

    for end in range(lookback, T - horizon):
        past_feats = features[end - lookback:end, :]  # [L,F]

        # anchor = last close of past window (log)
        anchor = log_close[end - 1]

        fut_log = log_ohlc[end:end + horizon, :]      # [H,4]
        fut_delta = fut_log - anchor                  # [H,4]  (stationary-ish)

        past_list.append(past_feats)
        fut_list.append(fut_delta)

    past = np.stack(past_list, axis=0).astype(np.float32)
    future = np.stack(fut_list, axis=0).astype(np.float32)
    return past, future


def walk_forward_split(past: np.ndarray, future: np.ndarray, train_ratio: float = 0.85):
    n = past.shape[0]
    cut = int(n * train_ratio)
    return (past[:cut], future[:cut]), (past[cut:], future[cut:])


def scale_train_only(past_tr: np.ndarray, past_te: np.ndarray, fut_tr: np.ndarray, fut_te: np.ndarray):
    """
    Standardize past features and future targets using TRAIN stats only.
    """
    # past scaling (per feature)
    mu_x = past_tr.reshape(-1, past_tr.shape[-1]).mean(axis=0)
    sd_x = past_tr.reshape(-1, past_tr.shape[-1]).std(axis=0) + 1e-8

    past_tr_s = (past_tr - mu_x) / sd_x
    past_te_s = (past_te - mu_x) / sd_x

    # future scaling (per OHLC dim) in delta-log space
    mu_y = fut_tr.reshape(-1, fut_tr.shape[-1]).mean(axis=0)
    sd_y = fut_tr.reshape(-1, fut_tr.shape[-1]).std(axis=0) + 1e-8

    fut_tr_s = (fut_tr - mu_y) / sd_y
    fut_te_s = (fut_te - mu_y) / sd_y

    scaler = {"mu_x": mu_x, "sd_x": sd_x, "mu_y": mu_y, "sd_y": sd_y}
    return past_tr_s, past_te_s, fut_tr_s, fut_te_s, scaler
