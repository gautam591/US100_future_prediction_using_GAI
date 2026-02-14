import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_stooq_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().capitalize() for c in df.columns]
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {df.columns.tolist()}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close"]:
        df = df[df[col] > 0]
    df["Volume"] = df["Volume"].fillna(0)
    return df.reset_index(drop=True)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["log_close"] = np.log(d["Close"])
    d["r_close"] = d["log_close"].diff()

    eps = 1.0
    d["log_vol"] = np.log(d["Volume"].astype(float) + eps)
    d["r_vol"] = d["log_vol"].diff()

    d["hl_range"] = (d["High"] - d["Low"]) / d["Close"]
    return d[["r_close", "r_vol", "hl_range"]].dropna().reset_index(drop=True)

def make_windows(X: np.ndarray, lookback: int, horizon: int, target_col: int = 0):
    T, F = X.shape
    if T <= lookback + horizon:
        raise ValueError("Not enough data for lookback/horizon.")
    past, fut = [], []
    for t in range(lookback, T - horizon):
        past.append(X[t - lookback:t, :])
        fut.append(X[t:t + horizon, target_col:target_col + 1])
    return np.stack(past).astype(np.float32), np.stack(fut).astype(np.float32)

def walk_forward_split(past: np.ndarray, future: np.ndarray, train_ratio: float = 0.8):
    n = past.shape[0]
    cut = int(n * train_ratio)
    return (past[:cut], future[:cut]), (past[cut:], future[cut:])

def scale_past_train_only(past_train: np.ndarray, past_test: np.ndarray):
    scaler = StandardScaler()
    scaler.fit(past_train.reshape(-1, past_train.shape[-1]))
    tr = scaler.transform(past_train.reshape(-1, past_train.shape[-1])).reshape(past_train.shape).astype(np.float32)
    te = scaler.transform(past_test.reshape(-1, past_test.shape[-1])).reshape(past_test.shape).astype(np.float32)
    return tr, te, scaler
