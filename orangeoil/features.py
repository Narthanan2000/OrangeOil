from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .ais import ships_daily
from .config import CFG
from .eia import load_eia_weekly, load_wti


def build_dataset(con):
    df = pd.DataFrame({"wti": load_wti()})
    log_price = np.log(df.wti)
    df["lr1"] = log_price.diff()
    df["lr5"] = log_price.diff(5)
    df["lr20"] = log_price.diff(20)
    df["vol10"] = df.lr1.rolling(10).std()
    df["dist_ma50"] = log_price - log_price.rolling(50).mean()

    weekly = load_eia_weekly()
    weekly = weekly.reindex(weekly.index.union(df.index)).ffill().reindex(df.index)
    for col in weekly.columns:
        rolling = weekly[col].rolling(250, min_periods=50)
        df[f"eia_{col}_chg4w"] = weekly[col].pct_change(20)
        df[f"eia_{col}_z"] = (weekly[col] - rolling.mean()) / rolling.std()

    firms = pd.read_sql("SELECT * FROM firms_daily", con, parse_dates=["date"])
    if len(firms):
        calendar = pd.date_range(firms.date.min(), df.index.max())
        pivot = firms.pivot(index="date", columns="region", values="frp_sum").reindex(calendar)
        smooth = np.log1p(pivot.rolling(7, min_periods=3).mean())
        for region in smooth.columns:
            df[f"frp_{region}"] = smooth[region].reindex(df.index)
            baseline = smooth[region].rolling(30, min_periods=10).mean()
            df[f"frp_{region}_chg"] = (smooth[region] - baseline).reindex(df.index)

    ships = ships_daily(con)
    if len(ships):
        ships = ships.rolling(3, min_periods=1).mean()
        for col in ships.columns:
            df[col] = ships[col].reindex(df.index)
    ais_cols = [c for c in df.columns if c.startswith("ais_")]
    df["ais_available"] = df[ais_cols].notna().any(axis=1).astype(float) if ais_cols else 0.0

    df["y"] = log_price.shift(-CFG["horizon"]) - log_price
    return df


def normalize(df, feats, mean, std):
    scaled = (df.reindex(columns=feats) - mean) / std
    return scaled.clip(-5, 5).fillna(0).values.astype(np.float32)


def make_splits(df, device):
    window, horizon = CFG["window"], CFG["horizon"]
    feats = [c for c in df.columns if c not in ("wti", "y")]
    feats = [c for c in feats if df[c].notna().sum() > 30]
    n = len(df)
    i_train, i_val = int(0.70 * n), int(0.85 * n)
    mean = df[feats].iloc[:i_train].mean()
    std = df[feats].iloc[:i_train].std().replace(0, 1)
    scaled = normalize(df, feats, mean, std)
    y_all = df["y"].values.astype(np.float32)
    y_std = float(np.nanstd(y_all[:i_train]))
    windows = np.lib.stride_tricks.sliding_window_view(scaled, window, axis=0).transpose(0, 2, 1)
    pad = np.zeros((window - 1, window, scaled.shape[1]), np.float32)
    x_all = np.concatenate([pad, windows])

    def indices(a, b):
        rows = range(max(a, window - 1), b - horizon)
        return np.array([t for t in rows if not np.isnan(y_all[t])], dtype=int)

    idx_train, idx_val, idx_test = indices(0, i_train), indices(i_train, i_val), indices(i_val, n)

    def tensor(array):
        return torch.tensor(np.ascontiguousarray(array), device=device)

    return SimpleNamespace(
        feats=feats,
        mean=mean,
        std=std,
        y_std=y_std,
        y_all=y_all,
        idx_test=idx_test,
        idx_train=idx_train,
        x_train=tensor(x_all[idx_train]),
        x_val=tensor(x_all[idx_val]),
        x_test=tensor(x_all[idx_test]),
        y_train=tensor(y_all[idx_train] / y_std),
        y_val=tensor(y_all[idx_val] / y_std),
        y_test=tensor(y_all[idx_test] / y_std),
        scaled=scaled,
    )
