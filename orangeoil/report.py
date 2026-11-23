import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .agent import evaluate
from .config import CFG


def _path(name):
    os.makedirs(CFG["reports_dir"], exist_ok=True)
    return os.path.join(CFG["reports_dir"], name)


def plot_history(history):
    df = pd.DataFrame(history).set_index("it")
    df[["train_reward", "reward"]].plot(figsize=(9, 3), title="Reward train vs validation")
    plt.tight_layout()
    plt.savefig(_path("reward_history.png"), dpi=130)
    plt.close()


def plot_correlations(data):
    scaled = pd.DataFrame(data.scaled[data.idx_train], columns=data.feats)
    target = pd.Series(data.y_all[data.idx_train])
    corr = scaled.corrwith(target, method="spearman").dropna()
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(20)[::-1]
    colors = np.where(top > 0, "tab:green", "tab:red")
    top.plot.barh(figsize=(8, 6), color=colors, title="Top corrélations (Spearman) avec le rendement futur")
    plt.tight_layout()
    plt.savefig(_path("correlations.png"), dpi=130)
    plt.close()
    return top[::-1]


def backtest(model, data, df):
    model.eval()
    with torch.no_grad():
        mu = model(data.x_test)[0].cpu().numpy()
    next_ret = df["lr1"].shift(-1).iloc[data.idx_test].values
    position = np.tanh(mu)
    pnl = position * next_ret - 0.0002 * np.abs(np.diff(position, prepend=0))
    ok = ~np.isnan(pnl)
    index = df.index[data.idx_test][ok]
    strategy = pd.Series(pnl[ok], index=index)
    hold = pd.Series(next_ret[ok], index=index)

    def sharpe(series):
        return float(series.mean() / (series.std() + 1e-12) * np.sqrt(252))

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
    strategy.cumsum().plot(ax=axes[0], label="OrangeOil")
    hold.cumsum().plot(ax=axes[0], label="Buy & Hold")
    axes[0].legend()
    axes[0].set_title("Rendement cumulé (test)")
    actual = data.y_test.cpu().numpy() * data.y_std
    axes[1].scatter(actual, mu * data.y_std, s=6, alpha=0.5)
    axes[1].axline((0, 0), slope=1, c="r")
    axes[1].set_xlabel("réel")
    axes[1].set_ylabel("prévu")
    plt.tight_layout()
    plt.savefig(_path("backtest.png"), dpi=130)
    plt.close()
    return {"sharpe_orangeoil": sharpe(strategy), "sharpe_buy_hold": sharpe(hold)}


def plot_importance(model, data):
    base = evaluate(model, data.x_val, data.y_val, data.y_std)["reward"]
    scores = {}
    for j, name in enumerate(data.feats):
        shuffled = data.x_val.clone()
        order = torch.randperm(len(shuffled))
        shuffled[:, :, j] = shuffled[order, :, j]
        scores[name] = base - evaluate(model, shuffled, data.y_val, data.y_std)["reward"]
    top = pd.Series(scores).sort_values(ascending=False).head(20)
    top[::-1].plot.barh(figsize=(8, 6), title="Importance des features (delta reward)")
    plt.tight_layout()
    plt.savefig(_path("importance.png"), dpi=130)
    plt.close()
    return top
