import numpy as np
import pandas as pd
import torch

from .agent import DEVICE, OrangeOil, evaluate, train
from .config import CFG
from .features import build_dataset, make_splits, normalize
from .report import backtest, plot_correlations, plot_history, plot_importance


def set_seed(seed=None):
    seed = CFG["seed"] if seed is None else seed
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_train(con):
    set_seed()
    df = build_dataset(con)
    data = make_splits(df, DEVICE)
    print(f"{len(df)} lignes, {len(data.feats)} features")
    print(f"train {len(data.idx_train)} | val {len(data.y_val)} | test {len(data.y_test)}")
    model, history = train(data, seed=CFG["seed"])
    plot_history(history)
    plot_correlations(data)
    metrics = evaluate(model, data.x_test, data.y_test, data.y_std)
    print(f"MAE modèle {metrics['mae']:.4f} | MAE naïf {metrics['mae_naive']:.4f}")
    print(f"Précision directionnelle {metrics['dir_acc']:.1%} | IC {metrics['ic']:+.3f}")
    print(backtest(model, data, df))
    plot_importance(model, data)
    torch.save(
        {
            "state": model.state_dict(),
            "feats": data.feats,
            "mu": data.mean.to_dict(),
            "sd": data.std.to_dict(),
            "y_std": data.y_std,
        },
        CFG["model_path"],
    )
    print(f"Modèle sauvegardé : {CFG['model_path']}")
    return model


def run_forecast(con, n=2000):
    checkpoint = torch.load(CFG["model_path"], map_location=DEVICE)
    feats = checkpoint["feats"]
    df = build_dataset(con)
    scaled = normalize(df, feats, pd.Series(checkpoint["mu"]), pd.Series(checkpoint["sd"]))
    x = torch.tensor(scaled[-CFG["window"]:][None], device=DEVICE)
    model = OrangeOil(len(feats)).to(DEVICE)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    with torch.no_grad():
        mu, log_std, _ = model(x)
        dist = torch.distributions.Normal(mu, log_std.exp())
        samples = dist.sample((n,)).squeeze().cpu().numpy() * checkpoint["y_std"]
    last = float(df.wti.iloc[-1])
    prices = last * np.exp(samples)
    print(f"Dernier WTI ({df.index[-1]:%Y-%m-%d}) : {last:.2f} $")
    print(
        f"Prévision à {CFG['horizon']} jours ouvrés : {np.median(prices):.2f} $ "
        f"(80% : {np.percentile(prices, 10):.2f} - {np.percentile(prices, 90):.2f})"
    )
    print(f"P(hausse) = {(samples > 0).mean():.1%}")
