import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REWARD_W = {"err": 1.0, "dir": 1.0, "pnl": 1.0}


def reward_fn(action, target, weights=REWARD_W):
    error = -(action - target).abs()
    agree = (torch.sign(action) == torch.sign(target)).float() * 2 - 1
    direction = agree * target.abs().clamp(max=2.0)
    pnl = torch.tanh(action) * target
    return weights["err"] * error + weights["dir"] * direction + weights["pnl"] * pnl


class ForecastEnv:
    def __init__(self, x, y, seed=0):
        self.x = x
        self.y = y
        self.rng = np.random.default_rng(seed)
        self.index = None

    def reset(self, batch):
        self.index = torch.tensor(self.rng.integers(0, len(self.x), batch), device=self.x.device)
        return self.x[self.index]

    def step(self, action):
        return reward_fn(action, self.y[self.index])


class OrangeOil(nn.Module):
    def __init__(self, n_feat, hidden=64, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.body = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.Tanh())
        self.mu = nn.Linear(hidden, 1)
        self.log_std = nn.Linear(hidden, 1)
        self.value = nn.Linear(hidden, 1)
        nn.init.constant_(self.log_std.bias, -0.5)

    def forward(self, x):
        _, h = self.gru(x)
        h = self.body(h[-1])
        return (
            self.mu(h).squeeze(-1),
            self.log_std(h).squeeze(-1).clamp(-3, 1),
            self.value(h).squeeze(-1),
        )


@torch.no_grad()
def evaluate(model, x, y, y_std):
    model.eval()
    mu, _, _ = model(x)
    corr = 0.0
    if mu.std() > 0:
        corr = float(np.corrcoef(mu.cpu().numpy(), y.cpu().numpy())[0, 1])
    return {
        "reward": reward_fn(mu, y).mean().item(),
        "mae": ((mu - y).abs().mean() * y_std).item(),
        "mae_naive": (y.abs().mean() * y_std).item(),
        "dir_acc": (torch.sign(mu) == torch.sign(y)).float().mean().item(),
        "ic": corr,
    }


def train(data, n_iters=1500, batch=256, lr=1e-3, ent0=0.02, patience=20, log_every=25, seed=42):
    model = OrangeOil(len(data.feats)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    env = ForecastEnv(data.x_train, data.y_train, seed)
    history = []
    best = -1e9
    best_state = None
    bad = 0
    for it in range(1, n_iters + 1):
        model.train()
        obs = env.reset(batch)
        mu, log_std, value = model(obs)
        dist = torch.distributions.Normal(mu, log_std.exp())
        action = dist.sample()
        reward = env.step(action)
        advantage = reward - value.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        entropy_coef = ent0 * (1 - it / n_iters)
        loss = (
            -(dist.log_prob(action) * advantage).mean()
            + 0.5 * (value - reward).pow(2).mean()
            - entropy_coef * dist.entropy().mean()
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if it % log_every:
            continue
        metrics = evaluate(model, data.x_val, data.y_val, data.y_std)
        history.append({"it": it, "train_reward": reward.mean().item(), **metrics})
        if metrics["reward"] > best:
            best = metrics["reward"]
            bad = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if it % (log_every * 4) == 0:
            print(
                f"it {it:5d} | R_train {reward.mean():+.3f} | R_val {metrics['reward']:+.3f} "
                f"| dir {metrics['dir_acc']:.1%} | IC {metrics['ic']:+.3f}"
            )
        if bad >= patience:
            print("Early stopping.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
