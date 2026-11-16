"""Is the repo's brier_loss a proper objective under right-censoring?

Controlled test: every sample shares the SAME true distribution, so the best
possible forecast is one shared logits tensor. Fit it by minimising (i) the
repo's survival_nll and (ii) the repo's brier_loss, then compare the implied
P(any terminal event by horizon h) with the known truth.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
from temporal_rlcd.model import survival_nll, brier_loss, forecast_distribution

H, R, CAUSE_P = 4, 0.15, np.array([0.5, 0.3, 0.2])


def simulate(n, censor_frac, seed):
    rng = np.random.default_rng(seed)
    outcome, obs = np.zeros(n, dtype=np.int64), np.full(n, H - 1, dtype=np.int64)
    for i in range(n):
        t_out, t_bin = 0, H - 1
        for j in range(H):
            if rng.random() < R:
                t_out, t_bin = int(rng.choice([1, 2, 3], p=CAUSE_P)), j
                break
        follow = int(rng.integers(0, H)) if rng.random() < censor_frac else H - 1
        if t_out and t_bin <= follow:
            outcome[i], obs[i] = t_out, t_bin
        else:
            outcome[i], obs[i] = 0, follow
    return torch.from_numpy(outcome), torch.from_numpy(obs)


def fit(loss_fn, outcome, obs):
    logits = torch.zeros(1, H, 4, requires_grad=True)
    opt = torch.optim.LBFGS([logits], lr=0.5, max_iter=500, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = loss_fn(logits.expand(len(outcome), H, 4), outcome, obs)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return (1 - forecast_distribution(logits)[0, :, 0]).numpy()


truth = 1 - (1 - R) ** np.arange(1, H + 1)
print('true P(any event by bin h):      ', np.round(truth, 3))
for frac in (0.0, 0.3, 0.6, 0.9):
    o, k = simulate(200_000, frac, seed=1)
    nll, br = fit(survival_nll, o, k), fit(brier_loss, o, k)
    print(f'censor_frac={frac:.1f} | survival_nll fit {np.round(nll, 3)} | brier_loss fit {np.round(br, 3)}'
          f' | brier bias at last bin {br[-1] - truth[-1]:+.3f}')
