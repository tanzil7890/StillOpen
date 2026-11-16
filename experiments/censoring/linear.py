"""Three competing methods that are linear in the features of the logistic-regression model (PROTOCOL_B.md, 4b).

Each takes one row per training history: its features, the delay to its outcome and the kind of outcome if one was
seen by the cutoff (kind 0 otherwise), and how long it has been followed. Each returns
P(none, each outcome, by each deadline) for the test histories, as [n, deadlines, classes].

  dfm   delayed-feedback model: a history ends in outcome k with probability p_k(x), or in none; given outcome k
        the delay is exponential with rate r_k(x)
  cox   cause-specific Cox regression with the Breslow baseline hazard; cumulative incidence by each deadline
  pu    positive-unlabelled learning per deadline and outcome, non-negative risk estimator

All use the penalty of the logistic regression (C = 1) and a fixed number of optimisation steps.
"""
import numpy as np
import torch

from models import ACT, COUNTS, NUM, PREV, _scale

torch.set_num_threads(4)
MIN_DELAY = 0.01  # days; logs with day stamps have delays of exactly zero


def design(Xtr, Xte):
    """The features of models.logit, without the window: scaled numbers and counts, the last two activities."""
    V = Xtr.shape[1] - COUNTS
    num, cnt = _scale(Xtr, Xte, NUM, True), _scale(Xtr, Xte, slice(COUNTS, None), False)

    def onehot(values):
        out = np.zeros((len(values), V + 1), dtype=np.float32)
        out[np.arange(len(values)), np.clip(values.astype(int), 0, V)] = 1
        return out

    return [torch.from_numpy(np.hstack([num[i], cnt[i], onehot(X[:, ACT]), onehot(X[:, PREV])]).astype(np.float32))
            for i, X in enumerate((Xtr, Xte))]


def minimise(loss, params, steps=200):
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=steps, history_size=20, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        value = loss()
        value.backward()
        return value

    opt.step(closure)


def dfm(Ftr, Fte, delay, kind, follow, deadlines, K):
    seen = torch.from_numpy(kind > 0)
    if not seen.any():
        return None
    n, d = Ftr.shape
    t = torch.from_numpy(np.maximum(np.where(kind > 0, delay, 0.0), MIN_DELAY)).float()
    u = torch.from_numpy(np.maximum(follow, 0.0)).float()
    k = torch.from_numpy(np.maximum(kind - 1, 0)).long()
    Wp, bp = torch.zeros(d, K, requires_grad=True), torch.zeros(K, requires_grad=True)
    Wr = torch.zeros(d, K, requires_grad=True)
    br = torch.full((K,), float(-np.log(max(float(t[seen].mean()), MIN_DELAY))), requires_grad=True)

    def parts(F):
        logp = torch.log_softmax(torch.cat([torch.zeros(len(F), 1), F @ Wp + bp], 1), 1)   # column 0: no outcome
        lograte = torch.clamp(F @ Wr + br, -15, 8)                                        # per day
        return logp, lograte

    def loss():
        logp, lograte = parts(Ftr)
        rate = lograte.exp()
        rows = torch.arange(n)
        with_outcome = logp[rows, k + 1] + lograte[rows, k] - rate[rows, k] * t
        without = torch.logsumexp(torch.cat([logp[:, :1], logp[:, 1:] - rate * u[:, None]], 1), 1)
        return -torch.where(seen, with_outcome, without).mean() + 0.5 * (Wp.square().sum() + Wr.square().sum()) / n

    minimise(loss, [Wp, bp, Wr, br])
    with torch.no_grad():
        logp, lograte = parts(Fte)
        by = logp[:, None, 1:].exp() * (1 - torch.exp(-lograte.exp()[:, None, :] * torch.tensor(deadlines).float()[None, :, None]))
        return torch.cat([1 - by.sum(-1, keepdim=True), by], -1).double().numpy()


def cox(Ftr, Fte, delay, kind, follow, deadlines, K, grid=200):
    if not (kind > 0).any():
        return None
    time = np.where(kind > 0, delay, follow)
    order = np.argsort(-time, kind='stable')                       # longest first: the risk set of a row is everything above it
    F, t, k = Ftr[torch.from_numpy(order)], time[order], kind[order]
    n, d = F.shape
    last_of_tie = np.r_[np.flatnonzero(t[1:] != t[:-1]), n - 1]    # rows with equal times share one risk set
    end = torch.from_numpy(last_of_tie[np.searchsorted(last_of_tie, np.arange(n))])
    is_k = [torch.from_numpy(k == c + 1) for c in range(K)]
    B = torch.zeros(d, K, requires_grad=True)

    def loss():
        eta = F @ B
        at_risk = torch.logcumsumexp(eta, 0)[end]                  # log of the summed exp(eta) over rows with time >= own
        ll = sum((eta[m, c] - at_risk[m, c]).sum() for c, m in enumerate(is_k))
        return -ll / n + 0.5 * B.square().sum() / n

    minimise(loss, [B])
    with torch.no_grad():
        at_risk = torch.logcumsumexp(F @ B, 0)[end].double().numpy()
        times = np.unique(np.r_[np.linspace(deadlines[-1] / grid, deadlines[-1], grid), deadlines])
        base = np.zeros((len(times), K))                           # Breslow: cumulative baseline hazard at each grid time
        for c in range(K):
            m = k == c + 1
            ts, inc = t[m][::-1], np.exp(-at_risk[m, c])[::-1]     # ascending times
            base[:, c] = np.r_[0.0, np.cumsum(inc)][np.searchsorted(ts, times, side='right')]
        step = torch.from_numpy(np.diff(np.r_[np.zeros((1, K)), base], axis=0)).float()   # [times, K]
        at = np.searchsorted(times, deadlines)
        out = []
        for s in range(0, len(Fte), 20000):                        # cumulative incidence, in pieces to bound memory
            hazard = step[None] * torch.clamp(Fte[s:s + 20000] @ B, -20, 20).exp()[:, None, :]   # [n, times, K]
            total = hazard.sum(-1)
            before = torch.exp(-(total.cumsum(1) - total))
            share = hazard / total.clamp_min(1e-30)[..., None]
            by = (before * (1 - torch.exp(-total)))[..., None] * share
            out.append(by.cumsum(1)[:, at])
        by = torch.cat(out).double().numpy()
        return np.concatenate([1 - by.sum(-1, keepdims=True), by], -1)


def pu(Ftr, Fte, delay, kind, follow, deadlines, K, steps=300):
    H = len(deadlines)
    fitted = [j for j in range(H) if (follow >= deadlines[j]).any()]
    if not fitted or not (kind > 0).any():
        return None
    n, d = Ftr.shape
    pos = np.stack([(kind == c + 1) & (delay <= deadlines[j]) for j in fitted for c in range(K)], 1)        # seen within the deadline
    prior = np.array([pos[follow >= deadlines[j], i * K + c].mean() for i, j in enumerate(fitted) for c in range(K)])
    usable = torch.from_numpy((pos.sum(0) > 0) & (prior > 0))
    P, pi = torch.from_numpy(pos).float(), torch.from_numpy(np.clip(prior, 1e-6, 1 - 1e-6)).float()
    W = torch.zeros(d, P.shape[1], requires_grad=True)
    b = torch.log(pi / (1 - pi)).clone().requires_grad_(True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    count = P.sum(0).clamp_min(1)
    for _ in range(steps):
        g = Ftr @ W + b
        plus = (torch.nn.functional.softplus(-g) * P).sum(0) / count      # positives, as positives
        minus = (torch.nn.functional.softplus(g) * P).sum(0) / count      # positives, as negatives
        unlabelled = torch.nn.functional.softplus(g).mean(0)              # every history, as negative
        risk = pi * plus + torch.clamp(unlabelled - pi * minus, min=0)
        loss = risk[usable].sum() + 0.5 * W.square().sum() / n
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.sigmoid(Fte @ W + b) * usable                           # a deadline and outcome never seen: zero
        p = p.view(len(Fte), len(fitted), K).double().numpy()
    p = p / np.maximum(p.sum(-1, keepdims=True), 1.0)                     # the outcomes exclude each other
    by = np.zeros((len(Fte), H, K))
    for j in range(H):                                                    # a deadline not fitted takes the last fitted one
        by[:, j] = p[:, max(i for i, f in enumerate(fitted) if f <= j)] if j >= fitted[0] else 0.0
    return np.concatenate([1 - by.sum(-1, keepdims=True), by], -1)
