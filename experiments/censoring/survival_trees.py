"""Boosted survival trees as a competing method (PROTOCOL_B.md, section 4e).

    python censoring/survival_trees.py helpdesk     -> censoring/results/helpdesk.methods_s.jsonl

cox_trees is cause-specific Cox regression in which the linear predictor of each outcome is a sum of boosted
regression trees. It takes one row per training history: its features, the delay to its outcome and the kind of
outcome if one was seen by the cutoff (kind 0 otherwise), and how long it has been followed. It is scored against
the window rule fitted with boosted trees, which is stored here as censor_aware and equals that of <log>.jsonl.

Read the results with:  python censoring/pooled.py --methods
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compete import clean
from datasets import DATASETS
from study import DAY, RESULTS, Log, boot_diff, fit_predict, labels, score, versions

ROUNDS, RATE = 60, 0.08  # as the main model


def tie_groups(t):
    """For times in ascending order: the first and the last position of the rows that share each row's time."""
    _, first, inverse, counts = np.unique(t, return_index=True, return_inverse=True, return_counts=True)
    return first[inverse], (first + counts - 1)[inverse]


def cox_trees(Xtr, Xte, delay, kind, follow, deadlines, K, rounds=ROUNDS, rate=RATE, grid=200):
    """-> P(none, each outcome, by each deadline) for the test histories, [n, deadlines, classes]."""
    if not (kind > 0).any():
        return None
    time = np.where(kind > 0, delay, follow)
    order = np.argsort(time, kind='stable')
    X, t, k = Xtr[order], time[order], kind[order]
    first, last = tie_groups(t)
    f_tr, f_te, at_risk = np.zeros((len(t), K)), np.zeros((len(Xte), K)), np.ones((len(t), K))
    for c in range(K):
        d = (k == c + 1).astype(float)
        if d.sum() == 0:                                            # an outcome never seen: its hazard stays zero
            continue
        for _ in range(rounds):
            e = np.exp(f_tr[:, c])
            S = np.cumsum(e[::-1])[::-1][first]                     # summed exp(f) of the rows with time >= own
            A, B = np.cumsum(d / S)[last], np.cumsum(d / S ** 2)[last]
            g, h = d - e * A, np.maximum(e * A - e ** 2 * B, 0.0)   # gradient and curvature of the partial likelihood
            tree = HistGradientBoostingRegressor(max_iter=1, learning_rate=1.0, max_leaf_nodes=31, l2_regularization=1.0,
                                                 early_stopping=False, categorical_features=[3, 4], random_state=0)
            tree.fit(X, g / np.maximum(h, 1e-12), sample_weight=h)  # a leaf's value is sum(g) / (sum(h) + penalty)
            f_tr[:, c] = np.clip(f_tr[:, c] + rate * tree.predict(X), -20, 20)
            f_te[:, c] = np.clip(f_te[:, c] + rate * tree.predict(Xte), -20, 20)
        at_risk[:, c] = np.cumsum(np.exp(f_tr[:, c])[::-1])[::-1][first]
    times = np.unique(np.r_[np.linspace(deadlines[-1] / grid, deadlines[-1], grid), deadlines])
    base = np.zeros((len(times), K))                                # Breslow: cumulative baseline hazard at each grid time
    for c in range(K):
        m = k == c + 1
        if m.any():
            base[:, c] = np.r_[0.0, np.cumsum(1 / at_risk[m, c])][np.searchsorted(t[m], times, side='right')]
    step, at, out = np.diff(np.r_[np.zeros((1, K)), base], axis=0), np.searchsorted(times, deadlines), []
    for s in range(0, len(Xte), 20000):                             # cumulative incidence, in pieces to bound memory
        hazard = step[None] * np.exp(f_te[s:s + 20000])[:, None, :]  # [n, times, K]
        total = hazard.sum(-1)
        before = np.exp(-(np.cumsum(total, 1) - total))
        share = hazard / np.maximum(total, 1e-30)[..., None]
        out.append(np.cumsum((before * (1 - np.exp(-total)))[..., None] * share, 1)[:, at])
    by = np.concatenate(out)
    return np.concatenate([1 - by.sum(-1, keepdims=True), by], -1)


def run(log, cutoff, history_days, n_boot=1000):
    edges, H, ncls = log.edges, len(log.edges), len(log.names) + 1
    tau = datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc).timestamp()
    te = (log.start > tau) & (log.start <= tau + log.cfg['test_days'] * DAY) & (log.tp <= log.log_end - edges[-1] * DAY)
    Y = np.where(((log.t_term[te] - log.tp[te]) / DAY)[:, None] <= edges[None, :], log.cause[te][:, None], 0)
    out = []
    for w in history_days:
        tr = (log.start > tau - w * DAY) & (log.start <= tau) & (log.tp <= tau)
        n_ident = int((edges < w).sum())
        Xtr, Xte = clean(log.X[tr], log.X[te])
        seen = log.t_term[tr] <= tau
        delay = np.where(seen, (log.t_term[tr] - log.tp[tr]) / DAY, np.inf)
        kind, follow = np.where(seen, log.cause[tr], 0), (tau - log.tp[tr]) / DAY
        keep, outcome, obs = labels(log.tp[tr], log.t_term[tr], log.cause[tr], tau, 'censor_aware', edges)
        dists = {'censor_aware': fit_predict(log.X[tr][keep], outcome[keep], obs[keep], log.X[te], H, ncls),
                 'cox_trees': cox_trees(Xtr, Xte, delay, kind, follow, edges, ncls - 1)}
        res, per = {}, {}
        for name, dist in dists.items():
            if dist is None:
                res[name] = None
                continue
            per[name], res[name] = score(np.clip(dist, 0, 1), Y, n_ident, log.names)
        diff = {'cox_trees': {'mean': round(float((per['cox_trees'] - per['censor_aware']).mean()), 4),
                              'ci95': boot_diff(per['cox_trees'], per['censor_aware'], log.cid[te], n_boot)}} if len(per) == 2 else {}
        out.append({'log': log.name, 'cutoff': cutoff, 'model': 'trees', 'versions': versions(), 'history_days': w,
                    'identifiable_deadlines': edges[:n_ident].tolist(), 'test_cases': int(len(np.unique(log.cid[te]))),
                    'brier_minus_censor_aware': diff, **res})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', choices=list(DATASETS))
    ap.add_argument('--cutoffs')
    ap.add_argument('--out')
    a = ap.parse_args()
    cfg, log = DATASETS[a.log], Log(a.log)
    cutoffs = a.cutoffs.split(',') if a.cutoffs else cfg['cutoffs'][::2] if a.log == 'bpi17' else cfg['cutoffs']
    with open(a.out or os.path.join(RESULTS, a.log + '.methods_s.jsonl'), 'w') as f:
        for cutoff in cutoffs:
            for r in run(log, cutoff, cfg['history_days']):
                f.write(json.dumps(r) + '\n'); f.flush()
                d = r['brier_minus_censor_aware'].get('cox_trees')
                print(f"{a.log} {cutoff} history {r['history_days']:>6.0f}d  window {r['censor_aware']['brier_ident']:.4f}  "
                      + (f"cox_trees {r['cox_trees']['brier_ident']:.4f} ({d['mean']:+.4f})" if d else 'cox_trees -'), flush=True)


if __name__ == '__main__':
    main()
