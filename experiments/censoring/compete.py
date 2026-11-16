"""Competing methods for training with open cases, under the protocol of study.py (PROTOCOL_B.md, section 4).

    python censoring/compete.py helpdesk          -> censoring/results/helpdesk.methods.jsonl

  ipcw    inverse probability of censoring weighting, per deadline. One classifier with the deadline as a feature.
          A (history, deadline) pair is used if its status at the deadline is known at the cutoff: the outcome was
          seen before the deadline, or the history was followed past it. Its weight is one over the share of
          training histories followed that long, capped at 20.
  pseudo  pseudo-labelling. A model trained on finished cases fills in the windows of open cases that had not ended
          with its own class probabilities; a second model is trained on everything.
  ipcw_x  the same weighting with the follow-up share estimated separately for histories taken early and late in a
          case (ten groups by time since the case began). Added after the first results of ipcw: a history taken
          late in a case has been followed for less time, so one pooled share is wrong for it.
          Run it with --methods ipcw_x; results go to <log>.methods_x.jsonl.

Both use the boosted trees of the main model and are scored exactly as the treatments of study.py. The window rule
is run alongside, so that every difference is paired on the same test histories.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS
from study import DAY, RESULTS, Log, boot_diff, compose, fit_predict, labels, score, versions

METHODS = ('ipcw', 'pseudo')   # the two fixed in the protocol before any was run
LATER = ('ipcw_x',)            # added afterwards, see PROTOCOL_B.md
ELAPSED = 1                    # column of study.build(): log hours since the case began
WEIGHT_CAP = 20.0


def trees():
    return HistGradientBoostingClassifier(max_iter=60, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                          early_stopping=False, categorical_features=[3, 4], random_state=0)


def clean(Xtr, Xte):
    empty = np.isnan(Xtr).all(axis=0)  # a feature this log does not have
    return (np.where(empty, 0.0, Xtr), np.where(empty, 0.0, Xte)) if empty.any() else (Xtr, Xte)


def predict(clf, X, H, ncls):
    """Class probabilities for every window index -> [n, H, classes]."""
    p = np.zeros((len(X), H, ncls))
    for j in range(H):
        p[:, j, clf.classes_] = clf.predict_proba(np.column_stack([X, np.full(len(X), j)]))
    p[:, :, 0] += 1 - p.sum(-1)
    return p


def ipcw(Xtr, tp, t_term, cause, tau, edges, Xte, ncls, groups=1):
    """-> P(none, each outcome, by each deadline) [n, H, classes], or None if it cannot be trained.
    groups > 1: the share followed is estimated within that many groups of time since the case began."""
    H = len(edges)
    u = (tau - tp) / DAY                                             # how long each history has been followed
    D = np.where(t_term <= tau, (t_term - tp) / DAY, np.inf)         # time to the outcome, if seen by the cutoff
    cuts = np.unique(np.quantile(Xtr[:, ELAPSED], np.linspace(0, 1, groups + 1))[1:-1]) if groups > 1 else []
    group = np.searchsorted(cuts, Xtr[:, ELAPSED], side='right') if len(cuts) else np.zeros(len(u), dtype=int)
    xs, ys, ws = [], [], []
    for g in np.unique(group):
        member = np.flatnonzero(group == g)
        followed = np.sort(u[member])

        def share_followed(t):                                       # share of these histories followed for at least t
            return 1 - np.searchsorted(followed, t, side='left') / len(followed)

        for j, d in enumerate(edges):
            if share_followed(d) <= 0:                               # none of them was followed past this deadline
                continue
            event = D[member] <= d
            known = event | (u[member] >= d)                         # status at the deadline is known
            idx = member[known]
            w = 1 / np.maximum(share_followed(np.where(event[known], D[idx], d)), 1e-12)
            xs.append(np.column_stack([Xtr[idx], np.full(len(idx), j)]))
            ys.append(np.where(event[known], cause[idx], 0))
            ws.append(np.minimum(w, WEIGHT_CAP))
    if not xs or len(np.unique(np.concatenate(ys))) < 2:
        return None
    clf = trees().fit(np.concatenate(xs), np.concatenate(ys), sample_weight=np.concatenate(ws))
    return predict(clf, Xte, H, ncls)                                # a direct forecast per deadline, not hazards


def pseudo(Xtr, tp, t_term, cause, tau, edges, Xte, ncls):
    """-> the same, from hazards as in study.py; None if no model can be trained on finished cases."""
    H = len(edges)
    keep, outcome, obs = labels(tp, t_term, cause, tau, 'complete_only', edges)
    xs, ys, ws = [], [], []
    for j in range(H):                                               # finished cases, as complete_only
        idx = np.flatnonzero(keep & (obs >= j))
        xs.append(np.column_stack([Xtr[idx], np.full(len(idx), j)]))
        ys.append(np.where((obs[idx] == j) & (outcome[idx] > 0), outcome[idx], 0))
        ws.append(np.ones(len(idx)))
    if len(np.unique(np.concatenate(ys))) < 2:
        return None
    first = trees().fit(np.concatenate(xs), np.concatenate(ys))
    still_open = np.flatnonzero(t_term > tau)
    if len(still_open):
        ended = np.searchsorted(edges, (tau - tp[still_open]) / DAY, side='right')  # windows that had ended
        guess = predict(first, Xtr[still_open], H, ncls)             # hazards the first model expects
        at_risk = np.ones(len(still_open))
        for j in range(H):
            known = ended > j                                        # the window ended and the case is open: no outcome
            xs.append(np.column_stack([Xtr[still_open[known]], np.full(known.sum(), j)]))
            ys.append(np.zeros(known.sum(), dtype=int)); ws.append(np.ones(known.sum()))
            for c in range(ncls):                                    # not ended: every class, weighted by its probability
                w = at_risk[~known] * guess[~known, j, c]
                xs.append(np.column_stack([Xtr[still_open[~known]], np.full((~known).sum(), j)]))
                ys.append(np.full((~known).sum(), c)); ws.append(w)
            at_risk = np.where(known, at_risk, at_risk * guess[:, j, 0])
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    second = trees().fit(x[w > 0], y[w > 0], sample_weight=w[w > 0])
    return compose(predict(second, Xte, H, ncls))


def run(log, cutoff, history_days, methods=METHODS, n_boot=1000):
    edges, H, ncls = log.edges, len(log.edges), len(log.names) + 1
    tau = datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc).timestamp()
    te = (log.start > tau) & (log.start <= tau + log.cfg['test_days'] * DAY) & (log.tp <= log.log_end - edges[-1] * DAY)
    Y = np.where(((log.t_term[te] - log.tp[te]) / DAY)[:, None] <= edges[None, :], log.cause[te][:, None], 0)
    out = []
    for w in history_days:
        tr = (log.start > tau - w * DAY) & (log.start <= tau) & (log.tp <= tau)
        n_ident = int((edges < w).sum())
        Xtr, Xte = clean(log.X[tr], log.X[te])
        args = (Xtr, log.tp[tr], log.t_term[tr], log.cause[tr], tau, edges, Xte, ncls)
        keep, outcome, obs = labels(log.tp[tr], log.t_term[tr], log.cause[tr], tau, 'censor_aware', edges)
        make = {'ipcw': lambda: ipcw(*args), 'pseudo': lambda: pseudo(*args), 'ipcw_x': lambda: ipcw(*args, groups=10)}
        dists = {'censor_aware': fit_predict(log.X[tr][keep], outcome[keep], obs[keep], log.X[te], H, ncls),
                 **{m: make[m]() for m in methods}}
        res, per = {}, {}
        for name, dist in dists.items():
            if dist is None:
                res[name] = None
                continue
            per[name], res[name] = score(dist, Y, n_ident, log.names)
        diff = {m: {'mean': round(float((per[m] - per['censor_aware']).mean()), 4),
                    'ci95': boot_diff(per[m], per['censor_aware'], log.cid[te], n_boot)} for m in methods if m in per}
        out.append({'log': log.name, 'cutoff': cutoff, 'model': 'trees', 'versions': versions(), 'history_days': w,
                    'identifiable_deadlines': edges[:n_ident].tolist(), 'test_cases': int(len(np.unique(log.cid[te]))),
                    'brier_minus_censor_aware': diff, **res})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', choices=list(DATASETS))
    ap.add_argument('--cutoffs')
    ap.add_argument('--methods', help='comma-separated; default: ipcw,pseudo')
    ap.add_argument('--out')
    a = ap.parse_args()
    cfg, log = DATASETS[a.log], Log(a.log)
    methods = tuple(a.methods.split(',')) if a.methods else METHODS
    name = a.log + ('.methods' if methods == METHODS else '.methods_x') + '.jsonl'
    with open(a.out or os.path.join(RESULTS, name), 'w') as f:
        for cutoff in (a.cutoffs.split(',') if a.cutoffs else cfg['cutoffs']):
            for r in run(log, cutoff, cfg['history_days'], methods):
                f.write(json.dumps(r) + '\n'); f.flush()
                d = r['brier_minus_censor_aware']
                print(f"{a.log} {cutoff} history {r['history_days']:>6.0f}d  window {r['censor_aware']['brier_ident']:.4f}  "
                      + '  '.join(f"{m} {r[m]['brier_ident']:.4f} ({d[m]['mean']:+.4f} {d[m]['ci95']})" if r.get(m) else f"{m} -" for m in methods), flush=True)


if __name__ == '__main__':
    main()
