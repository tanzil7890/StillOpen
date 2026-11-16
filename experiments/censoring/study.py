"""Does the handling of still-open cases matter? The pilot's experiment, on any log of datasets.py.

    python censoring/study.py traffic                 every cutoff and history length of that log
    python censoring/study.py bpi17 --cutoffs 2016-08-08 --history 28,91

A cutoff date stands in for "today". Training sees only events up to the cutoff, from cases that started in the last
so-many days. Test cases start after the cutoff and are scored against what really happened. The model is the same
for every treatment (boosted trees on a discrete-time hazard, one row per history and deadline window); the only
thing that differs is what is done with the cases still open at the cutoff:

  complete_only     drop them: train on finished cases, the usual practice
  open_as_negative  keep them and label them "no outcome in any window"
  mature_only       keep a history only if all of its windows had ended by the cutoff, open or not
  censor_aware      keep a (history, window) pair if that window had ended by the cutoff, open or not

A fifth treatment is a reference only and is not run unless asked for with --treatments:

  future_labels     keep every history and label it with what happened after the cutoff as well. No one has these
                    labels on the cutoff date; a train/test split by case start date that keeps open cases uses them.

One line per (cutoff, history length) goes to censoring/results/<log>.jsonl. Scores are multiclass Brier scores,
averaged over the deadlines shorter than the history length, which are the ones a training window can observe.

--model repeats the experiment with another kind of model (results in <log>.<model>.jsonl):
  trees     boosted trees, 60 rounds (the default)       logit  logistic regression, see models.py
  trees150  the same with 150 rounds                     gru    a small recurrent network, see models.py
"""
import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models
from datasets import DATASETS, derived

DAY = 86400.0
MAXK = 40  # histories of at most this many events
CATEGORIES = 255  # what the boosted trees accept in a categorical feature
TREATMENTS = ('complete_only', 'open_as_negative', 'mature_only', 'censor_aware')
REFERENCE = 'future_labels'
MODELS = ('trees', 'trees150', 'logit', 'gru')
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def versions():
    """What produced a result line. torch is listed only when a model has loaded it."""
    v = {'python': sys.version.split()[0], 'numpy': np.__version__, 'scikit-learn': sklearn.__version__}
    if 'torch' in sys.modules:
        v['torch'] = sys.modules['torch'].__version__
    return v


def build(cases, V):
    """One row per history (a case up to one of its events, before its outcome)."""
    cid, tp, X = [], [], []
    for i, c in enumerate(cases):
        n = min(c['term_idx'] if c['cause'] else len(c['acts']), MAXK)
        if n == 0:
            continue
        acts, ts = c['acts'][:n], c['ts'][:n]
        counts = np.cumsum(np.eye(V + 1)[acts], axis=0)[:, 1:]
        prev = np.concatenate([[0], acts[:-1]])
        gap = np.concatenate([[0.0], np.diff(ts)]) / 3600
        dt = [datetime.fromtimestamp(t, timezone.utc) for t in ts]
        # the two categorical features hold at most 255 values: activities numbered beyond 254 share the value 254
        X.append(np.column_stack([np.arange(1, n + 1), np.log1p((ts - ts[0]) / 3600), np.log1p(gap),
                                  np.minimum(acts, CATEGORIES - 1), np.minimum(prev, CATEGORIES - 1),
                                  np.full(n, np.log1p(c['amount'])), [d.hour for d in dt], [d.weekday() for d in dt], counts]))
        cid.append(np.full(n, i)); tp.append(ts)
    return np.concatenate(cid), np.concatenate(tp), np.concatenate(X)


def labels(tp, t_term, cause, tau, treatment, edges):
    """-> (keep mask, outcome, last window at risk), from what was visible at the cutoff tau."""
    H = len(edges)
    vis = t_term <= tau
    d = np.where(vis, (t_term - tp) / DAY, np.inf)
    ev_bin = np.searchsorted(edges, d, side='left')                  # H means "after the last window"
    n_full = np.searchsorted(edges, (tau - tp) / DAY, side='right')  # windows that had ended by tau
    in_h = vis & (ev_bin < H)
    outcome = np.where(in_h, cause, 0)
    if treatment == 'complete_only':
        keep, obs = vis, np.where(in_h, ev_bin, H - 1)
    elif treatment == 'open_as_negative':
        keep, obs = np.ones_like(vis), np.where(in_h, ev_bin, H - 1)
    elif treatment == 'mature_only':
        keep, obs = n_full == H, np.where(in_h, ev_bin, H - 1)
    elif treatment == REFERENCE:
        later = np.searchsorted(edges, (t_term - tp) / DAY, side='left')  # the outcome's window, seen by tau or not
        keep, outcome, obs = np.ones_like(vis), np.where(later < H, cause, 0), np.where(later < H, later, H - 1)
    else:
        # A window counts only if it had ended by tau. Counting an outcome seen part-way through a window, while
        # skipping the still-open cases in that same window, would over-count outcomes.
        seen = in_h & (ev_bin < n_full)
        outcome = np.where(seen, cause, 0)
        obs = np.where(seen, ev_bin, n_full - 1)
        keep = obs >= 0
    return keep, outcome, obs


def person_period(X, outcome, obs, H):
    xs, ys = [], []
    for j in range(H):
        m = obs >= j
        xs.append(np.column_stack([X[m], np.full(m.sum(), j)]))
        ys.append(np.where((obs[m] == j) & (outcome[m] > 0), outcome[m], 0))
    return np.concatenate(xs), np.concatenate(ys)


def compose(p):
    """Per-window hazards [n, H, classes] -> P(none, each outcome, by each deadline) [n, H, classes]."""
    surv = np.cumprod(p[:, :, 0], axis=1)
    before = np.concatenate([np.ones((len(p), 1)), surv[:, :-1]], axis=1)
    return np.concatenate([surv[:, :, None], np.cumsum(before[:, :, None] * p[:, :, 1:], axis=1)], axis=2)


def marginal(outcome, obs, n, H, ncls):
    """The same hazard estimate without features: outcome frequencies per window."""
    p = np.zeros((1, H, ncls))
    for j in range(H):
        at_risk = max((obs >= j).sum(), 1)
        for c in range(1, ncls):
            p[0, j, c] = ((obs == j) & (outcome == c)).sum() / at_risk
        p[0, j, 0] = 1 - p[0, j, 1:].sum()
    return np.repeat(compose(p), n, axis=0)


def fit_predict(Xtr, outcome, obs, Xte, H, ncls, model='trees', case_tr=None, case_te=None):
    """case_tr and case_te give the case of every row; only the recurrent network needs them."""
    empty = np.isnan(Xtr).all(axis=0)  # a feature this log does not have (no amount): sklearn cannot bin such a column
    if empty.any():
        Xtr, Xte = np.where(empty, 0.0, Xtr), np.where(empty, 0.0, Xte)
    xpp, ypp = person_period(Xtr, outcome, obs, H)
    if len(np.unique(ypp)) < 2:  # no outcome at all in the training rows: nothing to split on
        return marginal(outcome, obs, len(Xte), H, ncls)
    if model == 'logit':
        return compose(models.logit(Xtr, outcome, obs, Xte, H, ncls))
    if model == 'gru':
        return compose(models.gru(Xtr, outcome, obs, case_tr, Xte, case_te, H, ncls))
    # 60 rounds, no early stopping: sklearn's built-in early stopping splits the rows at random, so rows of one
    # case land on both sides and it stops far too late (checked in pilot/tree_check.py).
    clf = HistGradientBoostingClassifier(max_iter=150 if model == 'trees150' else 60, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                         early_stopping=False, categorical_features=[3, 4], random_state=0)
    clf.fit(xpp, ypp)
    p = np.zeros((len(Xte), H, ncls))
    for j in range(H):
        p[:, j, clf.classes_] = clf.predict_proba(np.column_stack([Xte, np.full(len(Xte), j)]))
    p[:, :, 0] += 1 - p.sum(-1)
    return compose(p)


def ece(p, y, bins=10):
    idx, tot = np.minimum((p * bins).astype(int), bins - 1), 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return tot


def score(dist, Y, n_ident, names):
    """n_ident = number of leading deadlines that the training window can observe."""
    H, ncls = dist.shape[1:]
    per_h = ((dist - np.eye(ncls)[Y]) ** 2).sum(-1)
    per_history = per_h[:, :n_ident].mean(-1)
    any_p, any_y = 1 - dist[:, :, 0], (Y > 0).astype(float)
    mid = H // 2
    return per_history, {
        'brier_ident': round(float(per_history.mean()), 4),
        'brier_all_deadlines': round(float(per_h.mean()), 4),
        'ece_any_ident': round(float(np.mean([ece(any_p[:, h], any_y[:, h]) for h in range(n_ident)])), 4),
        'auc_any_mid': round(float(roc_auc_score(any_y[:, mid], any_p[:, mid])), 3) if 0 < any_y[:, mid].mean() < 1 else None,
        'pred_minus_actual_last': {n: round(float(dist[:, -1, c].mean() - (Y[:, -1] == c).mean()), 3)
                                   for c, n in enumerate(names, 1)},
        'pred_minus_actual_any': [round(float(any_p[:, h].mean() - any_y[:, h].mean()), 3) for h in range(H)],
    }


def boot_diff(a, b, groups, n=1000, seed=0):
    """Mean of a - b over histories, with a 95% interval from resampling whole cases."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    sa, sb, cnt = (np.bincount(inv, w) for w in (a, b, np.ones_like(a)))
    d = []
    for _ in range(n):
        s = rng.integers(0, len(ug), len(ug))
        d.append((sa[s].sum() - sb[s].sum()) / cnt[s].sum())
    return [round(float(np.quantile(d, q)), 4) for q in (0.025, 0.975)]


class Log:
    """A parsed log with its history features and the arrays the experiment needs."""

    def __init__(self, name):
        self.name, self.cfg = name, DATASETS[name]
        with open(derived(name), 'rb') as f:
            data = pickle.load(f)
        cases = data['cases']
        self.cid, self.tp, self.X = build(cases, len(data['vocab']))
        self.start = np.array([c['ts'][0] for c in cases])[self.cid]
        self.t_term = np.array([c['ts'][c['term_idx']] if c['cause'] else np.inf for c in cases])[self.cid]
        self.cause = np.array([c['cause'] for c in cases])[self.cid]
        self.log_end = max(c['ts'][-1] for c in cases)
        self.edges = np.array(self.cfg['deadlines'], dtype=float)
        self.names = list(dict.fromkeys(self.cfg['outcomes'].values()))  # several activities may stand for one outcome

    def run(self, cutoff, history_days, treatments=TREATMENTS, n_boot=1000, model='trees'):
        """-> one result per history length, for the cutoff date 'YYYY-MM-DD'."""
        edges, H, ncls = self.edges, len(self.edges), len(self.names) + 1
        tau = datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc).timestamp()
        te = (self.start > tau) & (self.start <= tau + self.cfg['test_days'] * DAY) & (self.tp <= self.log_end - edges[-1] * DAY)
        d_te = (self.t_term[te] - self.tp[te]) / DAY
        Y = np.where(d_te[:, None] <= edges[None, :], self.cause[te][:, None], 0)
        head = {'log': self.name, 'cutoff': cutoff, 'test_histories': int(te.sum()), 'test_cases': int(len(np.unique(self.cid[te]))),
                'actual_any_by_deadline': np.round((Y > 0).mean(0), 3).tolist(),
                'actual_last_by_outcome': {n: round(float((Y[:, -1] == c).mean()), 3) for c, n in enumerate(self.names, 1)}}
        out = []
        for w in history_days:
            tr = (self.start > tau - w * DAY) & (self.start <= tau) & (self.tp <= tau)
            n_ident = int((edges < w).sum())
            res, per = {}, {}
            for treat in treatments:
                keep, outcome, obs = labels(self.tp[tr], self.t_term[tr], self.cause[tr], tau, treat, edges)
                if keep.sum() == 0:  # mature_only when the history is shorter than the longest deadline
                    res[treat] = None
                    continue
                t0 = time.time()
                dist = fit_predict(self.X[tr][keep], outcome[keep], obs[keep], self.X[te], H, ncls, model,
                                   self.cid[tr][keep], self.cid[te])
                per[treat], res[treat] = score(dist, Y, n_ident, self.names)
                res[treat].update(train_histories=int(keep.sum()), fit_s=round(time.time() - t0, 1))
                if treat == 'censor_aware':
                    _, res['no_features_censor_aware'] = score(marginal(outcome[keep], obs[keep], int(te.sum()), H, ncls), Y, n_ident, self.names)
            diff = {t: {'mean': round(float((per[t] - per['censor_aware']).mean()), 4),
                        'ci95': boot_diff(per[t], per['censor_aware'], self.cid[te], n_boot)}
                    for t in per if t != 'censor_aware'} if 'censor_aware' in per else {}
            out.append({**head, 'model': model, 'versions': versions(), 'history_days': w,
                        'identifiable_deadlines': edges[:n_ident].tolist(),
                        'train_histories': int(tr.sum()), 'train_cases': int(len(np.unique(self.cid[tr]))),
                        'share_of_train_histories_from_open_cases': round(float((self.t_term[tr] > tau).mean()), 3),
                        'brier_minus_censor_aware': diff, **res})
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', choices=list(DATASETS))
    ap.add_argument('--cutoffs', help='comma-separated dates; default: all of datasets.py')
    ap.add_argument('--history', help='comma-separated days; default: all of datasets.py')
    ap.add_argument('--treatments', help='comma-separated; default: the four of the study')
    ap.add_argument('--model', choices=MODELS, default='trees')
    ap.add_argument('--out', help='default: censoring/results/<log>.jsonl, or <log>.<model>.jsonl for another model')
    a = ap.parse_args()
    cfg = DATASETS[a.log]
    cutoffs = a.cutoffs.split(',') if a.cutoffs else cfg['cutoffs']
    history = [float(h) for h in a.history.split(',')] if a.history else cfg['history_days']
    treatments = tuple(a.treatments.split(',')) if a.treatments else TREATMENTS
    log = Log(a.log)
    os.makedirs(RESULTS, exist_ok=True)
    name = a.log if a.model == 'trees' else f'{a.log}.{a.model}'
    with open(a.out or os.path.join(RESULTS, name + '.jsonl'), 'w') as f:
        for cutoff in cutoffs:
            for r in log.run(cutoff, history, treatments, model=a.model):
                f.write(json.dumps(r) + '\n'); f.flush()
                d = r['brier_minus_censor_aware']
                print(f"{a.log} {cutoff} history {r['history_days']:>6.0f}d  test cases {r['test_cases']:5d}  open {r['share_of_train_histories_from_open_cases']:.2f}  "
                      + '  '.join(f"{t} {r[t]['brier_ident']:.4f}" if r.get(t) else f"{t} -" for t in treatments)
                      + '  | ' + '  '.join(f"{t} {v['mean']:+.4f} {v['ci95']}" for t, v in d.items()), flush=True)


if __name__ == '__main__':
    main()
