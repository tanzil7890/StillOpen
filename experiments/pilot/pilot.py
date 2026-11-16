"""Pseudo-prospective pilot on BPI 2012: does the handling of still-running cases matter?

"Now" is a cutoff tau. Training sees only events <= tau from cases that started in the
last W weeks. Test = cases that start after tau, scored against what really happened
(the log continues, so truth is known for 30 days after every test prediction).

Same model class for every treatment (gradient-boosted discrete-time hazard), so the
only thing that differs is what happens to cases still open at tau:
  complete_only   drop them (standard practice: train on finished traces)
  open_as_negative keep them, label "no outcome in any window"
  censor_aware    keep them, but only for the windows that were fully observed
"""
import json, pickle, sys, time
from datetime import datetime, timezone
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DAY = 86400.0
EDGES = np.array([1.0, 3.0, 7.0, 14.0, 30.0])   # outcome within 1/3/7/14/30 days
H, MAXK, NCLS = len(EDGES), 40, 4


def build(cases, V):
    cid, kk, tp, X = [], [], [], []
    for i, c in enumerate(cases):
        n_pre = c['term_idx'] if c['cause'] else len(c['acts'])
        n = min(n_pre, MAXK)
        if n == 0:
            continue
        acts, ts = c['acts'][:n], c['ts'][:n]
        counts = np.cumsum(np.eye(V + 1)[acts], axis=0)[:, 1:]
        prev = np.concatenate([[0], acts[:-1]])
        gap = np.concatenate([[0.0], np.diff(ts)]) / 3600
        dt = [datetime.fromtimestamp(t, timezone.utc) for t in ts]
        f = np.column_stack([np.arange(1, n + 1), np.log1p((ts - ts[0]) / 3600), np.log1p(gap), acts, prev,
                             np.full(n, np.log1p(c['amount'])), [d.hour for d in dt], [d.weekday() for d in dt], counts])
        X.append(f); cid.append(np.full(n, i)); kk.append(np.arange(1, n + 1)); tp.append(ts)
    return np.concatenate(cid), np.concatenate(kk), np.concatenate(tp), np.concatenate(X)


def labels(tp, t_term, cause, tau, treatment):
    """-> (keep mask, outcome, observed_bin) using only information available at tau."""
    vis = t_term <= tau
    d = np.where(vis, (t_term - tp) / DAY, np.inf)
    ev_bin = np.searchsorted(EDGES, d, side='left')              # H means "after the last window"
    n_full = np.searchsorted(EDGES, (tau - tp) / DAY, side='right')   # windows fully observed by tau
    in_h = vis & (ev_bin < H)
    outcome = np.where(in_h, cause, 0)
    if treatment == 'complete_only':
        keep, obs = vis, np.where(in_h, ev_bin, H - 1)
    elif treatment == 'open_as_negative':
        keep, obs = np.ones_like(vis), np.where(in_h, ev_bin, H - 1)
    else:
        # tau is an administrative cutoff, so every prefix's follow-up is known. A window enters
        # the risk set only if it was observable in full; counting an outcome seen partway through
        # a window while skipping still-open cases in that same window over-counts outcomes.
        seen = in_h & (ev_bin < n_full)
        outcome = np.where(seen, cause, 0)
        obs = np.where(seen, ev_bin, n_full - 1)
        keep = obs >= 0
    return keep, outcome, obs


def person_period(X, outcome, obs):
    xs, ys = [], []
    for j in range(H):
        m = obs >= j
        xs.append(np.column_stack([X[m], np.full(m.sum(), j)]))
        ys.append(np.where((obs[m] == j) & (outcome[m] > 0), outcome[m], 0))
    return np.concatenate(xs), np.concatenate(ys)


def compose(p):
    """Per-window hazards [n,H,4] -> P(none, cause1..3 by each horizon) [n,H,4]."""
    surv = np.cumprod(p[:, :, 0], axis=1)
    before = np.concatenate([np.ones((len(p), 1)), surv[:, :-1]], axis=1)
    return np.concatenate([surv[:, :, None], np.cumsum(before[:, :, None] * p[:, :, 1:], axis=1)], axis=2)


def fit_predict(Xtr, outcome, obs, Xte):
    xpp, ypp = person_period(Xtr, outcome, obs)
    # 60 rounds: sklearn's built-in early stopping splits person-period rows at random, so rows of
    # one case land on both sides and it stops far too late (checked in tree_check.py).
    clf = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                         early_stopping=False, categorical_features=[3, 4], random_state=0)
    clf.fit(xpp, ypp)
    p = np.zeros((len(Xte), H, NCLS))
    for j in range(H):
        p[:, j, clf.classes_] = clf.predict_proba(np.column_stack([Xte, np.full(len(Xte), j)]))
    p[:, :, 0] += 1 - p.sum(-1)
    return compose(p)


def marginal(outcome, obs, n):
    p = np.zeros((1, H, NCLS))
    for j in range(H):
        at_risk = max((obs >= j).sum(), 1)
        for c in (1, 2, 3):
            p[0, j, c] = ((obs == j) & (outcome == c)).sum() / at_risk
        p[0, j, 0] = 1 - p[0, j, 1:].sum()
    return np.repeat(compose(p), n, axis=0)


def ece(p, y, bins=10):
    idx, tot = np.minimum((p * bins).astype(int), bins - 1), 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return tot


def score(dist, Y, n_ident=H):
    """n_ident = number of leading horizons the training window can actually observe."""
    onehot = np.eye(NCLS)[Y]
    per_h = ((dist - onehot) ** 2).sum(-1)
    per_prefix = per_h[:, :n_ident].mean(-1)
    any_p, any_y = 1 - dist[:, :, 0], (Y > 0).astype(float)
    return per_prefix, {
        'brier_ident': round(float(per_prefix.mean()), 4),
        'brier_all_horizons': round(float(per_h.mean()), 4),
        'ece_any_ident': round(float(np.mean([ece(any_p[:, h], any_y[:, h]) for h in range(n_ident)])), 4),
        'auc_any_7d': round(float(roc_auc_score(any_y[:, 2], any_p[:, 2])), 3),
        'pred_minus_actual_30d': {n: round(float(dist[:, -1, c].mean() - (Y[:, -1] == c).mean()), 3)
                                  for c, n in ((1, 'approved'), (2, 'declined'), (3, 'cancelled'))},
        'pred_minus_actual_any': [round(float(any_p[:, h].mean() - any_y[:, h].mean()), 3) for h in range(H)],
    }


def boot_diff(a, b, groups, n=400, seed=0):
    """Case-cluster bootstrap CI for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    sa, sb, cnt = (np.bincount(inv, w) for w in (a, b, np.ones_like(a)))
    d = []
    for _ in range(n):
        s = rng.integers(0, len(ug), len(ug))
        d.append((sa[s].sum() - sb[s].sum()) / cnt[s].sum())
    return [round(float(np.quantile(d, q)), 4) for q in (0.025, 0.975)]


def main():
    data = pickle.load(open(sys.argv[1], 'rb'))
    tau = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc).timestamp()
    weeks = [float(w) for w in sys.argv[3].split(',')]
    cases, V = data['cases'], len(data['vocab'])
    cid, kk, tp, X = build(cases, V)
    start = np.array([c['ts'][0] for c in cases])[cid]
    t_term = np.array([c['ts'][c['term_idx']] if c['cause'] else np.inf for c in cases])[cid]
    cause = np.array([c['cause'] for c in cases])[cid]
    log_end = max(c['ts'][-1] for c in cases)

    te = (start > tau) & (start <= tau + 21 * DAY) & (tp <= log_end - 30 * DAY)
    d_te = (t_term[te] - tp[te]) / DAY
    Y = np.where(d_te[:, None] <= EDGES[None, :], cause[te][:, None], 0)
    print(json.dumps({'tau': sys.argv[2], 'test_prefixes': int(te.sum()), 'test_cases': int(len(np.unique(cid[te]))),
                      'actual_any_by_horizon': np.round((Y > 0).mean(0), 3).tolist(),
                      'actual_30d_by_cause': [round(float((Y[:, -1] == c).mean()), 3) for c in (1, 2, 3)]}))
    for w in weeks:
        tr = (start > tau - w * 7 * DAY) & (start <= tau) & (tp <= tau)
        open_frac = float((t_term[tr] > tau).mean())
        n_ident = int((EDGES < w * 7).sum())
        res, per = {}, {}
        for treat in ('complete_only', 'open_as_negative', 'censor_aware'):
            keep, out, obs = labels(tp[tr], t_term[tr], cause[tr], tau, treat)
            t0 = time.time()
            dist = fit_predict(X[tr][keep], out[keep], obs[keep], X[te])
            per[treat], res[treat] = score(dist, Y, n_ident)
            res[treat]['fit_s'] = round(time.time() - t0, 1)
            if treat == 'censor_aware':
                _, res['no_features_censor_aware'] = score(marginal(out[keep], obs[keep], int(te.sum())), Y, n_ident)
        ci = {t: boot_diff(per[t], per['censor_aware'], cid[te]) for t in ('complete_only', 'open_as_negative')}
        print(json.dumps({'train_weeks': w, 'identifiable_horizons_days': EDGES[:n_ident].tolist(), 'train_prefixes': int(tr.sum()),
                          'train_cases': int(len(np.unique(cid[tr]))), 'frac_prefixes_from_open_cases': round(open_frac, 3),
                          'brier_minus_censor_aware_95ci': ci, **res}))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
