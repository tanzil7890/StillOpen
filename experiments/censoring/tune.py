"""Tuning without leakage (PROTOCOL_B.md, section 4c): how should the number of boosting rounds be chosen when the
most recent cases have no outcome yet, and does tuning change how the treatments compare?

    python censoring/tune.py helpdesk          -> censoring/results/helpdesk.tuned.jsonl

For every setting, with the boosted trees and rounds chosen among 15, 30, 60, 100 and 150:

  each treatment, tuned      a random 20% of the cases in its own training rows is held out; fitted with 150 rounds on
                             the rest; the rounds with the lowest log loss on the held-out rows are chosen; fitted
                             again on all rows. Stored under the treatment's name.
  censor_aware_fixed         the window rule with the 60 rounds of version 1
  censor_aware_prospective   the window rule with the held-out part taken from the recent half of the training
                             period: fitted on what the window rule gives at the inner cutoff, checked on what it gives
                             at the cutoff for the cases that started after the inner cutoff

Read the results with:  python censoring/pooled.py --model tuned

    python censoring/tune.py helpdesk --wide   -> censoring/results/helpdesk.tuned_wide.jsonl

With --wide (section 4f) the learning rate (0.04, 0.08, 0.16) and the number of leaves (7, 31) are tuned as well, on
the same random hold-out. Stored next to the four treatments: censor_aware_rounds (rounds only, as above) and
censor_aware_fixed. Read with:  python censoring/pooled.py --model tuned_wide
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compete import clean, predict
from datasets import DATASETS
from study import DAY, RESULTS, TREATMENTS, Log, boot_diff, compose, labels, marginal, score, versions

GRID = (15, 30, 60, 100, 150)
WIDE = tuple((rate, leaves) for rate in (0.04, 0.08, 0.16) for leaves in (7, 31))


def trees(rounds, rate=0.08, leaves=31):
    return HistGradientBoostingClassifier(max_iter=rounds, learning_rate=rate, max_leaf_nodes=leaves, l2_regularization=1.0,
                                          early_stopping=False, categorical_features=[3, 4], random_state=0)


def rows(X, case, outcome, obs, H):
    """One row per (history, window at risk), with the case of each row."""
    xs, ys, cs = [], [], []
    for j in range(H):
        m = obs >= j
        xs.append(np.column_stack([X[m], np.full(m.sum(), j)]))
        ys.append(np.where((obs[m] == j) & (outcome[m] > 0), outcome[m], 0))
        cs.append(case[m])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)


def losses(fit, check, ncls, rate=0.08, leaves=31):
    """Log loss on the rows to check after each number of rounds in GRID; None if that cannot be measured."""
    (xf, yf), (xc, yc) = fit, check
    if len(xc) == 0 or len(np.unique(yf)) < 2:
        return None
    clf = trees(max(GRID), rate, leaves).fit(xf, yf)
    loss = {}
    for i, p in enumerate(clf.staged_predict_proba(xc), 1):
        if i in GRID:
            full = np.full((len(xc), ncls), 1e-12)
            full[:, clf.classes_] = np.clip(p, 1e-12, 1)
            loss[i] = float(-np.log(full[np.arange(len(xc)), yc]).mean())
    return loss


def choose(fit, check, ncls):
    """Rounds of GRID with the lowest log loss on the rows to check; None if that cannot be measured."""
    loss = losses(fit, check, ncls)
    return min(loss, key=loss.get) if loss else None


def choose_wide(fit, check, ncls):
    """-> (rounds, learning rate, leaves) with the lowest log loss, and the rounds chosen at the settings of
    version 1 (learning rate 0.08, 31 leaves); None if that cannot be measured."""
    table = {(rate, leaves): losses(fit, check, ncls, rate, leaves) for rate, leaves in WIDE}
    if table[0.08, 31] is None:
        return None
    best = min(((loss, rounds, rate, leaves) for (rate, leaves), by in table.items() for rounds, loss in by.items()))
    return best[1:], min(table[0.08, 31], key=table[0.08, 31].get)


def fit_all(x, y, Xte, H, ncls, rounds, without_features, rate=0.08, leaves=31):
    if len(np.unique(y)) < 2:  # no outcome at all in the training rows: outcome frequencies, as study.py does
        return without_features
    return compose(predict(trees(rounds, rate, leaves).fit(x, y), Xte, H, ncls))


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
        tp, t_term, cause, case, start = log.tp[tr], log.t_term[tr], log.cause[tr], log.cid[tr], log.start[tr]
        res, per, chosen = {}, {}, {}

        def keep_score(name, dist, rounds):
            if dist is None:
                res[name] = None
                return
            per[name], res[name] = score(dist, Y, n_ident, log.names)
            res[name]['rounds'] = rounds

        rng = np.random.default_rng(0)
        held = rng.permutation(np.unique(case))[:max(1, len(np.unique(case)) // 5)]   # the same cases for every treatment
        for treat in TREATMENTS:
            keep, outcome, obs = labels(tp, t_term, cause, tau, treat, edges)
            if keep.sum() == 0:
                res[treat] = None
                continue
            x, y, c = rows(Xtr[keep], case[keep], outcome[keep], obs[keep], H)
            plain = marginal(outcome[keep], obs[keep], len(Xte), H, ncls)
            out_of = np.isin(c, held)
            rounds = choose((x[~out_of], y[~out_of]), (x[out_of], y[out_of]), ncls) or 60
            keep_score(treat, fit_all(x, y, Xte, H, ncls, rounds, plain), rounds)
            if treat == 'censor_aware':
                keep_score('censor_aware_fixed', fit_all(x, y, Xte, H, ncls, 60, plain), 60)
                inner = tau - w * DAY / 2                             # the recent half of the training period is held out
                early = (start <= inner) & (tp <= inner)
                k1, o1, b1 = labels(tp[early], t_term[early], cause[early], inner, 'censor_aware', edges)
                late = start > inner
                k2, o2, b2 = labels(tp[late], t_term[late], cause[late], tau, 'censor_aware', edges)
                x1, y1, _ = rows(Xtr[early][k1], case[early][k1], o1[k1], b1[k1], H)
                x2, y2, _ = rows(Xtr[late][k2], case[late][k2], o2[k2], b2[k2], H)
                rounds = choose((x1, y1), (x2, y2), ncls) or 60
                keep_score('censor_aware_prospective', fit_all(x, y, Xte, H, ncls, rounds, plain), rounds)
        diff = {m: {'mean': round(float((per[m] - per['censor_aware']).mean()), 4),
                    'ci95': boot_diff(per[m], per['censor_aware'], log.cid[te], n_boot)}
                for m in per if m != 'censor_aware'} if 'censor_aware' in per else {}
        out.append({'log': log.name, 'cutoff': cutoff, 'model': 'tuned', 'versions': versions(), 'history_days': w,
                    'identifiable_deadlines': edges[:n_ident].tolist(), 'test_cases': int(len(np.unique(log.cid[te]))),
                    'brier_minus_censor_aware': diff, **res})
    return out


def run_wide(log, cutoff, history_days, n_boot=1000):
    """As run(), with the learning rate and the number of leaves tuned as well (section 4f)."""
    edges, H, ncls = log.edges, len(log.edges), len(log.names) + 1
    tau = datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc).timestamp()
    te = (log.start > tau) & (log.start <= tau + log.cfg['test_days'] * DAY) & (log.tp <= log.log_end - edges[-1] * DAY)
    Y = np.where(((log.t_term[te] - log.tp[te]) / DAY)[:, None] <= edges[None, :], log.cause[te][:, None], 0)
    out = []
    for w in history_days:
        tr = (log.start > tau - w * DAY) & (log.start <= tau) & (log.tp <= tau)
        n_ident = int((edges < w).sum())
        Xtr, Xte = clean(log.X[tr], log.X[te])
        tp, t_term, cause, case = log.tp[tr], log.t_term[tr], log.cause[tr], log.cid[tr]
        res, per = {}, {}

        def keep_score(name, dist, rounds, rate=0.08, leaves=31):
            per[name], res[name] = score(dist, Y, n_ident, log.names)
            res[name].update(rounds=rounds, rate=rate, leaves=leaves)

        rng = np.random.default_rng(0)
        held = rng.permutation(np.unique(case))[:max(1, len(np.unique(case)) // 5)]   # the same cases as in run()
        for treat in TREATMENTS:
            keep, outcome, obs = labels(tp, t_term, cause, tau, treat, edges)
            if keep.sum() == 0:
                res[treat] = None
                continue
            x, y, c = rows(Xtr[keep], case[keep], outcome[keep], obs[keep], H)
            plain = marginal(outcome[keep], obs[keep], len(Xte), H, ncls)
            out_of = np.isin(c, held)
            (rounds, rate, leaves), only_rounds = choose_wide((x[~out_of], y[~out_of]), (x[out_of], y[out_of]), ncls) or ((60, 0.08, 31), 60)
            keep_score(treat, fit_all(x, y, Xte, H, ncls, rounds, plain, rate, leaves), rounds, rate, leaves)
            if treat == 'censor_aware':
                keep_score('censor_aware_rounds', fit_all(x, y, Xte, H, ncls, only_rounds, plain), only_rounds)
                keep_score('censor_aware_fixed', fit_all(x, y, Xte, H, ncls, 60, plain), 60)
        diff = {m: {'mean': round(float((per[m] - per['censor_aware']).mean()), 4),
                    'ci95': boot_diff(per[m], per['censor_aware'], log.cid[te], n_boot)}
                for m in per if m != 'censor_aware'} if 'censor_aware' in per else {}
        out.append({'log': log.name, 'cutoff': cutoff, 'model': 'tuned_wide', 'versions': versions(), 'history_days': w,
                    'identifiable_deadlines': edges[:n_ident].tolist(), 'test_cases': int(len(np.unique(log.cid[te]))),
                    'brier_minus_censor_aware': diff, **res})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', choices=list(DATASETS))
    ap.add_argument('--cutoffs')
    ap.add_argument('--out')
    ap.add_argument('--wide', action='store_true')
    a = ap.parse_args()
    cfg, log = DATASETS[a.log], Log(a.log)
    cutoffs = a.cutoffs.split(',') if a.cutoffs else cfg['cutoffs'][::2] if a.log == 'bpi17' else cfg['cutoffs']
    shown = (*TREATMENTS, 'censor_aware_rounds', 'censor_aware_fixed') if a.wide else (*TREATMENTS, 'censor_aware_fixed', 'censor_aware_prospective')
    with open(a.out or os.path.join(RESULTS, a.log + ('.tuned_wide' if a.wide else '.tuned') + '.jsonl'), 'w') as f:
        for cutoff in cutoffs:
            for r in (run_wide if a.wide else run)(log, cutoff, cfg['history_days']):
                f.write(json.dumps(r) + '\n'); f.flush()
                print(f"{a.log} {cutoff} history {r['history_days']:>6.0f}d  " + '  '.join(
                    f"{k[:12]} {r[k]['brier_ident']:.4f} ({r[k]['rounds']}" + (f", {r[k]['rate']}, {r[k]['leaves']})" if a.wide else ')')
                    if r.get(k) else f"{k[:12]} -" for k in shown), flush=True)


if __name__ == '__main__':
    main()
