"""Why inverse probability of censoring weighting goes wrong on the traffic fines, in two tables.

    python censoring/ipcw_diagnosis.py

Found on 2026-11-15, after the first results of compete.py (PROTOCOL_B.md, section 4a). Not fixed in advance.

1. Without features, per position of the history in its case: a history taken late in a case has been followed for
   less time, so the weight must come from histories at the same position. With one pooled follow-up share the
   estimate is off; with the share of its own position it matches the truth, as the window rule does.
2. With features: the fine amount changed over the years, so it tells recent fines from old ones. Among recent
   fines only those already paid have a known status at a deadline of a year, and the weighted classifier, which
   can isolate them by their amount, predicts payment. Without the amount most of the error goes. The window rule
   has no rows for windows that have not ended, so it has nothing wrong to learn from recent fines.
"""
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compete
from study import DAY, Log, fit_predict, labels, score

AMOUNT, POSITION = 5, 0  # columns of study.build()


def by_position(log, tau, history, deadline=365.0):
    tr = (log.start > tau - history * DAY) & (log.start <= tau) & (log.tp <= tau)
    tp, tt, cause, pos = log.tp[tr], log.t_term[tr], log.cause[tr], log.X[tr][:, POSITION]
    u, true_d = (tau - tp) / DAY, (tt - tp) / DAY                    # true_d uses the whole log
    seen_d = np.where(tt <= tau, true_d, np.inf)
    print(f"  {'position':>8s} {'histories':>9s} {'truth':>6s} {'window rule':>11s} {'ipcw, pooled share':>18s} {'ipcw, own position':>18s}")
    for k in (1, 2, 3, 4):
        m = pos == k

        def weighted(follow):
            f = np.sort(follow)
            event = seen_d[m] <= deadline
            known = event | (u[m] >= deadline)
            share = 1 - np.searchsorted(f, np.where(event[known], seen_d[m][known], deadline), side='left') / len(f)
            w = np.minimum(1 / np.maximum(share, 1e-12), compete.WEIGHT_CAP)
            return float((w * (event[known] & (cause[m][known] == 1))).sum() / w.sum())

        old = m & (u >= deadline)
        print(f"  {k:8d} {m.sum():9d} {((true_d[m] <= deadline) & (cause[m] == 1)).mean():6.3f} "
              f"{((seen_d[old] <= deadline) & (cause[old] == 1)).mean():11.3f} {weighted(u):18.3f} {weighted(u[m]):18.3f}")


def with_and_without_amount(log, cutoff, tau, history):
    edges, H, ncls = log.edges, len(log.edges), len(log.names) + 1
    tr = (log.start > tau - history * DAY) & (log.start <= tau) & (log.tp <= tau)
    te = (log.start > tau) & (log.start <= tau + log.cfg['test_days'] * DAY) & (log.tp <= log.log_end - edges[-1] * DAY)
    Y = np.where(((log.t_term[te] - log.tp[te]) / DAY)[:, None] <= edges[None, :], log.cause[te][:, None], 0)
    u = (tau - log.tp[tr]) / DAY
    amount = lambda rows: np.expm1(log.X[rows][:, AMOUNT]).round(1)
    unseen = np.mean(~np.isin(amount(te), np.unique(amount(tr)[u >= edges[-1]])))
    print(f"  cutoff {cutoff}: {unseen:.0%} of the test histories have an amount that no training history followed for a year has")
    for label, drop in (('all features', False), ('without the amount', True)):
        X = log.X.copy()
        if drop:
            X[:, AMOUNT] = 0.0
        keep, outcome, obs = labels(log.tp[tr], log.t_term[tr], log.cause[tr], tau, 'censor_aware', edges)
        _, w = score(fit_predict(X[tr][keep], outcome[keep], obs[keep], X[te], H, ncls), Y, H, log.names)
        _, i = score(compete.ipcw(X[tr], log.tp[tr], log.t_term[tr], log.cause[tr], tau, edges, X[te], ncls), Y, H, log.names)
        first = log.names[0]
        print(f"    {label:19s} Brier: window rule {w['brier_ident']:.3f}, ipcw {i['brier_ident']:.3f}; predicted minus actual share "
              f"with a {first} within a year: window rule {w['pred_minus_actual_last'][first]:+.3f}, ipcw {i['pred_minus_actual_last'][first]:+.3f}")


def main():
    log = Log('traffic')
    print("1. Share of histories with a first payment within 365 days, cutoff 2007-09-13, two years of history, no features:")
    by_position(log, datetime(2007, 9, 13, tzinfo=timezone.utc).timestamp(), 730)
    print("\n2. With the boosted trees, two years of history:")
    for cutoff in ('2004-07-26', '2007-09-13'):
        with_and_without_amount(log, cutoff, datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc).timestamp(), 730)


if __name__ == '__main__':
    main()
