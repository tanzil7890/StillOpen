"""What each treatment of open cases costs in decisions (PROTOCOL_B.md, section 4d).

    python censoring/decisions.py traffic            -> censoring/results/traffic.decisions.jsonl
    python censoring/decisions.py traffic --replay   -> censoring/results/traffic.replay.jsonl  (a cutoff every month)
    python censoring/decisions.py --summary          tables over all logs that have results

For every setting of the grid, with the main model and its fixed settings, and for each of the four treatments:

  planning error  at the first history of each test case: forecast minus actual number of cases with an outcome by
                  the deadline, per 100 cases
  calibration     for "any outcome by the deadline" over all test histories: slope and intercept of a logistic
                  recalibration (1 and 0 are ideal); reliability, resolution and uncertainty of the Brier score
  value           logs with an amount: summed amount times forecast probability of the first-listed outcome (payment,
                  approval) against the summed amount of the cases where it happened, as a ratio minus one

All of them at the longest deadline that is shorter than the history.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS
from study import DAY, RESULTS, TREATMENTS, Log, fit_predict, labels, versions

POSITION, AMOUNT = 0, 5  # columns of study.build()


def calibration(p, y, bins=10):
    """p: forecast probability, y: 0/1 -> slope, intercept, reliability, resolution, uncertainty."""
    out = {'slope': None, 'intercept': None}
    if 0 < y.mean() < 1:
        z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        fit = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], y)
        out = {'slope': round(float(fit.coef_[0, 0]), 3), 'intercept': round(float(fit.intercept_[0]), 3)}
    idx, base = np.minimum((p * bins).astype(int), bins - 1), y.mean()
    rel = res = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            rel += m.mean() * (p[m].mean() - y[m].mean()) ** 2
            res += m.mean() * (y[m].mean() - base) ** 2
    return {**out, 'reliability': round(float(rel), 4), 'resolution': round(float(res), 4), 'uncertainty': round(float(base * (1 - base)), 4)}


def measure(log, cutoff, w, test_days):
    edges, H, ncls = log.edges, len(log.edges), len(log.names) + 1
    tau = datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc).timestamp()
    tr = (log.start > tau - w * DAY) & (log.start <= tau) & (log.tp <= tau)
    te = (log.start > tau) & (log.start <= tau + test_days * DAY) & (log.tp <= log.log_end - edges[-1] * DAY)
    j = int((edges < w).sum()) - 1                                    # the longest deadline shorter than the history
    if j < 0 or te.sum() == 0 or tr.sum() == 0:
        return None
    Y = np.where(((log.t_term[te] - log.tp[te]) / DAY)[:, None] <= edges[None, :], log.cause[te][:, None], 0)
    first = log.X[te][:, POSITION] == 1
    amount = np.expm1(log.X[te][first][:, AMOUNT]) if log.cfg.get('amount') else None
    out = {'log': log.name, 'cutoff': cutoff, 'history_days': w, 'deadline_days': float(edges[j]), 'test_cases': int(first.sum()),
           'actual_per_100': round(float((Y[first][:, j] > 0).mean() * 100), 2), 'versions': versions()}
    for treat in TREATMENTS:
        keep, outcome, obs = labels(log.tp[tr], log.t_term[tr], log.cause[tr], tau, treat, edges)
        if keep.sum() == 0:
            out[treat] = None
            continue
        dist = fit_predict(log.X[tr][keep], outcome[keep], obs[keep], log.X[te], H, ncls)
        any_p, any_y = 1 - dist[:, j, 0], (Y[:, j] > 0).astype(float)
        r = {'planning_error_per_100': round(float((any_p[first].mean() - any_y[first].mean()) * 100), 2), **calibration(any_p, any_y)}
        if amount is not None and np.isfinite(amount).all():
            realised = float((amount * (Y[first][:, j] == 1)).sum())
            r['value_ratio'] = round(float((amount * dist[first][:, j, 1]).sum() / realised - 1), 4) if realised > 0 else None
        out[treat] = r
    return out


def monthly(first, last):
    t, end = datetime.fromisoformat(first), datetime.fromisoformat(last)
    while t <= end:
        yield t.strftime('%Y-%m-%d')
        t += timedelta(days=30.4375)


def rank_of(r):
    return sorted(DATASETS[r['log']]['history_days']).index(r['history_days'])


def interval(values):
    """Mean with a 95% interval from the t distribution; the unit is the log."""
    v = np.asarray(list(values), dtype=float)
    if len(v) < 2:
        return f"{v.mean():+6.1f}" if len(v) else '     -'
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():+6.1f} ({v.mean() - half:+.1f} to {v.mean() + half:+.1f})"


def summary():
    names = [n for n in DATASETS if os.path.exists(os.path.join(RESULTS, n + '.decisions.jsonl'))]
    rows = {n: [json.loads(line) for line in open(os.path.join(RESULTS, n + '.decisions.jsonl'))] for n in names}

    def per_log(key, rank, treat, f=lambda x: x):
        out = {}
        for n in names:
            v = [f(r[treat][key]) for r in rows[n] if rank_of(r) == rank and r.get(treat) and r[treat].get(key) is not None]
            if v:
                out[n] = float(np.mean(v))
        return out

    labels_ = ['half', 'equal', 'twice', 'three times']
    for title, key, f, fmt in (("Planning error: |forecast - actual| cases with an outcome by the deadline, per 100 cases (mean over logs)", 'planning_error_per_100', abs, '{:6.1f}'),
                               ("The same, signed (positive: too many outcomes forecast)", 'planning_error_per_100', lambda x: x, '{:+6.1f}'),
                               ("Calibration slope (1 is ideal)", 'slope', lambda x: x, '{:6.2f}'),
                               ("Reliability part of the Brier score (0 is ideal)", 'reliability', lambda x: x, '{:6.3f}'),
                               ("Value forecast: |forecast / realised - 1| (logs with an amount)", 'value_ratio', abs, '{:6.2f}')):
        print(f"\n{title}\n  {'history':12s} " + ' '.join(f"{t:>17s}" for t in TREATMENTS) + '   logs where the window rule is closer than: ' + ', '.join(t for t in TREATMENTS[:-1]))
        for rank in range(4):
            cells, wins = [], []
            base = per_log(key, rank, 'censor_aware', f)
            for t in TREATMENTS:
                v = per_log(key, rank, t, f)
                cells.append(fmt.format(np.mean(list(v.values()))) if v else '     -')
                if t != 'censor_aware' and key != 'slope':
                    both = [n for n in v if n in base]
                    wins.append(f"{sum(base[n] < v[n] for n in both)}/{len(both)}" if both else '-')
            print(f"  {labels_[rank]:12s} " + ' '.join(f"{c:>17s}" for c in cells) + '   ' + '  '.join(wins))
    for title, f in (("Signed planning error per 100 cases: mean over logs (95% interval, the log as the unit)", None),
                     ("|Planning error| of the treatment minus that of the window rule, per 100 cases: mean over logs (95% interval)", abs)):
        print(f"\n{title}\n  {'history':12s} " + ' '.join(f"{t:>26s}" for t in (TREATMENTS if f is None else TREATMENTS[:-1])))
        for rank in range(4):
            base = per_log('planning_error_per_100', rank, 'censor_aware', abs)
            cells = []
            for t in (TREATMENTS if f is None else TREATMENTS[:-1]):
                v = per_log('planning_error_per_100', rank, t, f or (lambda x: x))
                cells.append(interval(v.values() if f is None else [v[n] - base[n] for n in v if n in base]))
            print(f"  {labels_[rank]:12s} " + ' '.join(f"{c:>26s}" for c in cells))
    print("\nPer log, longest history: signed planning error per 100 cases")
    for n in names:
        print(f"  {n:9s} " + ' '.join(f"{t[:13]} {np.mean([r[t]['planning_error_per_100'] for r in rows[n] if rank_of(r) == 3 and r.get(t)]):+6.1f}" for t in TREATMENTS))
    replay = [n for n in DATASETS if os.path.exists(os.path.join(RESULTS, n + '.replay.jsonl'))]
    if replay:
        print("\nReplay, a cutoff every month: mean |planning error| per 100 cases, and months")
        for n in replay:
            rs = [json.loads(line) for line in open(os.path.join(RESULTS, n + '.replay.jsonl'))]
            print(f"  {n:9s} months {len(rs):3d}  " + ' '.join(f"{t[:13]} {np.mean([abs(r[t]['planning_error_per_100']) for r in rs if r.get(t)]):5.1f}" for t in TREATMENTS))
        print("\nReplay, in cases per week: new cases per week, and mean |forecast - actual| cases with an outcome by the deadline")
        for n in replay:
            rs = [json.loads(line) for line in open(os.path.join(RESULTS, n + '.replay.jsonl'))]
            per_week = lambda r: r['test_cases'] / (30 / 7)
            print(f"  {n:9s} cases/week {np.mean([per_week(r) for r in rs]):7.1f}  " + ' '.join(
                f"{t[:13]} {np.mean([abs(r[t]['planning_error_per_100']) / 100 * per_week(r) for r in rs if r.get(t)]):6.1f}" for t in TREATMENTS))
        mean_abs = {t: {n: np.mean([abs(r[t]['planning_error_per_100']) for r in (json.loads(line) for line in open(os.path.join(RESULTS, n + '.replay.jsonl'))) if r.get(t)])
                        for n in replay} for t in TREATMENTS}
        print("\nReplay: mean |planning error| per 100 cases of the treatment minus that of the window rule, mean over logs (95% interval)")
        print('  ' + '   '.join(f"{t} {interval(mean_abs[t][n] - mean_abs['censor_aware'][n] for n in replay)}" for t in TREATMENTS[:-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', nargs='?', choices=list(DATASETS))
    ap.add_argument('--replay', action='store_true')
    ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    if a.summary or not a.log:
        return summary()
    cfg, log = DATASETS[a.log], Log(a.log)
    if a.replay:
        settings = [(c, sorted(cfg['history_days'])[2], 30) for c in monthly(cfg['cutoffs'][0], cfg['cutoffs'][-1])]
    else:
        settings = [(c, w, cfg['test_days']) for c in cfg['cutoffs'] for w in cfg['history_days']]
    with open(os.path.join(RESULTS, a.log + ('.replay' if a.replay else '.decisions') + '.jsonl'), 'w') as f:
        for cutoff, w, test_days in settings:
            r = measure(log, cutoff, w, test_days)
            if r is None:
                continue
            f.write(json.dumps(r) + '\n'); f.flush()
            print(f"{a.log} {cutoff} history {w:>6.0f}d deadline {r['deadline_days']:.0f}d cases {r['test_cases']:5d} actual {r['actual_per_100']:5.1f}/100 | "
                  + '  '.join(f"{t[:8]} {r[t]['planning_error_per_100']:+6.1f}" if r.get(t) else f"{t[:8]}      -" for t in TREATMENTS), flush=True)


if __name__ == '__main__':
    main()
