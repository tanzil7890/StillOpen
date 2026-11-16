"""Calibration in detail and the money case (PROTOCOL_B.md, section 4g).

    python censoring/calibration.py traffic      -> censoring/results/traffic.calibration.jsonl
    python censoring/calibration.py --summary    tables over all logs that have results
    python censoring/calibration.py --diagram    reliability tables, rows of all cutoffs put together

For every setting of the grid, with the main model and its fixed settings, for each of the four treatments and
every deadline shorter than the history:

  for "any outcome" and each outcome   the ten-bin reliability table over all test histories (count, mean forecast,
                                       share that happened), slope and intercept of a logistic recalibration, and
                                       the reliability, resolution and uncertainty parts of the Brier score
  at the first history of a test case  the planning error per 100 cases; for logs with an amount, the forecast and
                                       the realised value of the first-listed outcome (payment, approval)
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS
from decisions import AMOUNT, POSITION, calibration, interval, rank_of
from study import DAY, RESULTS, TREATMENTS, Log, fit_predict, labels, versions

BINS = 10


def table(p, y):
    """-> one [count, mean forecast, share that happened] per bin of the forecast; empty bins are [0, None, None]."""
    idx = np.minimum((p * BINS).astype(int), BINS - 1)
    return [[int((idx == b).sum()), round(float(p[idx == b].mean()), 4), round(float(y[idx == b].mean()), 4)] if (idx == b).any()
            else [0, None, None] for b in range(BINS)]


def measure(log, cutoff, w):
    edges, H, ncls = log.edges, len(log.edges), len(log.names) + 1
    tau = datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc).timestamp()
    tr = (log.start > tau - w * DAY) & (log.start <= tau) & (log.tp <= tau)
    te = (log.start > tau) & (log.start <= tau + log.cfg['test_days'] * DAY) & (log.tp <= log.log_end - edges[-1] * DAY)
    n_ident = int((edges < w).sum())
    if n_ident == 0 or te.sum() == 0 or tr.sum() == 0:
        return None
    Y = np.where(((log.t_term[te] - log.tp[te]) / DAY)[:, None] <= edges[None, :], log.cause[te][:, None], 0)
    first = log.X[te][:, POSITION] == 1
    amount = np.expm1(log.X[te][first][:, AMOUNT]) if log.cfg.get('amount') else None
    out = {'log': log.name, 'cutoff': cutoff, 'history_days': w, 'test_cases': int(first.sum()), 'versions': versions()}
    for treat in TREATMENTS:
        keep, outcome, obs = labels(log.tp[tr], log.t_term[tr], log.cause[tr], tau, treat, edges)
        if keep.sum() == 0:
            out[treat] = None
            continue
        dist, by = fit_predict(log.X[tr][keep], outcome[keep], obs[keep], log.X[te], H, ncls), {}
        for j in range(n_ident):
            targets = {'any': (1 - dist[:, j, 0], (Y[:, j] > 0).astype(float))}
            targets.update({name: (dist[:, j, k + 1], (Y[:, j] == k + 1).astype(float)) for k, name in enumerate(log.names)})
            r = {name: {**calibration(p, y), 'bins': table(p, y)} for name, (p, y) in targets.items()}
            p, y = targets['any']
            r['planning_error_per_100'] = round(float((p[first].mean() - y[first].mean()) * 100), 2)
            if amount is not None and np.isfinite(amount).all():
                r['forecast_value'] = round(float((amount * dist[first][:, j, 1]).sum()), 2)
                r['realised_value'] = round(float((amount * (Y[first][:, j] == 1)).sum()), 2)
            by[f"{edges[j]:g}"] = r
        out[treat] = by
    return out


def read():
    names = [n for n in DATASETS if os.path.exists(os.path.join(RESULTS, n + '.calibration.jsonl'))]
    return {n: [json.loads(line) for line in open(os.path.join(RESULTS, n + '.calibration.jsonl'))] for n in names}


def pooled_bins(rows, treat, target='any', rank=3):
    """Bins of the longest deadline shorter than the history, the rows of all given settings put together."""
    count, forecast, happened = np.zeros(BINS), np.zeros(BINS), np.zeros(BINS)
    for r in rows:
        if rank_of(r) != rank or not r.get(treat):
            continue
        last = max(r[treat], key=float)
        for b, (n, p, y) in enumerate(r[treat][last][target]['bins']):
            if n:
                count[b] += n; forecast[b] += n * p; happened[b] += n * y
    return count, forecast / np.maximum(count, 1), happened / np.maximum(count, 1)


def diagram():
    rows = read()
    print("Reliability at the longest history and the longest deadline shorter than it, any outcome; rows of all logs and cutoffs")
    print("together, every log weighted the same. Per bin: mean forecast -> share that happened (share of the rows).")
    for treat in TREATMENTS:
        parts = [pooled_bins(rs, treat) for rs in rows.values()]
        weight = np.mean([c / c.sum() for c, _, _ in parts], axis=0)
        cells = []
        for b in range(BINS):
            use = [(c[b] / c.sum(), f[b], h[b]) for c, f, h in parts if c[b]]
            if use:
                s = sum(u[0] for u in use)
                cells.append(f"{sum(u[0] * u[1] for u in use) / s:.2f}->{sum(u[0] * u[2] for u in use) / s:.2f} ({100 * weight[b]:.0f}%)")
        print(f"  {treat:17s} " + '  '.join(cells))


def summary():
    rows = read()
    labels_ = ['half', 'equal', 'twice', 'three times']
    print(f"{len(rows)} logs. Calibration slope (1 is ideal) by position of the deadline among those shorter than the history: the shortest, the longest.")
    for target in ('any', 'first-listed outcome'):
        print(f"\n  {target}\n  {'history':12s} " + ' '.join(f"{t:>26s}" for t in TREATMENTS))
        for rank in range(4):
            cells = []
            for t in TREATMENTS:
                short, long_ = [], []
                for n, rs in rows.items():
                    name = 'any' if target == 'any' else list(dict.fromkeys(DATASETS[n]['outcomes'].values()))[0]
                    s = [r[t][min(r[t], key=float)][name]['slope'] for r in rs if rank_of(r) == rank and r.get(t)]
                    g = [r[t][max(r[t], key=float)][name]['slope'] for r in rs if rank_of(r) == rank and r.get(t)]
                    s, g = [v for v in s if v is not None], [v for v in g if v is not None]
                    if s and g:
                        short.append(np.mean(s)); long_.append(np.mean(g))
                cells.append(f"{np.mean(short):.2f}, {np.mean(long_):.2f}" if short else '-')
            print(f"  {labels_[rank]:12s} " + ' '.join(f"{c:>26s}" for c in cells))
    print("\nReliability part of the Brier score per outcome (0 is ideal), longest history, longest deadline shorter than it; mean over logs")
    for t in TREATMENTS:
        vals = {'any': [], 'first-listed outcome': [], 'other outcomes': []}
        for n, rs in rows.items():
            names = list(dict.fromkeys(DATASETS[n]['outcomes'].values()))
            use = [r[t][max(r[t], key=float)] for r in rs if rank_of(r) == 3 and r.get(t)]
            if use:
                vals['any'].append(np.mean([u['any']['reliability'] for u in use]))
                vals['first-listed outcome'].append(np.mean([u[names[0]]['reliability'] for u in use]))
                if len(names) > 1:
                    vals['other outcomes'].append(np.mean([u[m]['reliability'] for u in use for m in names[1:]]))
        print(f"  {t:17s} " + '  '.join(f"{k} {np.mean(v):.4f} ({len(v)} logs)" for k, v in vals.items() if v))
    if 'traffic' in rows:
        print("\nMoney case, traffic fines: forecast against realised value of the fines with a first payment by the deadline,")
        print("per cutoff month of new fines; mean over the six cutoffs, in thousands of the log's currency; and forecast / realised - 1 with a 95% interval over cutoffs")
        for deadline in ('90', '365'):
            for rank in range(4):
                rs = [r for r in rows['traffic'] if rank_of(r) == rank and r.get('censor_aware') and deadline in r['censor_aware']]
                if not rs:
                    continue
                realised = np.mean([r['censor_aware'][deadline]['realised_value'] for r in rs]) / 1000
                cells = []
                for t in TREATMENTS:
                    use = [r[t][deadline] for r in rs if r.get(t) and deadline in r[t]]
                    if use:
                        ratio = [100 * (u['forecast_value'] / u['realised_value'] - 1) for u in use]
                        cells.append(f"{t[:13]} {np.mean([u['forecast_value'] for u in use]) / 1000:7.0f} {interval(ratio).strip()}%")
                print(f"  {deadline:>3s} days, history {labels_[rank]:11s} realised {realised:6.0f} | " + ' | '.join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', nargs='?', choices=list(DATASETS))
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--diagram', action='store_true')
    a = ap.parse_args()
    if a.diagram:
        return diagram()
    if a.summary or not a.log:
        return summary()
    cfg, log = DATASETS[a.log], Log(a.log)
    with open(os.path.join(RESULTS, a.log + '.calibration.jsonl'), 'w') as f:
        for cutoff in cfg['cutoffs']:
            for w in cfg['history_days']:
                r = measure(log, cutoff, w)
                if r is None:
                    continue
                f.write(json.dumps(r) + '\n'); f.flush()
                last = max(r['censor_aware'], key=float)
                print(f"{a.log} {cutoff} history {w:>6.0f}d deadline {last}d | " + '  '.join(
                    f"{t[:8]} slope {r[t][last]['any']['slope']}" if r.get(t) else f"{t[:8]} -" for t in TREATMENTS), flush=True)


if __name__ == '__main__':
    main()
