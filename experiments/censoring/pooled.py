"""Pooled statistics over logs: how much worse (or better) is a treatment than the window rule, with uncertainty?

    python censoring/pooled.py                    every log that has results, main model
    python censoring/pooled.py --phase 1          only the five logs of version 1
    python censoring/pooled.py --by org           logs of one organisation merged into one unit
    python censoring/pooled.py --model gru        another model's results
    python censoring/pooled.py --methods          the competing methods (results/<log>.methods.jsonl) as well
    python censoring/pooled.py --model linear     dfm, cox and pu against the window rule with logistic regression
    python censoring/pooled.py --model tuned      every treatment with tuned rounds against the tuned window rule;
                                                  also the window rule with fixed rounds and with a prospective hold-out
    python censoring/pooled.py --model tuned_wide the same with learning rate and leaves tuned as well

Follows PROTOCOL_B.md, section 3. The unit is the log. Per setting the effect is the log of the ratio of Brier
scores (treatment over window rule); per log it is the mean over that log's cutoffs with the standard error of that
mean; over logs it is a random-effects estimate (DerSimonian-Laird between-log variance, Hartung-Knapp interval in
its modified form, which is never narrower than the usual one). Also given: the 95% prediction interval for a new
log, a sign test in which every log has one vote, and Holm-corrected p-values over the cells of the table.
Ratios are printed as percentages: +12% means a Brier score 12% higher than the window rule's.

The last column is the plain mean over units with its t interval, every unit counting the same. It was added after
the tuning results (PROTOCOL_B.md, section 3): a unit in which the two things compared give the same forecasts at
every cutoff has a standard error of zero, and it carries the whole random-effects estimate when the other units are
within their own standard errors of it. Rows with such a unit are marked *.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS
from study import REFERENCE, RESULTS, TREATMENTS

BASE = 'censor_aware'
RANKS = ['half', 'equal', 'twice', 'three times']  # history length relative to the longest deadline


def read(log, model='trees', methods=False):
    """Result rows of a log, keyed by (cutoff, history); with methods=True the competing methods are merged in."""
    def load(suffix):
        path = os.path.join(RESULTS, log + suffix + '.jsonl')
        return [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    rows = {(r['cutoff'], r['history_days']): r for r in load('' if model == 'trees' else '.' + model)}
    if methods:
        for suffix in ('.methods', '.methods_x', '.methods_s'):   # the first two; the one added afterwards; survival trees
            for r in load(suffix if model == 'trees' else f'{suffix}.{model}'):
                key = (r['cutoff'], r['history_days'])
                if key in rows:
                    rows[key].update({k: v for k, v in r.items() if isinstance(v, dict) and 'brier_ident' in v and k != BASE})
    lengths = sorted(DATASETS[log]['history_days'])
    for (_, h), r in rows.items():
        r['rank'] = lengths.index(h)
    return list(rows.values())


def log_effect(rows, method, rank):
    """Mean over cutoffs of ln(Brier of the method / Brier of the window rule), its standard error, the count."""
    y = [np.log(r[method]['brier_ident'] / r[BASE]['brier_ident']) for r in rows
         if r['rank'] == rank and r.get(method) and r.get(BASE)]
    if len(y) < 2:
        return None
    return float(np.mean(y)), float(np.std(y, ddof=1) / np.sqrt(len(y))), len(y)


def random_effects(y, se):
    """-> pooled mean, 95% interval, 95% prediction interval, between-unit variance, p-value. Needs at least two units."""
    y, v = np.asarray(y, float), np.asarray(se, float) ** 2
    k = len(y)
    v = np.maximum(v, 1e-12)
    w = 1 / v
    fixed = (w * y).sum() / w.sum()
    q = (w * (y - fixed) ** 2).sum()
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))  # DerSimonian-Laird
    w = 1 / (v + tau2)
    mu = (w * y).sum() / w.sum()
    scale = max(1.0, (w * (y - mu) ** 2).sum() / (k - 1))                   # Hartung-Knapp, never below 1
    s = np.sqrt(scale / w.sum())
    t = stats.t.ppf(0.975, k - 1)
    p = 2 * stats.t.sf(abs(mu) / s, k - 1)
    pred = (mu - stats.t.ppf(0.975, k - 2) * np.sqrt(tau2 + s ** 2), mu + stats.t.ppf(0.975, k - 2) * np.sqrt(tau2 + s ** 2)) if k >= 3 else None
    return dict(mean=float(mu), ci=(float(mu - t * s), float(mu + t * s)), prediction=pred, tau2=float(tau2), p=float(p), units=k)


def plain_mean(y):
    """Mean over units, every unit counting the same, with its 95% t interval."""
    y = np.asarray(y, float)
    half = stats.t.ppf(0.975, len(y) - 1) * y.std(ddof=1) / np.sqrt(len(y))
    return float(y.mean()), (float(y.mean() - half), float(y.mean() + half))


def sign_test(y):
    """Two-sided: do the units agree in sign more than a fair coin would? -> (positive, negative, p)."""
    pos, neg = int(np.sum(np.asarray(y) > 0)), int(np.sum(np.asarray(y) < 0))
    return pos, neg, float(stats.binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else 1.0


def holm(p):
    """Holm's step-down correction; None entries stay None."""
    idx = [i for i, x in enumerate(p) if x is not None]
    order = sorted(idx, key=lambda i: p[i])
    out, running = list(p), 0.0
    for n, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - n) * p[i]))
        out[i] = running
    return out


def pool(logs, methods, model='trees', by='log', with_methods=False):
    """-> {(method, rank): result of random_effects plus the per-unit effects and the sign test}."""
    data = {log: read(log, model, with_methods) for log in logs}
    cells = {}
    for method in methods:
        for rank in range(4):
            unit = defaultdict(list)
            for log in logs:
                e = log_effect(data[log], method, rank)
                if e:
                    unit[DATASETS[log]['org'] if by == 'org' else log].append(e)
            # logs of one unit are averaged; their standard errors combine as for a mean of independent estimates
            names = list(unit)
            y = [np.mean([e[0] for e in unit[u]]) for u in names]
            se = [np.sqrt(np.sum([e[1] ** 2 for e in unit[u]])) / len(unit[u]) for u in names]
            if len(y) >= 2:
                cells[method, rank] = {**random_effects(y, se), 'per_unit': dict(zip(names, y)), 'sign': sign_test(y),
                                       'plain': plain_mean(y), 'zero_se': int(np.sum(np.asarray(se) < 1e-9))}
    adjusted = holm([c['p'] for c in cells.values()])
    for c, a in zip(cells.values(), adjusted):
        c['p_holm'] = a
    return cells


def pct(x):
    return f"{(np.exp(x) - 1) * 100:+.1f}%"


def main():
    argv = sys.argv[1:]
    def opt(name, default=None):
        if name in argv:
            i = argv.index(name)
            v = argv[i + 1]
            del argv[i:i + 2]
            return v
        return default
    model, by, phase = opt('--model', 'trees'), opt('--by', 'log'), opt('--phase')
    with_methods = '--methods' in argv
    logs = [log for log in DATASETS if read(log, model) and (phase is None or DATASETS[log]['phase'] == phase)]
    # whatever the result files hold besides the window rule: the treatments first, then the rest by name
    found = {k for log in logs for r in read(log, model, with_methods) for k, v in r.items() if isinstance(v, dict) and 'brier_ident' in v}
    found -= {BASE, REFERENCE, 'no_features_censor_aware'}
    methods = [t for t in TREATMENTS if t in found] + sorted(found - set(TREATMENTS))
    cells = pool(logs, methods, model, by, with_methods)
    print(f"model {model}; {len(logs)} logs ({', '.join(logs)}); unit: {'organisation' if by == 'org' else 'log'}")
    print(f"Brier score relative to the window rule. Pooled over units; in brackets the 95% interval.\n")
    print(f"{'method':17s} {'history':12s} {'units':>5s} {'pooled':>8s} {'95% interval':>20s} {'new log: 95% of cases':>24s} {'p':>7s} {'p (Holm)':>9s} {'higher/lower':>13s} {'sign p':>7s}   plain mean over units (95% interval)")
    for (method, rank), c in cells.items():
        pred = f"{pct(c['prediction'][0])} to {pct(c['prediction'][1])}" if c['prediction'] else '-'
        print(f"{method:17s} {RANKS[rank]:12s} {c['units']:5d} {pct(c['mean']):>8s} {pct(c['ci'][0]) + ' to ' + pct(c['ci'][1]):>20s} {pred:>24s} "
              f"{c['p']:7.4f} {c['p_holm']:9.4f} {str(c['sign'][0]) + '/' + str(c['sign'][1]):>13s} {c['sign'][2]:7.4f}"
              f"   {pct(c['plain'][0])} ({pct(c['plain'][1][0])} to {pct(c['plain'][1][1])}){' *' if c['zero_se'] else ''}")
    if any(c['zero_se'] for c in cells.values()):
        print("\n* a unit has a standard error of zero (the same forecasts at every cutoff) and may carry the pooled estimate; compare the plain mean")
    print("\nper unit, pooled over the four history lengths is not shown: effects by unit at each history length")
    for (method, rank), c in cells.items():
        print(f"  {method:17s} {RANKS[rank]:12s} " + '  '.join(f"{u} {pct(v)}" for u, v in c['per_unit'].items()))


if __name__ == '__main__':
    main()
