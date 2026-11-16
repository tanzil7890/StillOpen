"""The three linear competing methods under the protocol of study.py (PROTOCOL_B.md, section 4b).

    python censoring/compete_linear.py helpdesk     -> censoring/results/helpdesk.linear.jsonl

dfm, cox and pu (see linear.py) are scored against the window rule fitted with logistic regression, which is linear
in the same features. That rule is stored here as censor_aware; it equals the censor_aware of <log>.logit.jsonl.

Read the results with:  python censoring/pooled.py --model linear
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import linear
from compete import clean
from datasets import DATASETS
from study import DAY, RESULTS, Log, boot_diff, fit_predict, labels, score, versions

METHODS = ('dfm', 'cox', 'pu')


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
        seen = log.t_term[tr] <= tau
        delay = np.where(seen, (log.t_term[tr] - log.tp[tr]) / DAY, np.inf)
        kind = np.where(seen, log.cause[tr], 0)
        follow = (tau - log.tp[tr]) / DAY
        Ftr, Fte = linear.design(Xtr, Xte)
        keep, outcome, obs = labels(log.tp[tr], log.t_term[tr], log.cause[tr], tau, 'censor_aware', edges)
        dists = {'censor_aware': fit_predict(log.X[tr][keep], outcome[keep], obs[keep], log.X[te], H, ncls, 'logit')}
        for m in methods:
            torch_seed()
            dists[m] = getattr(linear, m)(Ftr, Fte, delay, kind, follow, edges, ncls - 1)
        res, per = {}, {}
        for name, dist in dists.items():
            if dist is None:
                res[name] = None
                continue
            per[name], res[name] = score(np.clip(dist, 0, 1), Y, n_ident, log.names)
        diff = {m: {'mean': round(float((per[m] - per['censor_aware']).mean()), 4),
                    'ci95': boot_diff(per[m], per['censor_aware'], log.cid[te], n_boot)} for m in methods if m in per}
        out.append({'log': log.name, 'cutoff': cutoff, 'model': 'linear', 'versions': versions(), 'history_days': w,
                    'identifiable_deadlines': edges[:n_ident].tolist(), 'test_cases': int(len(np.unique(log.cid[te]))),
                    'brier_minus_censor_aware': diff, **res})
    return out


def torch_seed():
    import torch
    torch.manual_seed(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', choices=list(DATASETS))
    ap.add_argument('--cutoffs')
    ap.add_argument('--out')
    a = ap.parse_args()
    cfg, log = DATASETS[a.log], Log(a.log)
    with open(a.out or os.path.join(RESULTS, a.log + '.linear.jsonl'), 'w') as f:
        for cutoff in (a.cutoffs.split(',') if a.cutoffs else cfg['cutoffs']):
            for r in run(log, cutoff, cfg['history_days']):
                f.write(json.dumps(r) + '\n'); f.flush()
                d = r['brier_minus_censor_aware']
                print(f"{a.log} {cutoff} history {r['history_days']:>6.0f}d  window (logit) {r['censor_aware']['brier_ident']:.4f}  "
                      + '  '.join(f"{m} {r[m]['brier_ident']:.4f} ({d[m]['mean']:+.4f})" if r.get(m) else f"{m} -" for m in METHODS), flush=True)


if __name__ == '__main__':
    main()
