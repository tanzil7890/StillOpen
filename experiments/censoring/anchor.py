"""Replication anchor (paper_journal/audit/PROTOCOL.md): one published number of the outcome benchmark of
Teinemaa et al. (2019), reproduced with this study's model, and the same model without labels from the future.

    python -I censoring/anchor.py            -> censoring/results/anchor.json

The benchmark's processed logs are downloaded from the link in its repository into data/benchmark_raw/. Its protocol
(split_data_strict, prefixes, single bucket, static plus aggregation encoding) is re-implemented here; its code is
not run. The classifier is the boosted trees of this study, not the tuned XGBoost of the benchmark.
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, '..', 'data', 'benchmark_raw')
URL = 'https://drive.usercontent.google.com/download?id=154hcH-HGThlcZJW5zBvCJMZvjOQDsnPR&export=download&confirm=t'
MD5 = '29722ccd1ad7fcc73cd4a603c51329c1'
CASE, ACTIVITY, TIME, LABEL = 'Case ID', 'Activity', 'Complete Timestamp', 'label'
NUM_EVENT = ['timesincelastevent', 'timesincecasestart', 'timesincemidnight', 'event_nr', 'month', 'weekday', 'hour', 'open_cases']
DATASETS = {   # columns as in the benchmark's dataset_confs.py; published: Table 8, single_agg, XGBoost
    'traffic': dict(file='traffic_fines_1.csv', published=0.66, max_prefix=10, static_cat=['article', 'vehicleClass'], static_num=['amount', 'points'],
                    event_cat=['Activity', 'Resource', 'lastSent', 'notificationType', 'dismissal'], event_num=['expense'] + NUM_EVENT),
    'bpic2012_1': dict(file='bpic2012_O_ACCEPTED-COMPLETE.csv', published=0.70, max_prefix=None, static_cat=[], static_num=['AMOUNT_REQ'],
                       event_cat=['Activity', 'Resource'], event_num=NUM_EVENT),
}
MODELS = {'study settings': dict(max_iter=60, learning_rate=0.08), '500 rounds': dict(max_iter=500, learning_rate=0.05)}


def fetch():
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, 'processed_benchmark_event_logs.zip')
    if not os.path.exists(path):
        subprocess.run(['curl', '-sL', '-m', '900', '-o', path, URL], check=True)
    with open(path, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    if md5 != MD5:
        raise SystemExit(f"{path}: checksum {md5} is not the recorded {MD5}")
    with zipfile.ZipFile(path) as z:
        for cfg in DATASETS.values():
            if not os.path.exists(os.path.join(RAW, cfg['file'])):
                z.extract(cfg['file'], RAW)


def read(cfg):
    """-> cases in the order of their first timestamp, each a list of its rows sorted by time and then activity."""
    cases = {}
    with open(os.path.join(RAW, cfg['file']), newline='') as f:
        for row in csv.DictReader(f, delimiter=';'):
            row['_t'] = datetime.fromisoformat(row[TIME]).replace(tzinfo=timezone.utc).timestamp()
            cases.setdefault(row[CASE], []).append(row)
    for rows in cases.values():
        rows.sort(key=lambda r: (r['_t'], r[ACTIVITY]))
    return sorted(cases.values(), key=lambda rows: rows[0]['_t'])   # a stable sort, as the benchmark's


def number(text):
    try:
        return float(text)
    except ValueError:
        return np.nan


def encode(cases, cfg, max_prefix, vocab=None):
    """One row per prefix of 1..max_prefix events. -> features, label, prefix length, index of the case, vocabulary."""
    first = vocab is None
    vocab = vocab or {c: {} for c in cfg['static_cat'] + cfg['event_cat']}
    events = [(i, n, row) for i, rows in enumerate(cases) for n, row in enumerate(rows[:max_prefix], 1)]
    if first:
        for _, _, row in events:
            for c in vocab:
                vocab[c].setdefault(row[c], len(vocab[c]))
    case, length = np.array([e[0] for e in events]), np.array([e[1] for e in events])
    start = np.flatnonzero(length == 1)[case]                        # row of each event's first event

    def within_case(x, op):
        out = x.copy()
        for n in range(2, max_prefix + 1):                           # row r of length n follows row r-1 of the same case
            m = np.flatnonzero(length == n)
            out[m] = op(out[m - 1], x[m])
        return out

    blocks = []
    for c in cfg['static_cat']:                                      # one indicator per value, from the first event
        hot = np.zeros((len(events), len(vocab[c])), dtype=np.float32)
        code = np.array([vocab[c].get(e[2][c], -1) for e in events])[start]
        hot[np.flatnonzero(code >= 0), code[code >= 0]] = 1
        blocks.append(hot)
    if cfg['static_num']:
        blocks.append(np.nan_to_num(np.array([[number(e[2][c]) for c in cfg['static_num']] for e in events], dtype=np.float32)[start]))
    for c in cfg['event_cat']:                                       # how often each value occurred in the prefix
        hot = np.zeros((len(events), len(vocab[c])), dtype=np.float32)
        code = np.array([vocab[c].get(e[2][c], -1) for e in events])
        hot[np.flatnonzero(code >= 0), code[code >= 0]] = 1
        blocks.append(within_case(hot, np.add))
    x = np.nan_to_num(np.array([[number(e[2][c]) for c in cfg['event_num']] for e in events], dtype=np.float64))
    total, squares, n = within_case(x, np.add), within_case(x ** 2, np.add), length[:, None]
    std = np.sqrt(np.maximum(squares - total ** 2 / n, 0) / np.maximum(n - 1, 1))      # pandas: one value has no deviation, filled with 0
    blocks += [total / n, within_case(x, np.maximum), within_case(x, np.minimum), total, np.where(n > 1, std, 0.0)]
    y = np.array([cases[i][0][LABEL] == 'deviant' for i in case], dtype=int)
    return np.hstack(blocks).astype(np.float32), y, length, case, vocab


def measures(p, y, length, case, n_boot=200, seed=0):
    def all_of(idx):
        pp, yy, ll = p[idx], y[idx], length[idx]
        per = [(roc_auc_score(yy[ll == n], pp[ll == n]), (ll == n).sum()) for n in np.unique(ll) if 0 < yy[ll == n].mean() < 1]
        return (sum(a * w for a, w in per) / sum(w for _, w in per), roc_auc_score(yy, pp), ((pp - yy) ** 2).mean(), pp.mean() - yy.mean())
    point = all_of(np.arange(len(p)))
    rng, ids = np.random.default_rng(seed), np.unique(case)
    rows_of = {c: np.flatnonzero(case == c) for c in ids}
    boot = np.array([all_of(np.concatenate([rows_of[c] for c in rng.choice(ids, len(ids))])) for _ in range(n_boot)])
    names = ('auc_weighted_by_prefix_length', 'auc_all_prefixes', 'brier', 'forecast_minus_positive_share')
    return {k: {'value': round(float(v), 4), 'ci95': [round(float(np.percentile(boot[:, i], q)), 4) for q in (2.5, 97.5)]} for i, (k, v) in enumerate(zip(names, point))}


def run(name, cfg):
    cases = read(cfg)
    positive = [len(rows) for rows in cases if rows[0][LABEL] == 'deviant']
    max_prefix = cfg['max_prefix'] or min(40, int(np.ceil(np.quantile(positive, 0.90))))
    cut = int(0.8 * len(cases))
    split = min(row['_t'] for rows in cases[cut:] for row in rows)
    train = {'benchmark': [[r for r in rows if r['_t'] < split] for rows in cases[:cut]],          # split_data_strict
             'strict': [rows for rows in cases[:cut] if rows[-1]['_t'] < split]}                   # split_data_discard
    train = {k: [rows for rows in v if rows] for k, v in train.items()}
    out = {'dataset': cfg['file'], 'published_auc': cfg['published'], 'cases': len(cases), 'max_prefix_length': max_prefix,
           'positive_share_all_cases': round(float(np.mean([rows[0][LABEL] == 'deviant' for rows in cases])), 3),
           'positive_share_test_cases': round(float(np.mean([rows[0][LABEL] == 'deviant' for rows in cases[cut:]])), 3),
           'split_time': datetime.fromtimestamp(split, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
    for kind, rows in train.items():
        Xtr, ytr, _, _, vocab = encode(rows, cfg, max_prefix)
        Xte, yte, length, case, _ = encode(cases[cut:], cfg, max_prefix, vocab)
        out[kind] = {'train_cases': len(rows), 'train_prefixes': int(len(ytr)), 'positive_share_train_cases': round(float(np.mean([r[0][LABEL] == 'deviant' for r in rows])), 3),
                     'test_cases': len(cases) - cut, 'test_prefixes': int(len(yte))}
        for label, settings in MODELS.items():
            clf = HistGradientBoostingClassifier(max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False, random_state=0, **settings)
            out[kind][label] = measures(clf.fit(Xtr, ytr).predict_proba(Xte)[:, 1], yte, length, case)
            print(name, kind, label, {k: v['value'] for k, v in out[kind][label].items()}, flush=True)
    return out


def main():
    fetch()
    results = {name: run(name, cfg) for name, cfg in DATASETS.items()}
    with open(os.path.join(HERE, 'results', 'anchor.json'), 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
