"""Download the event logs of datasets.py, check them and turn each into data/derived/<name>.pkl.

    python -I censoring/prepare.py            every log
    python -I censoring/prepare.py traffic    one log

Each log is saved in its own folder under data/ and its checksum is compared with the one published by
4TU.ResearchData. A log that is already there is not downloaded or parsed again. The pickle has the layout of the
pilot's bpi12.pkl: cases = [{acts, ts, term_idx, cause, amount}], vocab = {activity: id}.
"""
import hashlib
import os
import pickle
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATA, DATASETS, derived
from logs import read_log


def fetch(cfg):
    path = os.path.join(DATA, cfg['file'])
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(['curl', '-s', '-S', '-L', '--fail', '-o', path, cfg['url']], check=True)
    with open(path, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    if md5 != cfg['md5']:
        raise SystemExit(f"{path}: checksum {md5} is not the published {cfg['md5']}")
    return path


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return float('nan')


def parse(cfg, path):
    names = list(dict.fromkeys(cfg['outcomes'].values()))  # several activities may stand for one outcome
    cause_of = {activity: names.index(name) + 1 for activity, name in cfg['outcomes'].items()}
    cases, vocab = [], {}
    for attrs, events in read_log(path, **cfg.get('csv', {})):
        if not events:
            continue
        acts = []
        for name, state, _, _ in events:
            # lifecycle: None = never; 'W_' = for activities starting with W_; '' = for every activity
            token = name + '|' + state if cfg['lifecycle'] is not None and name.startswith(cfg['lifecycle']) else name
            acts.append(vocab.setdefault(token, len(vocab) + 1))
        term_idx, cause = next(((i, cause_of[e[0]]) for i, e in enumerate(events) if e[0] in cause_of), (-1, 0))
        where, key = cfg['amount'] or (None, None)
        amount = number({'case': attrs, 'first_event': events[0][3]}[where].get(key)) if where else float('nan')
        cases.append(dict(acts=np.array(acts, dtype=np.int64), ts=np.array([e[2] for e in events], dtype=np.float64),
                          term_idx=term_idx, cause=cause, amount=amount))
    if cfg.get('vocab') == 'frequency':  # number the activities from the most to the least frequent
        count = np.bincount(np.concatenate([c['acts'] for c in cases]), minlength=len(vocab) + 1)
        new = np.zeros(len(vocab) + 1, dtype=np.int64)
        new[np.argsort(-count[1:], kind='stable') + 1] = np.arange(1, len(vocab) + 1)
        for c in cases:
            c['acts'] = new[c['acts']]
        vocab = {token: int(new[i]) for token, i in vocab.items()}
    return dict(cases=cases, vocab=vocab)


def main():
    for name in sys.argv[1:] or DATASETS:
        cfg, out = DATASETS[name], derived(name)
        path = fetch(cfg)
        if not os.path.exists(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, 'wb') as f:
                pickle.dump(parse(cfg, path), f)
        with open(out, 'rb') as f:
            data = pickle.load(f)
        names = list(dict.fromkeys(cfg['outcomes'].values()))
        counts = np.bincount([c['cause'] for c in data['cases']], minlength=len(names) + 1).tolist()
        print(f"{name}: {len(data['cases'])} cases, {sum(len(c['acts']) for c in data['cases'])} events, "
              f"{len(data['vocab'])} activities; no outcome {counts[0]}, "
              + ', '.join(f"{label} {n}" for label, n in zip(names, counts[1:])))


if __name__ == '__main__':
    main()
