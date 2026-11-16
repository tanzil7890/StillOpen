"""The experiment grid of a new log, computed by the fixed rules of PROTOCOL_B.md (section 2).

    python -I censoring/design.py <log file> "<outcome activity>;<outcome activity>" [key=value options for a CSV]

Prints the deadlines, history lengths, test window and cutoff dates, and whether the log meets the inclusion
criteria that depend on them. Only descriptive statistics of the log are used; no model is trained.
"""
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logs import read_log

DAY = 86400.0
MAXK = 40
NICE = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365, 545, 730, 1095]


def nice(days):
    """Nearest value of NICE on a log scale; a value that repeats is dropped, so there can be fewer than five."""
    out = []
    for d in days:
        v = NICE[int(np.argmin(np.abs(np.log(NICE) - np.log(max(d, 0.5)))))]
        if v not in out:
            out.append(v)
    return out


def day(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d')


def propose(path, outcomes, not_before=None, **csv_columns):
    """not_before: 'YYYY-MM-DD'; cases that start earlier are left out (a documented change of the process)."""
    starts, to_outcome, from_history, events = [], [], [], []
    n_cases = n_with = 0
    t0 = datetime.fromisoformat(not_before).replace(tzinfo=timezone.utc).timestamp() if not_before else -np.inf
    for _, ev in read_log(path, **csv_columns):
        if not ev or ev[0][2] < t0:
            continue
        n_cases += 1
        starts.append(ev[0][2])
        events.extend(e[2] for e in ev)
        i = next((k for k, e in enumerate(ev) if e[0] in outcomes), None)
        if i is None:
            continue
        n_with += 1
        to_outcome.append((ev[i][2] - ev[0][2]) / DAY)
        from_history.extend((ev[i][2] - e[2]) / DAY for e in ev[:min(i, MAXK)])
    starts, events = np.array(starts), np.array(events)
    deadlines = nice(np.quantile(from_history, [.15, .35, .55, .75, .90]))
    longest = deadlines[-1]
    history = [max(1, round(longest * f)) for f in (0.5, 1, 2, 3)]
    first, last = np.quantile(starts, 0.01), np.quantile(events, 0.995)
    slack = float(np.quantile(to_outcome, 0.90))
    test = 90
    for cand in (21, 30, 60, 90):
        lo, hi = first + history[-1] * DAY, last - (longest + cand + slack) * DAY
        per_cutoff = ((starts > lo) & (starts <= hi)).sum() / max((hi - lo) / DAY, 1e-9) * cand if hi > lo else 0
        if per_cutoff >= 150:
            test = cand
            break
    lo, hi = first + history[-1] * DAY, last - (longest + test + slack) * DAY
    span = (hi - lo) / DAY
    per_cutoff = ((starts > lo) & (starts <= hi)).sum() / max(span, 1e-9) * test if span > 0 else 0
    k = 6 if span >= 5 * test else int(span // test) + 1 if span > 0 else 0
    cutoffs = [day(t) for t in np.linspace(lo, hi, k)] if k >= 2 else []
    return dict(cases=n_cases, with_outcome=n_with, deadlines=deadlines, history_days=history, test_days=test, cutoffs=cutoffs,
                usable_period=(day(lo), day(hi)) if span > 0 else None, usable_days=round(span), test_cases_per_cutoff=round(per_cutoff),
                quantiles_history_to_outcome=np.quantile(from_history, [.15, .35, .55, .75, .90]).round(1).tolist(),
                slack_days=round(slack), meets_grid_criterion=bool(len(deadlines) >= 3 and k >= 3 and per_cutoff >= 50))


if __name__ == '__main__':
    opts = dict(a.split('=', 1) for a in sys.argv[3:])
    r = propose(sys.argv[1], set(sys.argv[2].split(';')), **opts)
    for key, v in r.items():
        print(f"{key}: {v}")
