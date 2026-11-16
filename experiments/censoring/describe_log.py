"""What is in an event log: its size, time span, activities, how cases end and how long they take.

    python -I describe_log.py data/traffic_raw/Road_Traffic_Fine_Management_Process.xes.gz [Activity ...]

Use it to choose the outcome activities and the deadlines for a new log (see datasets.py). With activity names after
the path it also prints, for cases that contain one of them, the days from the first event to the first such event.
"""
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logs import read_log

DAY = 86400.0


def day(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d')


def main():
    path, terminal = sys.argv[1], set(sys.argv[2:])
    acts, last, first, life, n_ev, starts, ends, lens, to_term, which = Counter(), Counter(), Counter(), Counter(), 0, [], [], [], [], Counter()
    case_keys, event_keys, example = Counter(), Counter(), None
    for case, events in read_log(path):
        if not events:
            continue
        example = example or (case, events[0][3])
        case_keys.update(case.keys())
        for name, lc, t, ev in events:
            acts[name] += 1
            life[lc] += 1
            event_keys.update(ev.keys())
        first[events[0][0]] += 1
        last[events[-1][0]] += 1
        n_ev += len(events)
        starts.append(events[0][2]); ends.append(events[-1][2]); lens.append(len(events))
        hit = next(((name, t) for name, _, t, _ in events if name in terminal), None)
        which[hit[0] if hit else '(none)'] += 1
        if hit:
            to_term.append((hit[1] - events[0][2]) / DAY)
    starts, ends, lens = np.array(starts), np.array(ends), np.array(lens)
    q = [.1, .25, .5, .75, .9, .95, .99]
    print(f"cases {len(starts)}  events {n_ev}  activities {len(acts)}")
    print(f"first case starts {day(starts.min())}  last case starts {day(starts.max())}  last event {day(ends.max())}")
    print("case starts per year:", dict(sorted(Counter(datetime.fromtimestamp(t, timezone.utc).year for t in starts).items())))
    print("events per case, quantiles 10/25/50/75/90/95/99:", np.quantile(lens, q).round(0).tolist(), "max", int(lens.max()))
    print("days from first to last event, same quantiles:", np.quantile((ends - starts) / DAY, q).round(1).tolist())
    print("lifecycle values:", dict(life.most_common(8)))
    print("\nactivity: events / cases that start with it / cases that end with it")
    for name, n in acts.most_common(45):
        print(f"  {name[:60]:60s} {n:8d} {first[name]:7d} {last[name]:7d}")
    print("\ncase attributes:", {k: example[0].get(k) for k in list(case_keys)[:20]})
    print("event attributes (first event of the first case):", {k: example[1].get(k) for k in list(event_keys)[:25]})
    if terminal:
        d = np.array(to_term)
        print("\nfirst outcome activity per case:", dict(which.most_common()))
        print("days from first event to the outcome, quantiles 10/25/50/75/90/95/99:", np.quantile(d, q).round(1).tolist())


if __name__ == '__main__':
    main()
