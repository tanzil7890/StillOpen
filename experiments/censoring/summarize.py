"""Tables from censoring/results/*.jsonl.

    python censoring/summarize.py                  every log that has results
    python censoring/summarize.py traffic
    python censoring/summarize.py --model gru      the results of another model (trees150, logit, gru)

For each log and history length, averaged over the cutoff dates: the Brier score of each treatment, and each
treatment's difference from censor_aware. "worse" and "better" count the cutoff dates at which the 95% interval of
that difference (resampling test cases) lies wholly above or below zero. The last table adds the forecast bias at
the longest deadline: predicted minus actual share of test histories with any outcome by then.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS
from study import RESULTS, TREATMENTS

OTHERS = [t for t in TREATMENTS if t != 'censor_aware']


def path(name, model):
    return os.path.join(RESULTS, name + ('' if model == 'trees' else '.' + model) + '.jsonl')


def read(name, model='trees'):
    with open(path(name, model)) as f:
        return [json.loads(line) for line in f]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def cell(x, fmt='{:.3f}'):
    return '    -' if x is None else fmt.format(x)


def table(name, rows):
    cfg = DATASETS[name]
    by_w = defaultdict(list)
    for r in rows:
        by_w[r['history_days']].append(r)
    print(f"\n{name}: {cfg['title']}")
    print(f"  deadlines {cfg['deadlines']} days, {len(cfg['cutoffs'])} cutoff dates, test cases per cutoff "
          f"{min(r['test_cases'] for r in rows)} to {max(r['test_cases'] for r in rows)}")
    print(f"  {'history':>8s} {'open':>5s} | " + ' '.join(f"{t[:13]:>13s}" for t in TREATMENTS) + ' | '
          + ' '.join(f"{t[:13] + ' - c_a':>19s} {'worse/better':>12s}" for t in OTHERS))
    tally = defaultdict(lambda: [0, 0, 0])  # treatment -> [worse, better, settings]
    for w in sorted(by_w):
        rs = by_w[w]
        line = f"  {w:>7.0f}d {mean([r['share_of_train_histories_from_open_cases'] for r in rs]):5.2f} | "
        line += ' '.join(f"{cell(mean([r[t]['brier_ident'] if r.get(t) else None for r in rs])):>13s}" for t in TREATMENTS) + ' | '
        for t in OTHERS:
            ds = [r['brier_minus_censor_aware'][t] for r in rs if t in r['brier_minus_censor_aware']]
            worse, better = sum(d['ci95'][0] > 0 for d in ds), sum(d['ci95'][1] < 0 for d in ds)
            tally[t][0] += worse; tally[t][1] += better; tally[t][2] += len(ds)
            line += f"{cell(mean([d['mean'] for d in ds]), '{:+.4f}'):>19s} {f'{worse}/{better} of {len(ds)}':>12s} "
        print(line)
    print(f"  bias at {cfg['deadlines'][-1]} days (predicted - actual share with any outcome), by history length:")
    for t in TREATMENTS:
        print(f"    {t:17s} " + ' '.join(f"{cell(mean([r[t]['pred_minus_actual_any'][-1] if r.get(t) else None for r in by_w[w]]), '{:+.3f}'):>7s}"
                                         for w in sorted(by_w)))
    return tally


def main():
    argv, model = sys.argv[1:], 'trees'
    if '--model' in argv:
        i = argv.index('--model')
        model = argv[i + 1]
        del argv[i:i + 2]
    names = argv or [n for n in DATASETS if os.path.exists(path(n, model))]
    total = defaultdict(lambda: [0, 0, 0])
    print(f"model: {model}")
    for name in names:
        for t, v in table(name, read(name, model)).items():
            total[t] = [a + b for a, b in zip(total[t], v)]
    print(f"\nall logs: settings (log, cutoff date, history length) where a treatment is significantly worse / better "
          f"than censor_aware")
    for t in OTHERS:
        w, b, n = total[t]
        print(f"  {t:17s} worse in {w:3d}, better in {b:3d}, of {n}")


if __name__ == '__main__':
    main()
