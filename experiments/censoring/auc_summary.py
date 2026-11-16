"""AUC next to the Brier score, from the stored results of study.py (no model is run).

    python censoring/auc_summary.py             main model;  --model logit or gru for the others

The outcome benchmark of the field reports AUC. Every result line holds the AUC of "any outcome by the middle
deadline" over the test histories. It is used here where that deadline is shorter than the history, and the Brier
score is taken on the same settings. Shown per treatment and history length: the mean over logs, and the
difference to the window rule with a 95% interval over logs (t distribution, the log as the unit), next to the
same for the Brier score.
"""
import sys

import numpy as np

import pooled
from datasets import DATASETS
from decisions import interval
from study import TREATMENTS

BASE = 'censor_aware'


def per_log(rows, treat, rank, key):
    """Mean over the log's cutoffs, on the settings where the middle deadline is shorter than the history and both
    the treatment and the window rule have an AUC. The Brier score is taken on these same settings, so that the two
    columns can be compared (until 2026-11-15 it was taken on all settings, which differed at the shortest history)."""
    use = [(r[treat][key], r[BASE][key]) for r in rows if r['rank'] == rank and r.get(treat) and r.get(BASE)
           and r[treat].get('auc_any_mid') is not None and r[BASE].get('auc_any_mid') is not None
           and len(r['identifiable_deadlines']) > len(DATASETS[r['log']]['deadlines']) // 2]
    return (float(np.mean([a for a, _ in use])), float(np.mean([b for _, b in use]))) if use else None


def main():
    model = sys.argv[sys.argv.index('--model') + 1] if '--model' in sys.argv else 'trees'
    data = {log: pooled.read(log, model) for log in DATASETS}
    data = {log: rows for log, rows in data.items() if rows}
    print(f"model {model}; {len(data)} logs. Difference to the window rule: mean over logs (95% interval). AUC: higher is better; Brier: lower is better.")
    for key, name, scale in (('auc_any_mid', 'AUC, any outcome by the middle deadline', 1.0), ('brier_ident', 'Brier score', 1.0)):
        print(f"\n{name}\n  {'history':12s} {'window rule':>12s} " + ' '.join(f"{t:>34s}" for t in TREATMENTS[:-1]))
        for rank in range(4):
            cells, base = [], []
            for t in TREATMENTS[:-1]:
                pairs = [p for p in (per_log(rows, t, rank, key) for rows in data.values()) if p]
                if not pairs:
                    cells.append('-')
                    continue
                d = [100 * (a - b) for a, b in pairs]
                cells.append(f"{np.mean([a for a, _ in pairs]):.3f}, {interval(d).strip()} ({len(pairs)} logs)")
                base = [b for _, b in pairs] if len(pairs) >= len(base) else base
            print(f"  {pooled.RANKS[rank]:12s} {np.mean(base):12.3f} " + ' '.join(f"{c:>34s}" for c in cells))
    print("\nDifferences are in hundredths (AUC points, Brier points).")


if __name__ == '__main__':
    main()
