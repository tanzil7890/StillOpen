"""Two checks of study.py. Exits with an error if either fails.

    python censoring/check.py

1. Simulated cases with known outcome rates, cut off at a date: censor_aware and mature_only must recover the true
   probabilities, complete_only must overestimate them and open_as_negative must underestimate them.
2. On BPI Challenge 2012, study.py must give the same scores as the earlier pilot (pilot/results/).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study
from study import DAY, labels, marginal

HERE = os.path.dirname(os.path.abspath(__file__))


def simulated():
    """One history per case, at its start. Two competing outcomes with constant rates; a fifth of cases never end."""
    rng = np.random.default_rng(0)
    edges, n, tau, window = np.array([1.0, 3, 7, 14, 30]), 400_000, 200 * DAY, 45 * DAY
    rate = np.array([0.03, 0.05])  # per day
    start = rng.uniform(tau - window, tau, n)
    wait = rng.exponential(1 / rate.sum(), n) * DAY
    cause = np.where(rng.random(n) < 0.2, 0, 1 + (rng.random(n) < rate[1] / rate.sum()))
    t_term = np.where(cause > 0, start + wait, np.inf)
    truth = 0.8 * (rate / rate.sum())[None, :] * (1 - np.exp(-rate.sum() * edges))[:, None]  # [deadline, outcome]
    print("simulated cases, P(outcome 1 within 30 days), truth %.3f:" % truth[-1, 0])
    ok = True
    for treat, expect in (('complete_only', 'over'), ('open_as_negative', 'under'), ('mature_only', 'right'), ('censor_aware', 'right')):
        keep, outcome, obs = labels(start, t_term, cause, tau, treat, edges)
        est = marginal(outcome[keep], obs[keep], 1, len(edges), 3)[0, :, 1:]
        err = est - truth
        worst = float(np.abs(err).max())
        good = {'right': worst < 0.01, 'over': err[-1].min() > 0.02, 'under': err[-1].max() < -0.02}[expect]
        ok &= good
        print(f"  {treat:17s} estimate {est[-1, 0]:.3f}  largest error over deadlines and outcomes {worst:.3f}  "
              f"expected: {expect}  {'ok' if good else 'FAIL'}")
    return ok


def same_as_pilot():
    path = os.path.join(HERE, '..', 'pilot', 'results', 'pilot_2012-01-01.jsonl')
    if not os.path.exists(study.derived('bpi12')) or not os.path.exists(path):
        print("BPI 2012 or the pilot's results are not here: skipped")
        return True
    pilot = {r['train_weeks'] * 7: r for r in map(json.loads, open(path)) if 'train_weeks' in r}
    ok = True
    for r in study.Log('bpi12').run('2012-01-01', sorted(pilot), ('complete_only', 'open_as_negative', 'censor_aware'), n_boot=10):
        old = pilot[r['history_days']]
        for treat in ('complete_only', 'open_as_negative', 'censor_aware', 'no_features_censor_aware'):
            pairs = [(r[treat]['brier_ident'], old[treat]['brier_ident']), (r[treat]['brier_all_deadlines'], old[treat]['brier_all_horizons']),
                     (r[treat]['ece_any_ident'], old[treat]['ece_any_ident']), (r[treat]['pred_minus_actual_any'], old[treat]['pred_minus_actual_any'])]
            same = all(a == b for a, b in pairs)
            ok &= same
            print(f"  BPI 2012, {r['history_days']:>3.0f} days of history, {treat:24s} Brier {r[treat]['brier_ident']:.4f}  pilot {old[treat]['brier_ident']:.4f}  "
                  f"{'same' if same else 'DIFFERENT'}")
    return ok


if __name__ == '__main__':
    a = simulated()
    b = same_as_pilot()
    print("checks passed" if a and b else "CHECKS FAILED")
    sys.exit(0 if a and b else 1)
