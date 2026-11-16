"""Tests of the censoring study (experiments/censoring). Skipped where scikit-learn is not installed; the test
that reruns a log is skipped where that log has not been downloaded (python -I experiments/censoring/prepare.py)."""
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip('sklearn')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'experiments', 'censoring'))
import check  # noqa: E402
import study  # noqa: E402
from study import DAY, labels  # noqa: E402


def rows(treatment, started_days_ago, outcome_after_days):
    """Windows (1-based) and classes that a treatment gives the history at the start of one case, cutoff at day 0."""
    edges = np.array([3.0, 10.0, 20.0])
    tp = np.array([-started_days_ago * DAY])
    t_term = tp + outcome_after_days * DAY if outcome_after_days else np.array([np.inf])
    keep, outcome, obs = labels(tp, t_term, np.array([1]), 0.0, treatment, edges)
    if not keep[0]:
        return None
    return [(j + 1, int(outcome[0]) if j == obs[0] else 0) for j in range(obs[0] + 1)]


def test_figure_1_of_the_paper():
    """The five cases of Fig. 1: deadlines of 3, 10 and 20 days; C and E are open at the cutoff."""
    cases = {'A': (30, 5), 'B': (28, 17), 'C': (27, None), 'D': (7, 4), 'E': (6, None)}
    expected = {  # windows used; a window written (j, 1) holds the outcome
        'complete_only':    {'A': [(1, 0), (2, 1)], 'B': [(1, 0), (2, 0), (3, 1)], 'C': None, 'D': [(1, 0), (2, 1)], 'E': None},
        'open_as_negative': {'A': [(1, 0), (2, 1)], 'B': [(1, 0), (2, 0), (3, 1)], 'C': [(1, 0), (2, 0), (3, 0)],
                             'D': [(1, 0), (2, 1)], 'E': [(1, 0), (2, 0), (3, 0)]},
        'mature_only':      {'A': [(1, 0), (2, 1)], 'B': [(1, 0), (2, 0), (3, 1)], 'C': [(1, 0), (2, 0), (3, 0)], 'D': None, 'E': None},
        'censor_aware':     {'A': [(1, 0), (2, 1)], 'B': [(1, 0), (2, 0), (3, 1)], 'C': [(1, 0), (2, 0), (3, 0)],
                             'D': [(1, 0)], 'E': [(1, 0)]},  # the outcome of D, in window 2, is held back
    }
    for treatment, want in expected.items():
        for name, (ago, after) in cases.items():
            assert rows(treatment, ago, after) == want[name], (treatment, name)


def test_future_labels_uses_what_happened_after_the_cutoff():
    assert rows('future_labels', 6, 9) == [(1, 0), (2, 1)]   # open at the cutoff, outcome 3 days after it
    assert rows('censor_aware', 6, 9) == [(1, 0)]


def test_bias_of_each_treatment_on_simulated_cases():
    assert check.simulated()


def test_default_study_reproduces_the_stored_results():
    """The version-1 result file of the smallest log must come out again, apart from timings and version stamps."""
    if not os.path.exists(study.derived('helpdesk')):
        pytest.skip('the help desk log has not been downloaded')

    def clean(r):
        r = {k: v for k, v in r.items() if k not in ('versions', 'model')}
        for v in r.values():
            if isinstance(v, dict):
                v.pop('fit_s', None)
        return r

    with open(os.path.join(study.RESULTS, 'helpdesk.jsonl')) as f:
        stored = [clean(json.loads(line)) for line in f]
    log = study.Log('helpdesk')
    again = [clean(json.loads(json.dumps(r))) for cutoff in log.cfg['cutoffs'] for r in log.run(cutoff, log.cfg['history_days'])]
    assert again == stored


def test_pooled_statistics():
    import pooled
    # equal standard errors and clear heterogeneity: the pooled mean is the plain mean, and the Hartung-Knapp
    # interval is the ordinary t interval of that mean
    y, se = np.array([0.10, 0.30, 0.02, 0.45, 0.22, 0.38]), np.full(6, 0.01)
    r = pooled.random_effects(y, se)
    assert abs(r['mean'] - y.mean()) < 1e-12
    half = 2.5705818366147395 * y.std(ddof=1) / np.sqrt(6)   # t quantile, 5 degrees of freedom
    assert abs(r['ci'][0] - (y.mean() - half)) < 1e-9 and abs(r['ci'][1] - (y.mean() + half)) < 1e-9
    assert r['tau2'] > 0 and r['prediction'][0] < r['ci'][0] and r['prediction'][1] > r['ci'][1]
    # identical units: no between-unit variance, and the interval is not narrower than the fixed-effect one
    r = pooled.random_effects([0.2, 0.2, 0.2], [0.1, 0.1, 0.1])
    assert r['tau2'] == 0 and abs(r['mean'] - 0.2) < 1e-12 and r['ci'][1] - r['ci'][0] >= 2 * 1.96 * 0.1 / np.sqrt(3)
    assert pooled.sign_test([0.1] * 6) == (6, 0, 0.03125)
    assert pooled.sign_test([0.1, 0.2, -0.1, 0.3, 0.0])[:2] == (3, 1)
    assert pooled.holm([0.01, 0.04, None, 0.03]) == [0.03, 0.06, None, 0.06]


def test_competing_methods_on_simulated_cases():
    """Same simulation as check.py, with features that say nothing: ipcw must recover the true probabilities by every
    deadline; pseudo-labelling from a model of finished cases must stay too high."""
    import compete
    rng = np.random.default_rng(1)
    edges, n, tau, window = np.array([1.0, 3, 7, 14, 30]), 120_000, 200 * DAY, 45 * DAY
    rate = np.array([0.03, 0.05])
    start = rng.uniform(tau - window, tau, n)
    cause = np.where(rng.random(n) < 0.2, 0, 1 + (rng.random(n) < rate[1] / rate.sum()))
    t_term = np.where(cause > 0, start + rng.exponential(1 / rate.sum(), n) * DAY, np.inf)
    truth = 0.8 * (rate / rate.sum())[None, :] * (1 - np.exp(-rate.sum() * edges))[:, None]     # [deadline, outcome]
    X, Xte = np.zeros((n, 8)), np.zeros((1, 8))
    ipcw = compete.ipcw(X, start, t_term, cause, tau, edges, Xte, 3)[0, :, 1:]
    assert np.abs(ipcw - truth).max() < 0.01
    pseudo = compete.pseudo(X, start, t_term, cause, tau, edges, Xte, 3)[0, :, 1:]
    assert (pseudo[-1] - truth[-1]).min() > 0.02


def test_weighting_needs_follow_up_by_time_in_case():
    """Cases with a second history ten days in: the window rule and ipcw_x recover the truth for both histories;
    ipcw with one pooled follow-up share does not."""
    import compete
    from study import fit_predict
    rng = np.random.default_rng(2)
    edges, n, tau, window, late = np.array([1.0, 3, 7, 14, 30]), 150_000, 200 * DAY, 45 * DAY, 10.0
    rate = np.array([0.03, 0.05])
    start = rng.uniform(tau - window, tau, n)
    never, kind = rng.random(n) < 0.2, 1 + (rng.random(n) < rate[1] / rate.sum())
    d0 = np.where(never, np.inf, rng.exponential(1 / rate.sum(), n))
    b = np.flatnonzero((d0 > late) & (start + late * DAY <= tau))    # still without outcome ten days in, before the cutoff
    tp = np.concatenate([start, start[b] + late * DAY])
    t_term = np.concatenate([start + d0 * DAY, start[b] + d0[b] * DAY])
    cause = np.concatenate([np.where(never, 0, kind), np.where(never[b], 0, kind[b])])
    X = np.zeros((len(tp), 8)); X[n:, 1] = np.log1p(late * 24)
    Xte = np.zeros((2, 8)); Xte[1, 1] = np.log1p(late * 24)
    resolved_late = 1 - 0.2 / (0.2 + 0.8 * np.exp(-rate.sum() * late))
    truth = np.stack([0.8 * (1 - np.exp(-rate.sum() * edges)), resolved_late * (1 - np.exp(-rate.sum() * edges))])
    keep, outcome, obs = labels(tp, t_term, cause, tau, 'censor_aware', edges)
    by_window = 1 - fit_predict(X[keep], outcome[keep], obs[keep], Xte, 5, 3)[:, :, 0]
    pooled_share = 1 - compete.ipcw(X, tp, t_term, cause, tau, edges, Xte, 3)[:, :, 0]
    grouped = 1 - compete.ipcw(X, tp, t_term, cause, tau, edges, Xte, 3, groups=10)[:, :, 0]
    assert np.abs(by_window - truth).max() < 0.01
    assert np.abs(grouped - truth).max() < 0.015
    assert np.abs(pooled_share - truth).max() > 0.08


def test_linear_competing_methods_on_simulated_cases():
    """A feature doubles the rate of the first outcome. The delayed-feedback model and Cox regression must recover
    the true probabilities for both values of the feature; positive-unlabelled learning stays near them."""
    import linear
    rng = np.random.default_rng(3)
    edges, n, tau, window = np.array([1.0, 3, 7, 14, 30]), 60_000, 200 * DAY, 45 * DAY
    x = rng.integers(0, 2, n)
    h1, h2 = 0.03 * 2.0 ** x, 0.05
    start = rng.uniform(tau - window, tau, n)
    d = rng.exponential(1 / (h1 + h2))
    cause = 1 + (rng.random(n) < h2 / (h1 + h2))
    seen = start + d * DAY <= tau
    delay, kind, follow = np.where(seen, d, np.inf), np.where(seen, cause, 0), (tau - start) / DAY
    X = np.zeros((n, 11)); X[:, 0] = x
    Xte = np.zeros((2, 11)); Xte[1, 0] = 1
    truth = np.stack([np.stack([hk / (a + h2) * (1 - np.exp(-(a + h2) * edges)) for hk in (a, h2)], 1) for a in (0.03, 0.06)])
    Ftr, Fte = linear.design(X, Xte)
    for name, tolerance in (('dfm', 0.015), ('cox', 0.015), ('pu', 0.04)):
        p = getattr(linear, name)(Ftr, Fte, delay, kind, follow, edges, 2)
        assert np.allclose(p.sum(-1), 1) and np.abs(p[:, :, 1:] - truth).max() < tolerance, name


def test_calibration_measures():
    """Forecasts that are right on average have slope 1, intercept 0 and no reliability loss, and the three parts
    add up to the Brier score. Forecasts pushed towards 0 and 1 have a slope below 1 and a reliability loss."""
    import decisions
    rng = np.random.default_rng(5)
    p = rng.integers(0, 10, 200_000) / 10 + 0.05                      # one value per bin, so the split is exact
    y = (rng.random(len(p)) < p).astype(float)
    good = decisions.calibration(p, y)
    assert abs(good['slope'] - 1) < 0.03 and abs(good['intercept']) < 0.03 and good['reliability'] < 0.001
    assert abs(good['reliability'] - good['resolution'] + good['uncertainty'] - ((p - y) ** 2).mean()) < 0.001
    sharp = 1 / (1 + ((1 - p) / p) ** 2)                              # twice the log odds
    bad = decisions.calibration(sharp, y)
    assert abs(bad['slope'] - 0.5) < 0.03 and bad['reliability'] > 0.005


def test_tuning_rows_and_choice_of_rounds():
    """tune.rows builds the rows of study.person_period and keeps the case of each row; tune.choose returns a number
    of rounds from its grid, and nothing where the held-out loss cannot be measured."""
    import tune
    rng = np.random.default_rng(7)
    n, H = 3000, 3
    X = rng.normal(size=(n, 6))
    X[:, 3:5] = rng.integers(0, 4, (n, 2))
    obs, case = rng.integers(0, H, n), np.arange(n) // 3
    outcome = (rng.random(n) < 1 / (1 + np.exp(-2 * X[:, 0]))) * rng.integers(1, 3, n)
    x, y, c = tune.rows(X, case, outcome, obs, H)
    xs, ys = study.person_period(X, outcome, obs, H)
    assert np.array_equal(x, xs) and np.array_equal(y, ys) and np.array_equal(c[:n], case) and len(c) == len(x)
    held = c % 5 == 0
    assert tune.choose((x[~held], y[~held]), (x[held], y[held]), 3) in tune.GRID
    assert tune.choose((x[~held], np.zeros_like(y[~held])), (x[held], y[held]), 3) is None
    assert tune.choose((x[~held], y[~held]), (x[:0], y[:0]), 3) is None


def test_pooled_statistics_flag_a_unit_without_variance():
    """A unit whose standard error is zero carries the random-effects estimate when the other units are within their
    own standard errors of it (no between-unit variance is found); the plain mean does not follow it."""
    import pooled
    y, se = [0.0, 0.05, 0.03, 0.08, 0.04], [0.0, 0.06, 0.06, 0.06, 0.06]
    assert abs(pooled.random_effects(y, se)['mean']) < 1e-6
    mean, (low, high) = pooled.plain_mean(y)
    assert abs(mean - 0.04) < 1e-12 and low < mean < high


def test_boosted_survival_trees_on_simulated_cases():
    """A feature doubles the rate of the first outcome. Cox regression with boosted trees must recover the true
    probabilities for both values of the feature, from cases of which many are still open at the cutoff."""
    import survival_trees
    rng = np.random.default_rng(3)
    edges, n, tau, window = np.array([1.0, 3, 7, 14, 30]), 30_000, 200 * DAY, 45 * DAY
    x = rng.integers(0, 2, n)
    h1, h2 = 0.03 * 2.0 ** x, 0.05
    start = rng.uniform(tau - window, tau, n)
    d = rng.exponential(1 / (h1 + h2))
    cause = 1 + (rng.random(n) < h2 / (h1 + h2))
    seen = start + d * DAY <= tau
    delay, kind, follow = np.where(seen, d, np.inf), np.where(seen, cause, 0), (tau - start) / DAY
    X = np.zeros((n, 11)); X[:, 0] = x
    Xte = np.zeros((2, 11)); Xte[1, 0] = 1
    truth = np.stack([np.stack([hk / (a + h2) * (1 - np.exp(-(a + h2) * edges)) for hk in (a, h2)], 1) for a in (0.03, 0.06)])
    p = survival_trees.cox_trees(X, Xte, delay, kind, follow, edges, 2)
    assert np.allclose(p.sum(-1), 1) and np.abs(p[:, :, 1:] - truth).max() < 0.02
    first, last = survival_trees.tie_groups(np.array([1.0, 2.0, 2.0, 2.0, 5.0]))
    assert first.tolist() == [0, 1, 1, 1, 4] and last.tolist() == [0, 3, 3, 3, 4]


def test_reliability_table():
    """Ten bins of the forecast with count, mean forecast and share that happened; an empty bin is marked."""
    import calibration
    p, y = np.array([0.05, 0.05, 0.15, 0.95, 0.95, 0.95, 1.0]), np.array([0, 1, 0, 1, 1, 0, 1.0])
    bins = calibration.table(p, y)
    assert bins[0] == [2, 0.05, 0.5] and bins[1] == [1, 0.15, 0.0] and bins[2] == [0, None, None]
    assert bins[9][0] == 4 and abs(bins[9][1] - 0.9625) < 1e-9 and bins[9][2] == 0.75


def test_benchmark_encoding_of_the_replication():
    """The aggregation encoding of the outcome benchmark, on two hand-made cases: one row per prefix, indicators of
    the static attribute, the static number, counts of each activity, and mean, maximum, minimum, sum and standard
    deviation of the event number."""
    import anchor
    cfg = dict(static_cat=['s'], static_num=['a'], event_cat=['Activity'], event_num=['x'])
    make = lambda cid, label, events: [{'Case ID': cid, 'label': label, 's': 'u', 'a': '5', 'Activity': act, 'x': str(x)} for act, x in events]
    cases = [make('c1', 'deviant', [('A', 1), ('B', 3), ('A', 8)]), make('c2', 'regular', [('B', 2)])]
    X, y, length, case, vocab = anchor.encode(cases, cfg, 3)
    want = np.array([[1, 5, 1, 0, 1, 1, 1, 1, 0], [1, 5, 1, 1, 2, 3, 1, 4, np.std([1, 3], ddof=1)],
                     [1, 5, 2, 1, 4, 8, 1, 12, np.std([1, 3, 8], ddof=1)], [1, 5, 0, 1, 2, 2, 2, 2, 0]])
    assert np.allclose(X, want, atol=1e-5) and y.tolist() == [1, 1, 1, 0] and length.tolist() == [1, 2, 3, 1] and case.tolist() == [0, 0, 0, 1]
    # a value not seen in training gets no column, and prefixes are cut at the maximum length
    X2, _, length2, _, _ = anchor.encode([make('c3', 'regular', [('C', 4), ('A', 6), ('A', 7)])], cfg, 2, vocab)
    assert X2.shape == (2, 9) and length2.tolist() == [1, 2] and X2[:, 2:4].tolist() == [[0, 0], [1, 0]]


def test_the_check_script_pools_like_the_summary_and_ignores_run_times():
    """verify.py recomputes the pooled estimate with its own code; it has to agree with pooled.py on any numbers."""
    import pooled
    import verify
    y, se = [0.30, 0.10, 0.55, 0.02, 0.21], [0.05, 0.02, 0.20, 0.01, 0.08]
    assert verify.random_effects(y, se) == pytest.approx(pooled.random_effects(y, se)['mean'])
    line = {'cutoff': '2020-01-01', 'history_days': 14, 'versions': {'numpy': '2'}, 'seconds': 3.0,
            'a': {'brier_ident': 0.25, 'fit_s': 1.5, 'ci95': [0.1, 0.2]}, 'flag': True}
    assert dict(verify.flat(line)) == {'history_days': 14, 'a.brier_ident': 0.25, 'a.ci95[0]': 0.1, 'a.ci95[1]': 0.2}
