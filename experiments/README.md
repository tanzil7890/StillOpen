# Experiments

The experiments of this repository, on public event logs. Nothing here changes the `temporal_rlcd` package.

The first log is BPI Challenge 2012: 13,087 loan applications from a Dutch bank, each of which ends approved,
declined or cancelled. The task is to forecast, part-way through an application, which of those has happened
by a deadline of 1, 3, 7, 14 or 30 days. The `censoring/` study asks the same kind of question on ten more
public logs.

## Setup

```bash
bash experiments/setup.sh
```

This needs [uv](https://docs.astral.sh/uv/). It creates `experiments/.venv` and downloads and parses the BPI
Challenge 2012 log from 4TU.ResearchData for the pilot. The logs of the `censoring/` study are downloaded by
`censoring/prepare.py`. Every log has its own terms of use; check them before redistributing it.

## How every experiment is evaluated

A cutoff date stands in for "today". Training sees only events up to the cutoff. Test cases start after the
cutoff and are scored against what actually happened, which the rest of the log reveals.

Applications still open at the cutoff are the difficulty. The rule used here: a (case history, deadline) pair
may be used for training only if the whole window up to the deadline had passed by the cutoff, whatever
happened in it. Choosing on the window and never on the outcome keeps the training labels unbiased.

## `pilot/`: does the handling of open cases matter?

Boosted trees trained three ways (drop open cases, count them as "no outcome", censoring-aware), plus the
repository's Transformer on the same data. Results are in `pilot/results/`.

```bash
cd experiments
.venv/bin/python pilot/pilot.py data/derived/bpi12.pkl 2012-01-01 2,4,8,13
.venv/bin/python pilot/transformer_pilot.py .. pilot data/derived/bpi12.pkl 2012-01-01 4,13
```

`pilot/brier_bias.py` is the controlled test showing that `brier_loss` in `temporal_rlcd/model.py` is biased
when cases are censored.

## `censoring/`: the same question on five public logs

The pilot's experiment, generalised and run on five event logs from four organisations. For every log, cutoff
date and length of training history, boosted trees are trained four ways that differ only in what they do
with the cases still open at the cutoff:

| Treatment | What it does with open cases |
| --- | --- |
| `complete_only` | Drops them: trains on finished cases, the usual practice |
| `open_as_negative` | Keeps them, labelled "no outcome in any window" |
| `mature_only` | Keeps a case history only if all of its deadline windows had ended by the cutoff |
| `censor_aware` | Keeps a (case history, window) pair if that window had ended by the cutoff |

| Log | Cases | Outcomes | Deadlines (days) | Cutoff dates |
| --- | --- | --- | --- | --- |
| `bpi12`: loan applications, Dutch bank | 13,087 | approved, declined, cancelled | 1, 3, 7, 14, 30 | 3 |
| `bpi17`: loan applications, same bank, new system | 31,509 | approved, declined, cancelled | 1, 3, 7, 14, 30 | 6 |
| `traffic`: road traffic fines, Italian police | 150,370 | first payment, credit collection | 7, 30, 90, 180, 365 | 6 |
| `bpi20`: travel expense claims, Dutch university | 10,500 | paid | 3, 5, 7, 10, 14 | 6 |
| `helpdesk`: support tickets, Italian software company | 4,580 | resolved | 1, 3, 7, 14, 30 | 6 |

Each log has four history lengths, about 0.5, 1, 2 and 3 times its longest deadline, which gives 108
settings. The outcomes, deadlines, cutoff dates and history lengths are in `censoring/datasets.py`; they were
fixed from the shape of each log before any model was trained on it. BPI Challenge 2013 (incidents) was looked
at and left out, because 99% of its outcomes fall within two months.

```bash
cd experiments
.venv/bin/python -I censoring/prepare.py      # download the logs (about 40 MB), check them, parse them
.venv/bin/python censoring/check.py           # two checks, see below
.venv/bin/python censoring/study.py traffic   # one log; one result line per setting in censoring/results/
.venv/bin/python censoring/summarize.py       # the tables
.venv/bin/python censoring/study.py traffic --model gru          # another model: logit, gru or trees150
.venv/bin/python censoring/summarize.py --model gru
.venv/bin/python censoring/study.py traffic --treatments future_labels,censor_aware \
  --out censoring/results/traffic.future.jsonl                   # the reference with labels from after the cutoff
.venv/bin/python censoring/paper_inputs.py    # tables, figure data and quoted numbers of the paper
```

On a laptop `bpi17` takes about 30 minutes, `bpi12` and `traffic` about 6 minutes each, and the other two under
a minute. The other models (`models.py`) take about as long, and `trees150` about 40 minutes on `bpi17`. `describe_log.py` prints what is needed to add another log to `datasets.py`. The BPI Challenge 2020
log is licensed CC BY-NC 4.0 (no commercial use); the others are under the 4TU General Terms of Use.

`check.py` does two things. On simulated cases with known outcome rates it confirms that `censor_aware` and
`mature_only` recover the true probabilities, that `complete_only` overestimates them and that
`open_as_negative` underestimates them. On BPI Challenge 2012 it confirms that `study.py` gives exactly the
scores of the pilot.

### Results

Run on 2026-11-13. The score is the Brier score averaged over the deadlines shorter than the history, which
are the ones a training window can observe. A treatment counts as worse (or better) than `censor_aware` in a
setting when the 95% interval of the difference, from resampling test cases, lies wholly above (or below) zero.

| Log | `complete_only` worse | `open_as_negative` worse / better | `mature_only` worse |
| --- | --- | --- | --- |
| `bpi12` | 12 of 12 | 6 / 0 of 12 | 3 of 6 |
| `bpi17` | 22 of 24 | 19 / 2 of 24 | 11 of 12 |
| `traffic` | 24 of 24 | 12 / 4 of 24 | 4 of 12 |
| `bpi20` | 21 of 23 | 19 / 3 of 24 | 6 of 12 |
| `helpdesk` | 9 of 23 | 12 / 1 of 24 | 5 of 12 |
| All | 88 of 106 | 68 / 10 of 108 | 29 of 54 |

`complete_only` and `mature_only` are never significantly better than `censor_aware`. `mature_only` has no
training data at all in the other 54 settings, where the history is no longer than the longest deadline, and
`complete_only` has none in two.

Brier score minus that of `censor_aware`, averaged over the settings at each history length (27, or fewer
where a treatment had no data; median in brackets where the mean is dominated by one log):

| History, relative to the longest deadline | Open cases among training histories | `complete_only` | `open_as_negative` | `mature_only` |
| --- | --- | --- | --- | --- |
| about half | 77% | +0.494 (+0.357) | +0.015 | no data |
| about equal | 59% | +0.220 (+0.115) | +0.047 | no data |
| about twice | 37% | +0.127 (+0.023) | +0.019 | +0.012 |
| about three times | 25% | +0.120 (+0.013) | +0.009 | +0.006 |

Forecast bias at the longest deadline with the longest history: predicted minus actual share of test
histories with any outcome by then.

| Log | `complete_only` | `open_as_negative` | `mature_only` | `censor_aware` |
| --- | --- | --- | --- | --- |
| `bpi12` | +0.070 | -0.078 | +0.031 | +0.028 |
| `bpi17` | +0.056 | -0.107 | +0.002 | +0.001 |
| `traffic` | +0.572 | -0.096 | -0.018 | -0.014 |
| `bpi20` | +0.103 | -0.115 | +0.035 | +0.032 |
| `helpdesk` | +0.016 | -0.091 | -0.005 | -0.007 |

What this does and does not show:

- Training on finished cases only is the clear loser. Its loss grows with the share of open cases (rank
  correlation 0.82 over the 106 settings). On the traffic fines it roughly triples the Brier score even with
  three years of history, because fines that end in credit collection take over a year and so are missing
  from the finished cases. With a long history on the other four logs the loss is small, 1% to 7%.
- Counting open cases as "no outcome" costs little accuracy on average but under-predicts: by 8 to 11 points
  at the longest deadline even with the longest history.
- It is significantly better than `censor_aware` in 10 settings, 8 of them with a history no longer than the
  longest deadline, where `censor_aware` has few rows for the later windows. In 8 of the 10, `censor_aware`
  predicted more outcomes than followed the cutoff (averaged over the deadlines it can observe), which favours
  a method that under-predicts; two of the 10 also had very little training data (57 and 240 cases).
- `mature_only` is unbiased and, with a history of three times the longest deadline, nearly as good as
  `censor_aware` (+0.006). The gain of `censor_aware` is that it also works with a shorter history, where
  `mature_only` has little or no data.
- No treatment can forecast a deadline that is longer than its history: there `censor_aware` under-predicts
  too, and the score above leaves those deadlines out.
- Limits: fixed model settings; settings within a log share data, so the counts are not independent tests;
  `bpi20` and `helpdesk` have 78 to 745 test cases per cutoff; all five logs are public research logs, two
  of them from the same bank.

### Other models and the reference with labels from the future

The tables above are for boosted trees with 60 rounds. The same 108 settings with three other models, each
treatment against `censor_aware` (significantly worse / better / settings in which it could be trained):

| Model | `complete_only` | `open_as_negative` | `mature_only` |
| --- | --- | --- | --- |
| Boosted trees, 60 rounds | 88 / 0 / 106 | 68 / 10 / 108 | 29 / 0 / 54 |
| Boosted trees, 150 rounds | 88 / 0 / 106 | 68 / 9 / 108 | 33 / 2 / 54 |
| Logistic regression (`logit`) | 82 / 1 / 106 | 78 / 15 / 108 | 31 / 3 / 54 |
| Recurrent network (`gru`) | 85 / 1 / 106 | 57 / 18 / 108 | 29 / 4 / 54 |

The order of the treatments is the same for every model, and for every model and log `complete_only`
over-predicts and `open_as_negative` under-predicts at the longest deadline. The other models are less
clear-cut than the trees: with the recurrent network and the shortest history, `open_as_negative` is on
average slightly better than `censor_aware` (by 0.010), because `censor_aware` then has few rows for the later
windows.

`future_labels` is a reference, not a usable treatment: it labels every training history with what happened
after the cutoff as well, which is what a train/test split by case start date does when it keeps the cases
that are open at the split date. Mean Brier score over all logs (boosted trees, 60 rounds):

| History, relative to the longest deadline | `future_labels` | `censor_aware` | `complete_only` |
| --- | --- | --- | --- |
| about half | 0.291 | 0.329 | 0.815 |
| about equal | 0.307 | 0.316 | 0.536 |
| about twice | 0.306 | 0.309 | 0.436 |
| about three times | 0.302 | 0.304 | 0.423 |

Once the history is at least as long as the longest deadline, `censor_aware` is within 0.001 to 0.009 of
what labels from the future would give.

### Phase B: more logs, pooled statistics, competing methods

Work after version 1, on the branch `phase-b`. Its rules are in `censoring/PROTOCOL_B.md`, written before any
model was trained on a new log or any competing method was run. Version 1's result files are not touched, and
the paper's tables are still built from its five logs only.

```bash
cd experiments
.venv/bin/python -I censoring/design.py <log file> "<outcome activity>"   # the grid of a new log, by rule
.venv/bin/python -I censoring/prepare.py bpi14 bpi20r bpi15_1 bpi15_3 bpi15_4 bpi15_5
.venv/bin/python censoring/study.py bpi14        # as before; also --model logit or gru
.venv/bin/python censoring/pooled.py             # pooled over all logs; --by org, --phase 1, --model gru
.venv/bin/python censoring/compete.py bpi14      # the competing methods; --methods ipcw_x for the added one
.venv/bin/python censoring/pooled.py --methods
.venv/bin/python censoring/ipcw_diagnosis.py     # why the weighting fails on the traffic fines
.venv/bin/python censoring/compete_linear.py bpi14    # delayed-feedback model, Cox regression, positive-unlabelled
.venv/bin/python censoring/pooled.py --model linear
.venv/bin/python censoring/tune.py bpi14              # rounds chosen on held-out cases
.venv/bin/python censoring/pooled.py --model tuned
.venv/bin/python censoring/decisions.py bpi14         # planning error, calibration, value; --replay for monthly cutoffs
.venv/bin/python censoring/decisions.py --summary
```

**Logs.** 20 further public logs were examined. Six meet the criteria; with the five of version 1 that makes
eleven logs from nine organisations. The other fourteen are listed in `datasets.py` (`REJECTED`) with the
criterion that failed; BPI Challenge 2018 and 2019, the two largest, do not fit the grid because their
histories are too long for the time they cover.

| Log | Cases | Outcome | Deadlines (days) | Test cases per cutoff |
| --- | --- | --- | --- | --- |
| `bpi14`: IT incidents, service desk of a Dutch bank | 46,616 | closed | 1, 5, 14 | 4,173 to 6,075 |
| `bpi20r`: requests for payment, the university of `bpi20` | 6,886 | paid | 3, 5, 7, 10, 21 | 176 to 405 |
| `bpi15_1`: building permits, Dutch municipality 1 | 1,199 | decision sent | 1, 7, 21, 45, 90 | 49 to 86 |
| `bpi15_3`: building permits, municipality 3 | 1,409 | decision sent | 1, 2, 7, 30, 60 | 54 to 113 |
| `bpi15_4`: building permits, municipality 4 | 1,053 | decision sent | 1, 5, 14, 30, 60 | 49 to 75 |
| `bpi15_5`: building permits, municipality 5 | 1,156 | decision sent | 2, 5, 10, 45, 60 | 50 to 93 |

The outcome events were chosen from the activity lists; deadlines, history lengths, test windows and cutoff
dates are the output of `design.py`. The four permit logs are small, so their intervals are wide.

**Pooled result, boosted trees, eleven logs.** The unit is the log. The number is the Brier score relative to
`censor_aware` (+86% means 86% higher), pooled by a random-effects analysis, with its 95% interval; "logs" is
how many of the eleven have a higher score than `censor_aware`.

| Treatment | History about half the longest deadline | about equal | about twice | about three times |
| --- | --- | --- | --- | --- |
| `complete_only` | +86% (42 to 143), 11 logs | +35% (11 to 64), 11 logs | +16% (2 to 33), 11 logs | +14% (-1 to 31), 11 logs |
| `open_as_negative` | +2.0% (-0.2 to 4.2), 9 logs | +5.4% (-0.3 to 11.6), 11 logs | +4.8% (1.8 to 7.9), 11 logs | +2.4% (0.5 to 4.4), 11 logs |
| `mature_only` | no data | no data | +4.2% (1.4 to 7.1), 11 logs | +2.2% (0.1 to 4.3), 11 logs |

- Where all eleven logs agree, the sign test (each log one vote) gives p = 0.001. With the five logs of
  version 1 alone it cannot go below 0.06, which is why the logs were added.
- The intervals are wide because the logs differ a great deal: for `complete_only` with the longest history
  the traffic fines are near +200%, four logs are within 5%, and the other six are between 7% and 39%. After Holm's correction over
  the ten cells, `complete_only` at the shortest history (p = 0.004) and `open_as_negative` at twice the
  longest deadline (p = 0.045) remain below 0.05; `complete_only` at equal history and `mature_only` at
  twice are at 0.051.
- With organisations as the unit (nine) the direction is the same in every cell: nine of nine, except
  `open_as_negative` at the shortest history, seven of nine.
- Counting settings as in version 1, over all eleven logs: `complete_only` is significantly worse in 198 of
  250 and better in 3; `open_as_negative` worse in 149 of 252 and better in 20; `mature_only` worse in 72 of
  126 and never better.
- Logistic regression and the recurrent network agree: `complete_only` is +73% and +70% at the shortest
  history and +3.9% for both at the longest, worse on eleven of eleven logs in six of those eight cells
  and on ten in the other two.
- A second view, added after the tuning results (protocol, section 3): the plain mean over logs, each log
  counting the same, which `pooled.py` prints in its last column. It is larger than the pooled estimate in
  every cell of this table, because the small permit logs have the largest effects and the least weight in
  the pooled estimate: `complete_only` +88%, +41%, +25% and +22%; `open_as_negative` +5.3%, +9.6%, +12.1%
  and +9.3%; `mature_only` +8.9% and +4.6%. The pooled estimate is the cautious one of the two.

**Competing methods, boosted trees, eleven logs**, relative to `censor_aware`:

| Method | History about half | about equal | about twice | about three times |
| --- | --- | --- | --- | --- |
| `ipcw`: inverse probability of censoring weighting | +9.3% (-4.6 to 25.3) | +1.8% (-1.8 to 5.6) | +0.2% (-0.3 to 0.7) | +0.1% (-0.1 to 0.3) |
| `pseudo`: pseudo-labels from a model of finished cases | +30% (4 to 62) | +5.8% (1.3 to 10.5) | +1.7% (-0.4 to 3.9) | +0.2% (-0.3 to 0.7) |
| `ipcw_x`: weighting by time in the case (added afterwards) | +8.5% (-0.6 to 18.5) | +1.2% (-2.0 to 4.6) | +0.2% (-0.0 to 0.5) | +0.1% (0.0 to 0.3) |

- Neither method beats `censor_aware` at any history length. The expectation written down beforehand, that
  the weighting might win with a short history because it uses more of the recent data, did not hold.
- With a history of at least twice the longest deadline the weighting is within 1.1% of `censor_aware` on
  seven logs, within 6% on three of the small permit logs, and 33% worse on the traffic fines.
- Why the traffic fines: the fine amounts were raised over the years, so the amount tells recent fines from
  old ones. Among recent fines only those already paid have a known status at a deadline of a year; the
  weighted classifier can single them out by their amount and predicts payment (+56 points at one cutoff,
  where 99% of the test fines carry an amount that no fine followed for a year has). Without the amount
  feature most of the error goes. `censor_aware` has no rows for windows that have not ended and is within
  1 point at that cutoff. `ipcw_diagnosis.py` reproduces this.
- A second mechanism was confirmed on simulated cases (a test in `tests/test_censoring.py`): a history taken
  late in a case has been followed for less time, so one pooled weight is wrong for it. `ipcw_x` corrects
  that. On the real logs it changes little, so the drifting feature is the larger problem.
- `ipcw_x` and the diagnosis were not fixed in advance; the protocol records that (section 4a).

**Linear competing methods, eleven logs**, relative to `censor_aware` fitted with logistic regression (all of
them are linear in the same features; `linear.py`):

| Method | History about half | about equal | about twice | about three times |
| --- | --- | --- | --- | --- |
| `cox`: cause-specific Cox regression | -1.9% (-3.4 to -0.4) | -0.4% (-1.1 to 0.3) | +0.6% (-0.2 to 1.4) | +0.6% (0.0 to 1.2) |
| `dfm`: delayed-feedback model | +0.7% (-0.1 to 1.6) | +3.1% (1.1 to 5.2) | +5.7% (2.9 to 8.5) | +7.2% (4.2 to 10.3) |
| `pu`: positive-unlabelled learning | +8.5% (4.2 to 13.0) | +4.5% (1.8 to 7.3) | +2.7% (0.3 to 5.0) | +0.7% (-0.9 to 2.4) |

- Cox regression is the one linear method that comes out ahead of `censor_aware`, and only with the shortest
  history: better on 8 of 11 logs, p = 0.02 before and 0.14 after Holm's correction. With a longer history
  it is 0.6% behind. This is what was expected beforehand: it uses the time observed in windows that have
  not ended. It stays within 2% on the traffic fines, where the weighting failed. As a plain mean over logs
  the lead at the shortest history is -1.6% (-4.4 to 1.4), and Cox regression is 1.5% (0.0 to 3.0) and 1.4%
  (0.1 to 2.8) behind at the two longer ones. So the two are level, and neither is ahead throughout.
- The delayed-feedback model falls behind as the history grows (worse on eleven of eleven logs at the
  longest), which points at its assumption of exponential delays.
- Positive-unlabelled learning is worst with a short history, where most positives are still unseen.
- Random survival forests and boosted survival trees were not run: they need packages outside the pinned
  environment, and boosted survival trees are left for later work.

**Tuning without leakage, eleven logs** (`tune.py`). Version 1 fixed the number of boosting rounds at 60. Here
it is chosen among 15, 30, 60, 100 and 150 on a random fifth of the cases in each treatment's own training
rows. Every treatment tuned, relative to the tuned `censor_aware` (on the 2017 loan log every second cutoff,
to bound the running time):

| | History about half | about equal | about twice | about three times |
| --- | --- | --- | --- | --- |
| `complete_only`, tuned | +89% (45 to 146), 11 logs | +33% (14 to 56), 11 logs | +18% (2 to 36), 11 logs | +15% (-2 to 35), 11 logs |
| `open_as_negative`, tuned | +4.2% (-0.2 to 8.9), 9 logs | +6.3% (-0.6 to 13.7), 10 logs | +4.2% (2.0 to 6.5), 11 logs | +2.7% (0.7 to 4.8), 11 logs |
| `mature_only`, tuned | no data | no data | +3.5% (1.3 to 5.8), 11 logs | +1.4% (-0.1 to 3.0), 11 logs |
| `censor_aware` with the fixed 60 rounds | +4.6% (1.2 to 8.0) | +3.0% (0.6 to 5.4) | +0.3% (-0.6 to 1.2) | -0.0% (-1.0 to 0.9) |
| `censor_aware`, rounds chosen on the recent half of the history (plain mean) | +2.4% (-0.6 to 5.5) | -0.3% (-1.1 to 0.5) | -0.4% (-0.9 to 0.1) | +0.2% (-0.9 to 1.3) |

- Tuning does not change how the treatments compare. Every cell is within three points of the table with
  fixed rounds, and the number of logs on which a treatment is worse is the same in nine cells and one lower
  in one.
- `censor_aware` itself gains from tuning when the history is short: with the fixed 60 rounds it is 4.6% and
  3.0% worse at the two shorter histories, and no different at the two longer. Fewer than 60 rounds are
  chosen in 59% of the settings, so 60 was too many for the small training sets. That fits the
  over-confident forecasts found below.
- A hold-out taken from the recent half of the history (fitted on what was known at an inner cutoff, checked
  on the cases that started after it) is no better than the random one. It chooses fewer rounds (below 60 in
  72% of settings; the same number as the random hold-out in 89 of 240) and the scores differ by less than
  half a percent at three history lengths. At the shortest it is 2.4% worse on average, with an interval
  that includes zero and one permit log at +14.8%. The plan assumed that tuning would need such a
  hold-out; on these logs it does not.
- For that last row the plain mean is shown, not the pooled estimate: on two logs both hold-outs choose the
  same rounds at every cutoff of the shortest history, their standard error is zero, and they carry the
  pooled estimate (it prints -0.0%). `pooled.py` marks such rows.
- Only the number of rounds was tuned. The learning rate and the size of the trees stayed as they were.

**Cost in decisions, eleven logs** (`decisions.py`; main model). Per 100 new cases, the forecast number with an
outcome by the longest deadline shorter than the history, minus the actual number, at the first event of each
case; mean over logs:

| History | `complete_only` | `open_as_negative` | `mature_only` | `censor_aware` |
| --- | --- | --- | --- | --- |
| about half the longest deadline | +47.5 | -5.8 | no data | +0.5 |
| about equal | +37.6 | -13.1 | no data | -0.0 |
| about twice | +21.2 | -17.8 | -0.2 | -0.8 |
| about three times | +18.3 | -13.4 | -0.1 | -0.1 |

- With 95% intervals over the eleven logs: `complete_only` forecasts too many outcomes at every history
  length (+47.5, 37.8 to 57.2, down to +18.3, 9.2 to 27.5), `open_as_negative` too few (-5.8, -9.2 to -2.4,
  down to -17.8, -24.7 to -11.0, with twice the longest deadline). The intervals of `censor_aware` and
  `mature_only` all contain zero.
- In absolute terms the error of `censor_aware` is 6 to 9 cases per 100 in a single setting; it is the sign
  that averages out. `complete_only` is further off on 11 of 11 logs at the two shorter histories and on 10
  at the two longer (by 41.0, 28.5, 14.9 and 12.8 cases per 100; every interval excludes zero);
  `open_as_negative` on 7, 10, 11 and 10 (by 0.7, 5.1, 11.8 and 8.3; the first interval contains zero).
- Value, for the five logs with an amount, longest history: forecast value of the cases with a payment (or
  approval) by the deadline against the realised value. Averaged over cutoffs, `complete_only` is 138% too
  high on the traffic fines and between 4% too low and 31% too high on the other four logs;
  `open_as_negative` is 3.5% to 16% too low; `censor_aware` is 3% to 12% too high. At a single cutoff the
  error of `censor_aware` is larger than these averages suggest (mean absolute error 4% to 20%). It is
  smaller than that of `complete_only` on all five logs and than that of `open_as_negative` on three; on
  the traffic fines and the reimbursement log (`bpi20r`) `open_as_negative` is closer (14% against 20%, 11%
  against 15%). For the fines the outcome is a first payment, not payment in full.
- Calibration: the reliability part of the Brier score is lowest for `censor_aware` at every history length
  (0.009 with the longest history, against 0.052 for `complete_only`, 0.046 for `open_as_negative` and 0.011
  for `mature_only`). All treatments are over-confident: the calibration slope is below 1 for every one of
  them. For `censor_aware` it is 0.37, 0.47, 0.61 and 0.71 from the shortest to the longest history; that of
  `open_as_negative` is slightly nearer to 1 at the two longer histories (0.63 and 0.77), that of
  `complete_only` is the lowest throughout (0.26 to 0.47). Forecasts from any of them should be
  recalibrated before use.

**Replay.** In place of a real deployment: a cutoff every month over each log's usable period (313 months in
all), a history of twice the longest deadline, forecasts for the cases that start in the next 30 days.

| | `complete_only` | `open_as_negative` | `mature_only` | `censor_aware` |
| --- | --- | --- | --- | --- |
| Mean absolute planning error per 100 cases | 22.8 | 15.9 | 8.3 | 8.2 |
| Mean signed error | +22.6 | -15.2 | +1.6 | +0.9 |
| Months in which `censor_aware` is closer | 274 of 313 | 233 of 313 | 174 of 313 | |
| Logs on which `censor_aware` is closer on average | 11 of 11 | 11 of 11 | 7 of 11 | |

Over logs, the mean absolute error of `complete_only` is 14.5 cases per 100 above that of `censor_aware`
(95% interval 6.5 to 22.5), of `open_as_negative` 7.7 above (5.1 to 10.2), of `mature_only` 0.1 above
(-0.5 to 0.6). In cases per week, at each log's own rate of new cases: on the 2017 loan log (633 new cases
a week) the forecast is off by 95 cases a week with `complete_only`, 98 with `open_as_negative` and 11 with
`censor_aware`; on the traffic fines (232 a week) by 128, 19 and 15; on the incident log of BPI 2014 (1,777
a week) by 104, 96 and 23. The four municipal logs have about five new cases a week, so their monthly
figures rest on some twenty cases each.

With a history of twice the longest deadline, waiting for full observation (`mature_only`) plans about as
well as `censor_aware`; the difference between those two is in accuracy per case, not in the totals.

### Checking the results yourself

```bash
cd experiments
.venv/bin/python censoring/verify.py               # re-runs two logs from the raw data and compares every number
                                                   # with the stored results; then recomputes the paper's
                                                   # headline numbers with separate code (about 5 minutes)
.venv/bin/python censoring/verify.py --logs bpi20  # any other log of datasets.py
.venv/bin/python censoring/verify.py --no-rerun    # only the recomputation (seconds)
```

`verify.py` writes nothing into `results/` and exits with status 0 if everything agrees. On
2026-11-15 the re-run of `helpdesk`, `bpi12` and `bpi20` (60 settings, 4,433 numbers) matched the stored results
exactly, and the recomputation found one inconsistency, which is corrected (the AUC comparison below).

**What a sceptical reading of the design added (2026-11-15, after all results; `PROTOCOL_B.md`, section 4h).**
The direction of the main result holds everywhere; its size is firm only for short histories. `complete_only`
against `censor_aware`, Brier score, pooled over logs, at a history of about 0.5 / 1 / 2 / 3 times the longest
deadline:

| View | 0.5x | 1x | 2x | 3x |
| --- | --- | --- | --- | --- |
| All eleven logs (the protocol's analysis) | +86% (42 to 143) | +35% (11 to 64) | +16% (1.5 to 33) | +14% (-1.4 to 31) |
| p after Holm's correction | 0.004 | 0.051 | 0.165 | 0.188 |
| Prediction interval for a new log | -10% to +286% | -3% to +89% | -4% to +41% | -7% to +38% |
| The six logs with a grid derived by rule | +46% (24 to 73) | +26% (14 to 39) | +20% (8.5 to 33) | +17% (5.6 to 30) |
| The five logs of version 1 (grids chosen by hand) | +142% (37 to 327) | +50% (-16 to 170) | +14% (-21 to 66) | +10% (-23 to 58) |
| Four permit logs as one unit (8 units) | +102% (43 to 187) | +37% (4.4 to 80) | +14% (-4.0 to 35) | +11% (-7.7 to 34) |
| All deadlines scored, not only those shorter than the history | +40% (8.2 to 81) | +26% (1.8 to 57) | as above | as above |
| Against `mature_only` (unbiased, also fewer rows) | no data | no data | +7.7% (-5.5 to 23) | +7.0% (-4.2 to 19) |

`complete_only` is the worse one on 11 of 11 logs (8 of 8 units) in every column. With a long history, much of
its loss on the small logs is a loss of training rows; the bias dominates on the traffic fines (+185% and +194%
against `mature_only`). The paper states the result in this narrowed form.

### After phase B: the items left open

Done on 2026-11-15, each fixed in a protocol before it was run (`censoring/PROTOCOL_B.md`, sections 4e to 4g).

```bash
cd experiments
.venv/bin/python censoring/survival_trees.py bpi14     # Cox regression with boosted trees -> results/<log>.methods_s.jsonl
.venv/bin/python censoring/pooled.py --methods         # now with cox_trees
.venv/bin/python censoring/auc_summary.py              # AUC next to the Brier score, from stored results
.venv/bin/python censoring/calibration.py bpi14        # reliability tables, slope per deadline and outcome, values
.venv/bin/python censoring/calibration.py --summary    # and --diagram
.venv/bin/python censoring/tune.py bpi14 --wide        # learning rate and leaves tuned as well
.venv/bin/python censoring/pooled.py --model tuned_wide
.venv/bin/python -I censoring/anchor.py                # one number of the outcome benchmark, reproduced
.venv/bin/python censoring/journal_inputs.py           # tables and numbers of the paper
```

**Boosted survival trees** (`cox_trees`: cause-specific Cox regression in which each linear predictor is a sum of
boosted trees; built from scikit-learn, no new package), relative to `censor_aware` with boosted trees:

| | History about half | about equal | about twice | about three times |
| --- | --- | --- | --- | --- |
| `cox_trees` | -1.5% (-3.5 to 0.4) | +1.1% (-0.4 to 2.6) | +4.1% (1.6 to 6.7) | +3.6% (1.5 to 5.7) |
| logs on which it is worse | 4 of 11 | 8 of 11 | 11 of 11 | 11 of 11 |

- Level with `censor_aware` when the history is short, about 4% behind when it is long, on every log. On the
  traffic fines it stays within 1.2%, as expected of a method that weights nothing.
- So the survival model that uses all observed time does not beat the window rule with the main model class
  either. Random survival forests are still not run (no implementation in the pinned environment).
- The window rule stored next to it equals the one of the main results in all 240 settings.

**Tuning more than the rounds** (`tune.py --wide`: learning rate 0.04, 0.08 or 0.16, 7 or 31 leaves, and the
rounds; thirty settings, chosen on the same random hold-out). Every treatment tuned, relative to the tuned
`censor_aware`:

| | History about half | about equal | about twice | about three times |
| --- | --- | --- | --- | --- |
| `complete_only`, tuned | +90% (48 to 144) | +37% (16 to 63) | +21% (3 to 43) | +14% (-2 to 32) |
| `open_as_negative`, tuned | +4.4% (0.4 to 8.6) | +5.4% (-1.2 to 12.4) | +3.6% (1.7 to 5.7) | +2.7% (0.6 to 4.9) |
| `mature_only`, tuned | no data | no data | +3.1% (1.0 to 5.3) | +0.9% (0.0 to 1.7) |
| `censor_aware`, rounds tuned only | +0.6% (-1.1 to 2.2) | +0.5% (-0.6 to 1.6) | +0.1% (-0.5 to 0.7) | +0.1% (-0.2 to 0.4) |
| `censor_aware`, fixed settings | +5.7% (1.2 to 10.4) | +4.5% (0.3 to 9.0) | +0.6% (-1.3 to 2.5) | +0.1% (-1.4 to 1.6) |

- The comparison of the treatments does not change with the wider search either. `complete_only` is worse on
  11, 11, 10 and 11 of the eleven logs.
- It gives `censor_aware` at most 0.6% over tuning the rounds alone (2.9% at one history length as a plain mean
  over logs), although trees of 7 leaves are chosen in 54% of the settings.
- The window rule with fixed settings equals the one of the main results in all 240 settings.

**AUC next to the Brier score** (`auc_summary.py`; AUC of "any outcome by the middle deadline", where that
deadline is shorter than the history; the Brier score on the same settings, which at the shortest history
leaves out the 2020 travel declarations). Difference to `censor_aware` in hundredths, mean over logs:

| History | `complete_only`, AUC | `complete_only`, Brier | `open_as_negative`, AUC | `open_as_negative`, Brier |
| --- | --- | --- | --- | --- |
| about half | -3.2 (-7.7 to 1.2) | +33.1 (16.0 to 50.3) | -2.3 (-4.8 to 0.1) | +1.8 (-0.9 to 4.5) |
| about equal | -4.8 (-7.3 to -2.2) | +15.5 (3.1 to 28.0) | -2.4 (-4.9 to 0.1) | +3.7 (0.2 to 7.2) |
| about twice | -3.1 (-5.4 to -0.7) | +9.1 (-0.3 to 18.6) | -1.7 (-3.5 to 0.1) | +4.3 (1.3 to 7.3) |
| about three times | -2.3 (-4.4 to -0.3) | +8.0 (-1.7 to 17.8) | -0.7 (-1.6 to 0.1) | +3.2 (0.5 to 5.9) |

AUC, the main metric of the field's benchmark, moves by 2 to 5 points where the Brier score moves by 8 to 33: most
of the damage is to the level of the probabilities, which AUC ignores. This comparison was not in the protocol; it
was added after the results (PROTOCOL_B.md, section 4h), and until 2026-11-15 its Brier column used all settings
(+35.1 and +2.0 in the first row).

**Calibration in detail and the money case** (`calibration.py`).

- Reliability, longest history and longest deadline shorter than it, all logs weighted the same: the mean
  distance between forecast and what happened is 16.1 points for `complete_only`, 13.7 for `open_as_negative`,
  4.7 for `mature_only` and 3.7 for `censor_aware`. 81% of the forecasts of `complete_only` are above 0.9, and
  81% of those cases have the outcome. Of the cases to which `open_as_negative` gives about a quarter, 61% have it.
- The slope of a logistic recalibration is below 1 for every treatment at every deadline, and lower at the
  longest deadline than at the shortest: for `censor_aware` with the longest history 0.92 at the shortest
  deadline and 0.71 at the longest. Forecasts should be recalibrated before use.
- Per outcome the picture is the same; the three logs with more than one outcome add little.
- Traffic fines, forecast against realised value of the new fines with a first payment by the deadline, mean
  over six cutoffs (95% interval):

| Deadline | History | `complete_only` | `open_as_negative` | `mature_only` | `censor_aware` |
| --- | --- | --- | --- | --- | --- |
| 90 days | about half | +289% (147 to 432) | -14% (-42 to 15) | no data | +21% (-11 to 54) |
| 90 days | about three times | +172% (88 to 255) | +3% (-25 to 32) | +2% (-26 to 31) | +11% (-20 to 41) |
| 365 days | about twice | +141% (68 to 213) | -11% (-33 to 12) | -2% (-23 to 19) | +6% (-17 to 30) |
| 365 days | about three times | +138% (69 to 207) | -9% (-29 to 11) | 0% (-23 to 23) | +6% (-19 to 30) |

  `complete_only` expects two to four times what is collected. The other three cannot be told apart with six
  cutoffs. A first payment is not payment in full.

**How published studies build their training set** (`../paper_journal/audit/`, table `protocols.csv` with a
quotation and a source per paper). Of 70 papers selected by a fixed rule (they cite the outcome benchmark, 2019
to 2026, about outcome prediction), 13 have no open full text and 12 turned out to be out of scope (ten papers
whose publishers refuse automated requests were read on 2026-11-15 from open-access files the author saved by
hand). Of the 45 read:

| How the training set is built | Papers |
| --- | --- |
| Split in time; training cases keep the label of the completed trace | 16 |
| Random division of cases, or cross-validation | 10 |
| Split in time; only cases completed before the split train (two of them streams) | 6 |
| Open cases train, and their incompleteness is modelled | 0 |
| Not stated | 13 |

How the papers were read and how the table was checked is described in the paper (Sect. 2.3 and 5.7).

**One number of the outcome benchmark, reproduced** (`anchor.py`; the benchmark's own processed files and
protocol, re-implemented from its code, with this study's boosted trees instead of its tuned XGBoost). Overall
AUC, single bucket, aggregation encoding:

| Dataset | Published | Here | Here, completed cases only |
| --- | --- | --- | --- |
| Traffic fines | 0.66 | 0.651 (0.664 with 500 rounds) | 0.655 |
| BPI 2012, accepted | 0.70 | 0.678 | 0.672 |

- The tolerance set beforehand was 0.02: the first is reproduced, the second is 0.022 away.
- Under the benchmark's protocol, dropping the cases that are open at the split changes little (Brier score
  0.180 to 0.188 on the fines, 0.211 to 0.213 on the loans). On the fines only 4% of the training cases are
  open at the split, because the benchmark trains on the first 80% of a log of thirteen years; on the loans 18%
  are, but the share of positive cases hardly moves. A single split late in a long log shows little of the
  problem. It appears when the history is short relative to the cases, which is what this study varies.

Still open: random survival forests, and an estimator that uses all observed time. The results for Cox regression say where such an estimator could gain: with short histories.

### Generated tables and exact versions

`censoring/paper_inputs.py` writes the tables, figure data and quoted numbers of the first version of the study
(five logs) into `../paper/generated/`, and `censoring/journal_inputs.py` those of the paper (eleven logs) into
`../paper_journal/generated/`. Nothing that the results determine is typed by hand.

To reproduce with the exact package versions, install `requirements.lock.txt` instead of
`requirements.txt`. `pytest` at the repository root runs the study's tests too (`tests/test_censoring.py`).
