# Phase B protocol

Fixed on 2026-11-14, before any model was trained on a new log and before any competing method was run.
Version 1 (five logs, four treatments) is not changed by anything here; its result files stay as they are.

## 1. Which logs are added

A log is included if it meets all of these, judged from descriptive statistics only (`describe_log.py`,
`design.py`):

1. Public, with a licence that allows research use, and not clinical patient data.
2. An event that resolves the case from the business point of view can be read from the activities. The
   choice of that event, and the reason, are written in `datasets.py`. Where the outcome benchmark of
   Teinemaa et al. labels the same log, its notion of outcome is preferred.
3. The log is not an extract of finished cases only.
4. The grid of section 2 fits: at least three deadlines, at least three cutoff dates, and at least 50 test
   cases per cutoff on average.
5. It does not describe the same cases as another included log.

Logs examined and rejected are listed in `datasets.py` with the criterion that failed.

## 2. The experiment grid of a new log, by rule

`design.py` computes it; nothing is chosen by hand after the outcome events.

- **Histories**: every event of a case before its outcome, at most the first 40.
- **Deadlines**: the 15th, 35th, 55th, 75th and 90th percentile of the time from a history to the outcome of
  its case, over histories whose case has an outcome, each rounded to the nearest value (on a log scale) of
  1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365, 545, 730, 1095 days. A value that repeats
  after rounding is dropped, so a log can have fewer than five deadlines; it needs at least three.
- **History lengths**: 0.5, 1, 2 and 3 times the longest deadline.
- **Usable period**: starts when 1% of the cases have started plus the longest history; ends at the time by
  which 99.5% of the events have occurred, minus the longest deadline, the test window, and the 90th
  percentile of the time from the start of a case to its outcome. The percentiles guard against stray
  timestamps and against the thin end of a log.
- **Test window**: the shortest of 21, 30, 60 and 90 days that gives at least 150 test cases per cutoff on
  average over the usable period (90 days if none does).
- **Cutoffs**: six, evenly spaced over the usable period; fewer if the period is shorter than five test
  windows, never fewer than three.
- **A known change of the process**: for the logs of the university (BPI Challenge 2020), cases that started
  before 2018 are left out, as in version 1, because in 2017 the process ran as a trial in two departments.
- **Many activities**: in a log with more than 250 activities they are numbered by frequency, and the rarest
  ones beyond 250 share one category in the two features that hold the last two activities. A count per
  activity is kept for all of them.
- **Features, models, scores**: as in version 1. The main model is boosted trees with 60 rounds; logistic
  regression and the recurrent network are run as checks.

## 3. Pooled statistics

The unit of analysis is the log. For each treatment (and later each competing method) against the window
rule, and each of the four history lengths:

- **Effect per setting**: log of the ratio of Brier scores (treatment over window rule), on the deadlines
  shorter than the history. A ratio is used because the level of the Brier score differs between logs.
- **Effect per log**: the mean over that log's cutoffs, with the standard error of that mean.
- **Pooled effect**: random-effects meta-analysis over logs (DerSimonian-Laird estimate of the
  between-log variance, Hartung-Knapp interval, which is the cautious choice with few logs). Reported: the
  pooled ratio with its 95% interval, the 95% prediction interval for a new log, and the between-log variance.
- **A second, assumption-free test**: the sign test over logs, each log one vote.
- **Multiple comparisons**: Holm's correction over all treatment-by-history-length cells of a table.
- **Sensitivity**: the same with logs of one organisation merged into one unit (the two bank logs; the logs
  of the university).

The counts of significant settings of version 1 are still reported, as a description.

Added on 2026-11-15, after the tuning results of section 4c: the plain mean over logs with its t interval
(every log counting the same) is printed next to the pooled effect. Reason: when the two things compared
give the same forecasts at every cutoff of a log, that log's standard error is zero, and it carries the
whole random-effects estimate if the other logs are within their own standard errors of it. This happened
in one row (prospective against random hold-out, shortest
history, two logs). The pooled effect stays the headline number everywhere else; where the two disagree,
both are reported.

## 4. Competing methods

All use only what is known at the cutoff and go through the same protocol, with the main model class.
Each is compared with the window rule by the statistics of section 3.

- **`ipcw`: inverse probability of censoring weighting, per deadline.** One classifier with the deadline as a
  feature. A (history, deadline) pair is used if its status at the deadline is known at the cutoff: the
  outcome was seen before the deadline, or the history has been followed past the deadline. It is weighted
  by one over the probability of having been followed that long, estimated from the follow-up times of the
  training histories, and capped at 20. Unlike the window rule it uses outcomes seen in windows that have
  not ended.
- **`pseudo`: pseudo-labelling.** Train on finished cases only; use that model to fill in the unknown
  windows of open cases with its own predicted class probabilities; train again on everything.
- Planned next, not yet specified in detail: a delayed-feedback model, survival models (Cox, random
  survival forest, boosted survival trees), positive-unlabelled learning.

Expectations, stated before running: `ipcw` is close to the window rule when the history is long and may be
better when it is short, because it uses more of the recent data; `pseudo` inherits the bias of training on
finished cases. Both are tested two-sided.

## 4a. Added after the first results (2026-11-15)

The first results of `ipcw` showed it far behind the window rule on the traffic fines and close to it
elsewhere. The cause was then confirmed on simulated cases: a history taken late in a case has been followed
for less time than one taken at its start, so a single follow-up share for all histories gives the late ones
the wrong weight. `ipcw_x` estimates the share separately in ten groups of time since the case began.
It was not fixed in advance and is reported as such; `ipcw` stays in the tables as it was specified.

## 4b. Further competing methods, linear (fixed 2026-11-15, before any of them was run)

These three are linear in the features of the logistic-regression model of version 1 (standardised numbers,
the last two activities as indicators, a count per activity), with one row per history and the same penalty.
They are therefore compared with the window rule fitted with logistic regression, not with the boosted trees.

- **`dfm`: delayed-feedback model** (Chapelle 2014), extended to several outcomes. A history ends in outcome k
  with probability p_k(x), or in none; given outcome k, the delay is exponential with rate r_k(x). An outcome
  seen after delay t contributes p_k r_k exp(-r_k t); a history without a seen outcome, followed for u,
  contributes p_0 + sum_k p_k exp(-r_k u). The forecast by deadline d is p_k (1 - exp(-r_k d)).
- **`cox`: cause-specific Cox regression**, one linear predictor per outcome, Breslow baseline hazard. The
  time of a history is its delay if the outcome was seen and its follow-up otherwise; another outcome
  censors. The forecast by a deadline is the cumulative incidence, integrated on a grid of 200 times.
- **`pu`: positive-unlabelled learning**, per deadline and outcome, with the non-negative risk estimator
  (Kiryo et al. 2017; used for event logs by Peeperkorn et al. 2023). Positive: the outcome was seen within
  the deadline. Unlabelled: every history. The share of positives is estimated from the histories followed
  past the deadline. Only deadlines shorter than the history are fitted.

Expectations, stated before running: `cox` is close to the window rule and may be better with a short
history, because it uses the time observed in windows that have not ended; `dfm` depends on delays being
roughly exponential; `pu` is biased, because the positives it cannot see are the slow ones, not a random
sample. All tested two-sided with the statistics of section 3.

Not run: random survival forests and boosted survival trees. They need packages outside the pinned
environment, and boosted survival trees are the estimator planned for phase C.

## 4c. Tuning without leakage (fixed 2026-11-15, before it was run)

Main model. What is tuned is the number of boosting rounds: 15, 30, 60, 100 or 150 (60 is the fixed value
of version 1). Everything else stays as it is.

- **`tuned`**, for each of the four treatments: the usual practice. A random 20% of the cases in the
  treatment's own training rows is held out, the model is fitted with 150 rounds on the rest, the number of
  rounds with the lowest log loss on the held-out rows is chosen, and the model is fitted again on all rows
  with that number.
- **`prospective`**, for the window rule only: the held-out part is the recent half of the training period
  instead of a random part. The model is fitted on what the window rule gives at the inner cutoff (half the
  history before the cutoff) and checked on the rows that the window rule gives at the cutoff for cases
  that started after the inner cutoff.

Questions: does tuning change how the treatments compare with the window rule (each tuned, pooled as in
section 3); does the window rule gain from tuning (tuned against fixed); is the prospective hold-out better
than the random one for it. To bound the running time, the largest log (`bpi17`) uses every second cutoff.

## 4d. Cost in decisions (fixed 2026-11-15, before it was run)

Main model with its fixed settings, the four treatments, every setting of the grid.

- **Planning error**: at the first history of each test case, the forecast number of cases with an outcome
  by a deadline against the number that had one, per 100 cases. Reported for the longest deadline shorter
  than the history.
- **Calibration**, for "any outcome by that deadline" over all test histories: slope and intercept of a
  logistic recalibration (1 and 0 are ideal), and the split of the Brier score into reliability,
  resolution and uncertainty over ten bins.
- **Value**, for logs with an amount: at the first history, the amount times the forecast probability of the
  first-listed outcome by the deadline (payment or approval), summed over test cases, against the summed
  amount of the cases where that outcome happened. For the traffic fines the outcome is a first payment,
  which is not always payment in full.
- **Replay**, in place of a real deployment: a cutoff every month from the first to the last cutoff of the
  grid, a history of twice the longest deadline, forecasts for the cases that start in the following 30
  days. Reported: the planning error month by month and its mean absolute value.

Added to the report after the run, from the stored results and without running any model again: 95%
intervals over logs for the planning error (t distribution, the log as the unit), and the replay's error
converted into cases per week with each log's own rate of new cases.

## 4e. Boosted survival trees (fixed 2026-11-15, before it was run)

Section 4b left out the tree-based survival models because they need packages outside the pinned
environment. One of them can be built from what is installed, and it is the natural counterpart of the main
model, so it is added.

- **`cox_trees`: cause-specific Cox regression with boosted trees.** One row per history, the features of the
  main model. For each outcome the linear predictor of `cox` is replaced by a sum of regression trees. Each
  round fits one tree to the Newton step of the partial likelihood (Breslow form for ties): for history i
  with predictor f_i, gradient d_i - exp(f_i) A_i and curvature exp(f_i) A_i - exp(2 f_i) B_i, where A_i and
  B_i sum 1/S and 1/S^2 over the outcomes seen no later than i's time and S is the summed exp(f) of the
  histories still at risk. Settings as the main model: 60 rounds, learning rate 0.08, 31 leaves, penalty
  1.0, the two activity features as categories. Baseline hazard and cumulative incidence as for `cox`.
- Compared with the window rule fitted with boosted trees (the main model), under the statistics of section
  3. The largest log (`bpi17`) uses every second cutoff, as in 4c.
- Expectation, stated before running: close to the window rule; possibly better with the shortest history,
  as `cox` was among the linear models; close on the traffic fines as well, because nothing is weighted.
- Still not run: random survival forests (no implementation in the pinned environment), and the
  fake-negative losses of Ktena et al. (2019), which are defined for a stream in which every example first
  enters as a negative and have no deadline; under a single cutoff they reduce to `open_as_negative`.

## 4f. Tuning more than the rounds (fixed 2026-11-15, before it was run)

Section 4c tuned the number of rounds only. Here the learning rate (0.04, 0.08, 0.16) and the number of
leaves (7, 31) are tuned as well, with the rounds (15, 30, 60, 100, 150): thirty settings. The hold-out is
the random fifth of the cases of 4c, and each of the four treatments is tuned on its own rows. Stored next
to them: the window rule with the rounds-only tuning of 4c and with the fixed settings of version 1.
`bpi17` uses every second cutoff. Questions: does the comparison of the treatments change; how much does
the window rule gain over rounds-only tuning.

## 4g. Calibration in detail and the money case (fixed 2026-11-15, before it was run)

Section 4d measured calibration for "any outcome" at one deadline. Here, with the main model and its fixed
settings, for every treatment, every setting and every deadline shorter than the history:

- for "any outcome" and for each outcome on its own: the ten-bin reliability table over all test histories
  (count, mean forecast, share that happened), slope and intercept of the logistic recalibration, and the
  three parts of the Brier score;
- at the first history of each test case: the planning error per 100 cases, and for the logs with an amount
  the forecast and the realised value of the first-listed outcome.

Reported: reliability diagrams at the longest history and the longest deadline shorter than it, the rows of
all cutoffs and logs put together and per log; the slope per deadline; the money case on the traffic fines
at 90 days (every history length) and at 365 days (the two history lengths longer than a year), in the
log's currency. As before, a payment is a first payment, not payment in full.

## 4h. Found when the manuscript was checked (2026-11-15, after all results)

Two things came out of a check of the journal manuscript against the result files (`verify.py`).

- The comparison of AUC with the Brier score (`auc_summary.py`) is not in this protocol. The AUC of "any
  outcome by the middle deadline" has been stored with every result line since version 1, but the table that
  sets it next to the Brier score, and the rule to use it only where the middle deadline is shorter than the
  history, were written after the results had been seen. It is reported as an addition made afterwards.
- In that table the Brier score was first taken on all settings and the AUC only on the settings in which the
  middle deadline is shorter than the history. At the shortest history these differ (one log, the 2020 travel
  declarations, drops out: 56 instead of 63 settings). Both columns now use the same settings. The Brier
  difference of `complete_only` at the shortest history changes from +35.1 to +33.1 points and that of
  `open_as_negative` from +2.0 to +1.8; no other cell changes.

The same check included a sceptical re-reading of the design. It led to
four more views of the main comparison (`complete_only` against the window rule), all computed from the stored
results by `journal_inputs.py` (`robustness`) and all reported as additions made afterwards:

- the six logs whose grid was derived by the rule of section 2 taken alone, and the five logs of version 1,
  whose grids were chosen by hand, taken alone;
- the four permit logs, one process in four municipalities, taken as one unit (8 units);
- all deadlines scored, not only those shorter than the history;
- `complete_only` against `mature_only`, which is unbiased and also has fewer rows.

Also from that reading: the manuscript now reports the prediction interval and the between-log spread that
section 3 asks for, which it had left out. The number of logs examined for phase B was given as 21 with 15
rejected; `datasets.py` lists 14 rejected (counting the six CoSeLoG logs singly), so 20 were examined.

## 5. Order of work and what is left open

More logs first, then the pooled statistics on all logs, then the competing methods (sections 4, 4a, 4b),
then tuning without leakage (4c) and the cost in decisions (4d).
