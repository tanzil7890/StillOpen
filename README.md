# StillOpen

Outcome prediction on event logs when recent cases are still open.

A model that predicts how a running case will end is trained on past cases. On the day it is trained, many of the
most recent cases have not finished, so their outcomes are unknown. This repository holds the code, the protocol
and every result of a study of what to do with those open cases, on eleven public event logs.

The paper is *Outcome Prediction When Recent Cases Are Still Open: A Prospective Study on Eleven Event Logs*
(Mohammad Tanzil Idrisi, 2026).

## The question

Every experiment is run as it would be in use. A cutoff date stands in for today. Training sees only what was
recorded by the cutoff, and forecasts are scored on cases that start afterwards. The task is to forecast whether
an outcome (a loan approved, a fine paid, a ticket resolved) happens within a deadline.

Four ways of handling the cases still open at the cutoff are compared. They differ in nothing else.

| Treatment | What it does with open cases |
| --- | --- |
| `complete_only` | Drops them and trains on finished cases only |
| `open_as_negative` | Keeps them, labelled "no outcome" |
| `mature_only` | Keeps a case only if all of its deadline windows had ended by the cutoff |
| `censor_aware` (the window rule) | Keeps each (case, deadline) pair whose window had ended by the cutoff |

## Main result

Brier score relative to the window rule, pooled over the eleven logs with boosted trees. Positive means worse.
The columns give the length of the training history as a multiple of the longest deadline; brackets hold the 95%
interval.

| Treatment | 0.5x | 1x | 2x | 3x |
| --- | --- | --- | --- | --- |
| `complete_only` | +86% (42 to 143) | +35% (11 to 64) | +16% (1.5 to 33) | +14% (-1.4 to 31) |
| `open_as_negative` | +2.0% (-0.2 to 4.2) | +5.4% (-0.3 to 12) | +4.8% (1.8 to 7.9) | +2.4% (0.5 to 4.4) |
| `mature_only` | no data | no data | +4.2% (1.4 to 7.1) | +2.2% (0.1 to 4.3) |

- Training on finished cases only is worse on all eleven logs at every history length. The size of the loss is
  firm only for the shortest history; with long histories the interval reaches zero.
- Labelling open cases as negative costs less accuracy but forecasts too few outcomes.
- Five established methods for censored or delayed labels were run in seven variants. None improves on the
  window rule throughout, and the rule cannot be told apart from Cox regression.

Results that go against the window rule are reported with the rest. All tables are in
[experiments/README.md](experiments/README.md).

## What is here

| Path | Content |
| --- | --- |
| `experiments/censoring/` | The study: `study.py` runs one log, `datasets.py` holds the logs, outcomes, deadlines and cutoff dates, `pooled.py` the pooled statistics, `verify.py` the check |
| `experiments/censoring/PROTOCOL_B.md` | The rules fixed before the runs, and what was added after results were seen |
| `experiments/censoring/results/` | Every result file |
| `experiments/pilot/` | The first experiment, on one log |
| `experiments/README.md` | How to run each step, and all result tables |
| `paper_journal/generated/`, `paper/generated/` | The tables and quoted numbers of the paper, written by the scripts |
| `paper_journal/audit/protocols.csv` | How 70 published studies build their training set, with a quotation and a source per paper |
| `temporal_rlcd/` | The small event-sequence model the project started from, with its own README |
| `tests/` | Tests of the study code and of the model |

## Run it

The setup needs [uv](https://docs.astral.sh/uv/).

```bash
bash experiments/setup.sh                     # creates experiments/.venv
cd experiments
.venv/bin/python -I censoring/prepare.py      # download the logs, verify their checksums, parse them
.venv/bin/python censoring/study.py helpdesk  # one log (under a minute); results go to censoring/results/
.venv/bin/python censoring/pooled.py          # the pooled tables, from the stored results
```

To check the stored results, `verify.py` runs two logs again from the raw data, compares every number with the
stored files, and recomputes the headline numbers with separate code. It takes about five minutes.

```bash
.venv/bin/python censoring/verify.py
```

The tests run from the repository root:

```bash
experiments/.venv/bin/python -m pytest tests -q
```

## Data

No event log is stored here. The scripts download each one from its public source. Each log has its own terms of
use; the BPI Challenge 2020 logs, for one, are licensed CC BY-NC 4.0 (no commercial use). Check the terms of each
log before you redistribute it.

## Licence

The code is under the MIT licence; see [LICENSE](LICENSE).
