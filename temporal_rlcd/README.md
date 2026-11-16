# Temporal RLCD Starter: Typed Probabilistic Future-Event Decisions

**Status: educational, runnable research prototype (not a trained foundation model).**

This repository is an **original** tiny PyTorch architecture. It is inspired by the
publicly described System One / Jev design philosophy (no text decoding; typed probabilities),
not TypeSafe AI's undisclosed weights, architecture, or RLCD internals. It also includes
an **isolated RLCD-inspired** policy-gradient loss for choosing latent routes with strictly
proper probability scores. The default supervised training script **does not run RL**.

## Run

```bash
python -m pip install -r requirements.txt
pytest -q
python -m temporal_rlcd.train --epochs 4 --train-n 2000 --test-n 500 --output toy.pt
python -m temporal_rlcd.demo  # deliberately UNTRAINED model for JSON shape demo
```

## What is here

- `model.py`: event ID, log inter-event interval, and age embeddings; causal Transformer
  (no future attention); typed multi-head outputs for next event type, next-event time,
  and future first-terminal-event probabilities.
- Competing risk discrete-time hazard with mutually exclusive terminal outcomes:
  none / success / failure / escalation, across four observation intervals.
- `survival_nll`: event likelihood with **right censoring**, so an unobserved future
  outcome is not treated as a negative outcome.
- `brier_loss`: proper multiclass probability objective evaluated only at horizons
  with known outcomes; `forecast_distribution`: coherent cumulative probabilities.
- `rlcd.py`: a **separate** experimental REINFORCE objective on sampled internal
  routes, scored by negative multiclass Brier and optionally KL-anchored to a
  reference route policy. It is not integrated with the toy training entrypoint.
- `synthetic.py`: artificial workflow traces; synthetic labels must **never**
  be interpreted as evidence of real forecasting accuracy.
- `tests`: probability mass, monotonic event risk, censoring gradients, RL policy
  loss shape/gradients.

## Input contract

At decision time, the model sees only event prefixes:

| name | shape | meaning |
| --- | --- | --- |
| `events` | B × T int | event ids 1..7, 0 means padding on right |
| `gaps` | B × T float | nonnegative elapsed time between observed events |
| `lengths` | B int | number of valid events per sequence |

Outputs include `next_event_logits` and `hazard_logits` of size `B × 4 × 4`.
`forecast_distribution` returns `B × 4 × 4`, where last dimension is
`[no terminal outcome yet, success by horizon, failure by horizon, escalation by horizon]`.
This is a **first-terminal-outcome** model, not a generator of whole future traces.
Its event taxonomy is fixed. For arbitrary natural-language questions, add a shared
question encoder and a query-conditioned decision layer (e.g. a Jeva-style set scorer).

## Losses and why proper scores come before RL

Let `p[k,c]` be the distribution over no event (c=0) and event causes 1..C
conditional on survival to time bucket k. Then probability of event c **by**
horizon h is `sum_{k<=h} survival_before_k * p[k,c]`, and probability of none
is `product_{k<=h} p[k,0]`. These sum to 1 and preserve monotonicity.

Stage 1 loss: `L = survival_NLL + .25 * multiclass_Brier + .3 * next_event_CE + .05 * next_gap_NLL`.
This is differentiable proper-score training, **not RLCD by itself**.

Stage 2, when meaningful latent expert/route choices are added:
- Keep a calibrated answer head and freeze a stage-1 reference policy.
- Draw M>=2 discrete route samples per state.
- Score **each complete output distribution** with negative Brier using an
  observed outcome; compute leave-one-out advantage.
- Train route selection via `-advantage * log pi(route|state) + beta * KL(pi||pi_ref)`.
- Recalibrate and compare with no-RL stage 1, as RL adds variance and might not help.

OpenJev-RLCD (2026) describes a related *two-stage calibrate then reinforce*
training recipe for reasoning models. Our route choice is a proposed adaptation;
it is not a faithful reproduction of that paper's reasoning model implementation.

## How to turn this into actual research

1. **Use real event logs with timestamps and outcomes.** Select an outcome taxonomy
   and practical forecast windows. Split by both event time and held-out workflows.
2. **Fix the data contract.** One row per event: `(system_id, trace_id, event_time,
   event_type, numeric_context, textual_context, action, observed_outcome)`.
   Build rolling prefixes with only information known at that prefix timestamp.
   Track observation/end-of-follow-up and censoring precisely.
3. **Strong baselines.** Kaplan-Meier/cause-specific hazards, LightGBM with temporal
   aggregates, sequence Transformer without RL, temporal point processes, and FEP.
4. **New backbone.** Add a semantic text encoder or embeddings for event descriptions;
   explicit action encoding; a query and horizon embedding; dataset-invariant event
   representation and optional in-context examples from the target log.
5. **Generalization.** Pretrain on many distinct event domains, then test true zero-shot
   transfer on held-out systems with disjoint event vocabularies. Ablate each component.
6. **Proper evidence.** Measure negative log-likelihood, multiclass and binary Brier,
   ECE/reliability plots, next-event accuracy, timing MAE/NLL, top-k prediction,
   selective prediction, and latency vs baselines. Evaluate discrimination and
   calibration for every horizon, including censoring-aware methods.
7. **RLCD ablation.** Only if route uncertainty or costly internal actions genuinely
   require a policy-gradient estimator, test calibrate-then-reinforce with a KL anchor
   against direct supervised calibration. Report three or more seeds.

### Important limitations

- The toy event generator does not represent any real-world population or business.
- Proper scoring rules incentivize honest probabilities **in expectation with sufficient
  data/capacity** but do not guarantee calibration on a distribution shift.
- The toy validation is independent synthetic data; real deployments require temporal,
  entity-level and system-level holdout splits.
- This is observational forecasting, **not causal effects of actions**. A predicted
  result under a hypothetical intervention needs causal identification/experiments.
- No clinical patient data should be used without proper authorization, contracts,
  privacy controls, and domain-specific validation.

## Open research worth reading (as of October 24, 2026)

- TypeSafe / Jev: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Jeva (open decision-native architecture): https://github.com/LiQiiiii/Jeva
- OpenJev-RLCD (independent calibrated RL implementation): https://github.com/ZimmyGao/openjev-rlcd
- Foundation Event Predictor (FEP): https://arxiv.org/abs/2610.17503
- Public process log benchmark sources: https://github.com/erphq/pm-bench

## Typed probability API example

After loading a trained model, query multiple horizons/outcomes with one forward
pass (questions are fixed-schema, **not** free-form English):

```python
from temporal_rlcd.interface import BooleanForecastQuestion, answer_questions

qs = [
    BooleanForecastQuestion('failure_1h', 'failure', 2),
    BooleanForecastQuestion('completion_24h', 'success', 3),
]
answers = answer_questions(model, events, gaps, lengths, qs)
# [[{'id': 'failure_1h', 'type': 'Noul', 'horizon': '60m',
#    'outcome': 'failure', 'probability': 0.0..1.0}, ...], ...]
```

`model`, `events`, `gaps`, and `lengths` are PyTorch objects described above.
To see an actual JSON output shape with random **untrained** weights, run
`python -m temporal_rlcd.demo`. For real production use, enforce versioned
outcome schemas and link horizon-bin semantics to actual elapsed seconds.
