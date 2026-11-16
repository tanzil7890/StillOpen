"""Validated, no-token-generation 'System One' question interface for fixed outcomes.

This v0 supports typed Boolean outcome-by-horizon questions, not arbitrary
natural-language prompts or intervention/counterfactual predictions.
"""
from dataclasses import dataclass
from typing import Literal, Sequence
import torch
from .model import forecast_distribution

OutcomeName = Literal['success', 'failure', 'escalation', 'any_terminal_event']
OUTCOME_INDEX = {'success': 1, 'failure': 2, 'escalation': 3}
TIME_WINDOWS = ('5m', '15m', '60m', '24h')


@dataclass(frozen=True)
class BooleanForecastQuestion:
    question_id: str
    outcome: OutcomeName
    horizon_bin: int

    def __post_init__(self):
        if not self.question_id or not isinstance(self.question_id, str):
            raise ValueError('question_id must be a nonempty string')
        if self.outcome not in {*OUTCOME_INDEX, 'any_terminal_event'}:
            raise ValueError(f'Unknown outcome: {self.outcome}')
        if not 0 <= self.horizon_bin < len(TIME_WINDOWS):
            raise ValueError('horizon_bin must be in [0, 3]')


@torch.no_grad()
def answer_questions(model, events: torch.Tensor, gaps: torch.Tensor,
                     lengths: torch.Tensor, questions: Sequence[BooleanForecastQuestion]):
    """Return typed probabilities for a batched history, one model forward pass."""
    if len(questions) == 0:
        raise ValueError('At least one question required')
    if model.n_bins != len(TIME_WINDOWS) or model.n_outcomes != len(OUTCOME_INDEX):
        raise ValueError('This interface expects the toy 4-bin, 3-cause model')
    model.eval()
    predictions = model(events, gaps, lengths)
    distribution = forecast_distribution(predictions['hazard_logits'])
    batch_answers = []
    for b in range(events.shape[0]):
        answers = []
        for q in questions:
            dist = distribution[b, q.horizon_bin]
            p = 1.0 - dist[0] if q.outcome == 'any_terminal_event' else dist[OUTCOME_INDEX[q.outcome]]
            answers.append({'id': q.question_id, 'type': 'Noul',
                            'horizon': TIME_WINDOWS[q.horizon_bin],
                            'outcome': q.outcome, 'probability': float(p.item())})
        batch_answers.append(answers)
    return batch_answers
