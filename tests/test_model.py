import torch
from temporal_rlcd.model import (TemporalDecisionModel, survival_nll,
                                 brier_loss, forecast_distribution)
from temporal_rlcd.rlcd import route_policy_loss, negative_brier_reward


def test_shapes_and_probabilities():
    torch.manual_seed(7)
    model = TemporalDecisionModel(dropout=0.).eval()
    events = torch.tensor([[1, 3, 5, 0, 0], [2, 4, 1, 3, 7]])
    gaps = torch.tensor([[1., 2., 4., 0., 0.], [1., 1., 4., 2., 3.]])
    out = model(events, gaps, torch.tensor([3, 5]))
    assert out['next_event_logits'].shape == (2, 7)
    assert out['hazard_logits'].shape == (2, 4, 4)
    dist = forecast_distribution(out['hazard_logits'])
    assert torch.allclose(dist.sum(-1), torch.ones(2, 4), atol=1e-5)
    assert bool((dist[:, 1:, 1:] >= dist[:, :-1, 1:] - 1e-6).all())
    assert bool((dist[:, 1:, 0] <= dist[:, :-1, 0] + 1e-6).all())


def test_censoring_objectives_and_gradients():
    torch.manual_seed(9)
    x = torch.randn(2, 4, 4, requires_grad=True)
    outcome = torch.tensor([0, 2])
    k = torch.tensor([1, 3])
    loss = survival_nll(x, outcome, k) + brier_loss(x, outcome, k)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(x.grad).all()
    # Scores cannot depend on future bins after censoring.
    only_censored = torch.randn(1, 4, 4, requires_grad=True)
    brier_loss(only_censored, torch.tensor([0]), torch.tensor([1])).backward()
    assert torch.allclose(only_censored.grad[:, 2:], torch.zeros_like(only_censored.grad[:, 2:]))


def test_rlcd_like_policy_gradient():
    torch.manual_seed(10)
    logits = torch.randn(2, 3, requires_grad=True)
    routes = torch.tensor([[0, 1, 0], [1, 2, 2]])
    predictions = torch.softmax(torch.randn(2, 3, 4), -1)
    outcomes = torch.tensor([[0, 2, 0], [1, 1, 1]])
    reward = negative_brier_reward(predictions, outcomes)
    loss = route_policy_loss(logits, routes, reward, torch.zeros_like(logits))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_typed_questions_are_valid_and_one_forward():
    from temporal_rlcd.interface import BooleanForecastQuestion, answer_questions
    model = TemporalDecisionModel().eval()
    events = torch.tensor([[1, 3, 4]])
    gaps = torch.tensor([[1., 2., 2.]])
    qs = [BooleanForecastQuestion('fail60m', 'failure', 2),
          BooleanForecastQuestion('anything24h', 'any_terminal_event', 3)]
    ans = answer_questions(model, events, gaps, torch.tensor([3]), qs)
    assert len(ans) == 1 and len(ans[0]) == 2
    assert all(0. <= a['probability'] <= 1. for a in ans[0])
    assert ans[0][0]['horizon'] == '60m'
    import pytest
    with pytest.raises(ValueError):
        BooleanForecastQuestion('invalid', 'failure', 99)
