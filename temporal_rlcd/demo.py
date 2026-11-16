"""Show the typed JSON output from an untrained toy model, no claims of accuracy."""
import json
import torch
from .model import TemporalDecisionModel, forecast_distribution


def demo():
    torch.manual_seed(123)
    model = TemporalDecisionModel().eval()
    events = torch.tensor([[1, 2, 3, 2, 3, 4, 0, 0]])
    gaps = torch.tensor([[1., 8., 5., 14., 4., 6., 0., 0.]])
    with torch.no_grad():
        raw = model(events, gaps, torch.tensor([6]))
        dist = forecast_distribution(raw['hazard_logits'])[0]
        next_probs = raw['next_event_logits'].softmax(-1)[0]
    print(json.dumps({
        'notice': 'UNTRAINED toy model: probabilities are not meaningful',
        'outcomes': ['none', 'success', 'failure', 'escalation'],
        'time_bins': ['0-5m', '5-15m', '15-60m', '60m-1d'],
        'by_horizon_probability_distribution': dist.tolist(),
        'next_event_type_id': int(next_probs.argmax().item() + 1),
        'next_event_probabilities': next_probs.tolist(),
    }, indent=2))


if __name__ == '__main__':
    demo()
