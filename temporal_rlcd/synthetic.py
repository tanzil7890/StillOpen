"""Small *synthetic* agent workflow dataset for code/tests; no empirical claims.

Each case uses only a prefix as its input. Labels describe the future first
terminal event (success/failure/escalation) and are sometimes right-censored.
The generator is deliberately simple and must be replaced with real traces.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Example:
    events: np.ndarray
    gaps: np.ndarray
    next_event: int
    next_gap: float
    outcome: int
    observed_bin: int


class SyntheticWorkflowDataset(Dataset):
    def __init__(self, n: int = 2000, max_events: int = 32, n_bins: int = 4,
                 seed: int = 0):
        self.max_events = max_events
        self.n_bins = n_bins
        self.data = []
        rng = np.random.default_rng(seed)
        # event ids: 1 navigate, 2 API error, 3 retry, 4 progress,
        #            5 manual intervention request, 6 external wait, 7 validation
        for _ in range(n):
            length = int(rng.integers(6, max_events + 1))
            mode = rng.choice(3, p=[0.45, 0.35, 0.20])
            probs = np.array([
                [0.24, 0.05, 0.05, 0.34, 0.02, 0.13, 0.17],
                [0.12, 0.28, 0.26, 0.10, 0.09, 0.12, 0.03],
                [0.13, 0.12, 0.10, 0.14, 0.25, 0.20, 0.06]
            ])[mode]
            events = rng.choice(np.arange(1, 8), size=length, p=probs).astype('int64')
            gaps = rng.lognormal(mean=0.2 + 0.3 * mode, sigma=0.7, size=length).astype('float32')
            recent = events[-8:]
            failures = np.mean(np.isin(recent, [2, 3]))
            interventions = np.mean(recent == 5)
            progress = np.mean(np.isin(recent, [4, 7]))
            # Probability distribution over *the first terminal event*, not all events.
            type_logits = np.array([1.1 + 4 * progress - 3 * failures,
                                    0.2 + 5 * failures,
                                    -0.4 + 6 * interventions + 1.2 * failures])
            type_probs = np.exp(type_logits - type_logits.max())
            type_probs /= type_probs.sum()
            event_rate = float(np.clip(0.09 + 0.20 * failures + 0.09 * interventions +
                                       0.12 * progress, 0.06, 0.7))
            true_outcome, true_bin = 0, n_bins - 1
            for j in range(n_bins):
                if rng.random() < event_rate:
                    true_outcome = int(rng.choice(np.arange(1, 4), p=type_probs))
                    true_bin = j
                    break
            # Observe an independent random follow-up length, possibly censoring an event.
            followup_end = int(rng.integers(0, n_bins)) if rng.random() < 0.3 else n_bins - 1
            if true_outcome and true_bin <= followup_end:
                outcome, observed_bin = true_outcome, true_bin
            else:
                outcome, observed_bin = 0, followup_end
            next_probs = probs.copy()
            next_probs[1] += 0.1 * failures
            next_probs = next_probs / next_probs.sum()
            next_event = int(rng.choice(np.arange(1, 8), p=next_probs))
            next_gap = float(rng.lognormal(0.2 + 0.3 * mode, 0.7))
            self.data.append(Example(events, gaps, next_event, next_gap, outcome, observed_bin))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        pad = self.max_events - len(ex.events)
        return {
            'events': torch.from_numpy(np.pad(ex.events, (0, pad))),
            'gaps': torch.from_numpy(np.pad(ex.gaps, (0, pad))),
            'lengths': torch.tensor(len(ex.events), dtype=torch.long),
            'next_event': torch.tensor(ex.next_event - 1, dtype=torch.long),
            'next_gap': torch.tensor(ex.next_gap, dtype=torch.float32),
            'outcome': torch.tensor(ex.outcome, dtype=torch.long),
            'observed_bin': torch.tensor(ex.observed_bin, dtype=torch.long)
        }
