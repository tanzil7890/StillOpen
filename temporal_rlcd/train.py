"""Minimal supervised/proper-score stage. No claim to train a foundation model."""
import argparse
import json
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .model import TemporalDecisionModel, survival_nll, brier_loss, next_gap_nll, forecast_distribution
from .synthetic import SyntheticWorkflowDataset


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch['events'], batch['gaps'], batch['lengths'])
        loss = (survival_nll(out['hazard_logits'], batch['outcome'], batch['observed_bin'])
                + 0.25 * brier_loss(out['hazard_logits'], batch['outcome'], batch['observed_bin'])
                + 0.3 * F.cross_entropy(out['next_event_logits'], batch['next_event'])
                + 0.05 * next_gap_nll(out['next_gap_mu'], out['next_gap_logsigma'], batch['next_gap']))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(batch['outcome'])
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_brier, count = 0., 0
    # Calibration and discrimination reported only for fully observable final horizons.
    risks, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch['events'], batch['gaps'], batch['lengths'])
        total_brier += brier_loss(out['hazard_logits'], batch['outcome'],
                                  batch['observed_bin']).item() * len(batch['outcome'])
        count += len(batch['outcome'])
        valid = (batch['outcome'] > 0) | (batch['observed_bin'] == model.n_bins - 1)
        dist = forecast_distribution(out['hazard_logits'])[:, -1, :]
        risks.extend((1.0 - dist[valid, 0]).cpu().tolist())
        labels.extend((batch['outcome'][valid] > 0).float().cpu().tolist())
    import numpy as np
    p = np.asarray(risks)
    y = np.asarray(labels)
    binary_brier = float(np.mean((p - y)**2)) if len(p) else float('nan')
    ece = 0.
    if len(p):
        for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            sel = ((p >= lo) & (p < hi)) if hi < 1 else ((p >= lo) & (p <= hi))
            if np.any(sel):
                ece += sel.mean() * abs(float(y[sel].mean() - p[sel].mean()))
    return {'approx_mean_multiclass_brier': total_brier / max(count, 1),
            'final_horizon_binary_brier_uncensored_only': binary_brier,
            'final_horizon_ece_10bins_uncensored_only': float(ece),
            'final_horizon_n_scored': len(p)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-n', type=int, default=2000)
    parser.add_argument('--test-n', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='temporal_rlcd_toy.pt')
    args = parser.parse_args()
    torch.manual_seed(42)
    if args.device == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    tr = SyntheticWorkflowDataset(args.train_n, seed=101)
    val = SyntheticWorkflowDataset(args.test_n, seed=909)
    model = TemporalDecisionModel(d_model=args.d_model).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    for epoch in range(args.epochs):
        loss = train_epoch(model, tr_loader, optimizer, args.device)
        metrics = evaluate(model, val_loader, args.device)
        print(json.dumps({'epoch': epoch + 1, 'train_loss': round(loss, 4), **metrics}))
    torch.save({'model_state': model.state_dict(), 'config': {'d_model': args.d_model,
               'n_event_types': 8, 'n_outcomes': 3, 'n_bins': 4}}, args.output)
    print('Wrote:', args.output)


if __name__ == '__main__':
    main()
