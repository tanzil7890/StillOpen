"""Time-aware, no-text-generation, typed probabilistic forecasts.

This is an original reference implementation, not Jev's architecture or RLCD.
Input: event IDs, inter-event time gaps, valid lengths.
Output: next-event distribution, next-gap lognormal parameters, and competing-risk
hazards over preset future time buckets. First terminal event has mutually
exclusive causes 1..C, with cause 0 meaning no terminal event in that bucket.
"""
import torch
from torch import nn
from torch.nn import functional as F


class TemporalDecisionModel(nn.Module):
    def __init__(self, n_event_types: int = 8, n_outcomes: int = 3, n_bins: int = 4,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
                 max_events: int = 128):
        super().__init__()
        self.n_event_types = n_event_types
        self.n_outcomes = n_outcomes
        self.n_bins = n_bins
        self.max_events = max_events
        self.embedding = nn.Embedding(n_event_types, d_model, padding_idx=0)
        self.time_embed = nn.Sequential(nn.Linear(2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.position_embed = nn.Embedding(max_events, d_model)
        block = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                           dim_feedforward=d_model * 4, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.next_event_head = nn.Linear(d_model, n_event_types - 1)  # excludes PAD id 0
        self.next_gap_head = nn.Linear(d_model, 2)
        self.hazard_head = nn.Linear(d_model, n_bins * (n_outcomes + 1))

    def forward(self, events: torch.Tensor, gaps: torch.Tensor, lengths: torch.Tensor):
        """events [B,T] padded on right; gaps [B,T] nonnegative; lengths [B]."""
        if events.ndim != 2 or gaps.shape != events.shape:
            raise ValueError('events and gaps must both be [B, T]')
        b, t = events.shape
        if t > self.max_events:
            raise ValueError('sequence exceeds max_events')
        if torch.any(lengths < 1) or torch.any(lengths > t):
            raise ValueError('all examples require 1..T history events')
        positions = torch.arange(t, device=events.device)
        padding = positions.unsqueeze(0) >= lengths.unsqueeze(1)
        # Delta time and time since the first recorded event. Both are known at decision time.
        log_gap = torch.log1p(gaps.clamp(min=0))
        age = torch.log1p(gaps.clamp(min=0).cumsum(dim=1))
        time_features = torch.stack([log_gap, age], dim=-1)
        h = (self.embedding(events) + self.time_embed(time_features)
             + self.position_embed(positions).unsqueeze(0))
        # Attention only to current and previous observations; no future leakage.
        causal = torch.triu(torch.ones(t, t, device=events.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=causal, src_key_padding_mask=padding)
        final = self.norm(h[torch.arange(b, device=events.device), lengths - 1])
        gap_params = self.next_gap_head(final)
        # One parallel call yields the complete distribution, no string decoding.
        return {
            'next_event_logits': self.next_event_head(final),
            'next_gap_mu': gap_params[:, 0],
            'next_gap_logsigma': gap_params[:, 1].clamp(-4, 3),
            'hazard_logits': self.hazard_head(final).view(b, self.n_bins, self.n_outcomes + 1),
        }


def forecast_distribution(hazard_logits: torch.Tensor):
    """Return P(no terminal event, first event of cause 1,...,C) by each bin.

    shape: [B, H, C+1]; each slice sums to 1; cumulative event risk is monotone.
    """
    hazards = F.softmax(hazard_logits, dim=-1)
    no_event = hazards[..., :1]  # [B,H,1] conditional survival in each bin
    survival = no_event.cumprod(dim=1)
    before = torch.cat([torch.ones_like(no_event[:, :1]), survival[:, :-1]], dim=1)
    first_event_mass = before * hazards[..., 1:]
    first_event_cdf = first_event_mass.cumsum(dim=1)
    return torch.cat([survival, first_event_cdf], dim=-1)


def survival_nll(hazard_logits: torch.Tensor, outcome: torch.Tensor,
                 observed_bin: torch.Tensor):
    """Right-censored likelihood.

    outcome 0: none observed through observed_bin (right-censored).
    outcome 1..C: first terminal event observed in observed_bin.
    All bins are 0-based and represent full observed intervals, not partial ones.
    """
    logp = F.log_softmax(hazard_logits, dim=-1)
    b, h, c = logp.shape
    if torch.any(observed_bin < 0) or torch.any(observed_bin >= h):
        raise ValueError('invalid observed_bin')
    if torch.any(outcome < 0) or torch.any(outcome >= c):
        raise ValueError('invalid outcome')
    prefix = logp[..., 0].cumsum(dim=1)
    k = observed_bin.view(-1, 1)
    prev = prefix.gather(1, (k - 1).clamp(min=0)).squeeze(1)
    prev = torch.where(observed_bin > 0, prev, torch.zeros_like(prev))
    event_at_k = logp[torch.arange(b, device=logp.device), observed_bin, outcome]
    censored = prefix.gather(1, k).squeeze(1)
    loglik = torch.where(outcome != 0, prev + event_at_k, censored)
    return -loglik.mean()


def brier_loss(hazard_logits: torch.Tensor, outcome: torch.Tensor,
               observed_bin: torch.Tensor):
    """Multiclass, proper Brier objective; never score unobserved future horizons.

    The known label at each scored bin is 0 (no terminal event yet) or the
    observed first terminal cause. If right censored at k, exclude bins > k.
    """
    dist = forecast_distribution(hazard_logits)
    b, h, c = dist.shape
    horizon = torch.arange(h, device=dist.device).view(1, h)
    event_occurred = (outcome.unsqueeze(1) > 0) & (horizon >= observed_bin.unsqueeze(1))
    target = torch.where(event_occurred, outcome.unsqueeze(1), torch.zeros_like(horizon))
    target = target.expand(b, h)
    known = (outcome.unsqueeze(1) > 0) | (horizon <= observed_bin.unsqueeze(1))
    labels = F.one_hot(target.long(), num_classes=c).to(dist.dtype)
    scores = (dist - labels).square().sum(dim=-1)
    return (scores * known).sum() / known.sum().clamp(min=1)


def next_gap_nll(mu: torch.Tensor, logsigma: torch.Tensor, next_gap: torch.Tensor):
    """Gaussian NLL for log(1 + delta time); data >= 0."""
    x = torch.log1p(next_gap.clamp(min=0))
    return (0.5 * ((x - mu) / logsigma.exp()).square() + logsigma).mean()
