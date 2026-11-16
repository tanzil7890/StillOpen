"""Optional RLCD-inspired latent-policy optimization experiment.

Unlike direct supervised optimization of calibrated probabilities, this can
use policy gradient because it samples a discrete *internal route/expert*.
It does NOT reproduce Jev's unpublished RLCD algorithm. It is an isolated
research building block: integrate only after calibrated baselines exist.
"""
import torch
from torch.nn import functional as F


def route_policy_loss(policy_logits: torch.Tensor, sampled_routes: torch.Tensor,
                      observed_scores: torch.Tensor, reference_logits: torch.Tensor,
                      kl_beta: float = 0.04):
    """REINFORCE on sampled routes under a strictly proper probability score.

    policy_logits: [B, K] over internal routes (no string generation)
    sampled_routes: [B, M] routes sampled from that policy
    observed_scores: [B, M] negative Brier scores of route-specific predictions
    reference_logits: [B, K] frozen stage-1 policy logits

    M>=2 is required for a leave-one-out reward baseline. This updates the
    route policy only, not the scoring models that produce the probabilities.
    """
    if sampled_routes.ndim != 2 or sampled_routes.size(1) < 2:
        raise ValueError('sampled_routes must have >=2 samples per example')
    if sampled_routes.shape != observed_scores.shape:
        raise ValueError('routes and scores shapes differ')
    logpolicy = F.log_softmax(policy_logits, dim=-1)
    log_prob = logpolicy.gather(-1, sampled_routes)
    m = observed_scores.size(1)
    baseline = (observed_scores.sum(dim=1, keepdim=True) - observed_scores) / (m - 1)
    advantages = (observed_scores - baseline).detach()
    pg_loss = -(advantages * log_prob).mean()
    pi = logpolicy.exp()
    logref = F.log_softmax(reference_logits.detach(), dim=-1)
    kl = (pi * (logpolicy - logref)).sum(-1).mean()
    return pg_loss + kl_beta * kl


def negative_brier_reward(predicted_distribution: torch.Tensor,
                          observed_label: torch.Tensor):
    """Proper score applied to a complete categorical probability distribution."""
    target = F.one_hot(observed_label, predicted_distribution.size(-1))
    return -(predicted_distribution - target).square().sum(-1)
