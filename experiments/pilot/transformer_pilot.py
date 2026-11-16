"""The repo's TemporalDecisionModel + losses, unchanged, on BPI 2012 (same test set as pilot.py).

Compares: boosted trees restricted to the same information (event types + times),
the Transformer trained with survival_nll only, and with the repo's recipe
survival_nll + 0.25 * brier_loss.
"""
import json, pickle, sys, time
from datetime import datetime, timezone
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])            # repo (trusted)
sys.path.insert(0, sys.argv[2])            # my scripts dir
from temporal_rlcd.model import TemporalDecisionModel, survival_nll, brier_loss, forecast_distribution
from pilot import build, labels, score, fit_predict, boot_diff, EDGES, H, DAY, MAXK


def sequences(cases, cid, kk):
    E = np.zeros((len(cid), MAXK), dtype=np.int64)
    G = np.zeros((len(cid), MAXK), dtype=np.float32)
    for i in np.unique(cid):
        rows = np.where(cid == i)[0]
        acts, ts = cases[i]['acts'], cases[i]['ts']
        gaps = np.concatenate([[0.0], np.diff(ts)]) / 3600
        for r, k in zip(rows, kk[rows]):
            E[r, :k], G[r, :k] = acts[:k], gaps[:k]
    return E, G


def train_eval(E, G, L, out, obs, groups, Ete, Gte, Lte, V, use_brier, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    val_cases = set(rng.choice(ug, size=int(0.15 * len(ug)), replace=False).tolist())
    is_val = np.array([g in val_cases for g in groups])
    t = lambda a: torch.from_numpy(a)
    tr = [t(a[~is_val]) for a in (E, G, L, out, obs)]
    va = [t(a[is_val]) for a in (E, G, L, out, obs)]
    model = TemporalDecisionModel(n_event_types=V + 1, n_outcomes=3, n_bins=H, d_model=64, max_events=MAXK)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    best, best_state, bad, n = float('inf'), None, 0, len(tr[0])
    for epoch in range(12):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, 256):
            b = perm[s:s + 256]
            hz = model(tr[0][b], tr[1][b], tr[2][b])['hazard_logits']
            loss = survival_nll(hz, tr[3][b], tr[4][b])
            if use_brier:
                loss = loss + 0.25 * brier_loss(hz, tr[3][b], tr[4][b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = np.mean([survival_nll(model(va[0][s:s + 2048], va[1][s:s + 2048], va[2][s:s + 2048])['hazard_logits'],
                                      va[3][s:s + 2048], va[4][s:s + 2048]).item() * len(va[0][s:s + 2048])
                         for s in range(0, len(va[0]), 2048)]) * (len(range(0, len(va[0]), 2048)) / len(va[0]))
        if v < best - 1e-4:
            best, bad, best_state = v, 0, {k: x.clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        dist = torch.cat([forecast_distribution(model(t(Ete[s:s + 2048]), t(Gte[s:s + 2048]), t(Lte[s:s + 2048]))['hazard_logits'])
                          for s in range(0, len(Ete), 2048)]).numpy()
    return dist, epoch + 1


def main():
    data = pickle.load(open(sys.argv[3], 'rb'))
    tau = datetime.fromisoformat(sys.argv[4]).replace(tzinfo=timezone.utc).timestamp()
    torch.set_num_threads(4)
    cases, V = data['cases'], len(data['vocab'])
    cid, kk, tp, X = build(cases, V)
    E, G = sequences(cases, cid, kk)
    start = np.array([c['ts'][0] for c in cases])[cid]
    t_term = np.array([c['ts'][c['term_idx']] if c['cause'] else np.inf for c in cases])[cid]
    cause = np.array([c['cause'] for c in cases])[cid]
    log_end = max(c['ts'][-1] for c in cases)
    te = (start > tau) & (start <= tau + 21 * DAY) & (tp <= log_end - 30 * DAY)
    Y = np.where(((t_term[te] - tp[te]) / DAY)[:, None] <= EDGES[None, :], cause[te][:, None], 0)
    same_info = [c for c in range(X.shape[1]) if c not in (5, 6, 7)]   # drop amount, hour, weekday
    for w in [float(x) for x in sys.argv[5].split(',')]:
        tr = (start > tau - w * 7 * DAY) & (start <= tau) & (tp <= tau)
        n_ident = int((EDGES < w * 7).sum())
        keep, out, obs = labels(tp[tr], t_term[tr], cause[tr], tau, 'censor_aware')
        res = {'train_weeks': w, 'train_prefixes': int(keep.sum())}
        per_tree, res['trees_same_info'] = score(fit_predict(X[tr][keep][:, same_info], out[keep], obs[keep], X[te][:, same_info]), Y, n_ident)
        for name, use_brier in (('transformer_nll', False), ('transformer_nll_plus_repo_brier', True)):
            for seed in (0, 1, 2):
                t0 = time.time()
                dist, ep = train_eval(E[tr][keep], G[tr][keep], kk[tr][keep], out[keep], obs[keep], cid[tr][keep],
                                      E[te], G[te], kk[te], V, use_brier, seed)
                per, m = score(dist, Y, n_ident)
                m.update(epochs=ep, train_s=round(time.time() - t0),
                         brier_minus_trees_95ci=boot_diff(per, per_tree, cid[te]))
                res[f'{name}_seed{seed}'] = m
        print(json.dumps(res))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
