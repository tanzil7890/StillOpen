"""Tree baseline sensitivity on the corrected censor-aware labels (cutoff 2012-01-01)."""
import pickle, sys
from datetime import datetime, timezone
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, sys.argv[1])
import pilot
from pilot import build, labels, score, person_period, compose, EDGES, H, DAY, NCLS

def fit_predict(Xtr, out, obs, Xte, **kw):
    xpp, ypp = person_period(Xtr, out, obs)
    clf = HistGradientBoostingClassifier(learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                         categorical_features=[3, 4], random_state=0, **kw).fit(xpp, ypp)
    p = np.zeros((len(Xte), H, NCLS))
    for j in range(H):
        p[:, j, clf.classes_] = clf.predict_proba(np.column_stack([Xte, np.full(len(Xte), j)]))
    p[:, :, 0] += 1 - p.sum(-1)
    return compose(p)

data = pickle.load(open(sys.argv[2], 'rb'))
tau = datetime(2012, 1, 1, tzinfo=timezone.utc).timestamp()
cases, V = data['cases'], len(data['vocab'])
cid, kk, tp, X = build(cases, V)
start = np.array([c['ts'][0] for c in cases])[cid]
t_term = np.array([c['ts'][c['term_idx']] if c['cause'] else np.inf for c in cases])[cid]
cause = np.array([c['cause'] for c in cases])[cid]
log_end = max(c['ts'][-1] for c in cases)
te = (start > tau) & (start <= tau + 21 * DAY) & (tp <= log_end - 30 * DAY)
Y = np.where(((t_term[te] - tp[te]) / DAY)[:, None] <= EDGES[None, :], cause[te][:, None], 0)
same = [c for c in range(X.shape[1]) if c not in (5, 6, 7)]
cfgs = {'early_stop_400': dict(max_iter=400, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15),
        'fixed_150': dict(max_iter=150, early_stopping=False), 'fixed_60': dict(max_iter=60, early_stopping=False),
        'fixed_300': dict(max_iter=300, early_stopping=False)}
for w in (4.0, 13.0):
    tr = (start > tau - w * 7 * DAY) & (start <= tau) & (tp <= tau)
    n_ident = int((EDGES < w * 7).sum())
    keep, out, obs = labels(tp[tr], t_term[tr], cause[tr], tau, 'censor_aware')
    for feats, cols in (('same_info', same), ('rich', list(range(X.shape[1])))):
        for name, kw in cfgs.items():
            _, m = score(fit_predict(X[tr][keep][:, cols], out[keep], obs[keep], X[te][:, cols], **kw), Y, n_ident)
            print(f"weeks {w:>4.0f} {feats:<9} {name:<15} brier_id {m['brier_ident']:.4f} ece_id {m['ece_any_ident']:.3f} auc7d {m['auc_any_7d']:.3f}", flush=True)
