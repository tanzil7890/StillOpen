"""Two more model classes for study.py. Each takes what the boosted trees take and returns per-window hazards
[test histories, windows, classes], so every treatment of open cases can be repeated with a different kind of model.

  logit  multinomial logistic regression on one row per (history, window): the textbook discrete-time hazard model,
         linear in the features, with one intercept per window
  gru    a small recurrent network that reads the events of a case in order and gives the hazards of every window
         after each event

Both use fixed settings and no early stopping, so that nothing is tuned on data from after the cutoff.
"""
import warnings

import numpy as np

# columns of study.build(): history length, log hours since the case began, log hours since the previous event,
# activity, previous activity, log amount, hour of day, weekday, then one count per activity
NUM, ACT, PREV, COUNTS = [0, 1, 2, 5, 6, 7], 3, 4, 8


def _scale(Xtr, Xte, cols, centre):
    """Scale columns by what the training rows show; missing values become the training mean."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # a column that is missing everywhere
        mu = np.nan_to_num(np.nanmean(Xtr[:, cols], axis=0)) if centre else 0.0
        sd = np.nan_to_num(np.nanstd(Xtr[:, cols], axis=0))
    sd = np.where(sd > 0, sd, 1.0)
    return [np.nan_to_num((X[:, cols] - mu) / sd) for X in (Xtr, Xte)]


def logit(Xtr, outcome, obs, Xte, H, ncls):
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    V = Xtr.shape[1] - COUNTS
    num, cnt = _scale(Xtr, Xte, NUM, True), _scale(Xtr, Xte, slice(COUNTS, None), False)

    def onehot(values, width):
        n = len(values)
        return sparse.csr_matrix((np.ones(n), (np.arange(n), np.clip(values.astype(int), 0, width - 1))), shape=(n, width))

    def design(X, i):
        return sparse.hstack([sparse.csr_matrix(num[i]), sparse.csr_matrix(cnt[i]), onehot(X[:, ACT], V + 1),
                              onehot(X[:, PREV], V + 1)], format='csr')

    def in_window(F, j):
        return sparse.hstack([F, onehot(np.full(F.shape[0], j), H)], format='csr')

    Ftr, Fte = design(Xtr, 0), design(Xte, 1)
    xs, ys = [], []
    for j in range(H):
        m = obs >= j
        xs.append(in_window(Ftr[m], j))
        ys.append(np.where((obs[m] == j) & (outcome[m] > 0), outcome[m], 0))
    clf = LogisticRegression(C=1.0, max_iter=300)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # "did not converge" on the largest sets; 300 steps is the fixed budget
        clf.fit(sparse.vstack(xs, format='csr'), np.concatenate(ys))
    p = np.zeros((len(Xte), H, ncls))
    for j in range(H):
        p[:, j, clf.classes_] = clf.predict_proba(in_window(Fte, j))
    return p


def gru(Xtr, outcome, obs, case_tr, Xte, case_te, H, ncls, epochs=30, min_steps=300, seed=0):
    """The rows of one case are adjacent and in event order in both sets (study.py keeps them so)."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    V = Xtr.shape[1] - COUNTS
    num = _scale(Xtr, Xte, NUM, True)

    def pad(X, feats, case):
        new = np.r_[True, case[1:] != case[:-1]]
        which = np.cumsum(new) - 1                       # row -> position of its case
        step = np.arange(len(case)) - np.flatnonzero(new)[which]  # row -> position of its event in the case
        n, T = int(which[-1]) + 1, int(step.max()) + 1
        act, x = torch.zeros(n, T, dtype=torch.long), torch.zeros(n, T, len(NUM))
        act[which, step] = torch.from_numpy(np.clip(X[:, ACT], 0, V).astype(np.int64))
        x[which, step] = torch.from_numpy(feats).float()
        return act, x, which, step

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.gru = nn.Embedding(V + 1, 16), nn.GRU(16 + len(NUM), 64, batch_first=True)
            self.drop, self.out = nn.Dropout(0.1), nn.Linear(64, H * ncls)

        def forward(self, act, x):
            h, _ = self.gru(torch.cat([self.emb(act), x], -1))
            return self.out(self.drop(h)).view(*act.shape, H, ncls)

    act, x, which, step = pad(Xtr, num[0], case_tr)
    windows = np.arange(H)[None, :]
    target = torch.zeros(*act.shape, H, dtype=torch.long)
    at_risk = torch.zeros(*act.shape, H, dtype=torch.bool)
    target[which, step] = torch.from_numpy(np.where((obs[:, None] == windows) & (outcome[:, None] > 0), outcome[:, None], 0))
    at_risk[which, step] = torch.from_numpy(obs[:, None] >= windows)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    n, batch = len(act), 128
    for _ in range(max(epochs, -(-min_steps // -(-n // batch)))):
        order = torch.randperm(n)
        for s in range(0, n, batch):
            b = order[s:s + batch]
            if not at_risk[b].any():
                continue
            loss = nn.functional.cross_entropy(net(act[b], x[b])[at_risk[b]], target[b][at_risk[b]])
            opt.zero_grad(); loss.backward(); opt.step()

    net.eval()
    act, x, which, step = pad(Xte, num[1], case_te)
    with torch.no_grad():
        p = torch.cat([net(act[s:s + 2048], x[s:s + 2048]).softmax(-1) for s in range(0, len(act), 2048)])
    return p[which, step].double().numpy()
