"""Compact tabular MLP (PyTorch, CPU) used as a diverse ensemble member."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _prep(Xtr, Xva, Xte, cats):
    from sklearn.preprocessing import QuantileTransformer
    num_cols = [c for c in Xtr.columns if c not in cats]
    out = {}
    med = Xtr[num_cols].median() if num_cols else None
    na_cols = [c for c in num_cols if Xtr[c].isna().any()]
    qt = None
    if num_cols:
        qt = QuantileTransformer(n_quantiles=min(1000, max(10, len(Xtr) // 2)), output_distribution="normal", subsample=300000, random_state=0)
        qt.fit(Xtr[num_cols].fillna(med).fillna(0.0).values)
    card = []
    for c in cats:
        card.append(int(len(Xtr[c].cat.categories)) + 1 if isinstance(Xtr[c].dtype, pd.CategoricalDtype) else 2)
    for name, D in (("tr", Xtr), ("va", Xva), ("te", Xte)):
        if D is None:
            out[name] = None
            continue
        parts = []
        if num_cols:
            parts.append(qt.transform(D[num_cols].fillna(med).fillna(0.0).values))
            if na_cols:
                parts.append(D[na_cols].isna().values.astype(float))
        Xn = np.concatenate(parts, 1).astype(np.float32) if parts else np.zeros((len(D), 1), np.float32)
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=5.0, neginf=-5.0)
        Xc = np.zeros((len(D), len(cats)), np.int64)
        for j, c in enumerate(cats):
            if isinstance(D[c].dtype, pd.CategoricalDtype):
                Xc[:, j] = D[c].cat.codes.values.astype(np.int64) + 1
        out[name] = (Xn, Xc)
    return out, card


def fit_mlp(spec, problem, Xtr, ytr, Xva, yva, Xte, wtr, n_jobs, deadline, cats):
    import torch
    import torch.nn as nn
    torch.set_num_threads(max(1, n_jobs))
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    cats = [c for c in cats if c in Xtr.columns]
    data, card = _prep(Xtr, Xva, Xte, cats)
    task = problem.task
    K = problem.n_classes
    p = spec.params
    hidden = int(p.get("hidden", 256))
    drop = float(p.get("dropout", 0.15))
    lr = float(p.get("lr", 2e-3))
    epochs = int(p.get("epochs", 40))
    bs = int(p.get("batch_size", 512 if len(Xtr) < 100000 else 1024))
    patience = int(p.get("patience", 5))

    if task == "regression":
        mu, sd = float(np.mean(ytr)), float(np.std(ytr) + 1e-9)
        yt = ((ytr - mu) / sd).astype(np.float32)
        yv = ((yva - mu) / sd).astype(np.float32)
        out_dim = 1
    elif task == "binary":
        yt, yv, out_dim = ytr.astype(np.float32), yva.astype(np.float32), 1
    else:
        yt, yv, out_dim = ytr.astype(np.int64), yva.astype(np.int64), K

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(c, int(min(16, max(2, round(1.6 * c ** 0.56))))) for c in card])
            d = data["tr"][0].shape[1] + sum(e.embedding_dim for e in self.embs)
            self.mlp = nn.Sequential(
                nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(drop),
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(drop),
                nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.SiLU(), nn.Dropout(drop / 2),
                nn.Linear(hidden // 2, out_dim))

        def forward(self, xn, xc):
            z = [xn] + [e(xc[:, j].clamp(0, e.num_embeddings - 1)) for j, e in enumerate(self.embs)]
            return self.mlp(torch.cat(z, 1))

    net = Net()
    if task == "regression":
        loss_fn = nn.L1Loss(reduction="none") if problem.metric in ("mae", "medae") else nn.MSELoss(reduction="none")
    elif task == "binary":
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    else:
        loss_fn = nn.CrossEntropyLoss(reduction="none")
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    n = len(yt)
    steps = max(1, epochs * ((n + bs - 1) // bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    Xn_tr, Xc_tr = [torch.from_numpy(a) for a in data["tr"]]
    Xn_va, Xc_va = [torch.from_numpy(a) for a in data["va"]]
    y_tr_t = torch.from_numpy(yt)
    y_va_t = torch.from_numpy(yv)
    w_t = torch.from_numpy((wtr if wtr is not None else np.ones(n)).astype(np.float32))

    def predict(Xn, Xc):
        net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xn), 8192):
                outs.append(net(Xn[i:i + 8192], Xc[i:i + 8192]))
        o = torch.cat(outs)
        if task == "regression":
            return o.squeeze(1).numpy() * sd + mu
        if task == "binary":
            return torch.sigmoid(o.squeeze(1)).numpy()
        return torch.softmax(o, 1).numpy()

    def val_loss():
        net.eval()
        with torch.no_grad():
            o = net(Xn_va, Xc_va)
            if out_dim == 1:
                o = o.squeeze(1)
            return float(loss_fn(o, y_va_t).mean())

    best, best_state, bad, ep_done = np.inf, None, 0, 0
    step = 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            o = net(Xn_tr[b], Xc_tr[b])
            if out_dim == 1:
                o = o.squeeze(1)
            loss = (loss_fn(o, y_tr_t[b]) * w_t[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step < steps - 1:
                sched.step()
                step += 1
        ep_done = ep + 1
        vl = val_loss()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if deadline is not None and time.time() > deadline:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    pva = predict(Xn_va, Xc_va)
    pte = None
    if data["te"] is not None:
        Xn_te, Xc_te = [torch.from_numpy(a) for a in data["te"]]
        pte = predict(Xn_te, Xc_te)
    return pva, pte, ep_done, None
