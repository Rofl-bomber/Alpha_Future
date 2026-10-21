"""MLP на torch (CPU): quantile transform числовых, эмбеддинги категорий, early stopping."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from .base import ModelBase


class NNPrep:
    def __init__(self, cat_cols):
        self.cat_cols = list(cat_cols)

    def fit(self, X: pd.DataFrame, seed: int):
        self.num_cols = [c for c in X.columns if c not in self.cat_cols]
        num = X[self.num_cols].astype(float).values if self.num_cols else np.zeros((len(X), 0))
        self.na_idx = [i for i in range(num.shape[1]) if np.isnan(num[:, i]).any()]
        if num.shape[1]:
            self.qt = QuantileTransformer(n_quantiles=min(1000, max(10, len(X) // 2)),
                                          output_distribution="normal", subsample=100_000, random_state=seed)
            self.qt.fit(num)
        self.cards = []
        for c in self.cat_cols:
            n_cat = len(X[c].cat.categories) if isinstance(X[c].dtype, pd.CategoricalDtype) else int(X[c].nunique())
            self.cards.append(n_cat + 1)  # 0 — пропуск/неизвестное
        return self

    def transform(self, X: pd.DataFrame):
        if self.num_cols:
            num = X[self.num_cols].astype(float).values
            z = self.qt.transform(num)
            na = np.isnan(z)
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            if self.na_idx:
                z = np.hstack([z, na[:, self.na_idx].astype(float)])
        else:
            z = np.zeros((len(X), 0))
        cats = np.zeros((len(X), len(self.cat_cols)), dtype=np.int64)
        for j, c in enumerate(self.cat_cols):
            codes = X[c].cat.codes.values.astype(np.int64) + 1 if isinstance(X[c].dtype, pd.CategoricalDtype) \
                else pd.Categorical(X[c]).codes.astype(np.int64) + 1
            cats[:, j] = np.clip(codes, 0, self.cards[j] - 1)
        return z.astype(np.float32), cats

    @property
    def n_num(self):
        return (len(self.num_cols) + len(self.na_idx)) if self.num_cols else 0


def _emb_dim(card: int) -> int:
    return int(min(16, max(2, round(1.6 * card ** 0.56))))


def build_net(kind: str, n_num: int, cards: list[int], out_dim: int, p: dict):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(c, _emb_dim(c)) for c in cards])
            d_in = n_num + sum(_emb_dim(c) for c in cards)
            d_in = max(d_in, 1)
            drop = float(p.get("dropout", 0.1 if kind == "mlp" else 0.15))
            if kind == "mlp":
                h = int(p.get("hidden", 256))
                layers, d = [], d_in
                for _ in range(int(p.get("n_layers", 2))):
                    layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
                    d = h
                self.body = nn.Sequential(*layers)
                self.head = nn.Linear(d, out_dim)
            else:
                d = int(p.get("d_main", 192))
                self.inp = nn.Linear(d_in, d)
                hidden = int(d * float(p.get("d_hidden_factor", 2.0)))
                self.blocks = nn.ModuleList([
                    nn.Sequential(nn.BatchNorm1d(d), nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(hidden, d), nn.Dropout(drop))
                    for _ in range(int(p.get("n_blocks", 3)))
                ])
                self.head = nn.Sequential(nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, out_dim))

        def forward(self, xn, xc):
            parts = [xn] if xn.shape[1] else []
            parts += [e(xc[:, j]) for j, e in enumerate(self.embs)]
            x = torch.cat(parts, dim=1) if parts else torch.zeros((xn.shape[0], 1))
            if kind == "mlp":
                return self.head(self.body(x))
            x = self.inp(x)
            for b in self.blocks:
                x = x + b(x)
            return self.head(x)

    return Net()


class MLPModel(ModelBase):
    name = "mlp"
    kind = "mlp"
    defaults = dict(lr=1e-3, weight_decay=1e-5, batch_size=512, max_epochs=100, patience=8)
    param_schema = {
        "lr": ("float", 1e-5, 0.1),
        "weight_decay": ("float", 0.0, 0.1),
        "batch_size": ("int", 32, 8192),
        "max_epochs": ("int", 1, 500),
        "patience": ("int", 1, 50),
        "hidden": ("int", 16, 2048),
        "n_layers": ("int", 1, 8),
        "dropout": ("float", 0.0, 0.9),
    }
    param_aliases = {
        "learning_rate": "lr", "epochs": "max_epochs", "n_epochs": "max_epochs", "num_epochs": "max_epochs",
        "hidden_size": "hidden", "hidden_dim": "hidden", "units": "hidden", "d_main": "hidden",
        "num_layers": "n_layers", "layers": "n_layers", "depth": "n_layers",
        "early_stopping_rounds": "patience", "wd": "weight_decay",
    }
    doc_defaults = {"hidden": 256, "n_layers": 2, "dropout": 0.1}

    def fit(self, X_tr, y_tr, X_va, y_va, cat_cols):
        import torch
        from torch import nn

        torch.set_num_threads(self.n_threads)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        p = dict(self.params)
        self.prep = NNPrep(cat_cols).fit(X_tr, self.seed)
        xn_tr, xc_tr = self.prep.transform(X_tr)
        xn_va, xc_va = self.prep.transform(X_va)
        out_dim = self.n_classes if self.is_clf else 1
        if self.is_clf:
            yt = torch.tensor(np.asarray(y_tr), dtype=torch.long)
            yv = torch.tensor(np.asarray(y_va), dtype=torch.long)
            loss_fn = nn.CrossEntropyLoss()
        else:
            self.y_mu, self.y_sd = float(np.mean(y_tr)), float(np.std(y_tr) or 1.0)
            yt = torch.tensor((np.asarray(y_tr) - self.y_mu) / self.y_sd, dtype=torch.float32).view(-1, 1)
            yv = torch.tensor((np.asarray(y_va) - self.y_mu) / self.y_sd, dtype=torch.float32).view(-1, 1)
            loss_fn = nn.MSELoss()
        net = build_net(self.kind, self.prep.n_num, self.prep.cards, out_dim, p)
        opt = torch.optim.AdamW(net.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"]))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        Xn, Xc = torch.tensor(xn_tr), torch.tensor(xc_tr)
        Vn, Vc = torch.tensor(xn_va), torch.tensor(xc_va)
        bs = int(min(p["batch_size"], max(32, len(Xn) // 8)))
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for epoch in range(int(p["max_epochs"])):
            net.train()
            perm = torch.randperm(len(Xn), generator=g)
            for i in range(0, len(Xn), bs):
                idx = perm[i:i + bs]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss = loss_fn(net(Xn[idx], Xc[idx]), yt[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(np.mean([loss_fn(net(Vn[i:i + 4096], Vc[i:i + 4096]), yv[i:i + 4096]).item()
                                    for i in range(0, len(Vn), 4096)]))
            sched.step(vl)
            if vl < best - 1e-6:
                best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
                self.best_iteration = epoch + 1
            else:
                bad += 1
                if bad >= int(p["patience"]):
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net
        return self

    def predict(self, X):
        import torch

        xn, xc = self.prep.transform(X)
        self.net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(xn), 8192):
                o = self.net(torch.tensor(xn[i:i + 8192]), torch.tensor(xc[i:i + 8192]))
                outs.append(torch.softmax(o, dim=1).numpy() if self.is_clf else o.numpy().ravel())
        out = np.concatenate(outs, axis=0)
        if self.is_clf:
            return out.astype(float)
        return out.astype(float) * self.y_sd + self.y_mu
