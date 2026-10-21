"""MLP на torch (CPU) с ранней остановкой; без torch — sklearn MLP.

Вход: препроцессор из model_linear с квантильным преобразованием числовых признаков.
"""

import time

import numpy as np

import task_metrics
from model_linear import Preprocessor

FAMILY = "mlp"

DEFAULTS = {"hidden": 256, "layers": 2, "dropout": 0.15, "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 512,
            "epochs": 60, "patience": 6}
RANGES = {"hidden": (int, 32, 1024), "layers": (int, 1, 4), "dropout": (float, 0.0, 0.6), "lr": (float, 1e-4, 1e-2),
          "weight_decay": (float, 0.0, 1e-2), "batch_size": (int, 64, 4096), "epochs": (int, 5, 300),
          "patience": (int, 2, 30)}


def space(trial) -> dict:
    return {"hidden": trial.suggest_categorical("hidden", [128, 256, 512]),
            "layers": trial.suggest_int("layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True)}


def _has_torch():
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


class Model:
    family = FAMILY

    def __init__(self, task_type, n_classes, params=None, seed=0, n_jobs=8, fast=False, objective="l2",
                 deadline=None):
        self.task_type = task_type
        self.n_classes = n_classes
        self.params = {**DEFAULTS, **(params or {})}
        self.seed = seed
        self.n_jobs = n_jobs
        self.fast = fast
        self.deadline = deadline
        self.objective = objective or "l2"
        self.best_iteration = None

    # --------------------------------------------------------------- torch
    def _net(self, d_in, d_out):
        import torch.nn as nn

        h, L, dr = int(self.params["hidden"]), int(self.params["layers"]), float(self.params["dropout"])
        layers, d = [], d_in
        for i in range(L):
            width = max(16, h // (2 ** i))
            layers += [nn.Linear(d, width), nn.BatchNorm1d(width), nn.SiLU(), nn.Dropout(dr)]
            d = width
        layers.append(nn.Linear(d, d_out))
        return nn.Sequential(*layers)

    def _fit_torch(self, Z, y, w, Zv, yv):
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(max(1, self.n_jobs))
        p = self.params
        if self.task_type == "regression":
            self.y_mu, self.y_sd = float(np.mean(y)), float(np.std(y) + 1e-9)
            yt = (np.asarray(y, dtype=np.float32) - self.y_mu) / self.y_sd
            d_out = 1
        elif self.task_type == "binary":
            yt = np.asarray(y, dtype=np.float32)
            d_out = 1
        else:
            yt = np.asarray(y, dtype=np.int64)
            d_out = self.n_classes
        self.net = self._net(Z.shape[1], d_out)
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"]))
        X_t = torch.from_numpy(Z)
        y_t = torch.from_numpy(yt)
        w_t = torch.from_numpy(np.asarray(w if w is not None else np.ones(len(Z)), dtype=np.float32))
        bs = int(p["batch_size"])
        epochs = int(p["epochs"]) // (2 if self.fast else 1)
        best, best_state, bad = None, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for epoch in range(max(1, epochs)):
            self.net.train()
            perm = torch.randperm(len(X_t), generator=g)
            for i in range(0, len(perm), bs):
                idx = perm[i: i + bs]
                if len(idx) < 2:
                    continue
                out = self.net(X_t[idx])
                loss = self._loss(out, y_t[idx], w_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            if Zv is not None:
                val = self._val_loss(Zv, yv)
                if best is None or val < best - 1e-6:
                    best, bad = val, 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                    self.best_iteration = epoch + 1
                else:
                    bad += 1
                    if bad >= int(p["patience"]):
                        break
            if self.deadline and time.time() > self.deadline:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        if self.best_iteration is None:
            self.best_iteration = epochs

    def _loss(self, out, y, w):
        import torch.nn.functional as F

        if self.task_type in ("regression",):
            per = (out.squeeze(1) - y) ** 2
        elif self.task_type == "binary":
            per = F.binary_cross_entropy_with_logits(out.squeeze(1), y, reduction="none")
        else:
            per = F.cross_entropy(out, y, reduction="none")
        return (per * w).sum() / w.sum()

    def _val_loss(self, Zv, yv):
        import torch

        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.from_numpy(Zv))
            if self.task_type == "regression":
                yt = torch.from_numpy(((np.asarray(yv) - self.y_mu) / self.y_sd).astype(np.float32))
            elif self.task_type == "binary":
                yt = torch.from_numpy(np.asarray(yv, dtype=np.float32))
            else:
                yt = torch.from_numpy(np.asarray(yv, dtype=np.int64))
            return float(self._loss(out, yt, torch.ones(len(yt))).item())

    def _predict_torch(self, Z):
        import torch

        self.net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(Z), 8192):
                o = self.net(torch.from_numpy(Z[i: i + 8192]))
                if self.task_type == "regression":
                    outs.append(o.squeeze(1).numpy() * self.y_sd + self.y_mu)
                elif self.task_type == "binary":
                    outs.append(torch.sigmoid(o.squeeze(1)).numpy())
                else:
                    outs.append(torch.softmax(o, dim=1).numpy())
        return np.concatenate(outs).astype(float)

    # --------------------------------------------------------------- общий интерфейс
    def fit(self, X, y, w=None, X_val=None, y_val=None):
        self.pre = Preprocessor(quantile=True).fit(X)
        Z = self.pre.transform(X)
        Zv = self.pre.transform(X_val) if X_val is not None else None
        self.torch = _has_torch()
        if self.torch:
            self._fit_torch(Z, y, w, Zv, y_val)
        else:
            from sklearn.neural_network import MLPClassifier, MLPRegressor

            h = int(self.params["hidden"])
            kw = dict(hidden_layer_sizes=(h, max(16, h // 2)), alpha=float(self.params["weight_decay"]) + 1e-5,
                      learning_rate_init=float(self.params["lr"]), max_iter=int(self.params["epochs"]),
                      early_stopping=True, n_iter_no_change=int(self.params["patience"]), random_state=self.seed)
            if self.task_type == "regression":
                self.y_mu, self.y_sd = float(np.mean(y)), float(np.std(y) + 1e-9)
                self.sk = MLPRegressor(**kw).fit(Z, (np.asarray(y) - self.y_mu) / self.y_sd)
            else:
                self.sk = MLPClassifier(**kw).fit(Z, y)
        return self

    def predict(self, X):
        Z = self.pre.transform(X)
        if self.torch:
            return self._predict_torch(Z)
        if self.task_type == "regression":
            return self.sk.predict(Z) * self.y_sd + self.y_mu
        proba = self.sk.predict_proba(Z)
        full = task_metrics.expand_proba(proba, [int(c) for c in self.sk.classes_], max(self.n_classes, 2))
        return full[:, 1] if self.task_type == "binary" else full

    def importance(self) -> dict:
        return {}
