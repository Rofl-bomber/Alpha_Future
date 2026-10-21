"""Проверка редких путей харнесса на готовом рабочем каталоге прогона (только для стенда).

python tests/smoke_paths.py <workdir прогона>

Проверяет: модели linear / extratrees / mlp в режиме screen, CustomModel и fit_predict
через песочницу и эксперимент, Optuna (tune), refit, слияние и проверку патчей.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core_config  # noqa: E402
import core_timer  # noqa: E402
import exp_runner  # noqa: E402
import feat_sandbox  # noqa: E402
import search_operators  # noqa: E402
import task_spec  # noqa: E402

CUSTOM_MODEL = '''
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

class CustomModel:
    def __init__(self, task_type, n_classes, seed):
        self.task_type, self.n_classes, self.seed = task_type, n_classes, seed
    def _X(self, X):
        X = X.copy()
        for c in X.columns:
            if str(X[c].dtype) == "category":
                X[c] = X[c].cat.codes
        return X.astype(float).to_numpy()
    def fit(self, X, y, X_val=None, y_val=None):
        cls = HistGradientBoostingRegressor if self.task_type == "regression" else HistGradientBoostingClassifier
        self.m = cls(max_iter=200, random_state=self.seed).fit(self._X(X), y)
        self.classes_ = getattr(self.m, "classes_", None)
        return self
    def predict(self, X):
        return self.m.predict(self._X(X))
    def predict_proba(self, X):
        return self.m.predict_proba(self._X(X))
'''

ESCAPE = '''
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

def _X(df):
    df = df.copy()
    for c in df.columns:
        if str(df[c].dtype) in ("category", "object"):
            df[c] = df[c].astype("category").cat.codes
    return df.astype(float).to_numpy()

def fit_predict(train, y, predict_on, spec, time_budget_s):
    if spec["task_type"] == "regression":
        m = HistGradientBoostingRegressor(max_iter=150).fit(_X(train), y)
        return [m.predict(_X(d)) for d in predict_on]
    m = HistGradientBoostingClassifier(max_iter=150).fit(_X(train), y)
    out = []
    for d in predict_on:
        p = m.predict_proba(_X(d))
        full = np.zeros((len(d), len(spec["classes"])))
        full[:, m.classes_] = p
        out.append(full[:, 1] if spec["task_type"] == "binary" else full)
    return out
'''


def main():
    wd = Path(sys.argv[1]).resolve()
    cfg = core_config.load_config()
    cfg.workdir = str(wd)
    timer = core_timer.Timer(time.time(), 7200, cfg.phases)
    ctx = exp_runner.RunContext(wd, cfg, timer, core_timer.ProcessRegistry())
    spec = task_spec.TaskSpec.load(wd / "spec.json")
    base = search_operators.default_config(spec)
    ok = True

    for fam in ("linear", "extratrees", "mlp"):
        cfgm = search_operators.apply_patch(base, {"model.name": fam})
        r = exp_runner.run_experiment(f"t_{fam}", cfgm, ctx, "screen", budget_s=900)
        print(fam, r.get("ok"), r.get("mean"), r.get("time_s"), str(r.get("error"))[:300])
        ok &= bool(r.get("ok"))

    res = feat_sandbox.run("model", CUSTOM_MODEL, ctx, sample=True, slot_id="mc_test")
    print("custom model sandbox:", res)
    (wd / "codes" / "mc_test.py").write_text(CUSTOM_MODEL, encoding="utf-8")
    cfgc = search_operators.apply_patch(base, {"model.name": "custom", "model.custom_id": "mc_test"})
    r = exp_runner.run_experiment("t_custom", cfgc, ctx, "screen", budget_s=900)
    print("custom experiment:", r.get("ok"), r.get("mean"), str(r.get("error"))[:300])
    ok &= bool(res.get("ok")) and bool(r.get("ok"))

    res = feat_sandbox.run("escape", ESCAPE, ctx, sample=True, slot_id="es_test")
    print("escape sandbox:", res)
    (wd / "codes" / "es_test.py").write_text(ESCAPE, encoding="utf-8")
    cfge = search_operators.apply_patch(base, {"model.name": "escape"})
    cfge["model"]["escape_id"] = "es_test"
    r = exp_runner.run_experiment("t_escape", cfge, ctx, "full", budget_s=900)
    print("escape experiment:", r.get("ok"), r.get("mean"), str(r.get("error"))[:300])
    ok &= bool(res.get("ok")) and bool(r.get("ok"))

    r = exp_runner.run_job({"kind": "tune", "config": base}, ctx, 120, "t_tune")
    print("tune:", r.get("ok"), r.get("n_trials"), str(r.get("error"))[:300])
    ok &= bool(r.get("ok"))

    y_hold = np.load(wd / "y_holdout.npy") if (wd / "y_holdout.npy").exists() else None
    if y_hold is not None:
        r = exp_runner.run_job({"kind": "refit", "config": base, "node_id": "t_refit", "seeds": [0],
                                "y_holdout_path": str(wd / "y_holdout.npy"), "n_estimators": 200}, ctx, 600, "t_refit")
        print("refit:", r.get("ok"), str(r.get("error"))[:300])
        ok &= bool(r.get("ok"))

    a = search_operators.apply_patch(base, {"features.builtin": "+freq_enc", "model.params": {"num_leaves": 31}})
    b = search_operators.apply_patch(base, {"preprocess.missing_flags": True, "model.params": {"num_leaves": 127}})
    merged, conflicts = search_operators.merge(a, b, base)
    print("merge:", merged["features"], merged["preprocess"], merged["model"]["params"], conflicts)
    patch, notes = search_operators.validate_patch(
        {"model.params": {"num_leaves": 5000, "bogus": 1}, "features.builtin": "+nope", "target.transform": "log1p",
         "features.drop": ["not_a_col"]}, base, spec)
    print("validate:", patch, notes)
    print("ВСЁ OK" if ok else "ЕСТЬ ОШИБКИ")


if __name__ == "__main__":
    main()
