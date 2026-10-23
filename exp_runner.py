"""Эксперимент в отдельном процессе.

Родитель (оркестратор) вызывает run_experiment / run_job: пишет задание в JSON, запускает
`python exp_runner.py job.json` в новой группе процессов, следит за таймаутом и памятью,
читает результат. Ребёнок загружает данные, строит признаки по конфигурации пайплайна,
обучает модели по фолдам и сохраняет OOF, предсказания для holdout и test и скоры по фолдам.
Метрику считает только этот код; LLM получает лишь итоговые числа.

Виды заданий: experiment (режимы full / screen / alt), tune, refit, adversarial, materialize.

Данные фолда: базовые признаки (встроенные операции, features.drop) → цепочка transform Инженера
(config.features.chain; вызывается внутри фолда, таргет — только обучающей части; результат принятой
части цепочки кэшируется на диске по фолдам — job.cache_upto) → редкие категории, флаги пропусков,
удаление сырых текстов и дат → кодирование категорий → модель. В режиме screen оцениваются только
первые фолды основного разбиения (Splits.eval_folds).

Если обучение остановлено по дедлайну (модель недообучена или не все фолды пройдены), результат
помечается truncated=True: скоры по пройденным фолдам, среднее и время в нём есть, но
предсказания не сохраняются и оркестратор не использует такой результат в решениях.
Каждое задание пишет в logs/<задание>.log ход работы и итоговую сводку; родитель дописывает
туда же итог процесса (код возврата, время, таймаут, память).
"""

import copy
import importlib
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
log = logging.getLogger("agent")

MODEL_MODULES = {"lgbm": "model_lgbm", "catboost": "model_catboost", "xgb": "model_xgb",
                 "linear": "model_linear", "extratrees": "model_extratrees", "mlp": "model_mlp"}
BOOSTING = ("lgbm", "catboost", "xgb")


def model_module(name: str):
    return importlib.import_module(MODEL_MODULES[name])


# =====================================================================================
# Родитель: запуск дочерних процессов
# =====================================================================================


@dataclass
class RunContext:
    workdir: Path
    cfg: object
    timer: object
    registry: object

    def p(self, *parts) -> Path:
        path = Path(self.workdir).joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def spec_path(self):
        return self.p("spec.json")

    @property
    def data_path(self):
        return self.p("data.pkl")

    @property
    def splits_path(self):
        return self.p("splits.npz")

    @property
    def hints_path(self):
        return self.p("hints.json")


def child_env(n_jobs: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI_")}  # код LLM не видит ключей
    n = str(max(1, int(n_jobs)))
    env.update(OMP_NUM_THREADS=n, MKL_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n, NUMEXPR_NUM_THREADS=n,
               PYTHONHASHSEED="0", PYTHONWARNINGS="ignore", PYTHONIOENCODING="utf-8")
    return env


def launch(script: str, args: list, ctx: RunContext, timeout: float, tag: str) -> dict:
    """Запустить python-скрипт архива в отдельной группе процессов с таймаутом и лимитом памяти."""
    import psutil

    from core_timer import kill_tree

    log_path = ctx.p("logs", f"{tag}.log")
    hard_left = ctx.timer.hard_deadline() - time.time() - 1.0
    timeout = max(1.0, min(timeout, hard_left))
    mem_limit = ctx.cfg.mem_limit_gb * 1024**3
    t0 = time.time()
    timed_out = mem_killed = False
    peak = 0
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen([sys.executable, "-u", str(HERE / script)] + [str(a) for a in args],
                                stdout=lf, stderr=subprocess.STDOUT, cwd=str(ctx.workdir),
                                env=child_env(ctx.cfg.n_jobs), start_new_session=True)
        ctx.registry.add(proc)
        try:
            ps = psutil.Process(proc.pid)
        except Exception:
            ps = None
        last_mem = 0.0
        while proc.poll() is None:
            now = time.time()
            if now - t0 > timeout:
                timed_out = True
                kill_tree(proc.pid)
                break
            if ps is not None and now - last_mem > 1.0:
                last_mem = now
                try:
                    rss = ps.memory_info().rss + sum(c.memory_info().rss for c in ps.children(recursive=True))
                    peak = max(peak, rss)
                    if rss > mem_limit:
                        mem_killed = True
                        kill_tree(proc.pid)
                        break
                except Exception:
                    pass
            time.sleep(0.1)
        try:
            proc.wait(timeout=10)
        except Exception:
            kill_tree(proc.pid)
        ctx.registry.remove(proc)
    info = {"returncode": proc.returncode, "timed_out": timed_out, "mem_killed": mem_killed,
            "time_s": round(time.time() - t0, 2), "peak_gb": round(peak / 1024**3, 2)}
    reason = ("снят по таймауту" if timed_out else "снят по лимиту памяти" if mem_killed
              else f"завершился, код {proc.returncode}")
    append_log(log_path, f"== процесс {reason}: {info['time_s']:.1f} с из лимита {timeout:.0f} с, "
                         f"пик памяти {info['peak_gb']:.2f} GB")
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            info["tail"] = f.read()[-4000:]
    except Exception:
        info["tail"] = ""
    return info


def append_log(path, text: str):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text.rstrip("\n") + "\n")
    except Exception:
        pass


def summary_line(kind: str, res: dict) -> str:
    """Короткая сводка результата задания для лога."""
    if not res.get("ok"):
        err = str(res.get("error") or "").strip().splitlines()
        head = err[0][:300] if err else "без текста"
        extra = f"; пройдено фолдов: {res['progress']}" if res.get("progress") else ""
        return f"== итог: ОШИБКА — {head}{extra}"
    parts = []
    if res.get("truncated"):
        parts.append(f"ОБРЕЗАН ({res.get('stop_reason')})")
    else:
        parts.append("успех")
    if res.get("fold_scores") is not None:
        done = res.get("folds_done")
        folds = ", ".join(f"{v:.6g}" for v in res["fold_scores"])
        parts.append(f"CV {res['mean']:.6g} ± {res.get('std', 0):.3g}; фолды"
                     + (f" {[k + 1 for k in done]}" if done is not None else "") + f": [{folds}]")
        if res.get("seed_means") and len(res["seed_means"]) > 1:
            parts.append("по сидам: " + ", ".join(f"{v:.6g}" for v in res["seed_means"]))
    if kind == "tune":
        parts.append(f"проб Optuna: {res.get('n_trials')}, лучший результат {res.get('best_gain')}")
    if kind == "adversarial" and res.get("auc") is not None:
        parts.append(f"AUC {res['auc']:.4f}")
    if res.get("time_s") is not None:
        parts.append(f"время {res['time_s']:.1f} с")
    if res.get("budget_s") is not None:
        parts.append(f"бюджет {res['budget_s']:.0f} с")
    return "== итог: " + "; ".join(parts)


def run_job(job: dict, ctx: RunContext, timeout: float, tag: str) -> dict:
    job = dict(job)
    job.setdefault("workdir", str(ctx.workdir))
    job.setdefault("n_jobs", ctx.cfg.n_jobs)
    job.setdefault("seed", ctx.cfg.seed)
    job["deadline"] = min(time.time() + timeout, ctx.timer.hard_deadline() - 2.0)
    job_path = ctx.p("jobs", f"{tag}.json")
    res_path = ctx.p("jobs", f"{tag}.result.json")
    if res_path.exists():
        res_path.unlink()
    job["result_path"] = str(res_path)
    job_path.write_text(json.dumps(job, ensure_ascii=False, default=str), encoding="utf-8")
    progress_path = Path(str(res_path) + ".progress")
    if progress_path.exists():
        progress_path.unlink()
    info = launch("exp_runner.py", [job_path], ctx, timeout + 15.0, tag)
    if res_path.exists():
        try:
            res = json.loads(res_path.read_text(encoding="utf-8"))
            res.setdefault("wall_s", info["time_s"])
            res["peak_gb"] = info["peak_gb"]
            return res
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"не прочитан результат: {e}", "wall_s": info["time_s"]}
    reason = ("таймаут" if info["timed_out"] else "превышен лимит памяти" if info["mem_killed"]
              else f"процесс завершился с кодом {info['returncode']}")
    res = {"ok": False, "error": f"{reason}\n{info['tail'][-2500:]}", "wall_s": info["time_s"],
           "timed_out": info["timed_out"], "mem_killed": info["mem_killed"]}
    if progress_path.exists():  # процесс снят до записи результата — сколько успел сделать
        try:
            res["progress"] = progress_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    append_log(ctx.p("logs", f"{tag}.log"), summary_line(job.get("kind", ""), res))
    return res


def run_experiment(node_id: str, config: dict, ctx: RunContext, mode: str = "full", seeds=(0,),
                   budget_s: float = 600.0, extra: dict | None = None) -> dict:
    job = {"kind": "experiment", "node_id": node_id, "config": config, "mode": mode, "seeds": list(seeds)}
    if extra:
        job.update(extra)
    tag = f"{node_id}_{mode}" + ("" if list(seeds) == [0] else "_s" + "-".join(map(str, seeds)))
    res = run_job(job, ctx, budget_s, tag)
    res.setdefault("mode", mode)
    return res


# =====================================================================================
# Ребёнок: построение признаков и обучение
# =====================================================================================


def _load(job):
    import task_spec
    from eval_protocol import Splits

    wd = Path(job["workdir"])
    spec = task_spec.TaskSpec.load(wd / "spec.json")
    with open(wd / "data.pkl", "rb") as f:
        data = pickle.load(f)
    splits = Splits.load(wd / "splits.npz") if (wd / "splits.npz").exists() else None
    hints = json.loads((wd / "hints.json").read_text(encoding="utf-8")) if (wd / "hints.json").exists() else {}
    return spec, data, splits, hints


def build_features(config: dict, data: dict, spec, hints: dict, workdir: Path):
    """Базовые признаки до цепочки transform: встроенные операции (без таргета) и features.drop."""
    import feat_builtin

    X = data["X_all"]
    ctx = {"cat_cols": spec.cat_cols, "num_cols": spec.num_cols, "text_cols": spec.text_cols,
           "datetime_cols": spec.datetime_cols, "top_num": hints.get("top_num", []), "segments": []}
    feats = config.get("features", {})
    ops = list(dict.fromkeys(feat_builtin.default_ops(spec) + list(feats.get("builtin", []))))
    X = feat_builtin.apply_ops(X, ops, ctx)
    drop = [c for c in feats.get("drop", []) if c in X.columns]
    if drop:
        X = X.drop(columns=drop)
    return X


def finish_frames(frames: list, config: dict, spec) -> list:
    """После цепочки: редкие категории, флаги пропусков, удаление сырых текстов и дат — по всем кадрам вместе."""
    import pandas as pd

    import feat_builtin

    pre = config.get("preprocess", {})
    lens = [len(f) for f in frames]
    allX = pd.concat(frames, axis=0, ignore_index=True)
    allX = feat_builtin.rare_merge(allX, feat_builtin.category_columns(allX), int(pre.get("rare_min_count", 0) or 0))
    if pre.get("missing_flags"):
        allX = feat_builtin.missing_flags(allX, [c for c in allX.columns if pd.api.types.is_numeric_dtype(allX[c])])
    allX = feat_builtin.drop_raw_objects(allX)
    out, start = [], 0
    for n in lens:
        out.append(allX.iloc[start: start + n].reset_index(drop=True))
        start += n
    return out


def _cache_dir(workdir) -> Path:
    d = Path(workdir) / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_trim(workdir, max_gb: float):
    files = sorted(_cache_dir(workdir).glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    while files and total > max_gb * 1024**3:
        p = files.pop(0)
        total -= p.stat().st_size
        try:
            p.unlink()
        except OSError:
            pass


def prepare_fold(config: dict, spec, workdir, Xtr, ytr, frames: list, cache: dict | None = None) -> tuple:
    """Данные фолда для модели: цепочка transform (принятая часть — из кэша) и завершающая обработка."""
    import feat_transform

    chain = list(config.get("features", {}).get("chain", []) or [])
    keep_raw = set(spec.text_cols) | set(spec.datetime_cols)
    start = 0
    upto = min(int((cache or {}).get("upto", 0) or 0), len(chain))
    if cache and upto > 0:
        feats = config.get("features", {})
        key = feat_transform.cache_key([feats.get("builtin", []), feats.get("drop", []), chain[:upto],
                                        cache.get("tag"), len(frames), len(Xtr)])
        path = _cache_dir(workdir) / f"{key}.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    Xtr, frames = pickle.load(f)
                start = upto
            except Exception:
                start = 0
        if start == 0:
            Xtr, frames = feat_transform.run_chain(chain[:upto], workdir, Xtr, ytr, frames, keep_raw)
            start = upto
            size = sum(int(f.memory_usage(index=False).sum()) for f in [Xtr] + frames)
            if size < 1.5 * 1024**3:
                tmp = path.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    pickle.dump((Xtr, frames), f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)
                _cache_trim(workdir, float(cache.get("max_gb", 6.0)))
    Xtr, frames = feat_transform.run_chain(chain[start:], workdir, Xtr, ytr, frames, keep_raw)
    out = finish_frames([Xtr] + list(frames), config, spec)
    return out[0], out[1:]


def _safe_names(cols):
    out, seen = [], set()
    for c in cols:
        s = re.sub(r"[^0-9a-zA-Z_]", "_", str(c))[:60] or "f"
        base, i = s, 1
        while s in seen:
            s = f"{base}_{i}"
            i += 1
        seen.add(s)
        out.append(s)
    return out


def _target_fwd(y, how):
    if how == "log1p":
        return np.sign(y) * np.log1p(np.abs(y))
    if how == "sqrt":
        return np.sign(y) * np.sqrt(np.abs(y))
    return y


def _target_inv(y, how):
    if how == "log1p":
        return np.sign(y) * np.expm1(np.abs(y))
    if how == "sqrt":
        return np.sign(y) * y**2
    return y


def _weights(y, how, n_classes):
    if how in (None, "none") or n_classes < 2:
        return None
    counts = np.bincount(y.astype(int), minlength=n_classes).astype(float)
    w_c = len(y) / (n_classes * np.maximum(counts, 1))
    if how == "sqrt_balanced":
        w_c = np.sqrt(w_c)
    return w_c[y.astype(int)]


class _CustomModel:
    """Обёртка класса CustomModel, написанного Инженером, под внутренний интерфейс."""

    family = "custom"

    def __init__(self, code, task_type, n_classes, seed):
        ns = {}
        exec(compile(code, "<custom_model>", "exec"), ns)
        cls = ns["CustomModel"]
        try:
            self.m = cls(task_type=task_type, n_classes=n_classes, seed=seed)
        except TypeError:
            self.m = cls()
        self.task_type = task_type
        self.n_classes = n_classes
        self.best_iteration = None

    def fit(self, X, y, w=None, X_val=None, y_val=None):
        try:
            self.m.fit(X, y, X_val=X_val, y_val=y_val)
        except TypeError:
            self.m.fit(X, y)
        return self

    def predict(self, X):
        import task_metrics

        if self.task_type == "regression":
            return np.asarray(self.m.predict(X), dtype=float).reshape(-1)
        proba = np.asarray(self.m.predict_proba(X), dtype=float)
        classes = getattr(self.m, "classes_", None)
        K = max(self.n_classes, 2)
        if classes is not None and proba.ndim == 2 and proba.shape[1] != K:
            proba = task_metrics.expand_proba(proba, [int(c) for c in classes], K)
        if self.task_type == "binary":
            return proba[:, 1] if proba.ndim == 2 else proba
        return proba

    def importance(self):
        return {}


def _make_model(mcfg: dict, task_type: str, n_classes: int, seed: int, n_jobs: int, fast: bool, objective,
                deadline, workdir):
    name = mcfg.get("name", "lgbm")
    if name == "custom":
        code = (Path(workdir) / "codes" / f"{mcfg['custom_id']}.py").read_text(encoding="utf-8")
        return _CustomModel(code, task_type, n_classes, seed)
    mod = model_module(name)
    return mod.Model(task_type, n_classes, params=mcfg.get("params") or {}, seed=seed, n_jobs=n_jobs, fast=fast,
                     objective=objective, deadline=deadline)


def _single_tasks(spec):
    """Список (подзадача, тип, число классов, индекс столбца y)."""
    if spec.task_type == "multilabel":
        return [("binary", 2, j) for j in range(len(spec.targets))]
    if spec.task_type == "multi_regression":
        return [("regression", 0, j) for j in range(len(spec.targets))]
    return [(spec.task_type, spec.n_classes, None)]


def _fit_predict(config, spec, Xtr, ytr, Xva, yva, others, seed, n_jobs, fast, deadline, workdir, n_estimators=None):
    """Обучить модель(и) на подготовленных Xtr и предсказать для [Xva] + others.

    Возвращает (список предсказаний, best_iter, важности, число признаков, обучение остановлено по дедлайну).
    """
    import feat_builtin
    import task_metrics

    pre = config.get("preprocess", {})
    tcfg = config.get("target", {})
    train_cfg = config.get("training", {})
    metric = task_metrics.metric_for(spec)
    frames = ([Xva] if Xva is not None else []) + list(others)
    cats = feat_builtin.category_columns(Xtr)
    method = pre.get("cat_encoding", "native")
    mcfg = copy.deepcopy(config.get("model", {}))
    enc_y = ytr if ytr.ndim == 1 else ytr[:, 0] if spec.task_type == "multi_regression" else ytr
    Xtr_e, frames_e = feat_builtin.encode_fold(Xtr, frames, enc_y, cats, method, spec.task_type, spec.n_classes, seed)
    names = _safe_names(Xtr_e.columns)
    mapping = dict(zip(names, Xtr_e.columns))
    Xtr_e = Xtr_e.set_axis(names, axis=1)
    frames_e = [f.set_axis(names, axis=1) for f in frames_e]
    if n_estimators:
        mcfg.setdefault("params", {})
        mcfg["params"] = dict(mcfg["params"], n_estimators=int(n_estimators))
    preds_all = [[] for _ in frames_e]
    best_iters, imp = [], {}
    truncated = False
    for sub_type, n_cls, j in _single_tasks(spec):
        y_t = ytr if j is None else ytr[:, j]
        y_v = None if yva is None else (yva if j is None else yva[:, j])
        how = tcfg.get("transform", "none") if sub_type == "regression" else "none"
        y_fit = y_t.astype(float).copy() if sub_type == "regression" else y_t.astype(int)
        if sub_type == "regression" and float(tcfg.get("clip_quantile", 0) or 0) > 0:
            q = float(tcfg["clip_quantile"])
            lo, hi = np.quantile(y_fit, [q, 1 - q])
            y_fit = np.clip(y_fit, lo, hi)
        y_fit = _target_fwd(y_fit, how) if sub_type == "regression" else y_fit
        y_val = None
        if y_v is not None:
            y_val = _target_fwd(y_v.astype(float), how) if sub_type == "regression" else y_v.astype(int)
        w = _weights(y_fit, tcfg.get("class_weight", "none"), n_cls) if sub_type != "regression" else None
        model = _make_model(mcfg, sub_type, n_cls, seed, n_jobs, fast, metric.objective, deadline, workdir)
        Xv = frames_e[0] if Xva is not None else None
        if n_estimators:
            Xv, y_val = None, None
        model.fit(Xtr_e, y_fit, w, Xv, y_val)
        preds = [model.predict(f) for f in frames_e]
        # псевдоразметка: уверенные строки test (последний кадр) добавляются в обучение фолда
        if train_cfg.get("pseudo_label") and sub_type in ("binary", "multiclass") and len(frames_e) >= 2:
            p_te = preds[-1]
            conf = np.maximum(p_te, 1 - p_te) if sub_type == "binary" else p_te.max(axis=1)
            thr = float(train_cfg.get("pseudo_threshold", 0.95))
            sel = np.where(conf >= thr)[0]
            if len(sel) > len(Xtr_e):
                sel = np.random.default_rng(seed).choice(sel, len(Xtr_e), replace=False)
            if len(sel) > 0:
                import pandas as pd

                lab = (p_te[sel] >= 0.5).astype(int) if sub_type == "binary" else p_te[sel].argmax(axis=1)
                X_aug = pd.concat([Xtr_e, frames_e[-1].iloc[sel]], axis=0, ignore_index=True)
                y_aug = np.concatenate([y_fit, lab])
                w_aug = None if w is None else np.concatenate([w, _weights(lab, tcfg.get("class_weight"), n_cls)])
                model = _make_model(mcfg, sub_type, n_cls, seed, n_jobs, fast, metric.objective, deadline, workdir)
                model.fit(X_aug, y_aug, w_aug, Xv, y_val)
                preds = [model.predict(f) for f in frames_e]
        if sub_type == "regression":
            preds = [_target_inv(p, how) for p in preds]
        for i, p in enumerate(preds):
            preds_all[i].append(p)
        truncated = truncated or bool(getattr(model, "truncated", False))
        if model.best_iteration:
            best_iters.append(int(model.best_iteration))
        for k, v in (model.importance() or {}).items():
            imp[mapping.get(k, k)] = imp.get(mapping.get(k, k), 0.0) + v
    if len(_single_tasks(spec)) > 1:
        out = [np.stack(ps, axis=1) for ps in preds_all]
    else:
        out = [ps[0] for ps in preds_all]
    return out, (int(np.mean(best_iters)) if best_iters else None), imp, len(names), truncated


def _say(msg: str):
    """Строка в лог задания (stdout дочернего процесса перенаправлен в logs/<задание>.log)."""
    print(msg, flush=True)


def _progress(job, text: str):
    """Сколько сделано — на случай, если процесс снимут до записи результата."""
    try:
        Path(str(job["result_path"]) + ".progress").write_text(text, encoding="utf-8")
    except Exception:
        pass


def _cache_spec(job, folds, k) -> dict | None:
    """Кэш принятой части цепочки для фолда k: ключ — разбиение (хэш массива фолдов) и номер фолда."""
    import hashlib

    upto = int(job.get("cache_upto", 0) or 0)
    if upto <= 0:
        return None
    tag = hashlib.md5(np.ascontiguousarray(folds).tobytes()).hexdigest()[:10] + f":{k}"
    return {"upto": upto, "tag": tag, "max_gb": float(job.get("cache_max_gb", 6.0))}


def _experiment(job):
    import final_ensemble
    import task_metrics

    spec, data, splits, hints = _load(job)
    config = job["config"]
    mode = job["mode"]
    seeds = job.get("seeds", [0])
    deadline = job["deadline"]
    workdir = Path(job["workdir"])
    t0 = time.time()
    metric = task_metrics.metric_for(spec)
    model_name = config.get("model", {}).get("name", "lgbm")
    kind = mode if mode in ("full", "screen", "alt") else "full"
    folds = splits.fold_array(kind)
    if folds is None:
        return {"ok": False, "error": f"для режима {mode} нет разбиения (альтернативных фолдов нет)"}
    ks = splits.eval_folds(kind)
    K = len(ks)
    X = build_features(config, data, spec, hints, workdir)
    n_train = data["n_train"]
    Xtr_all = X.iloc[:n_train]
    Xs = Xtr_all.iloc[splits.search_idx].reset_index(drop=True)
    Xh = Xtr_all.iloc[splits.holdout_idx].reset_index(drop=True)
    Xte = X.iloc[n_train:].reset_index(drop=True)
    ys = data["y_search"]
    fast = mode in ("screen", "alt")
    need_ho_te = mode == "full"
    ho, te = None, None
    best_iters, imp_total = [], {}
    n_feat = None
    seed_runs = []  # (oof, пройденные фолды) по сидам
    truncated, stop_reason = False, "ok"
    _say(f"базовые признаки за {time.time() - t0:.1f} с: {X.shape[1]} колонок; цепочка "
         f"{config.get('features', {}).get('chain', [])}; фолды {ks}, сиды {seeds}")

    if model_name == "escape":
        import escape_contract

        code = (workdir / "codes" / f"{config['model']['escape_id']}.py").read_text(encoding="utf-8")
        raw = data["X_all"]
        Rs = raw.iloc[:n_train].iloc[splits.search_idx].reset_index(drop=True)
        Rh = raw.iloc[:n_train].iloc[splits.holdout_idx].reset_index(drop=True)
        Rte = raw.iloc[n_train:].reset_index(drop=True)
        oof, ho, te = escape_contract.run_escape(code, spec, Rs, ys, Rh, Rte, folds, deadline, need_ho_te, ks=ks)
        seed_runs = [(oof, list(ks))]
        n_feat = raw.shape[1]
    else:
        n_runs = K * len(seeds)
        prepared = {}
        for s in seeds:
            seed = int(job.get("seed", 42)) + int(s)
            oof, done = None, []
            for k in ks:
                if time.time() > deadline:
                    truncated, stop_reason = True, f"дедлайн наступил перед фолдом {k + 1} (сид {s})"
                    break
                tf = time.time()
                tr, va = folds != k, folds == k
                if k not in prepared or len(seeds) == 1:
                    prepared[k] = prepare_fold(config, spec, workdir, Xs[tr].reset_index(drop=True), ys[tr],
                                               [Xs[va].reset_index(drop=True), Xh, Xte], _cache_spec(job, folds, k))
                Xtr_k, (Xva_k, Xh_k, Xte_k) = prepared[k]
                if len(seeds) == 1:
                    prepared.pop(k, None)
                tprep = time.time() - tf
                others = [Xh_k, Xte_k] if need_ho_te else []
                preds, bi, imp, n_feat, trunc = _fit_predict(config, spec, Xtr_k, ys[tr], Xva_k, ys[va], others,
                                                             seed, job["n_jobs"], fast, deadline, workdir)
                if oof is None:
                    oof = np.zeros((len(ys),) + np.asarray(preds[0]).shape[1:])
                oof[va] = preds[0]
                done.append(k)
                if need_ho_te:
                    ho = preds[1] / n_runs if ho is None else ho + preds[1] / n_runs
                    te = preds[2] / n_runs if te is None else te + preds[2] / n_runs
                if bi:
                    best_iters.append(bi)
                for f, v in imp.items():
                    imp_total[f] = imp_total.get(f, 0.0) + v
                _say(f"сид {s}, фолд {k + 1}: {time.time() - tf:.1f} с (данные {tprep:.1f} с), итераций {bi}"
                     + (" — обучение остановлено по дедлайну" if trunc else ""))
                _progress(job, f"сид {s}: фолдов {len(done)}/{K}, {time.time() - t0:.0f} с")
                if trunc:
                    truncated, stop_reason = True, f"обучение остановлено по дедлайну на фолде {k + 1} (сид {s})"
                    break
            seed_runs.append((oof, done))
            if truncated:
                break

    complete = [o for o, d in seed_runs if len(d) == K]
    budget_s = round(deadline - job.get("_t_start", t0), 1)
    top_imp = sorted(imp_total.items(), key=lambda kv: -kv[1])[:30]
    total = sum(v for _, v in top_imp) or 1.0
    common = {"mode": mode, "time_s": round(time.time() - t0, 2), "budget_s": budget_s, "n_folds": K,
              "best_iter": int(np.mean(best_iters)) if best_iters else None, "n_features": n_feat,
              "importance": [(f, round(v / total, 4)) for f, v in top_imp]}

    def fold_score(o, k, post):
        m = folds == k
        return task_metrics.score(metric, ys[m], o[m], spec.task_type, post)

    if truncated:
        if complete:
            oof, done = np.mean(complete, axis=0), list(ks)
        else:
            oof, done = seed_runs[0] if seed_runs else (None, [])
        if oof is None or not done:
            return {"ok": False, "truncated": True, "stop_reason": stop_reason,
                    "error": f"TimeoutError: {stop_reason}; ни один фолд не пройден", **common}
        mask = np.isin(folds, done)
        post = final_ensemble.fit_post(oof[mask], ys[mask], spec, config.get("postprocess", {}), metric)
        fold_scores = [fold_score(oof, k, post) for k in done]
        seed_means = [float(np.mean([fold_score(o, k, post) for k in ks])) for o in complete]
        return {"ok": True, "truncated": True, "stop_reason": stop_reason, "folds_done": [ks.index(k) for k in done],
                "fold_scores": [float(v) for v in fold_scores], "mean": float(np.mean(fold_scores)),
                "std": float(np.std(fold_scores)), "seed_means": seed_means, "preds_path": None, "post": post,
                **common}

    oof = np.mean(complete, axis=0)
    mask = np.isin(folds, ks)
    post = final_ensemble.fit_post(oof[mask], ys[mask], spec, config.get("postprocess", {}), metric)
    fold_scores = [fold_score(oof, k, post) for k in ks]
    seed_means = [float(np.mean([fold_score(o, k, post) for k in ks])) for o in complete]
    import eval_acceptance

    bad = eval_acceptance.output_check(oof[mask], int(mask.sum()), spec.task_type if spec.task_type != "multi_regression"
                                       else "regression", spec.n_classes)
    if bad:
        raise ValueError(f"некорректный OOF: {bad}")
    suffix = "" if list(seeds) == [0] else "_s" + "-".join(map(str, seeds))
    preds_path = workdir / "preds" / f"{job['node_id']}_{mode}{suffix}.npz"
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"oof": oof}
    if need_ho_te:
        arrays.update(ho=ho, te=te)
    np.savez(preds_path, **arrays)
    return {"ok": True, "truncated": False, "stop_reason": "ok", "folds_done": list(range(K)),
            "fold_scores": [float(v) for v in fold_scores], "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)), "seed_means": seed_means, "preds_path": str(preds_path),
            "post": post, **common}


def _tune(job):
    """Optuna по параметрам модели лучшего узла на фолдах скрининга: до n_trials проб и до дедлайна.

    Данные фолдов (цепочка transform) готовятся один раз. Проба, в которой обучение остановлено
    по дедлайну, не засчитывается (TrialPruned).
    """
    import optuna

    import final_ensemble
    import task_metrics

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spec, data, splits, hints = _load(job)
    config = job["config"]
    workdir = Path(job["workdir"])
    deadline = job["deadline"]
    metric = task_metrics.metric_for(spec)
    X = build_features(config, data, spec, hints, workdir)
    Xs = X.iloc[: data["n_train"]].iloc[splits.search_idx].reset_index(drop=True)
    ys = data["y_search"]
    folds = splits.fold_array("screen")
    ks = splits.eval_folds("screen")
    prepared = {}
    for k in ks:
        tr, va = folds != k, folds == k
        prepared[k] = prepare_fold(config, spec, workdir, Xs[tr].reset_index(drop=True), ys[tr],
                                   [Xs[va].reset_index(drop=True)], _cache_spec(job, folds, k))
    mod = model_module(config["model"]["name"])
    base_params = dict(config["model"].get("params") or {})
    mask = np.isin(folds, ks)

    def objective(trial):
        tt = time.time()
        params = dict(base_params, **mod.space(trial))
        cfg2 = copy.deepcopy(config)
        cfg2["model"]["params"] = params
        oof = None
        for k in ks:
            if time.time() > deadline - 5:
                raise optuna.TrialPruned()
            tr, va = folds != k, folds == k
            Xtr_k, (Xva_k,) = prepared[k]
            preds, _, _, _, trunc = _fit_predict(cfg2, spec, Xtr_k, ys[tr], Xva_k, ys[va], [],
                                                 int(job.get("seed", 42)), job["n_jobs"], True, deadline, workdir)
            if trunc:
                _say(f"проба {trial.number}: обучение остановлено по дедлайну — не засчитана")
                raise optuna.TrialPruned()
            if oof is None:
                oof = np.zeros((len(ys),) + np.asarray(preds[0]).shape[1:])
            oof[va] = preds[0]
        post = final_ensemble.fit_post(oof[mask], ys[mask], spec, config.get("postprocess", {}), metric)
        s = float(np.mean([task_metrics.score(metric, ys[folds == k], oof[folds == k], spec.task_type, post)
                           for k in ks]))
        _say(f"проба {trial.number}: CV {s:.6g} за {time.time() - tt:.1f} с; {trial.params}")
        return task_metrics.gain(s, metric.greater_is_better)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=int(job.get("seed", 42))))
    n_trials = int(job["n_trials"]) if job.get("n_trials") else None
    study.optimize(objective, n_trials=n_trials, timeout=max(10.0, deadline - time.time() - 10), catch=(Exception,))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        return {"ok": False, "error": "ни одна проба Optuna не завершилась"}
    best = study.best_trial
    return {"ok": True, "best_params": dict(base_params, **best.params), "best_gain": float(best.value),
            "n_trials": len(done)}


def _refit(job):
    """Обучение на всех размеченных строках (поиск + holdout), предсказание test."""
    spec, data, splits, hints = _load(job)
    config = job["config"]
    workdir = Path(job["workdir"])
    X = build_features(config, data, spec, hints, workdir)
    n_train = data["n_train"]
    idx = np.concatenate([splits.search_idx, splits.holdout_idx])
    Xl = X.iloc[:n_train].iloc[idx].reset_index(drop=True)
    Xte = X.iloc[n_train:].reset_index(drop=True)
    y_hold = np.load(job["y_holdout_path"])
    yl = np.concatenate([data["y_search"], y_hold], axis=0)
    te = None
    truncated = False
    seeds = job.get("seeds", [0])
    if config.get("model", {}).get("name") == "escape":
        import escape_contract

        code = (workdir / "codes" / f"{config['model']['escape_id']}.py").read_text(encoding="utf-8")
        fn = escape_contract.load_fn(code)
        raw = data["X_all"]
        out = fn(raw.iloc[:n_train].iloc[idx].reset_index(drop=True), yl, [raw.iloc[n_train:].reset_index(drop=True)],
                 {"task_type": spec.task_type, "classes": list(spec.classes), "targets": list(spec.targets)},
                 max(5.0, job["deadline"] - time.time()))
        te = escape_contract.coerce(out[0], len(Xte), spec)
    else:
        Xl_p, (Xte_p,) = prepare_fold(config, spec, workdir, Xl, yl, [Xte])
        for s in seeds:
            seed = int(job.get("seed", 42)) + int(s)
            preds, _, _, _, trunc = _fit_predict(config, spec, Xl_p, yl, None, None, [Xte_p], seed, job["n_jobs"],
                                                 False, job["deadline"], workdir, n_estimators=job.get("n_estimators"))
            truncated = truncated or trunc
            te = preds[0] / len(seeds) if te is None else te + preds[0] / len(seeds)
    path = workdir / "preds" / f"{job['node_id']}_refit.npz"
    np.savez(path, te=te)
    return {"ok": True, "preds_path": str(path), "truncated": truncated,
            "stop_reason": "обучение остановлено по дедлайну" if truncated else "ok"}


def _materialize(job):
    """Текущие данные решения (D3): цепочка на всех строках поиска → cur_train (строки поиска) и cur_test."""
    spec, data, splits, hints = _load(job)
    config = job["config"]
    workdir = Path(job["workdir"])
    X = build_features(config, data, spec, hints, workdir)
    n_train = data["n_train"]
    Xs = X.iloc[:n_train].iloc[splits.search_idx].reset_index(drop=True)
    Xte = X.iloc[n_train:].reset_index(drop=True)
    tr, (te,) = prepare_fold(config, spec, workdir, Xs, data["y_search"], [Xte])
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tr.to_pickle(out / "cur_train.pkl")
    te.to_pickle(out / "cur_test.pkl")
    return {"ok": True, "train_path": str(out / "cur_train.pkl"), "test_path": str(out / "cur_test.pkl"),
            "n_columns": int(tr.shape[1])}


def _adversarial(job):
    import lightgbm as lgb
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    import feat_builtin

    spec, data, _, _ = _load(job)
    X = feat_builtin.drop_raw_objects(data["X_all"].drop(columns=[c for c in job.get("exclude", [])
                                                                   if c in data["X_all"].columns]))
    n_train = data["n_train"]
    y = np.r_[np.zeros(n_train), np.ones(len(X) - n_train)].astype(int)
    names = _safe_names(X.columns)
    mapping = dict(zip(names, X.columns))
    X = X.set_axis(names, axis=1)
    oof = np.zeros(len(X))
    imp = pd.Series(0.0, index=names)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, min_child_samples=50, verbose=-1, num_threads=job["n_jobs"], seed=0)
    for tr, va in skf.split(X, y):
        b = lgb.train(params, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=300,
                      valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        oof[va] = b.predict(X.iloc[va], num_iteration=b.best_iteration)
        imp += pd.Series(b.feature_importance("gain"), index=b.feature_name())
    auc = float(roc_auc_score(y, oof))
    top = [(mapping[k], round(float(v / max(imp.sum(), 1e-9)), 4)) for k, v in imp.sort_values(ascending=False).head(10).items()]
    p = np.clip(oof[:n_train], 1e-4, 1 - 1e-4)
    odds = p / (1 - p) * (n_train / max(1, len(X) - n_train))
    wpath = Path(job["workdir"]) / "adv_weights.npy"
    np.save(wpath, odds)
    return {"ok": True, "auc": auc, "top": top, "weights_path": str(wpath)}


def main():
    job_path = Path(sys.argv[1])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    import warnings

    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    kind = job["kind"]
    t_start = time.time()
    job["_t_start"] = t_start
    model = (job.get("config") or {}).get("model", {}).get("name")
    _say(f"== {kind} {job.get('node_id', '')}" + (f" [{job.get('mode')}]" if job.get("mode") else "")
         + (f", модель {model}" if model else "") + f"; бюджет {job['deadline'] - t_start:.0f} с")
    try:
        if kind == "experiment":
            res = _experiment(job)
        elif kind == "tune":
            res = _tune(job)
        elif kind == "refit":
            res = _refit(job)
        elif kind == "adversarial":
            res = _adversarial(job)
        elif kind == "materialize":
            res = _materialize(job)
        else:
            res = {"ok": False, "error": f"неизвестный вид задания {kind}"}
    except Exception as e:  # noqa: BLE001
        res = {"ok": False, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}"}
        _say(traceback.format_exc()[-3000:])
    res.setdefault("budget_s", round(job["deadline"] - t_start, 1))
    res.setdefault("time_s", round(time.time() - t_start, 2))
    _say(summary_line(kind, res))
    Path(job["result_path"]).write_text(json.dumps(res, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    main()
