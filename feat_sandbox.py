"""Песочница для кода Инженера: transform (Preprocessing, Features), CustomModel, fit_predict.

Код исполняется в отдельном процессе (`python feat_sandbox.py job.json`) с таймаутом и лимитом
памяти, без ключей LLM в окружении, на подвыборке строк поиска: обучающая часть (75%), валидационная
(25%) и кусочки holdout и test — те же роли кадров, что в настоящем фолде. Сначала применяется
принятая цепочка текущего решения, затем новый код. Проверяется форма выхода (feat_transform.normalize),
Инженеру возвращается, что код сделал с колонками, или ошибка.
"""

import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


# =====================================================================================
# Родитель
# =====================================================================================


def run(kind: str, code: str, ctx, slot_id: str, config: dict, timeout: float | None = None) -> dict:
    """kind: transform | model | escape. Запуск кода на подвыборке поверх текущего решения (config)."""
    import exp_runner

    cfg = ctx.cfg
    code_path = ctx.p("codes", f"{slot_id}_sample.py")
    code_path.write_text(code, encoding="utf-8")
    res_path = ctx.p("jobs", f"sandbox_{slot_id}.result.json")
    if res_path.exists():
        res_path.unlink()
    job = {"kind": kind, "code_path": str(code_path), "slot_id": slot_id, "config": config,
           "workdir": str(ctx.workdir), "sample_rows": cfg.sample_rows, "result_path": str(res_path),
           "seed": cfg.seed}
    job_path = ctx.p("jobs", f"sandbox_{slot_id}.json")
    job_path.write_text(json.dumps(job, ensure_ascii=False, default=str), encoding="utf-8")
    t = timeout or cfg.sandbox_timeout_s
    info = exp_runner.launch("feat_sandbox.py", [job_path], ctx, t, f"sandbox_{slot_id}")
    if res_path.exists():
        try:
            return json.loads(res_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"не прочитан результат песочницы: {e}"}
    reason = ("таймаут " + f"{t:.0f} с" if info["timed_out"] else "превышен лимит памяти" if info["mem_killed"]
              else f"код завершился аварийно (код {info['returncode']})")
    return {"ok": False, "error": f"{reason}\n{info['tail'][-1500:]}"}


# =====================================================================================
# Ребёнок
# =====================================================================================


def _sample_frames(job, spec, data, splits, hints):
    """Подвыборка в ролях фолда: (обучающая часть, её y, [валидация, holdout, test]) после принятой цепочки."""
    import numpy as np

    import exp_runner
    import feat_transform

    config = job["config"]
    rng = np.random.default_rng(int(job.get("seed", 0)))
    X = exp_runner.build_features(config, data, spec, hints, Path(job["workdir"]))
    n_train = data["n_train"]
    Xs = X.iloc[:n_train].iloc[splits.search_idx].reset_index(drop=True)
    ys = data["y_search"]
    n = min(int(job["sample_rows"]), len(Xs))
    idx = rng.choice(len(Xs), size=n, replace=False)
    cut = int(n * 0.75)
    tr, va = np.sort(idx[:cut]), np.sort(idx[cut:])
    Xh = X.iloc[:n_train].iloc[splits.holdout_idx].reset_index(drop=True)
    Xte = X.iloc[n_train:].reset_index(drop=True)
    h = Xh.iloc[np.sort(rng.choice(len(Xh), size=min(200, len(Xh)), replace=False))].reset_index(drop=True)
    t = Xte.iloc[np.sort(rng.choice(len(Xte), size=min(500, len(Xte)), replace=False))].reset_index(drop=True)
    keep_raw = set(spec.text_cols) | set(spec.datetime_cols)
    chain = list(config.get("features", {}).get("chain", []) or [])
    Xtr, frames = feat_transform.run_chain(chain, Path(job["workdir"]), Xs.iloc[tr].reset_index(drop=True), ys[tr],
                                           [Xs.iloc[va].reset_index(drop=True), h, t], keep_raw)
    return Xtr, ys[tr], frames, ys[va], keep_raw


def _child(job):
    import pickle

    import task_spec
    from eval_protocol import Splits

    wd = Path(job["workdir"])
    spec = task_spec.TaskSpec.load(wd / "spec.json")
    with open(wd / "data.pkl", "rb") as f:
        data = pickle.load(f)
    splits = Splits.load(wd / "splits.npz")
    hints = json.loads((wd / "hints.json").read_text(encoding="utf-8")) if (wd / "hints.json").exists() else {}
    code = Path(job["code_path"]).read_text(encoding="utf-8")
    t0 = time.time()
    if job["kind"] == "transform":
        import exp_runner
        import feat_transform

        Xtr, ytr, frames, _, keep_raw = _sample_frames(job, spec, data, splits, hints)
        tr2, frames2 = feat_transform.apply(code, Xtr, ytr, frames, keep_raw)
        exp_runner.finish_frames([tr2] + list(frames2), job["config"], spec)
        return {"ok": True, "time_s": round(time.time() - t0, 2), "n_columns": int(tr2.shape[1]),
                "describe": feat_transform.describe_diff(Xtr, tr2)}
    if job["kind"] == "model":
        import eval_acceptance
        import exp_runner
        import task_metrics

        Xtr, ytr, frames, yva, _ = _sample_frames(job, spec, data, splits, hints)
        out = exp_runner.finish_frames([Xtr] + list(frames), job["config"], spec)
        if spec.task_type in ("multilabel", "multi_regression"):
            raise ValueError("CustomModel поддерживается только для задач с одним таргетом")
        m = exp_runner._CustomModel(code, spec.task_type, spec.n_classes, 0)
        m.fit(out[0], ytr, X_val=out[1], y_val=yva)
        p = m.predict(out[1])
        bad = eval_acceptance.output_check(p, len(out[1]), spec.task_type, spec.n_classes)
        if bad:
            raise ValueError(f"выход CustomModel некорректен: {bad}")
        metric = task_metrics.metric_for(spec)
        s = task_metrics.score(metric, yva, p, spec.task_type)
        return {"ok": True, "score_on_sample": s, "time_s": round(time.time() - t0, 2)}
    if job["kind"] == "escape":
        import escape_contract

        Xs = data["X_all"].iloc[: data["n_train"]].iloc[splits.search_idx].reset_index(drop=True)
        return escape_contract.check_on_sample(code, spec, Xs, data["y_search"], n_rows=int(job["sample_rows"]),
                                               seed=int(job.get("seed", 0)))
    raise ValueError(f"неизвестный вид песочницы {job['kind']}")


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    import warnings

    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    t0 = time.time()
    print(f"== песочница {job['kind']} {job['slot_id']} (подвыборка)", flush=True)
    try:
        res = _child(job)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        # оставляем только кадры пользовательского кода, чтобы Инженер видел суть
        lines = [ln for ln in tb.splitlines() if "<transform>" in ln or "<custom_model>" in ln
                 or "<escape_fit_predict>" in ln or not ln.startswith("  File")]
        res = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": "\n".join(lines[-25:])}
        print(tb[-3000:], flush=True)
    if res.get("ok"):
        bits = [f"{k}={res[k]}" for k in ("n_columns", "score_on_sample", "shape") if k in res]
        print(f"== итог: успех; {', '.join(bits)}; время {time.time() - t0:.1f} с", flush=True)
        if res.get("notes"):
            print("   замечания: " + "; ".join(res["notes"]), flush=True)
    else:
        print(f"== итог: ОШИБКА — {res.get('error')}; время {time.time() - t0:.1f} с", flush=True)
    Path(job["result_path"]).write_text(json.dumps(res, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
