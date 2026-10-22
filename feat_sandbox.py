"""Песочница для кода, который пишет Инженер: функции признаков, CustomModel, fit_predict.

Код исполняется в отдельном процессе (`python feat_sandbox.py job.json`) с таймаутом и
лимитом памяти, без ключей LLM в окружении. Проверки для признаков: та же длина, нет inf,
нет полностью пустых и константных колонок, ограничение на число новых колонок.
На полных данных результат сохраняется в feats/<id>.pkl и затем подхватывается экспериментами.
"""

import json
import re
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


# =====================================================================================
# Родитель
# =====================================================================================


def run(kind: str, code: str, ctx, sample: bool, slot_id: str, timeout: float | None = None) -> dict:
    """kind: features | model | escape. sample=True — быстрый прогон на подвыборке для ReAct-цикла."""
    import exp_runner

    cfg = ctx.cfg
    code_path = ctx.p("codes", f"{slot_id}{'_sample' if sample else ''}.py")
    code_path.write_text(code, encoding="utf-8")
    res_path = ctx.p("jobs", f"sandbox_{slot_id}_{'s' if sample else 'f'}.result.json")
    if res_path.exists():
        res_path.unlink()
    job = {"kind": kind, "code_path": str(code_path), "sample": sample, "slot_id": slot_id,
           "workdir": str(ctx.workdir), "sample_rows": cfg.sample_rows, "max_new": cfg.max_new_features,
           "result_path": str(res_path), "seed": cfg.seed}
    job_path = ctx.p("jobs", f"sandbox_{slot_id}_{'s' if sample else 'f'}.json")
    job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    t = timeout or cfg.sandbox_timeout_s
    info = exp_runner.launch("feat_sandbox.py", [job_path], ctx, t, f"sandbox_{slot_id}_{'s' if sample else 'f'}")
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


def _safe(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", str(name))[:48] or "f"


def normalize_features(out, df_in, slot_id: str, max_new: int):
    """Привести выход make_features к таблице новых колонок той же длины."""
    import numpy as np
    import pandas as pd

    n = len(df_in)
    if isinstance(out, dict):
        out = pd.DataFrame({k: (v.to_numpy() if hasattr(v, "to_numpy") else v) for k, v in out.items()})
    elif isinstance(out, pd.Series):
        out = out.to_frame(name=out.name or "feature")
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"make_features должна вернуть DataFrame, получено {type(out).__name__}")
    if len(out) != n:
        raise ValueError(f"make_features вернула {len(out)} строк вместо {n}")
    out = out.reset_index(drop=True)
    new_cols = [c for c in out.columns if c not in df_in.columns]
    if not new_cols:
        raise ValueError("make_features не добавила ни одной новой колонки")
    notes, keep = [], {}
    for c in new_cols:
        s = out[c]
        if pd.api.types.is_bool_dtype(s):
            s = s.astype(float)
        if pd.api.types.is_numeric_dtype(s):
            s = s.astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            nun = s.nunique(dropna=True)
            if nun > 1000:
                notes.append(f"{c}: строковая колонка с {nun} значениями пропущена")
                continue
            s = s.astype(object).where(s.notna(), None).astype(str).where(s.notna(), np.nan).astype("category")
        if s.isna().all():
            notes.append(f"{c}: все значения пустые — пропущена")
            continue
        if s.nunique(dropna=True) <= 1:
            notes.append(f"{c}: константа — пропущена")
            continue
        keep[c] = s
        if len(keep) >= max_new:
            notes.append(f"оставлены первые {max_new} новых колонок")
            break
    if not keep:
        raise ValueError("после проверки не осталось полезных колонок: " + "; ".join(notes))
    res = pd.DataFrame(keep)
    res.columns = [f"{slot_id}__{_safe(c)}" for c in res.columns]
    return res, notes


def _describe(df):
    import pandas as pd

    lines = []
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            lines.append(f"{c}: категория, уник.={s.nunique()}, NaN={s.isna().mean():.1%}")
        else:
            lines.append(f"{c}: min={s.min():.4g}, mean={s.mean():.4g}, max={s.max():.4g}, NaN={s.isna().mean():.1%}")
    return "\n".join(lines)


def _child(job):
    import pickle

    import numpy as np
    import pandas as pd

    import task_spec

    wd = Path(job["workdir"])
    spec = task_spec.TaskSpec.load(wd / "spec.json")
    with open(wd / "data.pkl", "rb") as f:
        data = pickle.load(f)
    code = Path(job["code_path"]).read_text(encoding="utf-8")
    X = data["X_all"]
    rng = np.random.default_rng(int(job.get("seed", 0)))
    t0 = time.time()
    if job["kind"] == "features":
        if job["sample"]:
            idx = np.sort(rng.choice(len(X), size=min(int(job["sample_rows"]), len(X)), replace=False))
            df = X.iloc[idx].reset_index(drop=True)
        else:
            df = X.reset_index(drop=True)
        ns = {"np": np, "pd": pd}
        exec(compile(code, "<make_features>", "exec"), ns)
        fn = ns.get("make_features")
        if not callable(fn):
            raise ValueError("в коде нет функции make_features(df)")
        out = fn(df.copy())
        res, notes = normalize_features(out, df, job["slot_id"], int(job["max_new"]))
        result = {"ok": True, "n_new": res.shape[1], "columns": list(res.columns), "notes": notes,
                  "time_s": round(time.time() - t0, 2), "describe": _describe(res)}
        if job["sample"]:
            result["head"] = res.head(5).to_string(max_cols=12, max_colwidth=14)
        else:
            path = wd / "feats" / f"{job['slot_id']}.pkl"
            path.parent.mkdir(parents=True, exist_ok=True)
            res.to_pickle(path)
            result["path"] = str(path)
        return result
    if job["kind"] == "model":
        import exp_runner
        import feat_builtin
        import task_metrics

        from eval_protocol import Splits

        splits = Splits.load(wd / "splits.npz")
        Xs = feat_builtin.drop_raw_objects(X.iloc[: data["n_train"]].iloc[splits.search_idx].reset_index(drop=True))
        ys = data["y_search"]
        n = min(int(job["sample_rows"]), len(Xs))
        idx = rng.choice(len(Xs), size=n, replace=False)
        cut = int(n * 0.75)
        tr, te = idx[:cut], idx[cut:]
        if spec.task_type in ("multilabel", "multi_regression"):
            raise ValueError("CustomModel поддерживается только для задач с одним таргетом")
        m = exp_runner._CustomModel(code, spec.task_type, spec.n_classes, 0)
        m.fit(Xs.iloc[tr], ys[tr], X_val=Xs.iloc[te], y_val=ys[te])
        p = m.predict(Xs.iloc[te])
        import eval_acceptance

        bad = eval_acceptance.output_check(p, len(te), spec.task_type, spec.n_classes)
        if bad:
            raise ValueError(f"выход CustomModel некорректен: {bad}")
        metric = task_metrics.metric_for(spec)
        s = task_metrics.score(metric, ys[te], p, spec.task_type)
        return {"ok": True, "score_on_sample": s, "time_s": round(time.time() - t0, 2)}
    if job["kind"] == "escape":
        import escape_contract

        from eval_protocol import Splits

        splits = Splits.load(wd / "splits.npz")
        Xs = X.iloc[: data["n_train"]].iloc[splits.search_idx].reset_index(drop=True)
        return escape_contract.check_on_sample(code, spec, Xs, data["y_search"], n_rows=int(job["sample_rows"]),
                                               seed=int(job.get("seed", 0)))
    raise ValueError(f"неизвестный вид песочницы {job['kind']}")


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    import warnings

    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    t0 = time.time()
    print(f"== песочница {job['kind']} {job['slot_id']} ({'подвыборка' if job['sample'] else 'полные данные'})",
          flush=True)
    try:
        res = _child(job)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        # оставляем только кадры пользовательского кода, чтобы Инженер видел суть
        lines = [ln for ln in tb.splitlines() if "<make_features>" in ln or "<custom_model>" in ln
                 or "<escape_fit_predict>" in ln or not ln.startswith("  File")]
        res = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": "\n".join(lines[-25:])}
        print(tb[-3000:], flush=True)
    if res.get("ok"):
        bits = [f"{k}={res[k]}" for k in ("n_new", "score_on_sample", "shape") if k in res]
        print(f"== итог: успех; {', '.join(bits)}; время {time.time() - t0:.1f} с", flush=True)
        if res.get("notes"):
            print("   замечания: " + "; ".join(res["notes"]), flush=True)
    else:
        print(f"== итог: ОШИБКА — {res.get('error')}; время {time.time() - t0:.1f} с", flush=True)
    Path(job["result_path"]).write_text(json.dumps(res, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
