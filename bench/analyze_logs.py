"""Разбор логов прогона по ролям (шаг 3 проверки прототипа).

    python bench/analyze_logs.py путь/к/agent_runs/<время> [--samples 15] [--json out.json]

Читает journal.jsonl и llm_log.jsonl, печатает числа по Исследователю, Инженеру, Критику,
финалу, токенам и времени. Выборку rationale печатает для ручной оценки.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

MENU_RE = re.compile(r"частот|frequency|count[\s_-]*enc|target[\s_-]*enc|таргет[\w\s]*кодир|среднее\s+таргет|"
                     r"флаг\w*\s+пропуск|missing|isna|is_null|log1p\s*\(\s*таргет|group[\s_-]*by|агрегат|"
                     r"по\s+группам|mean\s+по|std\s+по", re.I)
ORDER_RE = re.compile(r"\.index\b|reset_index|\.shift\(|\.rolling\(|cumsum|cumcount|\.rank\(|range\(len|"
                      r"\.iloc\[\s*:\s*\d|np\.arange\(len", re.I)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else f"{a}/0"


def _reason_group(r: str) -> str:
    r = r or ""
    for key in ("невалидная гипотеза", "смена направления", "использует таргет", "использует id", "повтор",
                "невалидный патч", "неизвестн", "отклонён Критиком", "group_by", "group_cols", "target.transform"):
        if key in r:
            return key
    return r[:40]


def analyze(run_dir: Path, samples: int = 15) -> dict:
    J = _load(run_dir / "journal.jsonl")
    L = _load(run_dir / "llm_log.jsonl")
    final = json.loads((run_dir / "final.json").read_text()) if (run_dir / "final.json").exists() else {}
    spec = json.loads((run_dir / "spec.json").read_text()) if (run_dir / "spec.json").exists() else {}
    out: dict = {"run_dir": str(run_dir)}

    # ---------------- Исследователь
    calls = defaultdict(list)
    for r in L:
        calls[(r["role"], r["call_id"])].append(r)
    res_calls = {k: v for k, v in calls.items() if k[0] == "researcher"}
    first_ok = sum(1 for v in res_calls.values() if v[0].get("valid"))
    retry_ok = sum(1 for v in res_calls.values() if not v[0].get("valid") and any(x.get("valid") for x in v[1:]))
    failed = sum(1 for v in res_calls.values() if not any(x.get("valid") for x in v))
    err_kinds = Counter()
    for v in res_calls.values():
        for x in v:
            if x.get("valid") is False:
                e = x.get("error") or ""
                err_kinds["JSON не извлечён" if "JSON" in e else ("схема" if "validation" in e.lower() else e[:40])] += 1
    hyps = [e for e in J if e.get("hypothesis")]
    by_kind = Counter(e["kind"] for e in hyps)
    filtered = [e for e in hyps if e["status"] == "filtered"]
    executed = [e for e in hyps if e["status"] != "filtered"]
    kept = [e for e in hyps if e["status"] == "keep"]
    reasons = Counter(_reason_group(e.get("reason", "")) for e in filtered)
    switches = [e for e in J if e.get("event") == "direction_switch"]
    switch_violations = sum(1 for e in filtered if "смена направления" in (e.get("reason") or ""))
    fc = [e for e in hyps if e["kind"] == "feature_code"]
    menu_like = [e for e in fc if MENU_RE.search(e["hypothesis"].get("feature_spec") or "")]
    # повтор проваленных идей: та же (kind, patch, feature_spec), что раньше провалилась
    seen_failed, repeats_of_failed = set(), 0
    for e in hyps:
        h = e["hypothesis"]
        sig = json.dumps([h.get("kind"), h.get("patch"), (h.get("feature_spec") or "").strip().lower()],
                         sort_keys=True, ensure_ascii=False)
        if sig in seen_failed:
            repeats_of_failed += 1
        if e["status"] in ("discard", "critic_reject", "engineer_failed", "error"):
            seen_failed.add(sig)
    status_by_kind = defaultdict(Counter)
    for e in hyps:
        status_by_kind[e["kind"]][e["status"]] += 1
    out["researcher"] = {
        "calls": len(res_calls), "json_valid_first_try": first_ok, "json_valid_after_retry": retry_ok,
        "json_failed": failed, "invalid_kinds": dict(err_kinds),
        "hypotheses": len(hyps), "by_kind": dict(by_kind), "status_by_kind": {k: dict(v) for k, v in status_by_kind.items()},
        "filtered": len(filtered), "filter_reasons": dict(reasons), "executed": len(executed), "kept": len(kept),
        "repeats_of_failed": repeats_of_failed, "direction_switches": len(switches),
        "proposed_banned_after_switch": switch_violations,
        "feature_code_menu_like": len(menu_like),
        "rationale_samples": [f"[{e['kind']}|{e['status']}] {e['hypothesis'].get('rationale', '')[:200]}"
                              for e in hyps[:samples]],
    }

    # ---------------- Инженер
    eng = [e for e in J if e.get("engineer")]
    first, after_fix, fail = 0, 0, 0
    err_types = Counter()
    for e in eng:
        att = e["engineer"].get("attempts", [])
        if att and att[0].get("ok"):
            first += 1
        elif any(a.get("ok") for a in att):
            after_fix += 1
        else:
            fail += 1
        for a in att:
            if not a.get("ok"):
                msg = a.get("error") or ""
                m = re.search(r"(\w+Error|inf|пуст|строк|таймаут|Слишком много|нет кода|не добавила)", msg)
                err_types[(a.get("stage"), m.group(1) if m else msg[:30])] += 1
    codes = sorted((run_dir / "features").glob("fe_*.py"))
    order_use = []
    for p in codes:
        src = re.sub(r"index\s*=\s*\w+\.index", "", p.read_text(encoding="utf-8"))
        if ORDER_RE.search(src) or (spec.get("id_col") and re.search(rf"\b{re.escape(spec['id_col'])}\b", src)):
            order_use.append(p.stem)
    out["engineer"] = {"requests": len(eng), "ok_first_try": first, "ok_after_fix": after_fix, "failed": fail,
                       "error_types": {f"{k[0]}:{k[1]}": v for k, v in err_types.items()},
                       "registered": len(codes), "suspicious_order_or_id": order_use}

    # ---------------- Критик
    jumps = [e for e in J if e.get("critic")]
    out["critic"] = {
        "jump_calls": len(jumps),
        "verdicts": dict(Counter(e["critic"]["verdict"] for e in jumps)),
        "details": [{"exp": e.get("exp_id"), "rel_delta": e.get("rel_delta"), "verdict": e["critic"]["verdict"],
                     "assoc": e["critic"].get("assoc"), "reason": e["critic"].get("reason", "")[:200]} for e in jumps],
    }
    fin = next((e for e in J if e.get("event") == "final"), {})
    better_blend = None
    if final.get("blend_score") is not None and final.get("best_single_score") is not None:
        gib = spec.get("greater_is_better", False)
        better_blend = final["blend_score"] > final["best_single_score"] if gib else final["blend_score"] < final["best_single_score"]
    out["final"] = {"choice": fin.get("choice"), "reason": fin.get("reason"), "blend_better_by_oof": better_blend,
                    "blend_chosen_when_better": (fin.get("choice") == "blend") if better_blend else None,
                    "l1_score": final.get("l1_score"), "final_score": final.get("final_score"),
                    "best_single_score": final.get("best_single_score"), "blend_score": final.get("blend_score")}

    # ---------------- токены и время
    tok = Counter()
    lat = defaultdict(list)
    for r in L:
        if not r.get("cached"):
            tok[(r["role"], "prompt")] += int(r.get("usage", {}).get("prompt_tokens", 0) or 0)
            tok[(r["role"], "completion")] += int(r.get("usage", {}).get("completion_tokens", 0) or 0)
            lat[r["role"]].append(r.get("latency_s", 0))
    its = final.get("iterations", [])
    out["cost"] = {
        "tokens": {f"{k[0]}.{k[1]}": v for k, v in tok.items()},
        "latency_mean_s": {k: round(sum(v) / len(v), 2) for k, v in lat.items() if v},
        "iter_time_mean_s": round(sum(i["time_s"] for i in its) / len(its), 1) if its else None,
        "tokens_per_iter": round(sum(tok.values()) / len(its), 0) if its else None,
        "experiments": sum(1 for e in J if e.get("exp_id")),
        "experiment_errors": sum(1 for e in J if e.get("status") == "error"),
        "total_time_s": final.get("total_time_s"),
    }
    return out


def print_report(a: dict):
    r, en, c, f, cost = a["researcher"], a["engineer"], a["critic"], a["final"], a["cost"]
    print(f"# Разбор прогона {a['run_dir']}\n")
    print("## Исследователь")
    print(f"- вызовов: {r['calls']}; валидный JSON с первой попытки: {_pct(r['json_valid_first_try'], r['calls'])}, "
          f"после повтора: {r['json_valid_after_retry']}, не удалось: {r['json_failed']}; ошибки: {r['invalid_kinds']}")
    print(f"- гипотез: {r['hypotheses']}; по типам: {r['by_kind']}")
    print(f"- отфильтровано: {_pct(r['filtered'], r['hypotheses'])} — {r['filter_reasons']}")
    print(f"- исполнено: {r['executed']}, принято (keep): {_pct(r['kept'], r['executed'])}")
    print(f"- статусы по типам: {r['status_by_kind']}")
    print(f"- повторы проваленных идей: {r['repeats_of_failed']}; смен направления: {r['direction_switches']}, "
          f"после смены предложил запрещённое: {r['proposed_banned_after_switch']}")
    print(f"- feature_code, похожие на пункты меню: {r['feature_code_menu_like']}")
    print("- rationale для ручной оценки:")
    for s in r["rationale_samples"]:
        print(f"  - {s}")
    print("\n## Инженер")
    print(f"- запросов: {en['requests']}; с первой попытки: {_pct(en['ok_first_try'], en['requests'])}, "
          f"после исправлений: {en['ok_after_fix']}, не справился: {en['failed']}")
    print(f"- типичные ошибки: {en['error_types']}")
    print(f"- зарегистрировано признаков: {en['registered']}; подозрение на id/порядок строк: {en['suspicious_order_or_id']}")
    print("\n## Критик")
    print(f"- вызовов на скачок: {c['jump_calls']}; вердикты: {c['verdicts']}")
    for d in c["details"]:
        print(f"  - {d['exp']}: прирост {d['rel_delta']}, {d['verdict']}; связь с таргетом {d['assoc']}; {d['reason']}")
    print(f"- финал: выбор {f['choice']} ({(f['reason'] or '')[:150]}); бленд лучше по OOF: {f['blend_better_by_oof']}; "
          f"бленд выбран, когда лучше: {f['blend_chosen_when_better']}")
    print(f"- L1 {f['l1_score']} -> финал {f['final_score']} (лучшая одиночная {f['best_single_score']}, бленд {f['blend_score']})")
    print("\n## Стоимость")
    print(f"- токены: {cost['tokens']}")
    print(f"- средняя задержка LLM, с: {cost['latency_mean_s']}; время итерации, с: {cost['iter_time_mean_s']}; "
          f"токенов на итерацию: {cost['tokens_per_iter']}")
    print(f"- экспериментов: {cost['experiments']}, с ошибкой: {cost['experiment_errors']}; всего {cost['total_time_s']} с")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--json")
    args = ap.parse_args()
    res = analyze(Path(args.run_dir), args.samples)
    print_report(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
