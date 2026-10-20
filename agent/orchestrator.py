"""Оркестратор: L1 -> цикл Исследователь/Инженер/Критик -> финал (бленд + выбор Критика) -> submission.csv."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .cv import make_folds
from .ensemble import blend
from .features.builtin import build_feature_frame
from .features.sandbox import FeatureRegistry, run_make_features
from .journal import Journal, LLMLog
from .llm.client import LLMClient
from .llm.mock import MockLLM
from .llm.roles import Critic, Engineer, Researcher, dumps
from .metrics import improvement, relative_improvement
from .models import MODEL_MENU
from .pipeline import apply_patch, config_hash, default_config, diff_sections
from .profiler import profile
from .runner import rerun_postprocess, run_experiment
from .spec import build_spec, load_data
from .submission import write_submission

FAIL_STATUSES = ("discard", "critic_reject", "error", "engineer_failed")


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fmt(x, nd=6):
    return "-" if x is None else f"{x:.{nd}g}"


def _norm_text(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


class Orchestrator:
    def __init__(self, task_md: str | Path, cfg: Config):
        self.cfg = cfg
        self.t0 = time.time()
        self.spec = build_spec(task_md)
        base = Path(cfg.runs_dir) if cfg.runs_dir else self.spec.task_dir / "agent_runs"
        self.run_dir = base / time.strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.run_dir, self.spec.greater_is_better)
        self.llm_log = LLMLog(self.run_dir)
        self.registry = FeatureRegistry(self.run_dir)
        self.streak: dict[str, int] = {k: 0 for k in ("config", "feature_code", "model", "postprocess")}
        self.tried_configs: set[str] = set()
        self.seen_hyp: set[str] = set()
        self.seen_specs: set[str] = set()
        self.n_exp = 0
        self.iter_stats: list[dict] = []

    # ================================================================== запуск
    def run(self) -> dict:
        spec, cfg = self.spec, self.cfg
        _log(f"Задача: {spec.task_md} | {spec.task_type} | метрика {spec.metric} | таргет {spec.target}")
        (self.run_dir / "spec.json").write_text(json.dumps(spec.short(), ensure_ascii=False, indent=1, default=str))
        (self.run_dir / "agent_config.json").write_text(json.dumps(cfg.__dict__, ensure_ascii=False, indent=1, default=str))

        self.data = load_data(spec, cfg.subsample, cfg.seed)
        self.prof = profile(spec, self.data, cfg.profile_max_chars)
        self.folds = make_folds(spec, self.data.y, cfg.n_folds, cfg.seed)
        _log(f"Данные: train {self.data.n_train}, test {self.data.n_test}, признаков {len(spec.feature_cols)}; "
             f"фолды {self.folds.scheme}")

        # ---------------- L1: LightGBM с дефолтами, без LLM
        l1_cfg = default_config(spec)
        res = self._run(l1_cfg)
        if not res.ok:
            raise RuntimeError(f"L1 упал: {res.error}")
        self.tried_configs.add(config_hash(l1_cfg))
        self.l1_importances = {k: v for k, v in (res.importances or {}).items() if k in spec.feature_cols}
        self.prof.set_importances(self.l1_importances)
        self.journal.add(None, res, "baseline", iter=0)
        write_submission(res.test, spec, self.data)
        (self.run_dir / "profile.txt").write_text(self.prof.summary(), encoding="utf-8")
        _log(f"L1 {res.exp_id}: score={_fmt(res.score)} фолды={res.fold_scores} ({res.time_s} c); submission записан")

        # ---------------- LLM-роли
        mock = MockLLM() if cfg.mock_llm else None
        cache_dir = Path(cfg.llm_cache_dir) if cfg.llm_cache_dir else spec.task_dir / ".agent_cache" / "llm"
        self.client = LLMClient(cfg, self.llm_log, cache_dir, mock=mock)
        self.researcher = Researcher(self.client)
        self.engineer = Engineer(self.client, cfg.max_new_cols)
        self.critic = Critic(self.client)

        # ---------------- цикл
        for it in range(1, cfg.n_iters + 1):
            self._iteration(it)

        # ---------------- финал
        final = self._final()
        final["llm_tokens"] = self.llm_log.tokens
        final["iterations"] = self.iter_stats
        final["total_time_s"] = round(time.time() - self.t0, 1)
        final["run_dir"] = str(self.run_dir)
        (self.run_dir / "final.json").write_text(json.dumps(final, ensure_ascii=False, indent=1, default=str))
        return final

    # ================================================================== итерация
    def _iteration(self, it: int):
        cfg = self.cfg
        t_it = time.time()
        tok_before = json.loads(json.dumps(self.llm_log.tokens))
        banned = [k for k, s in self.streak.items() if s >= cfg.fail_streak_switch]
        if banned:
            for k in banned:
                self.streak[k] = 0
            self.journal.add(None, None, "event", event="direction_switch", kinds=banned, iter=it)
            _log(f"Итерация {it}: смена направления, запрещены категории {banned}")
        best_e = self.journal.best()
        ctx = self._researcher_ctx(it, banned, best_e)
        out = self.researcher.propose(ctx)
        if not out:
            _log(f"Итерация {it}: Исследователь не вернул валидных гипотез")
            self.journal.add(None, None, "event", event="researcher_failed", iter=it)
            self.iter_stats.append({"iter": it, "time_s": round(time.time() - t_it, 1), "n_hyp": 0})
            return
        hyps, invalid = out
        for bad in invalid:
            raw = bad["raw"] if isinstance(bad["raw"], dict) else {"raw": bad["raw"]}
            self.journal.add({**raw, "kind": raw.get("kind")}, None, "filtered",
                             reason=f"невалидная гипотеза: {bad['error'][:300]}", iter=it)
        hyps = hyps[: cfg.max_hypotheses]
        _log(f"Итерация {it}: {len(hyps)} гипотез: " + ", ".join(f"{h.id}:{h.kind}" for h in hyps))
        for j, h in enumerate(hyps):
            self._process(h, it, j, banned)
        tok_after = self.llm_log.tokens
        d_tok = {r: {"prompt": v["prompt"] - tok_before.get(r, {}).get("prompt", 0),
                     "completion": v["completion"] - tok_before.get(r, {}).get("completion", 0)}
                 for r, v in tok_after.items()}
        b = self.journal.best()
        self.iter_stats.append({"iter": it, "time_s": round(time.time() - t_it, 1), "n_hyp": len(hyps),
                                "tokens": d_tok, "best": b["exp_id"], "best_score": b["score"]})
        _log(f"Итерация {it} завершена за {time.time() - t_it:.0f} c; лучший {b['exp_id']} score={_fmt(b['score'])}")

    def _researcher_ctx(self, it: int, banned: list[str], best_e: dict) -> dict:
        spec, cfg = self.spec, self.cfg
        switch_note = ""
        if banned:
            switch_note = ("СМЕНА НАПРАВЛЕНИЯ: последние гипотезы категорий " + ", ".join(banned)
                           + f" ({cfg.fail_streak_switch}+ неудач подряд) не дали улучшения. В этой итерации НЕ "
                             "предлагай гипотезы этих категорий — переключись на другие.")
        notes = (" Примечания: " + "; ".join(spec.notes)) if spec.notes else ""
        prompt = dict(
            task_brief=spec.brief() + notes,
            task_excerpt=spec.task_text[:1500],
            profile=self.prof.summary(with_brief=False),
            best_id=best_e["exp_id"], best_score=_fmt(best_e["score"]),
            best_config=json.dumps(best_e["config"], ensure_ascii=False),
            journal=self.journal.summary(10),
            models="\n".join(f"- {k}: {v}" for k, v in MODEL_MENU.items()),
            features=self.registry.menu(),
            iteration=it, n_iters=cfg.n_iters, remaining=cfg.n_iters - it,
            switch_note=switch_note or "Ограничений по категориям нет.",
            n_hyp=f"от 3 до {cfg.max_hypotheses}",
        )
        mock = dict(iteration=it, n_iters=cfg.n_iters, banned_kinds=banned, top_num=self.prof.top_numeric(5),
                    top_cat=self.prof.top_categorical(3), target=spec.target, id_col=spec.id_col,
                    task_type=spec.task_type, n_hyp=cfg.max_hypotheses)
        return {"prompt": prompt, "mock": mock}

    # ================================================================== гипотеза
    def _filter(self, h, banned: list[str]) -> str | None:
        spec = self.spec
        if h.kind in banned:
            return f"смена направления: категория {h.kind} в этой итерации запрещена"
        text = json.dumps(h.patch, ensure_ascii=False) + " " + (h.feature_spec or "")
        for name, label in ((spec.target, "таргет"), (spec.id_col, "id")):
            if name and re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", text):
                return f"использует {label} ({name})"
        sig = json.dumps([h.kind, h.patch, _norm_text(h.feature_spec)], sort_keys=True, ensure_ascii=False, default=str)
        if sig in self.seen_hyp:
            return "повтор: такая гипотеза уже предлагалась"
        self.seen_hyp.add(sig)
        if h.kind == "feature_code":
            ns = _norm_text(h.feature_spec)
            if ns in self.seen_specs:
                return "повтор: такой признак уже запрашивался"
            self.seen_specs.add(ns)
        return None

    def _resolve_patch(self, patch: dict, own_alias: str | None, own_id: str | None) -> dict:
        out = {}
        for k, v in patch.items():
            if k in ("features.custom", "features.drop"):
                items = v if isinstance(v, list) else [v]
                new_items = []
                for it in items:
                    s = str(it)
                    pref, name = (s[0], s[1:]) if s[:1] in "+-" else ("", s)
                    if own_alias and name == own_alias:
                        name = own_id
                    else:
                        name = self.registry.resolve(name) or name
                    new_items.append(pref + name)
                out[k] = new_items
            else:
                out[k] = v
        return out

    def _semantic_check(self, cfg_new: dict) -> str | None:
        spec = self.spec
        f = cfg_new["features"]
        for fe in f["custom"]:
            if fe not in self.registry:
                return f"неизвестный признак {fe}"
            if self.registry.meta[fe]["status"] == "critic_reject":
                return f"признак {fe} отклонён Критиком"
        allowed_drop = set(spec.feature_cols) | set(self.registry.meta)
        bad = [c for c in f["drop"] if c not in allowed_drop]
        if bad:
            return f"неизвестные колонки в features.drop: {bad}"
        bad = [c for c in f["group_by"] if c not in spec.cat_cols]
        if bad:
            return f"features.group_by должны быть категориальными колонками: {bad}"
        bad = [c for c in f["group_cols"] if c not in spec.num_cols]
        if bad:
            return f"features.group_cols должны быть числовыми колонками: {bad}"
        if spec.is_clf and cfg_new["target"]["transform"] != "none":
            return "target.transform применим только к регрессии"
        return None

    def _process(self, h, it: int, j: int, banned: list[str]):
        cfg, spec = self.cfg, self.spec
        hd = h.model_dump()
        hd["uid"] = f"i{it}_{j + 1}_{h.id or 'h'}"
        reason = self._filter(h, banned)
        if reason:
            self.journal.add(hd, None, "filtered", reason=reason, iter=it)
            _log(f"  {hd['uid']} [{h.kind}] отфильтрована: {reason}")
            return
        best_e = self.journal.best()
        best_res = self.journal.result(best_e["exp_id"])
        base_cfg = best_e["config"]
        patch = dict(h.patch)
        fe_id, alias, eng_log = None, None, None

        if h.kind == "feature_code":
            custom = patch.get("features.custom", [])
            custom = custom if isinstance(custom, list) else [custom]
            unknown = [str(x).lstrip("+-") for x in custom if not self.registry.resolve(str(x).lstrip("+-"))]
            alias = unknown[0] if unknown else f"fe_{re.sub(r'[^0-9a-zA-Z_]', '_', hd['uid'])}"
            fe_id, err, eng_log = self._engineer(h, alias, hd["uid"])
            if fe_id is None:
                self._fail(h.kind)
                self.journal.add(hd, None, "engineer_failed", reason=err, engineer=eng_log, iter=it)
                _log(f"  {hd['uid']} [feature_code] Инженер не справился: {' '.join(str(err).split())[-160:]}")
                return
            if not unknown:  # имя не указано или совпало с уже зарегистрированным — добавляем новый признак явно
                patch["features.custom"] = list(custom) + [f"+{alias}"]
        patch = self._resolve_patch(patch, alias, fe_id)
        try:
            cfg_new = apply_patch(base_cfg, patch)
        except Exception as e:
            self.journal.add(hd, None, "filtered", reason=f"невалидный патч: {str(e)[:300]}", iter=it, feature=fe_id)
            _log(f"  {hd['uid']} [{h.kind}] невалидный патч: {str(e)[:120]}")
            return
        sem = self._semantic_check(cfg_new)
        if sem:
            self.journal.add(hd, None, "filtered", reason=sem, iter=it, feature=fe_id)
            _log(f"  {hd['uid']} [{h.kind}] отфильтрована: {sem}")
            return
        chash = config_hash(cfg_new)
        if chash in self.tried_configs:
            self.journal.add(hd, None, "filtered", reason="повтор: такая конфигурация уже проверялась", iter=it,
                             feature=fe_id)
            _log(f"  {hd['uid']} [{h.kind}] отфильтрована: повтор конфигурации")
            return
        self.tried_configs.add(chash)

        if diff_sections(cfg_new, base_cfg) == {"postprocess"} and best_res is not None:
            self.n_exp += 1
            res = rerun_postprocess(best_res, cfg_new, spec, self.folds, self.data, exp_id=f"e{self.n_exp:03d}")
        else:
            res = self._run(cfg_new)

        extra = dict(iter=it, base_exp=best_e["exp_id"], feature=fe_id, engineer=eng_log, applied_patch=patch)
        if not res.ok:
            self._fail(h.kind)
            self.journal.add(hd, res, "error", reason=str(res.error).splitlines()[0][:300], **extra)
            _log(f"  {hd['uid']} [{h.kind}] {res.exp_id} ошибка: {str(res.error).splitlines()[0][:150]}")
            return

        delta = improvement(spec.metric, res.score, best_e["score"])
        rel = relative_improvement(spec.metric, res.score, best_e["score"])
        folds_better = int(sum(improvement(spec.metric, a, b) > 0
                               for a, b in zip(res.fold_scores, best_e["fold_scores"])))
        keep = delta > 0 and folds_better >= cfg.min_folds_improved
        status, critic = ("keep" if keep else "discard"), None
        if keep and rel > cfg.jump_threshold:
            critic = self._critic_jump(h, hd, res, best_e, best_res, fe_id, delta, rel)
            if critic["verdict"] != "accept":
                status = "critic_reject"
        if fe_id:
            self.registry.meta[fe_id]["status"] = status
        self.journal.add(hd, res, status, delta=round(float(delta), 8), rel_delta=round(float(rel), 6),
                         folds_improved=folds_better, critic=critic, **extra)
        mark = {"keep": "KEEP", "discard": "discard", "critic_reject": "REJECT(критик)"}[status]
        _log(f"  {hd['uid']} [{h.kind}] {res.exp_id} {res.config['model']['name']}: score={_fmt(res.score)} "
             f"дельта={delta:+.5g} ({rel:+.1%}) фолдов лучше {folds_better}/{len(self.folds)} -> {mark} ({res.time_s} c)")
        if status == "keep":
            self.streak[h.kind] = 0
            write_submission(res.test, spec, self.data)
        else:
            self._fail(h.kind)

    def _fail(self, kind: str):
        self.streak[kind] = self.streak.get(kind, 0) + 1

    def _run(self, config: dict):
        self.n_exp += 1
        return run_experiment(config, self.spec, self.folds, self.data, registry=self.registry,
                              importances=getattr(self, "l1_importances", None), n_threads=self.cfg.n_threads,
                              seed=self.cfg.seed, exp_id=f"e{self.n_exp:03d}")

    # ================================================================== Инженер
    def _columns_text(self) -> str:
        rows = []
        for c in self.prof.columns:
            rows.append(f"- {c.name} | {'категориальная (str)' if c.kind == 'cat' else 'числовая (float)'} | "
                        f"{c.missing_train:.2f} | {', '.join(c.examples[:4])} | {(c.description or '')[:100]}")
        return "\n".join(rows)

    def _engineer(self, h, alias: str, uid: str):
        cfg = self.cfg
        mock_ctx = {"alias": alias, "feature_spec": h.feature_spec, "top_num": self.prof.top_numeric(5),
                    "top_cat": self.prof.top_categorical(3)}
        turn = self.engineer.start(h.feature_spec, h.rationale, self._columns_text(), mock_ctx)
        attempts, err = [], None
        sandbox_dir = self.run_dir / "features" / "sandbox"
        for attempt in range(1 + cfg.engineer_max_fixes):
            if turn.code is None:
                err, stage = turn.error or "нет кода", "llm"
            else:
                sb = run_make_features(turn.code, self.data.X, sandbox_dir, f"{uid}_a{attempt}",
                                       timeout=cfg.sandbox_timeout, max_new_cols=cfg.max_new_cols)
                if sb.ok:
                    fe_id = self.registry.next_id()
                    cols = self.registry.register(fe_id, sb.frame, turn.code, h.feature_spec, alias, uid)
                    attempts.append({"attempt": attempt, "ok": True, "columns": cols})
                    _log(f"  {uid} Инженер: {fe_id} ({alias}) готов с попытки {attempt + 1}, колонки {cols}")
                    return fe_id, None, {"call_id": turn.call_id, "attempts": attempts}
                err, stage = sb.error, sb.stage
            attempts.append({"attempt": attempt, "ok": False, "stage": stage, "error": str(err)[-800:]})
            if attempt < cfg.engineer_max_fixes:
                turn = self.engineer.fix(turn, str(err), stage, mock_ctx, attempt + 1)
        return None, err, {"call_id": turn.call_id, "attempts": attempts}

    # ================================================================== Критик: скачок
    def _new_columns_assoc(self, res, best_res) -> tuple[dict, str]:
        spec, data = self.spec, self.data
        new_cols = [c for c in res.feature_names if c not in set(best_res.feature_names or [])]
        out = {}
        if not new_cols:
            return out, "|corr|" if not spec.is_clf else "eta"
        try:
            ff = build_feature_frame(res.config, spec, data, self.registry, self.l1_importances)
        except Exception:
            return out, "-"
        y = data.y
        for c in new_cols[:20]:
            if c not in ff.X.columns:
                out[c] = "вычисляется внутри фолдов (target encoding)"
                continue
            s = ff.X[c].iloc[: data.n_train]
            if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object:
                out[c] = "категориальная"
                continue
            v = pd.to_numeric(s, errors="coerce").values.astype(float)
            m = np.isfinite(v)
            if m.sum() < 10 or np.nanstd(v[m]) == 0:
                out[c] = 0.0
                continue
            if spec.is_clf:
                yy, vv = y[m], v[m]
                mu = vv.mean()
                between = sum(((vv[yy == k].mean() - mu) ** 2) * (yy == k).sum() for k in np.unique(yy))
                out[c] = round(float(np.sqrt(between / ((vv - mu) ** 2).sum())), 4)
            else:
                out[c] = round(float(abs(np.corrcoef(v[m], y[m])[0, 1])), 4)
        return out, ("eta (корреляционное отношение с классом)" if spec.is_clf else "|corr| Пирсона с таргетом")

    def _critic_jump(self, h, hd, res, best_e, best_res, fe_id, delta, rel) -> dict:
        assoc, kind = self._new_columns_assoc(res, best_res)
        nums = [v for v in assoc.values() if isinstance(v, (int, float))]
        max_abs = max(nums) if nums else None
        hyp_view = {k: hd.get(k) for k in ("kind", "rationale", "patch", "feature_spec", "expected")}
        prompt_kw = dict(
            threshold=f"{self.cfg.jump_threshold:.0%}", task_brief=self.spec.brief(), hypothesis=dumps(hyp_view),
            code=self.registry.code(fe_id) if fe_id else "-",
            base_folds=best_e["fold_scores"], new_folds=res.fold_scores,
            base_score=_fmt(best_e["score"]), new_score=_fmt(res.score), delta=f"{delta:+.6g}", rel=f"{rel:+.1%}",
            corr_kind=kind, corr="\n".join(f"- {k}: {v}" for k, v in assoc.items()) or "новых колонок нет",
        )
        v = self.critic.judge_jump(prompt_kw, {"max_abs_corr": max_abs})
        if v is None:
            return {"verdict": "reject", "reason": "Критик не дал валидного ответа — скачок не подтверждён",
                    "assoc": assoc}
        _log(f"  Критик: {v.verdict} — {v.reason[:150]}")
        return {"verdict": v.verdict, "reason": v.reason, "assoc": assoc}

    # ================================================================== финал
    def _final(self) -> dict:
        spec, cfg = self.spec, self.cfg
        rejected = {k for k, m in self.registry.meta.items() if m["status"] == "critic_reject"}
        cands = []
        for e in self.journal.ranked():
            if e["status"] == "critic_reject" or self.journal.result(e["exp_id"]) is None:
                continue
            if set(e["config"]["features"]["custom"]) & rejected:
                continue
            r = self.journal.result(e["exp_id"])
            if any(np.allclose(r.oof_raw, self.journal.result(c["exp_id"]).oof_raw, rtol=1e-7, atol=1e-10)
                   for c in cands):
                continue  # те же предсказания (патч ничего не изменил) — для бленда бесполезно
            cands.append(e)
            if len(cands) >= cfg.blend_top_k:
                break
        best_single = cands[0]
        results = [self.journal.result(e["exp_id"]) for e in cands]
        out = {"best_single": best_single["exp_id"], "best_single_score": best_single["score"],
               "l1_score": next(e["score"] for e in self.journal.entries if e["status"] == "baseline"),
               "candidates": [e["exp_id"] for e in cands]}
        br = None
        if len(results) >= 2:
            br = blend(results, spec, self.data, self.folds, default_config(spec)["postprocess"])
            out.update(blend_score=br.score, blend_weights=dict(zip(br.members, br.weights)),
                       blend_folds=br.fold_scores)
            np.savez_compressed(self.run_dir / "preds" / "blend.npz", oof=br.oof, test=br.test,
                                oof_raw=br.oof_raw, test_raw=br.test_raw)
        _log("Финал: кандидаты " + ", ".join(f"{e['exp_id']}({e['model']}, {_fmt(e['score'])})" for e in cands)
             + (f"; бленд {_fmt(br.score)} веса {dict(zip(br.members, br.weights))}" if br else ""))

        choice, reason = best_single["exp_id"], "кандидат один"
        if br is not None:
            cand_lines = "\n".join(
                f"- {e['exp_id']} | {e['model']} | score={_fmt(e['score'])} | фолды={e['fold_scores']} | "
                f"статус={e['status']}" for e in cands)
            blend_line = (f"score={_fmt(br.score)} | фолды={br.fold_scores} | "
                          f"веса={dict(zip(br.members, br.weights))}")
            direction = "больше — лучше" if spec.greater_is_better else "меньше — лучше"
            v = self.critic.choose_final(
                dict(metric=spec.metric, direction=direction, candidates=cand_lines, blend=blend_line),
                {"blend_score": br.score, "best_single_id": best_single["exp_id"],
                 "best_single_score": best_single["score"], "greater_is_better": spec.greater_is_better})
            valid = {e["exp_id"] for e in cands} | {"blend"}
            if v is not None and v.choice.strip() in valid:
                choice, reason = v.choice.strip(), v.reason
            else:
                better = improvement(spec.metric, br.score, best_single["score"]) > 0
                choice = "blend" if better else best_single["exp_id"]
                reason = "Критик не дал валидного выбора — взят лучший по OOF"
            _log(f"Критик (финал): {choice} — {reason[:150]}")
        if choice == "blend":
            preds, final_score = br.test, br.score
        else:
            r = self.journal.result(choice)
            preds, final_score = r.test, r.score
        path = write_submission(preds, spec, self.data)
        self.journal.add(None, None, "event", event="final", choice=choice, reason=reason, score=None,
                         final_score=final_score, blend_score=br.score if br else None,
                         best_single=best_single["exp_id"])
        out.update(choice=choice, choice_reason=reason, final_score=final_score, submission=path)
        _log(f"submission.csv: {path} (выбор {choice}, CV {_fmt(final_score)}, L1 {_fmt(out['l1_score'])})")
        return out
