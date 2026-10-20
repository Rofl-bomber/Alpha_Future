"""Оркестратор по фазам, как работает DS:

    A. Разведка моделей (без LLM): L1 и другие семейства с дефолтами -> рабочая модель.
    B. Предобработка (без LLM): короткий список применимых вариантов.
    C. Признаки (Исследователь + Инженер + Критик): модель зафиксирована.
    D. Тюнинг (Исследователь): только параметры рабочей модели и постобработка.
    Перенос (без LLM): лучшая конфигурация на другие семейства — для бленда.
    Финал: бленд + выбор Критика -> submission.csv.

Изменение принимается (keep), только если средняя парная разница по фолдам больше noise_k·SE
и улучшение есть минимум на min_folds_improved фолдах.
"""

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
from .models import MODEL_MENU, MODEL_NAMES
from .pipeline import apply_patch, config_hash, default_config, diff_sections
from .profiler import profile
from .runner import rerun_postprocess, run_experiment
from .spec import build_spec, load_data
from .submission import write_submission

FAIL_STATUSES = ("discard", "critic_reject", "error", "engineer_failed")

PHASES = {
    "models": {"title": "A. Разведка моделей", "auto": True},
    "preprocess": {"title": "B. Предобработка", "auto": True},
    "features": {
        "title": "C. Признаки", "kinds": ("feature_code", "config"), "keys": ("features.",),
        "goal": "Улучшить метрику новыми признаками на зафиксированной модели: доменные признаки через "
                "feature_code, готовые операции и удаление колонок через config с ключами features.*. "
                "Модель, её параметры и предобработку не меняй — они выбраны на предыдущих фазах.",
    },
    "tuning": {
        "title": "D. Тюнинг", "kinds": ("model", "postprocess"), "keys": ("model.params", "postprocess."),
        "goal": "Подобрать параметры зафиксированной модели (model.params) и постобработку на финальном "
                "наборе признаков. Семейство модели (model.name) и признаки не меняй.",
    },
    "transfer": {"title": "Перенос на другие семейства", "auto": True},
}

# Изменения, через которые в модель может попасть новая информация, — только их проверяет Критик.
# Смена модели, её параметров или постобработки утечку таргета внести не может.
CRITIC_SECTIONS = {"features", "preprocess", "target"}


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
        self.it = 0                       # сквозной номер LLM-итерации
        self.iter_stats: list[dict] = []
        self.phase_stats: list[dict] = []

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
        families = [m.strip() for m in cfg.phase_models.split(",") if m.strip() in MODEL_NAMES] or ["lightgbm"]

        # ---------------- A. Разведка моделей (первое семейство — L1)
        t = self._phase_start("models")
        l1_cfg = apply_patch(default_config(spec), {"model.name": families[0]})
        res = self._run(l1_cfg)
        if not res.ok:
            raise RuntimeError(f"L1 упал: {res.error}")
        self.tried_configs.add(config_hash(l1_cfg))
        self.l1_importances = {k: v for k, v in (res.importances or {}).items() if k in spec.feature_cols}
        self.prof.set_importances(self.l1_importances)
        self.journal.add({"id": "L1", "uid": "A_L1", "kind": "model", "phase": "models",
                          "rationale": f"L1: {families[0]} с параметрами по умолчанию", "patch": {}},
                         res, "baseline", iter=0, phase="models")
        write_submission(res.test, spec, self.data)
        (self.run_dir / "profile.txt").write_text(self.prof.summary(), encoding="utf-8")
        _log(f"L1 {res.exp_id} {families[0]}: score={_fmt(res.score)} фолды={res.fold_scores} ({res.time_s} c); "
             "submission записан")
        for fam in families[1:]:
            self._auto("models", "model", {"model.name": fam}, f"разведка: {fam} с параметрами по умолчанию")
        self._phase_end("models", t)

        # ---------------- B. Предобработка
        t = self._phase_start("preprocess")
        for patch, why in self._preprocess_candidates():
            self._auto("preprocess", "config", patch, why)
        self._phase_end("preprocess", t)

        # ---------------- LLM-роли
        mock = MockLLM() if cfg.mock_llm else None
        cache_dir = Path(cfg.llm_cache_dir) if cfg.llm_cache_dir else spec.task_dir / ".agent_cache" / "llm"
        self.client = LLMClient(cfg, self.llm_log, cache_dir, mock=mock)
        self.researcher = Researcher(self.client)
        self.engineer = Engineer(self.client, cfg.max_new_cols)
        self.critic = Critic(self.client)

        # ---------------- C. Признаки и D. Тюнинг: общий бюджет AGENT_ITERS
        n_c = min(cfg.n_iters, max(1, round(cfg.n_iters * cfg.features_share))) if cfg.n_iters > 0 else 0
        used = self._llm_phase("features", n_c, cfg.plateau_features)
        self._llm_phase("tuning", cfg.n_iters - used, cfg.plateau_tuning)

        # ---------------- Перенос лучшей конфигурации на другие семейства
        if cfg.transfer:
            t = self._phase_start("transfer")
            best_model = self.journal.best()["config"]["model"]["name"]
            for fam in families:
                if fam != best_model:
                    self._auto("transfer", "model", {"model.name": fam},
                               f"перенос лучших признаков и предобработки на {fam}")
            self._phase_end("transfer", t)

        # ---------------- финал
        final = self._final()
        final["llm_tokens"] = self.llm_log.tokens
        final["iterations"] = self.iter_stats
        final["phases"] = self.phase_stats
        final["total_time_s"] = round(time.time() - self.t0, 1)
        final["run_dir"] = str(self.run_dir)
        (self.run_dir / "final.json").write_text(json.dumps(final, ensure_ascii=False, indent=1, default=str))
        return final

    # ================================================================== фазы
    def _phase_start(self, phase: str) -> float:
        b = self.journal.best()
        _log(f"=== Фаза {PHASES[phase]['title']}" + (f" (лучший {b['exp_id']} {_fmt(b['score'])})" if b else ""))
        self.journal.add(None, None, "event", event="phase_start", phase=phase)
        return time.time()

    def _phase_end(self, phase: str, t0: float, reason: str = "бюджет фазы исчерпан", iters: int | None = None):
        b = self.journal.best()
        n = sum(1 for e in self.journal.entries if e.get("phase") == phase and e.get("exp_id"))
        kept = sum(1 for e in self.journal.entries if e.get("phase") == phase and e["status"] == "keep")
        st = {"phase": phase, "time_s": round(time.time() - t0, 1), "experiments": n, "kept": kept,
              "reason": reason, "best": b["exp_id"], "best_score": b["score"], "iters": iters}
        self.phase_stats.append(st)
        self.journal.add(None, None, "event", event="phase_end", **st)
        _log(f"=== Фаза {PHASES[phase]['title']} завершена ({reason}): экспериментов {n}, принято {kept}, "
             f"лучший {b['exp_id']} {_fmt(b['score'])}, {st['time_s']:.0f} c")

    def _preprocess_candidates(self) -> list[tuple[dict, str]]:
        """Короткий список применимых вариантов предобработки (по профилю данных, без знания задачи)."""
        spec, data = self.spec, self.data
        Xtr = data.X.iloc[: data.n_train]
        out: list[tuple[dict, str]] = []
        if Xtr.isna().any().any():
            out.append(({"preprocess.missing_flags": True}, "в данных есть пропуски — флаги пропусков"))
        if spec.cat_cols:
            out.append(({"preprocess.cat_encoding": "target"}, "OOF target encoding вместо категорий"))
            rare = 20
            if any((Xtr[c].value_counts() < rare).any() for c in spec.cat_cols):
                out.append(({"preprocess.rare_min_count": rare}, f"есть категории реже {rare} в train — объединение"))
        if not spec.is_clf:
            y = pd.Series(data.y)
            if y.min() > -1 and y.skew() > 1:
                out.append(({"target.transform": "log1p"}, f"скошенный таргет (skew={y.skew():.2f}) — log1p"))
        return out[: self.cfg.preprocess_max]

    def _llm_phase(self, phase: str, max_iters: int, plateau: int) -> int:
        """LLM-фаза: до max_iters итераций; досрочный выход по плато или когда все категории запрещены."""
        if max_iters <= 0:
            return 0
        t = self._phase_start(phase)
        allowed = PHASES[phase]["kinds"]
        self.streak = {k: 0 for k in self.streak}
        no_keep, used, reason = 0, 0, "бюджет фазы исчерпан"
        for k in range(1, max_iters + 1):
            banned = [x for x in allowed if self.streak.get(x, 0) >= self.cfg.fail_streak_switch]
            if banned:
                for x in banned:
                    self.streak[x] = 0
                if set(banned) >= set(allowed):
                    reason = "все категории фазы исчерпаны"
                    break
                self.journal.add(None, None, "event", event="direction_switch", kinds=banned, iter=self.it + 1,
                                 phase=phase)
                _log(f"Смена направления: запрещены категории {banned}")
            n_keep = self._iteration(phase, k, max_iters, banned)
            used += 1
            no_keep = 0 if n_keep else no_keep + 1
            if no_keep >= plateau and k < max_iters:
                reason = f"плато: {no_keep} итерации подряд без улучшений"
                self.journal.add(None, None, "event", event="plateau", phase=phase, iter=self.it)
                break
        self._phase_end(phase, t, reason, iters=used)
        return used

    # ================================================================== итерация
    def _iteration(self, phase: str, k: int, max_k: int, banned: list[str]) -> int:
        cfg = self.cfg
        self.it += 1
        it = self.it
        t_it = time.time()
        tok_before = json.loads(json.dumps(self.llm_log.tokens))
        best_e = self.journal.best()
        ctx = self._researcher_ctx(phase, k, max_k, banned, best_e)
        out = self.researcher.propose(ctx)
        if not out:
            _log(f"Итерация {it} ({PHASES[phase]['title']} {k}/{max_k}): Исследователь не вернул валидных гипотез")
            self.journal.add(None, None, "event", event="researcher_failed", iter=it, phase=phase)
            self.iter_stats.append({"iter": it, "phase": phase, "time_s": round(time.time() - t_it, 1), "n_hyp": 0})
            return 0
        hyps, invalid = out
        for bad in invalid:
            raw = bad["raw"] if isinstance(bad["raw"], dict) else {"raw": bad["raw"]}
            self.journal.add({**raw, "kind": raw.get("kind"), "phase": phase}, None, "filtered",
                             reason=f"невалидная гипотеза: {bad['error'][:300]}", iter=it, phase=phase)
        hyps = hyps[: cfg.max_hypotheses]
        _log(f"Итерация {it} ({PHASES[phase]['title']} {k}/{max_k}): {len(hyps)} гипотез: "
             + ", ".join(f"{h.id}:{h.kind}" for h in hyps))
        n_keep = 0
        for j, h in enumerate(hyps):
            n_keep += int(self._process(h, phase, it, j, banned))
        tok_after = self.llm_log.tokens
        d_tok = {r: {"prompt": v["prompt"] - tok_before.get(r, {}).get("prompt", 0),
                     "completion": v["completion"] - tok_before.get(r, {}).get("completion", 0)}
                 for r, v in tok_after.items()}
        b = self.journal.best()
        self.iter_stats.append({"iter": it, "phase": phase, "time_s": round(time.time() - t_it, 1),
                                "n_hyp": len(hyps), "kept": n_keep, "tokens": d_tok, "best": b["exp_id"],
                                "best_score": b["score"]})
        _log(f"Итерация {it} завершена за {time.time() - t_it:.0f} c; принято {n_keep}; "
             f"лучший {b['exp_id']} score={_fmt(b['score'])}")
        return n_keep

    def _phase_menu(self, phase: str, best_cfg: dict) -> str:
        spec = self.spec
        if phase == "features":
            return "\n".join([
                f"Модель зафиксирована: {best_cfg['model']['name']}; предобработка выбрана: "
                f"{json.dumps(best_cfg['preprocess'], ensure_ascii=False)}, target={best_cfg['target']['transform']}.",
                "- kind=feature_code: новый признак по feature_spec; патч {\"features.custom\": [\"+fe_<имя>\"]}",
                "- features.builtin: freq_enc (частоты категорий на train+test), target_enc (OOF target encoding "
                "в дополнение к категориям), group_stats (mean/std/отклонение числовых признаков внутри категорий; "
                "features.group_by — категориальные колонки, features.group_cols — числовые; иначе по важности)",
                "- features.drop: удалить исходные колонки или fe_XXX",
                "- features.custom: включить или выключить уже зарегистрированные fe_XXX",
            ])
        name = best_cfg["model"]["name"]
        pp = ("- postprocess.class_weights_opt: true/false — множители вероятностей классов под macro F1"
              if spec.is_clf else "- postprocess.clip_to_train_range: true/false — клиппинг в диапазон train")
        return "\n".join([
            f"Модель зафиксирована: {name}. Параметры по умолчанию: {MODEL_MENU.get(name, '-')}.",
            f"Текущие явные параметры: {json.dumps(best_cfg['model']['params'], ensure_ascii=False)}.",
            "- model.params: словарь параметров или ключи вида model.params.<имя>; model.name не меняй",
            pp,
        ])

    def _researcher_ctx(self, phase: str, k: int, max_k: int, banned: list[str], best_e: dict) -> dict:
        spec, cfg = self.spec, self.cfg
        ph = PHASES[phase]
        switch_note = ""
        if banned:
            switch_note = ("СМЕНА НАПРАВЛЕНИЯ: последние гипотезы категорий " + ", ".join(banned)
                           + f" ({cfg.fail_streak_switch}+ неудач подряд) не дали улучшения. В этой итерации НЕ "
                             "предлагай гипотезы этих категорий — переключись на другие разрешённые.")
        notes = (" Примечания: " + "; ".join(spec.notes)) if spec.notes else ""
        prompt = dict(
            task_brief=spec.brief() + notes,
            task_excerpt=spec.task_text[:1500],
            profile=self.prof.summary(with_brief=False),
            best_id=best_e["exp_id"], best_score=_fmt(best_e["score"]),
            best_config=json.dumps(best_e["config"], ensure_ascii=False),
            journal=self.journal.summary(10),
            phase_title=ph["title"], phase_goal=ph["goal"],
            phase_kinds=", ".join(ph["kinds"]), phase_keys=", ".join(f"{x}*" for x in ph["keys"]),
            phase_menu=self._phase_menu(phase, best_e["config"]),
            features=self.registry.menu(),
            phase_iter=k, phase_max=max_k, noise_k=_fmt(cfg.noise_k, 3), min_folds=cfg.min_folds_improved,
            n_folds=cfg.n_folds,
            switch_note=switch_note or "Ограничений по категориям нет.",
            n_hyp=f"от 3 до {cfg.max_hypotheses}",
        )
        mock = dict(phase=phase, phase_iter=k, iteration=self.it, n_iters=cfg.n_iters, banned_kinds=banned,
                    top_num=self.prof.top_numeric(5), top_cat=self.prof.top_categorical(3), target=spec.target,
                    id_col=spec.id_col, task_type=spec.task_type, n_hyp=cfg.max_hypotheses,
                    model_name=best_e["config"]["model"]["name"])
        return {"prompt": prompt, "mock": mock}

    # ================================================================== гипотеза
    def _filter(self, h, banned: list[str], phase: str, current_model: str) -> str | None:
        spec = self.spec
        ph = PHASES[phase]
        if h.kind not in ph["kinds"]:
            return f"не та фаза: в фазе «{ph['title']}» разрешены {', '.join(ph['kinds'])}"
        bad = [k for k in h.patch if not str(k).startswith(ph["keys"])
               and not (k == "model.name" and h.patch[k] == current_model)]
        if bad:
            return f"не та фаза: ключи {bad} не разрешены в фазе «{ph['title']}»"
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

    def _process(self, h, phase: str, it: int, j: int, banned: list[str]) -> bool:
        hd = h.model_dump()
        hd["uid"] = f"i{it}_{j + 1}_{h.id or 'h'}"
        hd["phase"] = phase
        best_e = self.journal.best()
        current_model = best_e["config"]["model"]["name"]
        reason = self._filter(h, banned, phase, current_model)
        if reason:
            self.journal.add(hd, None, "filtered", reason=reason, iter=it, phase=phase)
            _log(f"  {hd['uid']} [{h.kind}] отфильтрована: {reason}")
            return False
        patch = {k: v for k, v in h.patch.items() if k != "model.name"}  # совпадающее имя модели — не изменение
        fe_id, alias, eng_log = None, None, None
        if h.kind == "feature_code":
            custom = patch.get("features.custom", [])
            custom = custom if isinstance(custom, list) else [custom]
            unknown = [str(x).lstrip("+-") for x in custom if not self.registry.resolve(str(x).lstrip("+-"))]
            alias = unknown[0] if unknown else f"fe_{re.sub(r'[^0-9a-zA-Z_]', '_', hd['uid'])}"
            fe_id, err, eng_log = self._engineer(h, alias, hd["uid"])
            if fe_id is None:
                self._fail(h.kind)
                self.journal.add(hd, None, "engineer_failed", reason=err, engineer=eng_log, iter=it, phase=phase)
                _log(f"  {hd['uid']} [feature_code] Инженер не справился: {' '.join(str(err).split())[-160:]}")
                return False
            if not unknown:  # имя не указано или совпало с уже зарегистрированным — добавляем новый признак явно
                patch["features.custom"] = list(custom) + [f"+{alias}"]
        patch = self._resolve_patch(patch, alias, fe_id)
        return self._evaluate(hd, h.kind, patch, phase, it, fe_id=fe_id, eng_log=eng_log)

    def _auto(self, phase: str, kind: str, patch: dict, rationale: str) -> bool:
        """Гипотеза без LLM (фазы A, B и перенос)."""
        n = sum(1 for e in self.journal.entries if e.get("phase") == phase and e.get("hypothesis"))
        hd = {"id": f"{phase}_{n + 1}", "uid": f"{phase[0].upper()}_{n + 1}", "kind": kind, "phase": phase,
              "rationale": rationale, "patch": patch, "feature_spec": None, "expected": "", "cost": "auto"}
        return self._evaluate(hd, kind, patch, phase, self.it)

    def _decide(self, res, base_e: dict) -> dict:
        """keep, если средняя парная разница по фолдам > noise_k·SE и улучшение минимум на min_folds фолдах."""
        cfg, metric = self.cfg, self.spec.metric
        d = np.array([improvement(metric, a, b) for a, b in zip(res.fold_scores, base_e["fold_scores"])])
        delta = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        t = delta / se if se > 0 else (float("inf") if delta > 0 else 0.0)
        folds_better = int((d > 0).sum())
        if delta <= 0:
            why = "хуже или равно базе"
        elif folds_better < cfg.min_folds_improved:
            why = f"лучше только на {folds_better}/{len(d)} фолдах"
        elif delta <= cfg.noise_k * se:
            why = f"в пределах шума: дельта {delta:.4g} ≤ {cfg.noise_k:g}·SE {se:.4g}"
        else:
            why = None
        return {"keep": why is None, "why": why, "delta": delta, "se": se, "t": t, "folds_better": folds_better,
                "rel": relative_improvement(metric, res.score, base_e["score"])}

    def _evaluate(self, hd: dict, kind: str, patch: dict, phase: str, it: int, fe_id=None, eng_log=None) -> bool:
        spec, cfg = self.spec, self.cfg
        best_e = self.journal.best()
        best_res = self.journal.result(best_e["exp_id"])
        base_cfg = best_e["config"]
        try:
            cfg_new = apply_patch(base_cfg, patch)
        except Exception as e:
            self.journal.add(hd, None, "filtered", reason=f"невалидный патч: {str(e)[:300]}", iter=it, phase=phase,
                             feature=fe_id)
            _log(f"  {hd['uid']} [{kind}] невалидный патч: {str(e)[:120]}")
            return False
        sem = self._semantic_check(cfg_new)
        if sem:
            self.journal.add(hd, None, "filtered", reason=sem, iter=it, phase=phase, feature=fe_id)
            _log(f"  {hd['uid']} [{kind}] отфильтрована: {sem}")
            return False
        chash = config_hash(cfg_new)
        if chash in self.tried_configs:
            self.journal.add(hd, None, "filtered", reason="повтор: такая конфигурация уже проверялась", iter=it,
                             phase=phase, feature=fe_id)
            _log(f"  {hd['uid']} [{kind}] отфильтрована: повтор конфигурации")
            return False
        self.tried_configs.add(chash)

        changed = diff_sections(cfg_new, base_cfg)
        if changed == {"postprocess"} and best_res is not None:
            self.n_exp += 1
            res = rerun_postprocess(best_res, cfg_new, spec, self.folds, self.data, exp_id=f"e{self.n_exp:03d}")
        else:
            res = self._run(cfg_new)

        extra = dict(iter=it, phase=phase, base_exp=best_e["exp_id"], feature=fe_id, engineer=eng_log,
                     applied_patch=patch)
        if not res.ok:
            self._fail(kind)
            self.journal.add(hd, res, "error", reason=str(res.error).splitlines()[0][:300], **extra)
            _log(f"  {hd['uid']} [{kind}] {res.exp_id} ошибка: {str(res.error).splitlines()[0][:150]}")
            return False

        dec = self._decide(res, best_e)
        status, critic = ("keep" if dec["keep"] else "discard"), None
        if dec["keep"] and dec["rel"] > cfg.jump_threshold and changed & CRITIC_SECTIONS:
            critic = self._critic_jump(hd, res, best_e, best_res, fe_id, dec["delta"], dec["rel"])
            if critic["verdict"] != "accept":
                status = "critic_reject"
        if fe_id:
            self.registry.meta[fe_id]["status"] = status
        self.journal.add(hd, res, status, delta=round(dec["delta"], 8), rel_delta=round(dec["rel"], 6),
                         se=round(dec["se"], 8), t_stat=round(dec["t"], 3) if np.isfinite(dec["t"]) else None,
                         folds_improved=dec["folds_better"], reason=dec["why"], critic=critic, **extra)
        mark = {"keep": "KEEP", "discard": "discard", "critic_reject": "REJECT(критик)"}[status]
        t_txt = f"t={dec['t']:.1f}" if np.isfinite(dec["t"]) else "t=inf"
        _log(f"  {hd['uid']} [{kind}] {res.exp_id} {res.config['model']['name']}: score={_fmt(res.score)} "
             f"дельта={dec['delta']:+.5g} ({dec['rel']:+.1%}, {t_txt}) фолдов лучше {dec['folds_better']}/"
             f"{len(self.folds)} -> {mark}" + (f" [{dec['why']}]" if dec["why"] and dec["delta"] > 0 else "")
             + f" ({res.time_s} c)")
        if status == "keep":
            self.streak[kind] = 0
            write_submission(res.test, spec, self.data)
            return True
        self._fail(kind)
        return False

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

    def _critic_jump(self, hd, res, best_e, best_res, fe_id, delta, rel) -> dict:
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
