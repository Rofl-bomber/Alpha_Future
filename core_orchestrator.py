"""Оркестратор: конечный автомат фаз, решения keep/discard, лестница сабмитов.

Фазы (доли бюджета от старта solution.py, см. core_config.PHASES):
  задача и данные → L1 без LLM → диагностика → поиск по популяции → финалисты на сидах →
  финальная сборка и выбор на selection-holdout → запись.
LLM только предлагает (Аналитик, Исследователь, Инженер, Критик); метрику считает харнесс,
решение «оставить / откатить» принимает этот модуль.
"""

import concurrent.futures as cf
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

import diag_ablation
import diag_errors
import eval_acceptance
import eval_protocol
import exp_journal
import exp_runner
import final_ensemble
import final_submission
import llm_client
import llm_engineer
import llm_roles
import search_operators
import search_population
import search_scheduler
import task_metrics
import task_profiler
import task_spec

log = logging.getLogger("agent")

TIME_FACTOR = {"lgbm": 1.0, "xgb": 1.5, "catboost": 2.0, "extratrees": 1.0, "linear": 0.6, "mlp": 2.0,
               "custom": 2.0, "escape": 3.0}


class Orchestrator:
    def __init__(self, task_path, cfg, timer, registry):
        self.task_path = Path(task_path).resolve()
        self.cfg = cfg
        self.timer = timer
        self.registry = registry
        self.workdir = Path(cfg.workdir)
        self.ctx = exp_runner.RunContext(self.workdir, cfg, timer, registry)
        fh = logging.FileHandler(self.workdir / "run.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger("agent").addHandler(fh)
        self.writer = None
        self.llm = None
        self.summary = {"workdir": str(self.workdir)}
        self.queue = []
        self.tried_sigs = set()
        self.evaluated = {}
        self.custom_feats = {}
        self.diag_text = "Диагностика ещё не проводилась."
        self.focus = None
        self.best_written = None
        self.screen_cache = {}
        self.alt_cache = {}
        self.pool = cf.ThreadPoolExecutor(max_workers=1)  # фоновый Исследователь
        self.sync_pool = cf.ThreadPoolExecutor(max_workers=2)  # синхронные вызовы с таймаутом
        self.research_future = None
        self.library_queue = []
        self.slot_n = 0
        self.branch_n = 0
        self.min_delta = 0.0
        self.l1 = None
        self.hints = {}
        self.segments = []
        self.research_empty = 0
        self.research_skip = 0

    # ================================================================== точка входа
    def run(self) -> dict:
        # (имя фазы, функция, что нужно для запуска: prep — подготовка, l1 — успешный L1)
        steps = [("подготовка", self._prepare, None), ("L1", self._l1, "prep"),
                 ("диагностика", self._initial_diagnosis, "l1"), ("поиск", self._search, "l1"),
                 ("финалисты", self._reseed, "l1"), ("финал", self._final, "l1")]
        prepared = False
        for name, fn, need in steps:
            if (need == "prep" and not prepared) or (need == "l1" and self.l1 is None):
                continue
            try:
                log.info("[%s] фаза: %s", self.timer.stamp(), name)
                fn()
                if name == "подготовка":
                    prepared = True
            except Exception as e:  # noqa: BLE001
                log.exception("фаза «%s» прервана: %s", name, e)
                if name == "подготовка":
                    if self.writer is None:
                        self._emergency_constant()
                    break
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.sync_pool.shutdown(wait=False, cancel_futures=True)
        self._save_summary()
        return self.summary

    # ================================================================== подготовка
    def _prepare(self):
        spec, train, test, dd, sample, text = task_spec.build_rules_spec(self.task_path)
        self.task_md_text = text
        data = task_spec.prepare_data(spec, train, test)
        self.spec, self.data, self.sample = spec, data, sample
        self.writer = final_submission.SubmissionWriter(spec, data.test_ids, len(data.test))
        self._write_constant()
        log.info("TaskSpec (правила): %s; метрика из: %s", spec.short(), spec.metric_source)

        self.llm = llm_client.LLMClient(self.cfg, self.workdir, deadline_fn=self.timer.hard_deadline)
        if self.llm.available:
            try:
                rules = spec.to_json()[:4000]
                cols = task_spec.dictionary_summary(spec, data)
                sample_txt = sample.head(5).to_string() if sample is not None else "нет"
                out = self._with_timeout(lambda: llm_roles.analyst(self.llm, text, rules, cols, sample_txt),
                                         max(30.0, min(240.0, self.timer.left("l1") * 0.4)))
                before = (spec.task_type, spec.metric, tuple(spec.classes))
                spec = task_spec.apply_analyst(spec, out, train)
                if (spec.task_type, spec.metric, tuple(spec.classes)) != before:
                    log.info("Аналитик уточнил спецификацию: %s", spec.short())
                    data = task_spec.prepare_data(spec, train, test)
                    self.writer = final_submission.SubmissionWriter(spec, data.test_ids, len(data.test))
                    self._write_constant()
                if out.get("output_file"):
                    p = (self.task_path.parent / Path(str(out["output_file"])).name)
                    if p.suffix.lower() == ".csv" and p.resolve() != Path(spec.output_path) and \
                            "sample" not in p.name.lower():
                        spec.conflicts.append(f"Аналитик предлагает файл {p.name}; пишем {spec.output_path}")
                self.spec, self.data = spec, data
            except Exception as e:  # noqa: BLE001
                log.warning("Аналитик недоступен (%s), остаёмся на правилах", str(e)[:200])
        else:
            log.warning("LLM не настроена (нет OPENAI_MODEL/OPENAI_API_KEY): работаем без ролей LLM")
        for c in spec.conflicts:
            log.info("конфликт: %s", c)

        X = data.X_all
        const = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        if const:
            data.X_all = X.drop(columns=const)
        spec.save(self.ctx.spec_path)
        with open(self.ctx.data_path, "wb") as f:
            pickle.dump({"X_all": data.X_all, "n_train": data.n_train}, f)
        prof = task_profiler.profile_data(spec, data, self.cfg)
        prof["adversarial"] = task_profiler.run_adversarial(spec, prof, self.ctx)
        self.prof = prof
        splits, vault = eval_protocol.make_splits(spec, data, prof, self.cfg)
        self.splits, self.vault = splits, vault
        splits.save(self.ctx.splits_path)
        self.y_search = data.y[splits.search_idx]
        with open(self.ctx.data_path, "wb") as f:
            pickle.dump({"X_all": data.X_all, "n_train": data.n_train, "y_search": self.y_search}, f)
        self.metric = task_metrics.metric_for(spec)
        self.gib = self.metric.greater_is_better
        self.profile_text = task_profiler.summary_text(spec, prof)
        self.task_text = (f"{spec.title}\n{spec.short()}\nВалидация: {splits.describe()}\n"
                          + ("Заметки: " + "; ".join(spec.notes[:6]) if spec.notes else ""))
        self.columns_text = llm_engineer.columns_text(spec, data.X_all)
        (self.workdir / "profile.txt").write_text(self.profile_text, encoding="utf-8")
        log.info("валидация: %s", splits.describe())
        adv = prof.get("adversarial") or {}
        if adv.get("auc") is not None:
            log.info("adversarial AUC=%.3f", adv["auc"])
        self.journal = exp_journal.Journal(self.workdir / "journal.jsonl", self.gib)
        self.pop = search_population.Population(self.cfg.population_k, self.cfg.max_per_family, self.gib,
                                                self.cfg.freeze_after, self.cfg.seed)
        self.scheduler = search_scheduler.Scheduler(self.cfg, self.pop, self.timer, spec)
        self.library_queue = search_operators.library(spec, prof)
        self.engineer = llm_engineer.Engineer(self.llm, self.ctx, spec, data.X_all, self.columns_text, self.task_text)
        self.summary.update(spec=spec.short(), validation=splits.describe())

    def _write_constant(self):
        pred = final_submission.constant_pred(self.spec, self.data.y, len(self.data.test))
        self.writer.write(pred, final_submission.constant_post(self.spec, self.data.y), "константа (уровень 0)")

    def _with_timeout(self, fn, timeout):
        fut = self.sync_pool.submit(fn)
        return fut.result(timeout=timeout)

    def _emergency_constant(self):
        """Подготовка упала до появления сабмита: пишем константу по правилам, без LLM."""
        try:
            spec, train, test, _, _, _ = task_spec.build_rules_spec(self.task_path)
            data = task_spec.prepare_data(spec, train, test)
            self.spec, self.data = spec, data
            self.writer = final_submission.SubmissionWriter(spec, data.test_ids, len(data.test))
            self._write_constant()
        except Exception as e:  # noqa: BLE001
            log.error("не удалось записать даже константный сабмит: %s", e)

    # ================================================================== L1
    def _l1(self):
        base = search_operators.default_config(self.spec)
        tries = [("L1: LightGBM, надёжные дефолты", base)]
        fast = search_operators.apply_patch(base, {"model.params": {"learning_rate": 0.15, "n_estimators": 600}})
        tries.append(("L1: LightGBM, быстрый", fast))
        tries.append(("L1: ExtraTrees", search_operators.apply_patch(base, {"model.name": "extratrees"})))
        for desc, cfg in tries:
            budget = max(60.0, self.timer.left("reseed") * 0.5 if self.l1 is None else 0)
            node = self._new_node(cfg, None, "anchor", desc, branch="b0")
            res = exp_runner.run_experiment(node.id, cfg, self.ctx, "full", budget_s=budget)
            node.result = res
            if res.get("ok"):
                self.l1 = node
                self.pop.set_anchor(node)
                self.pop.members.append(node.id)
                self.evaluated[search_operators.config_hash(cfg)] = node.id
                self.min_delta = eval_acceptance.noise_floor(res["fold_scores"], self.cfg)
                self._journal(node, "anchor", mode="full")
                self._maybe_write(node, force=True)
                self._write_hints(res)
                log.info("[%s] L1 %s: CV %.6g ± %.3g за %.0f с", self.timer.stamp(), node.id, res["mean"],
                         res["std"], res["time_s"])
                return
            node.status = "failed"
            self._journal(node, "failed", mode="full", error=res.get("error"))
            log.warning("L1 (%s) не удался: %s", desc, str(res.get("error"))[:500])

    def _write_hints(self, res):
        nums = set(self.spec.num_cols)
        top = [f for f, _ in res.get("importance", []) if f in nums][:10]
        self.hints = {"top_num": top}
        self.ctx.hints_path.write_text(json.dumps(self.hints, ensure_ascii=False), encoding="utf-8")

    # ================================================================== узлы и журнал
    def _new_node(self, config, parent, op, desc, branch=None, hyp=None):
        node = search_population.Node(id=self.pop.new_id(), config=config, parent=parent.id if parent else None,
                                      op=op, desc=desc, branch=branch or (parent.branch if parent else "b0"),
                                      hypothesis=hyp)
        self.pop.register(node)
        return node

    def _new_branch(self):
        self.branch_n += 1
        return f"b{self.branch_n}"

    def _journal(self, node, status, mode="full", error=None, delta=None, extra=None):
        res = node.result if mode == "full" else (node.screen if mode == "screen" else node.alt)
        rec = {"node": node.id, "parent": node.parent, "op": node.op, "desc": node.desc, "status": status,
               "mode": mode, "family": node.family, "branch": node.branch,
               "patch": (node.hypothesis or {}).get("patch"), "kind": (node.hypothesis or {}).get("kind"),
               "source": (node.hypothesis or {}).get("source"), "flags": node.flags, "reason": node.reason,
               "error": (error or (res or {}).get("error") or "")[:1500] or None, "delta": delta}
        if res and res.get("ok"):
            rec.update(fold_scores=res["fold_scores"], mean=res["mean"], std=res["std"], time_s=res["time_s"],
                       best_iter=res.get("best_iter"), n_features=res.get("n_features"))
        elif res and res.get("wall_s") and (res.get("timed_out") or "исчерпан" in str(res.get("error", ""))):
            rec["time_s"] = res["wall_s"]  # прогон не уложился в бюджет — оценка времени семейства растёт
        if extra:
            rec.update(extra)
        self.journal.add(rec)

    def _load_preds(self, node, mode="full"):
        res = node.result if mode == "full" else node.screen
        z = np.load(res["preds_path"])
        return {k: z[k] for k in z.files}

    def _maybe_write(self, node, force=False):
        """Новый лучший по CV узел сразу перезаписывает сабмит."""
        if node.mean is None or node.flags:
            return
        g = task_metrics.gain(node.mean, self.gib)
        if force or self.best_written is None or g > self.best_written:
            preds = self._load_preds(node)
            if self.writer.write(preds["te"], node.result.get("post"), f"{node.id} {node.desc[:60]}", node.mean):
                self.best_written = g

    def _est_time(self, family):
        t = self.journal.times(family, "full")
        if t:
            return t
        base = self.l1.result["time_s"] if self.l1 and self.l1.result else 60.0
        return base * TIME_FACTOR.get(family, 1.5)

    # ================================================================== диагностика
    def _initial_diagnosis(self):
        self._diagnose(self.l1, phase_deadline=max(self.timer.deadline("diag"),
                                                   time.time() + 0.08 * max(0.0, self.timer.left("search"))))
        self.scheduler.diag_pending = False
        # сегменты с крупнейшими ошибками фиксируются один раз, до поиска: признаки узлов не меняются по ходу
        if self.segments:
            self.hints["segments"] = self.segments[:5]
            self.ctx.hints_path.write_text(json.dumps(self.hints, ensure_ascii=False), encoding="utf-8")
            self.library_queue.insert(0, {"kind": "config", "block": "features", "cost": "low", "source": "library",
                                          "rationale": f"флаги сегментов с крупнейшими ошибками {self.segments[:5]}",
                                          "patch": {"features.builtin": "+segment_flags"}})

    def _diagnose(self, node, phase_deadline=None):
        if node is None or node.mean is None:
            return
        t0 = time.time()
        deadline = phase_deadline or (time.time() + min(0.12 * self.timer.left("search"), 900.0))
        est = self._est_time(node.family) * 0.45
        texts = []
        try:
            preds = self._load_preds(node)
            Xs = self.data.X_all.iloc[: self.data.n_train].iloc[self.splits.search_idx].reset_index(drop=True)
            err, err_text = diag_errors.analyze(self.spec, self.y_search, preds["oof"], node.result.get("post"), Xs,
                                                preds.get("te"), self.prof)
            if node is self.l1 and not self.segments:
                self.segments = [list(sg) for sg in err.get("segments", [])]
            texts.append("Анализ ошибок:\n" + err_text)
        except Exception as e:  # noqa: BLE001
            log.warning("анализ ошибок не удался: %s", e)

        def screen(cfg, tag):
            budget = min(max(30.0, deadline - time.time()), max(60.0, est * 3))
            if deadline - time.time() < 10:
                return None
            return exp_runner.run_experiment(tag, cfg, self.ctx, "screen", budget_s=budget)

        if deadline - time.time() > est:
            abl = diag_ablation.run(node, self.spec, self.prof, screen, self.gib, lambda: deadline - time.time(), est)
            texts.insert(0, abl["text"])
            self.focus = abl.get("focus")
            if abl.get("base"):
                self.screen_cache[node.id] = abl["base"]
            for r in sorted(abl.get("improving", []), key=lambda r: -r["delta_gain"]):
                self.queue.insert(0, {"kind": "config", "block": r["block"], "patch": r["patch"],
                                      "rationale": f"абляция: {r['variant']} улучшила быстрый скор на {r['delta_gain']:+.4g}",
                                      "cost": "low", "source": "ablation", "parent_id": node.id})
        self.diag_text = "\n\n".join(texts) or self.diag_text
        (self.workdir / f"diag_{node.id}.txt").write_text(self.diag_text, encoding="utf-8")
        log.info("[%s] диагностика %s за %.0f с; фокус: %s", self.timer.stamp(), node.id, time.time() - t0,
                 self.focus)

    # ================================================================== поиск
    def _search(self):
        while True:
            left = self.timer.left("search")
            best = self.pop.best() or self.l1
            est = self._est_time(best.family)
            if left < max(30.0, 0.6 * est):
                break
            tried = {self.pop.nodes[i].family for i in self.pop.nodes}
            restart = [o for o in search_operators.restart_options(self.spec, tried)
                       if self._sig({"kind": "restart", "patch": o[1]}) not in self.tried_sigs]
            action = self.scheduler.next(restart)
            try:
                if action.kind == "diagnose":
                    self._diagnose(self.pop.best() or self.l1)
                elif action.kind == "escape":
                    self._op_escape()
                elif action.kind == "merge":
                    self._op_merge(action.payload["a"], action.payload["b"])
                elif action.kind == "restart":
                    desc, patch = action.payload["option"]
                    self.tried_sigs.add(self._sig({"kind": "restart", "patch": patch}))
                    self._evaluate_hypothesis({"kind": "restart", "block": "model", "patch": patch,
                                               "rationale": desc, "source": "scheduler"})
                elif action.kind == "tune":
                    self._op_tune(action.payload["node"])
                else:
                    hyp = self._next_hypothesis()
                    if hyp is None:
                        # гипотез нет: новая ветка другого семейства, затем Optuna лучшего узла, иначе — финал
                        best = self.pop.best() or self.l1
                        if restart:
                            desc, patch = restart[0]
                            self.tried_sigs.add(self._sig({"kind": "restart", "patch": patch}))
                            self._evaluate_hypothesis({"kind": "restart", "block": "model", "patch": patch,
                                                       "rationale": desc, "source": "scheduler"})
                        elif best.id not in self.scheduler.tuned:
                            self.scheduler.tuned.add(best.id)
                            self._op_tune(best)
                        else:
                            log.info("гипотезы закончились")
                            break
                        continue
                    self._evaluate_hypothesis(hyp)
            except Exception as e:  # noqa: BLE001
                log.exception("ошибка в операторе %s: %s", action.kind, e)
                self.scheduler.feedback(False)
        log.info("[%s] поиск завершён: %s", self.timer.stamp(), self.journal.stats())

    # ------------------------------------------------------------------ очередь гипотез
    def _sig(self, hyp) -> str:
        return json.dumps([hyp.get("kind"), hyp.get("patch"), (hyp.get("feature_spec") or "")[:300]],
                          sort_keys=True, ensure_ascii=False, default=str)

    def _research_context(self, parent):
        custom = "\n".join(f"- {k}: {v}" for k, v in self.custom_feats.items()) or "нет"
        return dict(task=self.task_text, profile=self.profile_text[:9000], diagnostics=self.diag_text[:6000],
                    population=self.pop.summary(), journal=self.journal.summary(), custom=custom,
                    time_left=max(0.0, self.timer.left("search")) / 60, exp_time=self._est_time(parent.family),
                    focus=self.focus or "нет", parent=f"{parent.id} ({parent.family}): {parent.desc}")

    def _start_research(self):
        if self.research_future is not None or not self.llm or not self.llm.available or not self.llm.budget_left():
            return
        if self.research_skip > 0:  # Исследователь дважды подряд не дал нового — пропускаем пару раундов
            self.research_skip -= 1
            return
        parent = self.pop.best() or self.l1
        ctx = self._research_context(parent)
        schema = search_operators.schema_text(self.spec, self.hints)

        def call():
            return llm_roles.researcher(self.llm, schema, 4, ctx)

        self.research_future = self.pool.submit(call)

    def _collect_research(self, wait: float = 0.0):
        fut = self.research_future
        if fut is None:
            return
        try:
            hyps = fut.result(timeout=wait) if wait > 0 else (fut.result() if fut.done() else None)
        except cf.TimeoutError:
            return
        except Exception as e:  # noqa: BLE001
            log.warning("Исследователь не ответил: %s", str(e)[:300])
            hyps = []
            self.research_future = None
            return
        if hyps is None:
            return
        self.research_future = None
        added = 0
        for h in hyps:
            h["source"] = "researcher"
            sig = self._sig(h)
            if sig not in self.tried_sigs:
                self.queue.append(h)
                added += 1
        log.info("[%s] новых гипотез от Исследователя: %d", self.timer.stamp(), added)
        self.research_empty = 0 if added else self.research_empty + 1
        if self.research_empty >= 2:
            self.research_empty, self.research_skip = 0, 3

    def _next_hypothesis(self):
        self._collect_research()
        if len(self.queue) < 2:
            self._start_research()
        if not self.queue and self.research_future is not None:
            self._collect_research(wait=max(1.0, min(self.cfg.llm_timeout_s * 1.5, self.timer.left("search") - 30)))
        while self.queue:
            h = self.queue.pop(0)
            sig = self._sig(h)
            if sig in self.tried_sigs:
                continue
            self.tried_sigs.add(sig)
            return h
        while self.library_queue:
            h = self.library_queue.pop(0)
            sig = self._sig(h)
            if sig not in self.tried_sigs:
                self.tried_sigs.add(sig)
                return h
        return None

    # ------------------------------------------------------------------ Improve / Restart
    def _evaluate_hypothesis(self, hyp):
        kind = hyp.get("kind", "config")
        if kind == "restart":
            parent = self.l1
            branch = self._new_branch()
        else:
            parent = self.pop.nodes.get(hyp.get("parent_id")) or self.pop.select_parent()
            branch = parent.branch
        desc = (hyp.get("rationale") or kind)[:200]
        patch, notes = search_operators.validate_patch(hyp.get("patch") or {}, parent.config, self.spec,
                                                       known_custom=set(self.custom_feats), hints=self.hints)
        if kind in ("feature_code", "model_code"):
            slot = "features" if kind == "feature_code" else "model"
            slot_id = self._slot_id("fe" if slot == "features" else "mc")
            deadline = min(self.timer.deadline("search") - 30, time.time() + max(300.0, 0.15 * self.timer.left("search")))
            impl = self._implement(slot, slot_id, hyp, deadline)
            if not impl:
                return
            if slot == "features":
                flags = self._leak_check(slot_id)
                if flags:
                    node = self._new_node(parent.config, parent, "improve", desc, branch, hyp)
                    node.flags, node.status, node.reason = flags, "flagged", "признак похож на утечку"
                    self._journal(node, "flagged", mode="none")
                    self.scheduler.feedback(False)
                    return
                self.custom_feats[slot_id] = (hyp.get("feature_spec") or desc)[:200]
                patch = dict(patch, **{"features.custom": "+" + slot_id})
            else:
                patch = dict(patch, **{"model.name": "custom", "model.custom_id": slot_id})
        if not patch:
            node = self._new_node(parent.config, parent, "improve", desc, branch, hyp)
            node.status, node.reason = "invalid", "; ".join(notes)[:300] or "пустой патч"
            self._journal(node, "invalid", mode="none")
            return
        config = search_operators.apply_patch(parent.config, patch)
        hyp = dict(hyp, patch=patch)
        self._evaluate(config, parent, "restart" if kind == "restart" else "improve", desc, branch, hyp)

    def _slot_id(self, prefix):
        self.slot_n += 1
        return f"{prefix}_{self.slot_n:03d}"

    def _implement(self, slot, slot_id, hyp, deadline):
        if not self.llm or not self.llm.available:
            return None
        try:
            res = self.engineer.implement(slot, slot_id, hyp, deadline)
        except llm_client.LLMError as e:
            log.warning("Инженер недоступен: %s", str(e)[:200])
            return None
        if not res.get("ok"):
            node = self._new_node(self.l1.config, self.l1, "improve", (hyp.get("rationale") or "")[:200], None, hyp)
            node.status, node.reason = "failed", "Инженер: " + str(res.get("error"))[:300]
            self._journal(node, "failed", mode="none", error=res.get("error"))
            self.scheduler.feedback(False)
            log.info("Инженер не реализовал %s: %s", slot_id, str(res.get("error"))[:200])
            return None
        log.info("Инженер реализовал %s за %d шагов", slot_id, len(res.get("steps", [])))
        return res

    def _leak_check(self, slot_id):
        import pandas as pd

        path = self.workdir / "feats" / f"{slot_id}.pkl"
        if not path.exists():
            return []
        add = pd.read_pickle(path)
        rows = self.splits.search_idx
        ids = self.data.train_ids.iloc[rows].to_numpy() if self.data.train_ids is not None else None
        return eval_acceptance.leak_check(add.iloc[: self.data.n_train].iloc[rows].reset_index(drop=True),
                                          self.y_search, ids, self.spec.task_type, self.cfg.leak_corr)

    # ------------------------------------------------------------------ оценка узла
    def _screen_of(self, node):
        if node.id not in self.screen_cache:
            res = exp_runner.run_experiment(node.id, node.config, self.ctx, "screen",
                                            budget_s=max(60.0, self._est_time(node.family)))
            self.screen_cache[node.id] = res
        return self.screen_cache[node.id]

    def _alt_of(self, node):
        if node.id not in self.alt_cache:
            res = exp_runner.run_experiment(node.id, node.config, self.ctx, "alt",
                                            budget_s=max(60.0, self._est_time(node.family)))
            self.alt_cache[node.id] = res
        return self.alt_cache[node.id]

    def _cmp(self, a, b):
        return eval_acceptance.paired_compare(a, b, self.gib, self.min_delta, self.cfg.accept_min_fold_frac,
                                              self.cfg.noise_se_mult)

    def _evaluate(self, config, parent, op, desc, branch, hyp=None):
        h = search_operators.config_hash(config)
        if h in self.evaluated:
            log.info("конфигурация уже проверялась (%s) — пропуск", self.evaluated[h])
            return None
        fam = search_operators.family(config)
        est = self._est_time(fam)
        left = self.timer.left("search")
        if left < 1.1 * est:
            log.info("на эксперимент %s не хватает времени поиска (%.0f с из ~%.0f с)", fam, left, est)
            return None
        node = self._new_node(config, parent, op, desc, branch, hyp)
        self.evaluated[h] = node.id
        # скрининг: дешёвая оценка на 3 фолдах, в полную оценку проходят не явно худшие
        if est > self.cfg.screen_if_full_s and parent.fold_scores and left > 1.6 * est:
            ps = self._screen_of(parent)
            sb = min(max(60.0, est), self.timer.left("search") - est)
            node.screen = exp_runner.run_experiment(node.id, config, self.ctx, "screen", budget_s=sb)
            if node.screen.get("ok") and ps.get("ok"):
                c = self._cmp(node.screen["fold_scores"], ps["fold_scores"])
                if c["mean_gain"] < -c["se"]:
                    node.status, node.reason = "screen_rejected", f"скрининг Δ={c['mean_gain']:+.4g}"
                    self._journal(node, "screen_rejected", mode="screen", delta=c["mean_gain"])
                    self.pop.note_child(parent.id, False)
                    self.scheduler.feedback(False)
                    return node
        left = self.timer.left("search")
        if left < 1.05 * est:
            node.status, node.reason = "skipped", "после скрининга не осталось времени на полную оценку"
            self._journal(node, "skipped", mode="screen")
            return node
        budget = min(left - 5, max(3 * est, 120.0))
        node.result = exp_runner.run_experiment(node.id, config, self.ctx, "full", budget_s=budget)
        if not node.result.get("ok"):
            node.status = "failed"
            self._journal(node, "failed", mode="full")
            self.pop.note_child(parent.id, False)
            self.scheduler.feedback(False)
            log.info("[%s] %s %s: ошибка — %s", self.timer.stamp(), node.id, op,
                     str(node.result.get("error")).splitlines()[0][:200])
            return node
        cmp = self._cmp(node.result["fold_scores"], parent.fold_scores)
        accepted = cmp["accepted"]
        rel = eval_acceptance.relative_jump(node.mean, parent.mean, self.gib)
        if accepted and rel > self.cfg.jump_rel:
            node.flags.append(f"jump:{rel:+.1%}")
            accepted = self._recheck_jump(node, parent, rel)
        p_time, c_time = parent.result.get("time_s", 0), node.result.get("time_s", 0)
        if accepted and c_time < self.cfg.fast_train_ratio * p_time and fam == parent.family:
            node.flags.append("fast_train")
            accepted = self._critic_ok(node, parent, rel, "обучение подозрительно быстрое")
        node.status = "accepted" if accepted else ("flagged" if node.reason else "rejected")
        if node.status == "flagged":
            pass
        elif not accepted:
            node.flags = []
        self.pop.note_child(parent.id, accepted)
        joined = self.pop.consider(node) if accepted else False
        self._journal(node, node.status, mode="full", delta=cmp["mean_gain"],
                      extra={"wins": cmp["wins"], "se": cmp["se"], "joined": joined})
        log.info("[%s] %s %s [%s]: CV %.6g (Δ=%+.4g ± %.2g, %d/%d фолдов) — %s%s", self.timer.stamp(), node.id, op,
                 fam, node.mean, cmp["mean_gain"], cmp["se"], cmp["wins"], cmp["k"], node.status,
                 " → в популяции" if joined else "")
        if accepted:
            node.flags = [f for f in node.flags if not f.startswith("jump") and f != "fast_train"]
            self._maybe_write(node)
        self.scheduler.feedback(accepted)
        return node

    def _recheck_jump(self, node, parent, rel) -> bool:
        if self.timer.left("search") < self._est_time(node.family):
            node.reason = "не хватило времени перепроверить скачок"
            return False
        node.alt = exp_runner.run_experiment(node.id, node.config, self.ctx, "alt",
                                             budget_s=max(60.0, self._est_time(node.family)))
        pa = self._alt_of(parent)
        if not (node.alt.get("ok") and pa.get("ok")):
            node.reason = "перепроверка скачка не выполнена"
            return False
        c = self._cmp(node.alt["fold_scores"], pa["fold_scores"])
        if c["mean_gain"] <= max(self.min_delta, 0.0):
            node.reason = f"скачок {rel:+.1%} не повторился на других фолдах (Δ={c['mean_gain']:+.4g})"
            return False
        return self._critic_ok(node, parent, rel, f"скачок повторился на других фолдах: Δ={c['mean_gain']:+.4g}")

    def _critic_ok(self, node, parent, rel, alt_text) -> bool:
        if not self.llm or not self.llm.available or not self.llm.budget_left():
            return True
        code = json.dumps((node.hypothesis or {}).get("patch"), ensure_ascii=False)
        for key in ("custom_id",):
            cid = node.config.get("model", {}).get(key)
            if cid:
                p = self.workdir / "codes" / f"{cid}.py"
                code += "\n" + (p.read_text(encoding="utf-8")[:4000] if p.exists() else "")
        for fid in node.config.get("features", {}).get("custom", []):
            p = self.workdir / "codes" / f"{fid}.py"
            if p.exists():
                code += f"\n# {fid}\n" + p.read_text(encoding="utf-8")[:3000]
        ev = dict(task=self.task_text, parent_desc=parent.desc, parent_score=f"{parent.mean:.6g}",
                  cand_desc=node.desc, cand_score=f"{node.mean:.6g}", rel=rel,
                  parent_folds=[round(x, 5) for x in parent.fold_scores],
                  cand_folds=[round(x, 5) for x in node.fold_scores],
                  parent_time=parent.result.get("time_s"), cand_time=node.result.get("time_s"), alt=alt_text,
                  flags=node.flags, code=code[-8000:])
        try:
            v = self._with_timeout(lambda: llm_roles.critic_jump(self.llm, ev), self.cfg.llm_timeout_s)
        except Exception as e:  # noqa: BLE001
            log.warning("Критик не ответил: %s", str(e)[:200])
            return True
        node.critic = v
        if v["verdict"] == "reject":
            node.reason = "Критик: " + v.get("reason", "")[:300]
            return False
        return True

    # ------------------------------------------------------------------ Merge / Tune / Escape
    def _op_merge(self, a, b):
        merged, conflicts = search_operators.merge(a.config, b.config, self.l1.config)
        desc = f"слияние {a.id}+{b.id}" + (f" (конфликты: {conflicts[:3]})" if conflicts else "")
        self._evaluate(merged, a, "merge", desc, a.branch, {"kind": "merge", "patch": {"merge": [a.id, b.id]}})

    def _op_tune(self, node):
        budget = min(900.0, 0.2 * self.timer.left("search"))
        if budget < 6 * 0.45 * self._est_time(node.family):  # меньше ~6 проб Optuna — не имеет смысла
            return
        res = exp_runner.run_job({"kind": "tune", "config": node.config}, self.ctx, budget, f"{node.id}_tune")
        if not res.get("ok"):
            log.info("Tune %s не удался: %s", node.id, str(res.get("error"))[:200])
            return
        cfg = search_operators.apply_patch(node.config, {"model.params": res["best_params"]})
        log.info("Tune %s: %d проб Optuna", node.id, res["n_trials"])
        self._evaluate(cfg, node, "tune", f"Optuna ({res['n_trials']} проб) для {node.id}", node.branch,
                       {"kind": "tune", "patch": {"model.params": res["best_params"]}})

    def _op_escape(self):
        if not self.llm or not self.llm.available:
            return
        reason = self.spec.out_of_library_reason or "поиск вышел на плато"
        hyp = {"kind": "escape", "rationale": f"аварийный оператор: полный пайплайн ({reason})",
               "feature_spec": ("Напиши сильный пайплайн целиком: подготовка признаков (включая текст и даты, если есть), "
                                "модель и предсказания. Учитывай метрику " + self.spec.metric + "."),
               "expected": "работоспособное решение для нестандартной задачи"}
        slot_id = self._slot_id("es")
        deadline = min(self.timer.deadline("search") - 60, time.time() + max(600.0, 0.2 * self.timer.left("search")))
        impl = self._implement("escape", slot_id, hyp, deadline)
        if not impl:
            return
        cfg = search_operators.apply_patch(self.l1.config, {"model.name": "escape"})
        cfg["model"]["escape_id"] = slot_id
        self._evaluate(cfg, self.l1, "escape", hyp["rationale"], self._new_branch(), hyp)

    # ================================================================== финалисты
    def _final_pool(self):
        ok = [n for n in self.pop.nodes.values() if n.mean is not None and n.status in ("accepted", "anchor", "rejected")
              and not n.flags and n.result.get("preds_path")]
        ok.sort(key=lambda n: -task_metrics.gain(n.mean, self.gib))
        seen, pool = set(), []
        for n in ok:
            h = search_operators.config_hash(n.config)
            if h not in seen:
                seen.add(h)
                pool.append(n)
        return pool

    def _reseed(self):
        pool = self._final_pool()
        finalists = pool[: self.cfg.finalists]
        if self.l1 not in finalists:
            finalists.append(self.l1)
        for n in finalists:
            if n.op == "escape":
                continue
            est = self._est_time(n.family) * self.cfg.reseed_seeds
            if self.timer.left("reseed") < est * 0.9:
                continue
            seeds = list(range(1, self.cfg.reseed_seeds + 1))
            res = exp_runner.run_experiment(n.id, n.config, self.ctx, "full", seeds=seeds,
                                            budget_s=min(self.timer.left("reseed"), est * 2))
            if res.get("ok"):
                n.reseed = res
                means = [n.mean] + list(res.get("seed_means", []))
                log.info("[%s] сиды %s: %s", self.timer.stamp(), n.id, [round(m, 6) for m in means])
        self.finalists = finalists

    def _combined(self, node):
        """Предсказания узла с усреднением по сидам (если перепрогон был)."""
        base = self._load_preds(node)
        if node.reseed and node.reseed.get("ok"):
            z = np.load(node.reseed["preds_path"])
            S = len(node.reseed.get("seed_means", [1]))
            out = {}
            for k in ("oof", "ho", "te"):
                out[k] = (base[k] + S * z[k]) / (1 + S)
            return out
        return base

    # ================================================================== финал
    def _final(self):
        if self.l1 is None:
            log.warning("L1 нет — на диске константный сабмит")
            return
        pool = self._final_pool()
        finalists = getattr(self, "finalists", pool[: self.cfg.finalists])
        extra = [n for n in pool if n not in finalists][: max(0, 8 - len(finalists))]
        cands = [n for n in finalists + extra if n.mean is not None]
        if self.l1 not in cands:
            cands.append(self.l1)
        cands.sort(key=lambda n: -task_metrics.gain(n.mean, self.gib))
        # нестабильные по сидам узлы (кроме L1) не участвуют в бленде
        spreads = {n.id: float(np.std([n.mean] + n.reseed.get("seed_means", []))) for n in cands
                   if n.reseed and n.reseed.get("ok")}
        if len(spreads) >= 3:
            med = float(np.median(list(spreads.values())))
            cands = [n for n in cands if n is self.l1 or spreads.get(n.id, 0) <= max(3 * med, 1e-12)]
        excluded = self._critic_final(cands)
        cands = [n for n in cands if n.id not in excluded or n is self.l1]
        preds = {n.id: self._combined(n) for n in cands}
        ys = self.y_search
        oofs = [preds[n.id]["oof"] for n in cands]
        weights = final_ensemble.caruana(oofs, ys, self.spec, self.metric, self.cfg.blend_max_steps) \
            if len(cands) > 1 else np.array([1.0])
        post_cfg = self.l1.config.get("postprocess", {})
        options = []
        l1p = preds[self.l1.id]
        options.append({"name": "L1", "complexity": 0, "nodes": [self.l1], "w": [1.0], "ho": l1p["ho"],
                        "te": l1p["te"], "post": final_ensemble.fit_post(l1p["oof"], ys, self.spec, post_cfg, self.metric)})
        best_single = cands[0] if cands else self.l1
        if best_single is not self.l1:
            bp = preds[best_single.id]
            options.append({"name": f"лучший узел {best_single.id}", "complexity": 1, "nodes": [best_single], "w": [1.0],
                            "ho": bp["ho"], "te": bp["te"],
                            "post": final_ensemble.fit_post(bp["oof"], ys, self.spec,
                                                            best_single.config.get("postprocess", post_cfg), self.metric)})
        used = [i for i, w in enumerate(weights) if w > 0]
        if len(used) >= 2:
            b_oof = final_ensemble.blend(oofs, weights)
            options.append({"name": "бленд " + "+".join(f"{cands[i].id}:{weights[i]:.2f}" for i in used),
                            "complexity": 2, "nodes": [cands[i] for i in used], "w": [weights[i] for i in used],
                            "ho": final_ensemble.blend([preds[n.id]["ho"] for n in cands], weights),
                            "te": final_ensemble.blend([preds[n.id]["te"] for n in cands], weights),
                            "post": final_ensemble.fit_post(b_oof, ys, self.spec, post_cfg, self.metric),
                            "oof_score": task_metrics.score(self.metric, ys, b_oof, self.spec.task_type,
                                                            final_ensemble.fit_post(b_oof, ys, self.spec, post_cfg,
                                                                                    self.metric))})
        winner, report = final_ensemble.select_final(options, self.vault, self.spec, self.metric, self.cfg)
        log.info("[%s] выбор на selection-holdout: %s → %s", self.timer.stamp(),
                 json.dumps(report, ensure_ascii=False), winner["name"])
        post = dict(winner["post"])
        shift = final_ensemble.check_prior_shift(winner["te"], winner["ho"], post, self.vault, self.spec, self.metric)
        if shift.get("apply"):
            post["prior"] = shift["prior"]
            log.info("коррекция сдвига приоров принята: TV=%.3f, выигрыш на пересэмплированном holdout %.4g",
                     shift["tv"], shift["mean_gain"])
        self.writer.write(winner["te"], post, f"финал: {winner['name']}")
        self.summary.update(choice=winner["name"], holdout=report, prior_shift=shift,
                            l1_cv=self.l1.mean, best_cv=(self.pop.best() or self.l1).mean)
        self._refit(winner, post)

    def _critic_final(self, cands) -> set:
        if not self.llm or not self.llm.available or not self.llm.budget_left() or len(cands) < 2:
            return set()
        lines = []
        for n in cands:
            lines.append(f"- {n.id} [{n.family}] CV={n.mean:.6g} folds={[round(x, 5) for x in n.fold_scores]} "
                         f"time={n.result.get('time_s')}s op={n.op}: {n.desc[:150]}; "
                         f"changes vs L1: {search_operators.diff(self.l1.config, n.config)}")
        try:
            out = self._with_timeout(lambda: llm_roles.critic_final(self.llm, self.task_text, "\n".join(lines)),
                                     min(self.cfg.llm_timeout_s, max(30.0, self.timer.left("final") * 0.2)))
        except Exception as e:  # noqa: BLE001
            log.warning("Критик (финал) не ответил: %s", str(e)[:200])
            return set()
        ex = {str(x) for x in out.get("exclude", [])} - {self.l1.id}
        if ex:
            log.info("Критик исключил из финала: %s (%s)", ex, out.get("reason", "")[:200])
        return ex

    def _refit(self, winner, post):
        """Переобучение членов победителя на всех размеченных строках; test = 0.5·фолды + 0.5·refit."""
        members = winner["nodes"]
        est = sum(self._est_time(n.family) / max(1, n.result.get("n_folds", 5)) * 1.3 for n in members)
        if self.timer.left("final") < est * 1.5 + 20:
            log.info("refit пропущен: не хватает времени (нужно ~%.0f с)", est * 1.5)
            return
        y_hold = self.vault.release()
        np.save(self.workdir / "y_holdout.npy", y_hold)
        te_refit = []
        for n in members:
            bi = n.result.get("best_iter")
            n_est = int(bi * 1.1) + 1 if bi and n.family in exp_runner.BOOSTING else None
            job = {"kind": "refit", "config": n.config, "node_id": n.id, "seeds": [0],
                   "y_holdout_path": str(self.workdir / "y_holdout.npy"), "n_estimators": n_est}
            budget = max(30.0, self.timer.left("final") - 10)
            res = exp_runner.run_job(job, self.ctx, budget, f"{n.id}_refit")
            if not res.get("ok"):
                log.info("refit %s не удался: %s", n.id, str(res.get("error"))[:200])
                return
            te_refit.append(np.load(res["preds_path"])["te"])
        te_r = final_ensemble.blend(te_refit, winner["w"])
        te = 0.5 * np.asarray(winner["te"]) + 0.5 * te_r
        if self.writer.write(te, post, f"финал + refit: {winner['name']}"):
            self.summary["refit"] = True

    # ================================================================== итог
    def _save_summary(self):
        try:
            self.summary["submission"] = (self.writer.current if self.writer else None)
            self.summary["llm"] = self.llm.usage_summary() if self.llm else None
            self.summary["journal"] = self.journal.stats() if hasattr(self, "journal") else None
            self.summary["elapsed_s"] = round(self.timer.elapsed(), 1)
            (self.workdir / "final.json").write_text(json.dumps(self.summary, ensure_ascii=False, indent=1,
                                                                default=str), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log.warning("итоговый отчёт не сохранён: %s", e)
