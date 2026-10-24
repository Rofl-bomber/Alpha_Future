"""Оркестратор: четыре этапа, решения keep/discard, лестница сабмитов.

Этапы (границы — доли бюджета от старта solution.py, см. core_config.PHASES; границы анализа и
L1 — потолки, следующий этап начинается сразу, как только закончился предыдущий):
  0. TaskSpec по правилам + один JSON-вызов LLM, константный сабмит;
  1. анализ данных: статистика харнесса → Аналитик в Python-сессии (без ограничений харнесса,
     кроме потолка времени) → разбиение (submit_split) и отчёт; параллельно — страховочный L1;
  2. L1: детерминированный LightGBM на разбиении Аналитика — корень цепочки поиска;
  3. поиск: подэтапы Preprocessing → Features → Model; Исследователь (с инструментами) предлагает
     гипотезы подэтапа, Инженер пишет код, харнесс сравнивает с текущим решением на тех же фолдах;
     повторный вызов Аналитика после Preprocessing, после Features и на плато;
  4. финалисты: сиды, бленд, выбор на selection-holdout, refit (без изменений).
LLM только предлагает; метрику считает харнесс, решение «оставить / откатить» принимает этот модуль.
Поиск жадный: принятая гипотеза сразу становится текущим решением.

Бюджеты действий: в обычном режиме — оценки по времени прошлых прогонов; в режиме AGENT_UNLIMITED
каждое действие может идти до конца своей фазы (_budget, _can_start). Результат, обученный не до
конца из-за дедлайна (truncated), пишется в журнал, но в решениях не участвует.
"""

import concurrent.futures as cf
import copy
import json
import logging
import pickle
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

import core_timer
import eval_acceptance
import eval_protocol
import exp_journal
import exp_kernel
import exp_runner
import final_ensemble
import final_submission
import llm_analyst
import llm_client
import llm_engineer
import llm_researcher
import llm_roles
import search_operators
import search_population
import search_registry
import search_scheduler
import task_metrics
import task_profiler
import task_spec
from core_config import STAGES

log = logging.getLogger("agent")

TIME_FACTOR = {"lgbm": 1.0, "xgb": 1.5, "catboost": 5.0, "extratrees": 1.0, "linear": 0.6, "mlp": 2.0,
               "custom": 2.0, "escape": 3.0}
MIN_START_S = 5.0  # AGENT_UNLIMITED: действие не начинается, если до конца фазы меньше этого


class _SubRegistry:
    """Реестр процессов страховочного L1: виден watchdog (общий реестр), убивается отдельно."""

    def __init__(self, parent):
        self.parent = parent
        self.own = core_timer.ProcessRegistry()

    def add(self, proc):
        self.parent.add(proc)
        self.own.add(proc)

    def remove(self, proc):
        self.parent.remove(proc)
        self.own.remove(proc)

    def kill_all(self):
        self.own.kill_all()


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
        self.write_lock = threading.Lock()
        self.anchor_written = False
        self.llm = None
        self.kernel = None
        self.reg = search_registry.Registry(self.workdir)
        self.summary = {"workdir": str(self.workdir)}
        self.queue = []
        self.tried_sigs = set()
        self.evaluated = {}
        self.codes_desc = {}  # id кода transform → что он делает (из гипотезы)
        self.best_written = None
        self.screen_cache = {}
        self.alt_cache = {}
        self.pool = cf.ThreadPoolExecutor(max_workers=2)  # фоновый Исследователь (устаревший вызов не мешает новому)
        self.recall_pool = cf.ThreadPoolExecutor(max_workers=1)  # фоновый повторный вызов Аналитика
        self.sync_pool = cf.ThreadPoolExecutor(max_workers=2)  # синхронные вызовы с таймаутом
        self.research_future = None
        self.research_stage = None
        self.recall_future = None
        self.recall_source = "Аналитик"
        self.slot_n = 0
        self.branch_n = 0
        self.min_delta = 0.0
        self.l1 = None
        self.hints = {}
        self.stage = None
        self.research_empty = 0
        self.research_fail = 0
        self.lessons = []  # выводы Исследователя из прошлых вызовов; он сам их обновляет
        self.restart_done = set()
        self.retried = set()
        self.near_miss_used = 0
        self.split_source = "profiler"
        self.insurance = None  # {"thread", "result", "ctx"}
        self.dataset_text = "Текущий датасет ещё не построен."
        self.columns_cur = ""
        self.sample_cur = ""  # несколько строк текущих данных — Инженеру
        self.journal = None

    # ================================================================== точка входа
    def run(self) -> dict:
        # (имя этапа, функция, что нужно для запуска: prep — подготовка, l1 — успешный L1)
        steps = [("TaskSpec и данные", self._prepare, None), ("анализ данных", self._analysis, "prep"),
                 ("L1", self._l1, "prep"), ("поиск", self._search, "l1"), ("финалисты", self._reseed, "l1"),
                 ("финал", self._final, "l1")]
        prepared = False
        try:
            for name, fn, need in steps:
                if (need == "prep" and not prepared) or (need == "l1" and self.l1 is None):
                    continue
                try:
                    log.info("[%s] этап: %s", self.timer.stamp(), name)
                    fn()
                    if name == "TaskSpec и данные":
                        prepared = True
                except Exception as e:  # noqa: BLE001
                    log.exception("этап «%s» прерван: %s", name, e)
                    if name == "TaskSpec и данные":
                        if self.writer is None:
                            self._emergency_constant()
                        break
        finally:
            self._stop_insurance()
            if self.kernel is not None:
                self.kernel.close()
            for p in (self.pool, self.recall_pool, self.sync_pool):
                p.shutdown(wait=False, cancel_futures=True)
            self._save_summary()
        return self.summary

    # ================================================================== этап 0: TaskSpec и данные
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
                t_sp = self._budget("spec", max(30.0, min(180.0, self.timer.left("spec"))))
                out = self._with_timeout(lambda: llm_roles.spec_reader(self.llm, text, rules, cols, sample_txt,
                                                                       timeout=t_sp), t_sp)
                before = (spec.task_type, spec.metric, tuple(spec.classes))
                spec = task_spec.apply_analyst(spec, out, train)
                if (spec.task_type, spec.metric, tuple(spec.classes)) != before:
                    log.info("TaskSpec уточнён: %s", spec.short())
                    data = task_spec.prepare_data(spec, train, test)
                    self.writer = final_submission.SubmissionWriter(spec, data.test_ids, len(data.test))
                    self._write_constant()
                if out.get("output_file"):
                    p = (self.task_path.parent / Path(str(out["output_file"])).name)
                    if p.suffix.lower() == ".csv" and p.resolve() != Path(spec.output_path) and \
                            "sample" not in p.name.lower():
                        spec.conflicts.append(f"LLM предлагает файл {p.name}; пишем {spec.output_path}")
                self.spec, self.data = spec, data
            except Exception as e:  # noqa: BLE001
                log.warning("уточнение TaskSpec через LLM не удалось (%s), остаёмся на правилах", str(e)[:200])
        else:
            log.warning("LLM не настроена (нет OPENAI_MODEL/OPENAI_API_KEY): работаем без ролей LLM")
        for c in spec.conflicts:
            log.info("конфликт: %s", c)

        X = data.X_all
        const = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        if const:
            data.X_all = X.drop(columns=const)
        self.raw_train, self.raw_test, self.raw_dict = train, test, dd
        self.raw_valid = task_spec.target_mask(spec, train)
        spec.save(self.ctx.spec_path)
        with open(self.ctx.data_path, "wb") as f:
            pickle.dump({"X_all": data.X_all, "n_train": data.n_train}, f)
        prof = task_profiler.profile_data(spec, data, self.cfg)
        prof["adversarial"] = task_profiler.run_adversarial(spec, prof, self.ctx)
        self.prof = prof
        self.metric = task_metrics.metric_for(spec)
        self.gib = self.metric.greater_is_better
        splits, vault = eval_protocol.make_splits(spec, data, prof, self.cfg)  # запасная схема
        self._set_splits(splits, vault)
        self.stats_text = task_profiler.stats_text(spec, data, prof, train, test)
        (self.workdir / "stats.txt").write_text(self.stats_text, encoding="utf-8")
        adv = prof.get("adversarial") or {}
        if adv.get("auc") is not None:
            log.info("adversarial AUC=%.3f", adv["auc"])
        self.journal = exp_journal.Journal(self.workdir / "journal.jsonl", self.gib)
        self.pop = search_population.Population(self.gib)
        self.scheduler = search_scheduler.Scheduler(self.cfg, self.pop, self.timer, spec)
        self.library_queue = search_operators.library(spec, prof)
        self.engineer = llm_engineer.Engineer(self.llm, self.ctx, spec, self.task_text)
        self.summary.update(spec=spec.short())
        self._start_insurance()

    def _set_splits(self, splits, vault):
        self.splits, self.vault = splits, vault
        splits.save(self.ctx.splits_path)
        self.y_search = self.data.y[splits.search_idx]
        with open(self.ctx.data_path, "wb") as f:
            pickle.dump({"X_all": self.data.X_all, "n_train": self.data.n_train, "y_search": self.y_search}, f)
        self.task_text = (f"{self.spec.title}\n{self.spec.short()}\nВалидация: {splits.describe()}\n"
                          + ("Заметки: " + "; ".join(self.spec.notes[:6]) if self.spec.notes else ""))
        if getattr(self, "engineer", None) is not None:
            self.engineer.task_text = self.task_text
        log.info("валидация: %s", splits.describe())

    def _write_constant(self):
        pred = final_submission.constant_pred(self.spec, self.data.y, len(self.data.test))
        self.writer.write(pred, final_submission.constant_post(self.spec, self.data.y), "константа (уровень 0)")

    def _with_timeout(self, fn, timeout):
        fut = self.sync_pool.submit(fn)
        return fut.result(timeout=timeout)

    def _budget(self, phase: str, limited: float) -> float:
        """Бюджет действия, с: в AGENT_UNLIMITED — до конца фазы, иначе — переданная оценка."""
        if self.cfg.unlimited:
            return max(1.0, self.timer.left(phase))
        return limited

    def _can_start(self, phase: str, need: float) -> bool:
        left = self.timer.left(phase)
        return left >= MIN_START_S if self.cfg.unlimited else left >= need

    def _deadline(self, phase: str, limited: float) -> float:
        return self.timer.deadline(phase) if self.cfg.unlimited else limited

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

    # ------------------------------------------------------------------ страховочный L1
    def _start_insurance(self):
        """L1 на запасной схеме в фоне, пока идёт анализ: сабмит на случай долгого анализа."""
        ins_dir = self.workdir / "insurance"
        ins_dir.mkdir(parents=True, exist_ok=True)
        for name in ("spec.json", "data.pkl", "splits.npz"):
            shutil.copy(self.workdir / name, ins_dir / name)
        cfg = copy.copy(self.cfg)
        cfg.n_jobs = max(1, self.cfg.n_jobs - (self.cfg.kernel_threads if self.llm.available else 0))
        ctx = exp_runner.RunContext(ins_dir, cfg, self.timer, _SubRegistry(self.registry))
        state = {"result": None, "ctx": ctx, "config": search_operators.default_config(self.spec)}

        def work():
            budget = max(60.0, self.timer.left("l1"))
            res = exp_runner.run_experiment("ins_l1", state["config"], ctx, "full", budget_s=budget)
            state["result"] = res
            if res.get("ok") and not res.get("truncated"):
                with self.write_lock:
                    if not self.anchor_written:
                        te = np.load(res["preds_path"])["te"]
                        self.writer.write(te, res.get("post"), "страховочный L1", res["mean"])
                log.info("[%s] страховочный L1: CV %.6g за %.0f с", self.timer.stamp(), res["mean"], res["time_s"])
            else:
                log.info("страховочный L1 не удался: %s", str(res.get("error"))[:200])

        t = threading.Thread(target=work, daemon=True, name="insurance-l1")
        state["thread"] = t
        self.insurance = state
        t.start()

    def _stop_insurance(self):
        ins = self.insurance
        if ins and ins["thread"].is_alive():
            ins["ctx"].registry.kill_all()
            ins["thread"].join(timeout=10)

    # ================================================================== этап 1: анализ данных
    def _kernel_conf(self) -> dict:
        s = self.spec
        return {"paths": {"train": s.train_path, "test": s.test_path, "dict": s.dict_path, "sample": s.sample_path},
                "targets": list(s.targets), "id_col": s.id_col, "task_type": s.task_type, "n_classes": s.n_classes,
                "workdir": str(self.workdir), "holdout_min": self.cfg.holdout_min_frac,
                "holdout_max": self.cfg.holdout_max_frac, "seed": self.cfg.seed,
                "step_timeout": self.cfg.analysis_step_timeout_s, "max_out": self.cfg.analysis_max_out,
                "threads": self.cfg.kernel_threads}

    def _analysis(self):
        if not self.llm.available:
            log.info("анализ данных пропущен: LLM недоступна — запасная схема валидации профайлера")
            return
        try:
            self.kernel = exp_kernel.Kernel(self.ctx, self._kernel_conf(), log)
            self.kernel.start()
        except Exception as e:  # noqa: BLE001
            log.warning("Python-сессия Аналитика не запустилась: %s — анализ пропущен", str(e)[:300])
            self.kernel = None
            return
        files = [f"train: {self.raw_train.shape[0]} строк × {self.raw_train.shape[1]} колонок ({Path(self.spec.train_path).name})",
                 f"test: {self.raw_test.shape[0]} × {self.raw_test.shape[1]} ({Path(self.spec.test_path).name})"]
        if self.spec.dict_path:
            files.append(f"data_dictionary: {Path(self.spec.dict_path).name}")
        if self.spec.sample_path:
            files.append(f"sample_submission: {Path(self.spec.sample_path).name}")
        spec_text = (f"{self.spec.short()}; таргет {self.spec.targets}; id {self.spec.id_col}; "
                     f"классы {list(self.spec.classes)[:20]}")
        deadline = self._deadline("analysis", self.timer.deadline("analysis"))
        t0 = time.time()
        report = llm_analyst.run_analysis(self.llm, self.kernel, self.cfg, self.workdir, self.task_md_text, spec_text,
                                          "\n".join(files), self.stats_text, deadline)
        if report is None:
            log.warning("Аналитик не сдал отчёт — дальше работаем по статистике харнесса")
        else:
            short = llm_analyst.short_report(self.llm, self.cfg, report["part_a"])
            self.reg.load_report(report, short)
            (self.workdir / "analyst_short.md").write_text(short, encoding="utf-8")
            log.info("[%s] Аналитик: отчёт за %.0f с, шагов %d; гипотез %d, отклонённых %d, ловушек %d",
                     self.timer.stamp(), time.time() - t0, report["steps"], len(report["json"].get("hypotheses", [])),
                     len(report["json"].get("rejected", [])), len(report["json"].get("traps", [])))
        res = eval_protocol.splits_from_analyst(self.spec, self.data, self.raw_valid, self.workdir, self.cfg)
        if res is None:
            log.info("разбиение Аналитика не получено — остаётся запасная схема профайлера")
            return
        self.split_source = "analyst"
        self._set_splits(*res)
        self.summary.update(validation=self.splits.describe())

    # ================================================================== этап 2: L1
    def _l1(self):
        ins = self.insurance
        if ins and ins["thread"].is_alive():
            if self.split_source == "profiler":
                ins["thread"].join(timeout=max(1.0, self.timer.left("l1")))
            else:
                self._stop_insurance()
        base = search_operators.default_config(self.spec)
        if self.split_source == "profiler" and ins and ins.get("result") and ins["result"].get("ok") \
                and not ins["result"].get("truncated"):
            node = self._new_node(ins["config"], None, "anchor", "L1: LightGBM, надёжные дефолты (страховочный)", "b0")
            node.result = ins["result"]
            self._set_anchor(node)
            return
        tries = [("L1: LightGBM, надёжные дефолты", base)]
        fast = search_operators.apply_patch(base, {"model.params": {"learning_rate": 0.15, "n_estimators": 600}})
        tries.append(("L1: LightGBM, быстрый", fast))
        tries.append(("L1: ExtraTrees", search_operators.apply_patch(base, {"model.name": "extratrees"})))
        for desc, cfg in tries:
            budget = self._budget("search", max(60.0, self.timer.left("reseed") * 0.5))
            node = self._new_node(cfg, None, "anchor", desc, branch="b0")
            res = exp_runner.run_experiment(node.id, cfg, self.ctx, "full", budget_s=budget)
            node.result = res
            if res.get("ok") and res.get("truncated"):
                node.status, node.reason = "truncated", f"обучение остановлено по дедлайну: {res.get('stop_reason')}"
                self._journal(node, "truncated", mode="full")
                log.warning("L1 (%s) обрезан по времени — пробую следующий вариант", desc)
                continue
            if res.get("ok"):
                self._set_anchor(node)
                return
            node.status = "failed"
            self._journal(node, "failed", mode="full", error=res.get("error"))
            log.warning("L1 (%s) не удался: %s", desc, str(res.get("error"))[:500])
        if ins and ins.get("result") and ins["result"].get("ok") and not ins["result"].get("truncated"):
            log.warning("якорный L1 не получился — запасная схема и страховочный L1 как якорь")
            splits, vault = eval_protocol.make_splits(self.spec, self.data, self.prof, self.cfg)
            self.split_source = "profiler"
            self._set_splits(splits, vault)
            node = self._new_node(ins["config"], None, "anchor", "L1 (страховочный)", "b0")
            node.result = ins["result"]
            self._set_anchor(node)

    def _set_anchor(self, node):
        res = node.result
        self.l1 = node
        self.pop.set_anchor(node)
        self.evaluated[search_operators.config_hash(node.config)] = node.id
        self.min_delta = eval_acceptance.noise_floor(res["fold_scores"], self.cfg)
        self._journal(node, "anchor", mode="full")
        with self.write_lock:
            self.anchor_written = True
            self._maybe_write(node, force=True)
        nums = set(self.spec.num_cols)
        self.hints = {"top_num": [f for f, _ in res.get("importance", []) if f in nums][:10]}
        self.ctx.hints_path.write_text(json.dumps(self.hints, ensure_ascii=False), encoding="utf-8")
        log.info("[%s] L1 %s: CV %.6g ± %.3g за %.0f с", self.timer.stamp(), node.id, res["mean"], res["std"],
                 res["time_s"])
        self._refresh_current(node)

    # ------------------------------------------------------------------ текущие данные решения (D3)
    def _refresh_current(self, node):
        """Текущие данные решения → статистика D3, данные в сессию, колонки и строки Инженеру.

        cur_train — цепочка на строках поиска; cur_test — цепочка на всех строках train (AGENT_CUR_TEST_FULL),
        как при финальном переобучении, иначе тоже на строках поиска."""
        out_dir = self.workdir / "current"
        job = {"kind": "materialize", "config": node.config, "out_dir": str(out_dir)}
        y_path = None
        if self.cfg.cur_test_full and len(self.splits.holdout_idx):
            # метки holdout — только дочернему процессу, файл удаляется сразу после построения
            y_path = self.ctx.p("jobs", f"{node.id}_materialize_yh.npy")
            np.save(y_path, self.vault.labels_for_features())
            job["y_holdout_path"] = str(y_path)
        try:
            res = exp_runner.run_job(job, self.ctx,
                                     timeout=self._budget("search", max(60.0, min(600.0, 0.1 * self.timer.left("search")))),
                                     tag=f"{node.id}_materialize")
        finally:
            if y_path is not None:
                try:
                    y_path.unlink()
                except OSError:
                    pass
        if not res.get("ok"):
            log.warning("текущие данные решения не построены: %s", str(res.get("error"))[:300])
            return
        try:
            tr = pd.read_pickle(res["train_path"])
            te = pd.read_pickle(res["test_path"])
            self.dataset_text = task_profiler.current_stats(tr, te, list(self.spec.feature_cols),
                                                            node.result.get("importance"))
            self.columns_cur = llm_engineer.columns_text(tr, self.spec)
            self.sample_cur = llm_engineer.sample_text(tr)
            (self.workdir / "dataset_current.txt").write_text(self.dataset_text, encoding="utf-8")
            if self.kernel is not None:
                pos = np.where(self.raw_valid)[0]
                np.save(out_dir / "y_search.npy", np.asarray(self.y_search))
                np.save(out_dir / "search_rows.npy", pos[self.splits.search_idx])
                np.save(out_dir / "holdout_rows.npy", pos[self.splits.holdout_idx])
                items = {"cur_train": res["train_path"], "cur_test": res["test_path"],
                         "y_search": str(out_dir / "y_search.npy"), "search_rows": str(out_dir / "search_rows.npy"),
                         "holdout_rows": str(out_dir / "holdout_rows.npy")}
                if node.result.get("preds_path"):
                    np.save(out_dir / "oof.npy", np.load(node.result["preds_path"])["oof"])
                    items["oof"] = str(out_dir / "oof.npy")
                self.kernel.load(items)
        except Exception as e:  # noqa: BLE001
            log.warning("текущие данные решения не загружены: %s", str(e)[:300])

    # ================================================================== узлы и журнал
    def _new_node(self, config, parent, op, desc, branch=None, hyp=None):
        node = search_population.Node(id=self.pop.new_id(), config=config, parent=parent.id if parent else None,
                                      op=op, desc=desc, branch=branch or (parent.branch if parent else "b0"),
                                      hypothesis=hyp, stage=self.stage or "")
        self.pop.register(node)
        return node

    def _new_branch(self):
        self.branch_n += 1
        return f"b{self.branch_n}"

    def _incumbent(self):
        """Текущее решение: L1 плюс все принятые изменения. Все гипотезы применяются к нему."""
        return self.pop.incumbent() or self.l1

    def _journal(self, node, status, mode="full", error=None, delta=None, extra=None):
        res = node.result if mode == "full" else (node.screen if mode == "screen" else node.alt)
        hyp = node.hypothesis or {}
        rec = {"node": node.id, "parent": node.parent, "op": node.op, "stage": node.stage, "ref": hyp.get("ref"),
               "desc": node.desc, "status": status, "mode": mode, "family": node.family, "branch": node.branch,
               "change": node.change or None, "patch": hyp.get("patch"), "kind": hyp.get("kind"),
               "source": hyp.get("source"), "flags": node.flags, "reason": node.reason,
               "error": (error or (res or {}).get("error") or "")[:1500] or None, "delta": delta}
        if res and res.get("ok"):
            rec.update(fold_scores=res["fold_scores"], mean=res["mean"], std=res["std"], time_s=res["time_s"],
                       best_iter=res.get("best_iter"), n_features=res.get("n_features"))
        if res:
            rec.update(truncated=bool(res.get("truncated")), folds_done=res.get("folds_done"),
                       stop_reason=res.get("stop_reason"), budget_s=res.get("budget_s"))
            if not res.get("ok") and res.get("wall_s") and (res.get("timed_out") or "исчерпан" in str(res.get("error"))):
                rec["time_s"] = res["wall_s"]  # прогон не уложился в бюджет — оценка времени семейства растёт
        chk = hyp.get("code_check")
        if chk:
            rec["code_check"] = {"verdict": chk.get("verdict"), "issues": chk.get("issues"), "fixed": chk.get("fixed"),
                                 "summary": (chk.get("summary") or "")[:4000]}
        if extra:
            rec.update(extra)
        self.journal.add(rec)
        if hyp.get("ref") and status not in ("anchor",):
            self.reg.on_result(hyp.get("ref"), node.id, status, delta, node.stage)

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

    @staticmethod
    def _chain_len(config) -> int:
        return len(config.get("features", {}).get("chain", []) or [])

    def _exp(self, node_id, config, mode, budget, cache_from=None, seeds=(0,)):
        """Эксперимент с кэшем принятой части цепочки transform (cache_from — конфиг, чья цепочка принята)."""
        upto = self._chain_len(cache_from if cache_from is not None else config)
        extra = {"cache_upto": upto, "cache_max_gb": self.cfg.cache_max_gb} if upto else None
        return exp_runner.run_experiment(node_id, config, self.ctx, mode, seeds=seeds, budget_s=budget, extra=extra)

    # ================================================================== этап 3: поиск
    def _search(self):
        for i, _ in enumerate(STAGES):
            if not self._can_start("search", 30.0):
                break
            self.stage = self.scheduler.enter(i)
            self.queue = [h for h in self.queue if h.get("stage") == self.stage]
            self.research_empty = 0
            if self.research_future is not None and self.research_stage != self.stage:
                self.research_future = None  # ответ для прошлого подэтапа не ждём: он будет отброшен
            log.info("[%s] подэтап поиска: %s (до %.0f мин от старта)", self.timer.stamp(), self.stage,
                     (self.scheduler.stage_end - self.timer.t0) / 60)
            self._stage_loop()
            self._collect_recall(wait=0.0)
            if self.stage != "model" and self._can_start("search", 120.0):
                self._recall(blocking=True, reason=f"завершён подэтап {self.stage}; дальше — {STAGES[i + 1]}")
        log.info("[%s] поиск завершён: %s", self.timer.stamp(), self.journal.stats())

    def _stage_loop(self):
        while True:
            best = self._incumbent()
            if not self._can_start("search", max(30.0, 0.6 * self._est_time(best.family))):
                return
            action = self.scheduler.next()
            try:
                if action.kind == "advance":
                    log.info("[%s] подэтап %s завершён: %s", self.timer.stamp(), self.stage, action.payload.get("why"))
                    return
                if action.kind == "recall":
                    self._recall(blocking=False, reason=f"{action.payload.get('why')} в подэтапе {self.stage}")
                    continue
                if action.kind == "escape":
                    self._op_escape()
                    continue
                if action.kind == "tune":
                    self._op_tune(action.payload["node"])
                    continue
                hyp = self._next_hypothesis()
                if hyp is None:
                    if self.stage == "model":
                        restart = [o for o in search_operators.restart_options(self.spec, self.pop.families_tried())
                                   if self._restart_key(o[1]) not in self.restart_done]
                        best = self._incumbent()
                        if restart:
                            self._op_restart(*restart[0])
                            continue
                        if best.id not in self.scheduler.tuned:
                            self.scheduler.tuned.add(best.id)
                            self._op_tune(best)
                            continue
                    log.info("[%s] гипотезы подэтапа %s закончились", self.timer.stamp(), self.stage)
                    return
                self._evaluate_hypothesis(hyp)
                self._collect_recall(wait=0.0)
            except Exception as e:  # noqa: BLE001
                log.exception("ошибка в действии %s: %s", action.kind, e)
                self.scheduler.feedback(False)

    # ------------------------------------------------------------------ очередь гипотез
    def _sig(self, hyp, base=None) -> str:
        """Подпись гипотезы с учётом решения, к которому её применяют: та же идея на новом решении — новая."""
        base = base or self._incumbent().id
        return json.dumps([base, hyp.get("stage") or self.stage, hyp.get("kind"), hyp.get("patch"),
                           (hyp.get("spec") or "")[:300]], sort_keys=True, ensure_ascii=False, default=str)

    @staticmethod
    def _restart_key(patch) -> str:
        return json.dumps(patch, sort_keys=True, ensure_ascii=False, default=str)

    def _solution_text(self) -> str:
        """D1: текущее решение и история принятых изменений."""
        inc, l1 = self._incumbent(), self.l1
        lines = [f"Узел {inc.id}: CV {self.spec.metric} = {inc.mean:.6g} ± {np.std(inc.fold_scores):.3g} "
                 f"(L1 {l1.id}: {l1.mean:.6g}; {'больше' if self.gib else 'меньше'} — лучше)."]
        chain = self.pop.lineage()
        if chain:
            lines.append("Принятые изменения по порядку (подэтап; выигрыш к предыдущему решению; фолды с улучшением):")
            for i, n in enumerate(chain, 1):
                c = n.cmp or {}
                gain = (f"выигрыш {c['mean_gain']:+.4g} ({c.get('wins')}/{c.get('k')})"
                        if isinstance(c.get("mean_gain"), (int, float)) else "выигрыш не измерен")
                lines.append(f"  {i}. {n.id} [{n.stage}]: {n.change} → {gain}" + (f" (проверка {n.ref})" if n.ref else ""))
                if n.desc and n.op not in ("tune", "restart"):
                    lines.append(f"     идея: {' '.join(n.desc.split())[:160]}")
        else:
            lines.append("Принятых изменений пока нет: текущее решение — это L1.")
        lines.append("Что сейчас в пайплайне:")
        lines.append(search_operators.state_text(inc.config, self.codes_desc, inc.result.get("n_features")))
        return "\n".join(lines)

    def _lessons_text(self) -> str:
        if not self.lessons:
            return "Пока нет — это первый вызов."
        return "\n".join(f"- {x}" for x in self.lessons)

    def _short_text(self) -> str:
        if self.reg.short:
            return self.reg.short
        return "Отчёта Аналитика нет. Статистика харнесса по сырым данным:\n" + self.stats_text[:6000]

    def _research_context(self, parent):
        return dict(task=self.task_text, analyst_json=self.reg.json_text(), short=self._short_text(),
                    registry=self.reg.text_open(), solution=self._solution_text(), dataset=self.dataset_text,
                    journal=self.journal.summary(self.stage), lessons=self._lessons_text(),
                    stage_left=self.scheduler.stage_left() / 60, search_left=max(0.0, self.timer.left("search")) / 60,
                    exp_time=self._est_time(parent.family))

    def _research_usable(self) -> bool:
        return bool(self.llm and self.llm.available and self.llm.budget_left("researcher") and self.research_fail < 3)

    def _start_research(self):
        if self.research_future is not None or not self._research_usable():
            return
        parent = self._incumbent()
        ctx = self._research_context(parent)
        stage = self.stage
        schema = search_operators.schema_text(self.spec, stage, self.hints)
        timeout = max(60.0, min(self.scheduler.stage_left(), self.timer.left("search")) - 10)
        contract = self.engineer._contract("transform", len(self.data.X_all))  # тот же контракт, что видит Инженер

        def call():
            return llm_researcher.run(self.llm, self.kernel, self.cfg, stage, schema, ctx, self.reg.full_report,
                                      timeout=timeout, log_path=self.workdir / "researcher_log.jsonl",
                                      contract=contract)

        self.research_stage = stage
        self.research_future = self.pool.submit(call)

    def _collect_research(self, wait: float = 0.0):
        fut = self.research_future
        if fut is None:
            return
        try:
            out = fut.result(timeout=wait) if wait > 0 else (fut.result() if fut.done() else None)
        except cf.TimeoutError:
            return
        except Exception as e:  # noqa: BLE001
            log.warning("Исследователь не ответил: %s", str(e)[:300])
            self.research_future = None
            self.research_fail += 1
            return
        if out is None:
            return
        self.research_future = None
        self.research_fail = 0
        if out.get("stage") != self.stage:
            log.info("ответ Исследователя для подэтапа %s пришёл после перехода — отброшен", out.get("stage"))
            return
        lessons = [" ".join(str(x).split())[:240] for x in out.get("lessons") or []]
        if lessons:
            self.lessons = lessons[:8]
            (self.workdir / "lessons.txt").write_text("\n".join(self.lessons), encoding="utf-8")
            log.info("[%s] уроки Исследователя: %s", self.timer.stamp(), " | ".join(self.lessons)[:600])
        self.reg.mark_obsolete(out.get("obsolete") or [])
        added = 0
        for h in out.get("hypotheses", []):
            if h.get("stage") != self.stage:
                log.info("гипотеза другого подэтапа (%s) отброшена: %s", h.get("stage"), h.get("rationale", "")[:120])
                continue
            h["source"] = "researcher"
            if self._sig(h) not in self.tried_sigs:
                self.queue.append(h)
                added += 1
        log.info("[%s] новых гипотез от Исследователя (%s, инструментов %d): %d", self.timer.stamp(), self.stage,
                 out.get("tools", 0), added)
        self.research_empty = 0 if added else self.research_empty + 1

    def _pop_queue(self):
        while self.queue:
            h = self.queue.pop(0)
            if h.get("stage", self.stage) != self.stage:
                continue
            sig = self._sig(h)
            if sig in self.tried_sigs:
                continue
            self.tried_sigs.add(sig)
            return h
        return None

    def _next_hypothesis(self):
        """Очередь подэтапа (Исследователь, повторы «почти прошедших»); библиотека — запасной путь.

        Если Исследователь ничего нового не дал, его спрашивают ещё раз; после двух пустых ответов
        подряд (или если LLM недоступна) берётся библиотека подэтапа; иначе подэтап заканчивается.
        """
        while True:
            self._collect_research()
            if len(self.queue) < 2 and self.research_empty < 2:
                self._start_research()
            h = self._pop_queue()
            if h is not None:
                return h
            if self.research_future is None:
                break
            limited = min(self.scheduler.stage_left() - 20, self.timer.left("search") - 30)
            if limited <= 1 and not self.cfg.unlimited:
                break
            self._collect_research(wait=max(1.0, self._budget("search", limited)))
            h = self._pop_queue()
            if h is not None:
                return h
            if self.research_future is not None:  # не дождались
                break
        for cand in [c for c in self.library_queue if c.get("stage") == self.stage]:
            self.library_queue.remove(cand)
            sig = self._sig(cand)
            if sig not in self.tried_sigs:
                self.tried_sigs.add(sig)
                log.info("гипотеза из библиотеки подэтапа %s: %s", self.stage, cand.get("rationale"))
                return dict(cand)
        return None

    # ------------------------------------------------------------------ повторный вызов Аналитика
    def _recall_context(self):
        return dict(task=self.task_text, analyst_json=self.reg.json_text(), short=self._short_text(),
                    registry=self.reg.text_open(), solution=self._solution_text(), dataset=self.dataset_text,
                    journal=self.journal.summary(self.stage))

    def _recall(self, blocking: bool, reason: str):
        if self.kernel is None or not self.llm or not self.llm.available or not self.reg.available \
                or not self.llm.budget_left("analyst_delta") or self.recall_future is not None:
            return
        ctx = self._recall_context()
        deadline = min(time.time() + self.cfg.recall_minutes * 60, self.timer.deadline("search") - 60)
        if deadline - time.time() < 60:
            return
        stage = self.stage

        def call():
            return llm_analyst.run_delta(self.llm, self.kernel, self.cfg, self.workdir, ctx, deadline, reason)

        log.info("[%s] повторный вызов Аналитика (%s): %s", self.timer.stamp(), "ждём" if blocking else "в фоне", reason)
        self.recall_future = self.recall_pool.submit(call)
        self.recall_source = f"Аналитик ({stage}: {reason[:80]})"
        if blocking:
            self._collect_recall(wait=max(1.0, deadline - time.time() + 30))

    def _collect_recall(self, wait: float = 0.0):
        fut = self.recall_future
        if fut is None:
            return
        try:
            delta = fut.result(timeout=wait) if wait > 0 else (fut.result() if fut.done() else "pending")
        except cf.TimeoutError:
            return
        except Exception as e:  # noqa: BLE001
            log.warning("повторный вызов Аналитика не удался: %s", str(e)[:300])
            self.recall_future = None
            return
        if isinstance(delta, str) and delta == "pending":
            return
        self.recall_future = None
        if not delta:
            log.info("повторный вызов Аналитика: дельты нет")
            return
        n = self.reg.apply_delta(delta, self.recall_source)
        log.info("[%s] дельта Аналитика применена: изменений %d (снято %d, добавлено гипотез %d)", self.timer.stamp(), n,
                 len(delta.get("retire") or []), len((delta.get("add") or {}).get("hypotheses") or []))

    # ------------------------------------------------------------------ Improve / Restart
    def _evaluate_hypothesis(self, hyp):
        """Гипотеза применяется к текущему решению: все принятые изменения остаются на месте."""
        kind = hyp.get("kind", "config")
        parent = self._incumbent()
        branch = self._new_branch() if kind == "restart" else parent.branch
        desc = (hyp.get("rationale") or kind)[:200]
        if kind == "fit_predict":
            if self.stage != "model":
                return
            self._op_escape(hyp)
            return
        patch, notes = search_operators.validate_patch(hyp.get("patch") or {}, parent.config, self.spec,
                                                       known_custom=set(self.codes_desc), hints=self.hints,
                                                       allow_params=hyp.get("source") not in ("researcher",),
                                                       stage=self.stage if kind != "restart" else None)
        if notes:
            log.info("патч %s: %s", hyp.get("source"), "; ".join(notes)[:300])
        if kind in ("code", "custom_model"):
            slot = "transform" if kind == "code" else "model"
            if slot == "model" and self.stage != "model":
                return
            slot_id = self._slot_id("tf" if slot == "transform" else "mc")
            deadline = self._deadline("search", min(self.timer.deadline("search") - 30,
                                                    time.time() + max(300.0, 0.15 * self.timer.left("search"))))
            impl = self._implement(slot, slot_id, hyp, deadline)
            if not impl:
                return
            if impl.get("check"):
                hyp = dict(hyp, code_check=impl["check"])
            if slot == "transform":
                self.codes_desc[slot_id] = (hyp.get("spec") or desc)[:200]
                patch = dict(patch, **{"features.chain": "+" + slot_id})
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
        """Код Инженера для гипотезы. Неудача ничего не выбрасывает: в журнал пишется «не реализовано» с причиной,
        подпись гипотезы снимается, чтобы Исследователь мог предложить её снова (как есть или уточнив), и неудача
        не считается отклонением для плато подэтапа."""
        if not self.llm or not self.llm.available:
            return None
        try:
            res = self.engineer.implement(slot, slot_id, hyp, self._incumbent().config, self.columns_cur,
                                          self.reg.analyst_notes(hyp.get("ref")), deadline, len(self.data.X_all),
                                          context=self._engineer_context(slot))
        except llm_client.LLMError as e:  # implement сам ловит ошибки LLM; это страховка
            res = {"ok": False, "error": f"вызов LLM не удался: {e}"}
        if not res.get("ok"):
            parent = self._incumbent()
            self.tried_sigs.discard(self._sig(hyp))
            node = self._new_node(parent.config, parent, "improve", (hyp.get("rationale") or "")[:200], None, hyp)
            node.status = "failed"
            node.reason = (f"не реализовано Инженером ({res.get('attempts', 0)} попыток): "
                           + str(res.get("error"))[:300])
            node.change = f"{slot}: {(hyp.get('spec') or hyp.get('rationale') or '')[:120]}"
            self._journal(node, "failed", mode="none", error=res.get("error"))
            log.info("Инженер не реализовал %s: %s", slot_id, str(res.get("error"))[:200])
            return None
        log.info("Инженер реализовал %s с попытки %d", slot_id, res.get("attempts", 1))
        return res

    def _engineer_context(self, slot) -> dict:
        """Что Инженеру нужно знать, чтобы не гадать: как формируются кадры, код принятых transform, строки данных."""
        if slot == "escape":
            return {"folds": self._folds_text(), "chain": "не используется: fit_predict получает сырые данные",
                    "sample": "нет: fit_predict получает сырые данные"}
        return {"folds": self._folds_text(), "chain": self._chain_code_text(), "sample": self.sample_cur}

    def _folds_text(self) -> str:
        sp = self.splits
        kind = sp.scheme.split(":")[-1]
        lines = [f"Scheme: {sp.describe()}. Test: {len(self.data.X_all) - self.data.n_train} rows.",
                 "In every experiment, for each fold k: train_df = the search rows of all other folds, y_train = their "
                 "target; others = [the rows of fold k (validation), all holdout rows, all test rows]."]
        if kind == "ordered" and sp.group_col:
            lines.append(f"Folds are periods of {sp.group_col} (a period or a run of consecutive periods per fold): "
                         "train_df contains periods both before and after the validation period.")
        elif kind == "group" and sp.group_col:
            lines.append(f"Folds are groups by {sp.group_col}: a group is never split between train_df and the "
                         "validation part.")
        lines.append("For the final model transform is called once more with train_df = all train rows (search + "
                     "holdout) and others = [test rows].")
        return "\n".join(lines)

    def _chain_code_text(self, limit: int = 16000) -> str:
        """Код принятых transform по порядку; если не помещается — старые сокращаются первыми."""
        chain = list(self._incumbent().config.get("features", {}).get("chain", []) or [])
        if not chain:
            return "нет: это первый transform"
        parts = []
        for tid in chain:
            try:
                code = (self.workdir / "codes" / f"{tid}.py").read_text(encoding="utf-8")
            except OSError:
                code = "(код не найден)"
            parts.append([tid, (self.codes_desc.get(tid) or "")[:200], code])
        total = sum(len(c) for _, _, c in parts)
        for p in parts[:-1]:
            if total <= limit:
                break
            cut = p[2][:600] + "\n# ... (сокращено)"
            total -= len(p[2]) - len(cut)
            p[2] = cut
        return "\n\n".join(f"## {tid}: {desc}\n```python\n{code}\n```" for tid, desc, code in parts)

    # ------------------------------------------------------------------ оценка узла
    def _screen_of(self, node):
        if node.id not in self.screen_cache:
            self.screen_cache[node.id] = self._exp(node.id, node.config, "screen",
                                                   self._budget("search", max(60.0, self._est_time(node.family))))
        return self.screen_cache[node.id]

    def _alt_of(self, node):
        if node.id not in self.alt_cache:
            self.alt_cache[node.id] = self._exp(node.id, node.config, "alt",
                                                self._budget("search", max(60.0, self._est_time(node.family))))
        return self.alt_cache[node.id]

    def _cmp(self, a, b):
        return eval_acceptance.paired_compare(a, b, self.gib, self.min_delta, self.cfg.accept_min_fold_frac,
                                              self.cfg.noise_se_mult)

    def _record_truncated(self, node, parent, mode, res, ref_scores):
        done = res.get("folds_done") or []
        ref = [ref_scores[k] for k in done if k < len(ref_scores)] if ref_scores else []
        cmp = self._cmp(res["fold_scores"], ref) if ref and len(ref) == len(res["fold_scores"]) else None
        node.status = "truncated"
        node.reason = f"обучение остановлено по дедлайну: {res.get('stop_reason')}"
        extra = {"wins": cmp["wins"], "se": cmp["se"], "joined": False} if cmp else None
        self._journal(node, "truncated", mode=mode, delta=cmp["mean_gain"] if cmp else None, extra=extra)
        log.info("[%s] %s %s [%s, %s]: ОБРЕЗАН по времени — CV %.6g; в решениях не участвует", self.timer.stamp(),
                 node.id, node.op, node.family, mode, res["mean"])
        self.scheduler.feedback(False, counts_for_plateau=False)

    def _evaluate(self, config, parent, op, desc, branch, hyp=None):
        h = search_operators.config_hash(config)
        if h in self.evaluated:
            log.info("конфигурация уже проверялась (%s) — пропуск", self.evaluated[h])
            return None
        fam = search_operators.family(config)
        est = self._est_time(fam)
        if not self._can_start("search", 1.1 * est):
            log.info("на эксперимент %s не хватает времени поиска (%.0f с из ~%.0f с)", fam,
                     self.timer.left("search"), est)
            return None
        node = self._new_node(config, parent, op, desc, branch, hyp)
        self.evaluated[h] = node.id
        if op == "escape":
            node.change = "аварийный оператор: полный пайплайн Инженера (fit_predict) вместо пайплайна харнесса"
        elif op == "tune":
            node.change = f"{desc}: гиперпараметры {fam}"
        else:
            node.change = search_operators.describe_change(parent.config, config, self.codes_desc)
        # скрининг: дешёвая оценка на первых фолдах, в полную оценку проходят не явно худшие
        screen_room = self.cfg.unlimited or self.timer.left("search") > 1.6 * est
        if est > self.cfg.screen_if_full_s and parent.fold_scores and screen_room:
            ps = self._screen_of(parent)
            sb = self._budget("search", min(max(60.0, est), self.timer.left("search") - est))
            node.screen = self._exp(node.id, config, "screen", sb, cache_from=parent.config)
            if node.screen.get("ok") and node.screen.get("truncated"):
                ref = ps.get("fold_scores") if ps.get("ok") and not ps.get("truncated") else None
                self._record_truncated(node, parent, "screen", node.screen, ref)
                return node
            if node.screen.get("ok") and ps.get("ok") and not ps.get("truncated"):
                c = self._cmp(node.screen["fold_scores"], ps["fold_scores"])
                if c["mean_gain"] < -c["se"]:
                    node.status, node.reason = "screen_rejected", f"скрининг Δ={c['mean_gain']:+.4g}"
                    self._journal(node, "screen_rejected", mode="screen", delta=c["mean_gain"])
                    self.scheduler.feedback(False)
                    return node
        if not self._can_start("search", 1.05 * est):
            node.status, node.reason = "skipped", "после скрининга не осталось времени на полную оценку"
            self._journal(node, "skipped", mode="screen")
            return node
        left = self.timer.left("search")
        budget = self._budget("search", min(left - 5, max(3 * est, 120.0)))
        node.result = self._exp(node.id, config, "full", budget, cache_from=parent.config)
        if node.result.get("ok") and node.result.get("truncated"):
            self._record_truncated(node, parent, "full", node.result, parent.fold_scores)
            return node
        if not node.result.get("ok"):
            node.status = "failed"
            self._journal(node, "failed", mode="full")
            self.scheduler.feedback(False)
            log.info("[%s] %s %s: ошибка — %s", self.timer.stamp(), node.id, op,
                     str(node.result.get("error")).splitlines()[0][:300] if node.result.get("error") else "")
            return node
        cmp = self._cmp(node.result["fold_scores"], parent.fold_scores)
        node.cmp = cmp
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
        if not accepted and node.status != "flagged":
            node.flags = []
        joined = bool(accepted and op != "escape")
        if joined:
            self.pop.accept(node)
        self._journal(node, node.status, mode="full", delta=cmp["mean_gain"],
                      extra={"wins": cmp["wins"], "se": cmp["se"], "joined": joined})
        log.info("[%s] %s %s [%s/%s] на %s: %s — CV %.6g (Δ=%+.4g ± %.2g, %d/%d фолдов) — %s%s", self.timer.stamp(),
                 node.id, op, self.stage, fam, parent.id, node.change[:200], node.mean, cmp["mean_gain"], cmp["se"],
                 cmp["wins"], cmp["k"], node.status, " → текущее решение" if joined else "")
        if accepted:
            node.flags = [f for f in node.flags if not f.startswith("jump") and f != "fast_train"]
            self._maybe_write(node)
        if joined:
            self.research_empty = 0  # новое текущее решение — Исследователя снова есть о чём спросить
            self._refresh_current(node)
            self._requeue_near_misses(node)
        self.scheduler.feedback(accepted)
        return node

    def _requeue_near_misses(self, new_best):
        """После принятия: повторить на новом решении «почти прошедшие» гипотезы этого подэтапа."""
        left = self.cfg.near_miss_total - self.near_miss_used
        if left <= 0:
            return
        cands = []
        for n in self.pop.nodes.values():
            c = n.cmp or {}
            if n.status != "rejected" or n.id in self.retried or n.op in ("tune", "escape") or not c \
                    or n.stage != self.stage:
                continue
            patch = (n.hypothesis or {}).get("patch") or {}
            if not patch:
                continue
            gain, se = c.get("mean_gain", 0.0), c.get("se", 0.0)
            if gain > 0 and gain >= 0.5 * se and 2 * c.get("wins", 0) >= c.get("k", 1):
                cands.append(n)
        cands.sort(key=lambda n: -n.cmp["mean_gain"] / max(n.cmp["se"], 1e-12))
        for n in cands[: min(left, self.cfg.near_miss_per_accept)]:
            self.retried.add(n.id)
            self.near_miss_used += 1
            hyp = n.hypothesis or {}
            self.queue.insert(0, {"kind": "config", "stage": self.stage, "ref": hyp.get("ref"),
                                  "block": hyp.get("block", "features"), "patch": dict(hyp["patch"]),
                                  "rationale": f"повтор на новом решении {new_best.id}: {n.change[:120]} "
                                               f"(на {n.parent} было Δ={n.cmp['mean_gain']:+.4g}, "
                                               f"{n.cmp['wins']}/{n.cmp['k']} фолдов)",
                                  "cost": "low", "source": "near_miss"})
            log.info("почти прошедшая гипотеза %s поставлена в очередь на %s", n.id, new_best.id)

    def _recheck_jump(self, node, parent, rel) -> bool:
        if self.splits.fold_array("alt") is None:
            node.flags.append("no_alt_folds")
            return self._critic_ok(node, parent, rel, "альтернативного разбиения нет — перепроверка невозможна")
        if not self._can_start("search", self._est_time(node.family)):
            node.reason = "не хватило времени перепроверить скачок"
            return False
        node.alt = self._exp(node.id, node.config, "alt", self._budget("search", max(60.0, self._est_time(node.family))),
                             cache_from=parent.config)
        pa = self._alt_of(parent)
        if node.alt.get("truncated") or pa.get("truncated"):
            node.reason = "перепроверка скачка обрезана по времени — без неё скачок не принимается"
            return False
        if not (node.alt.get("ok") and pa.get("ok")):
            node.reason = "перепроверка скачка не выполнена"
            return False
        c = self._cmp(node.alt["fold_scores"], pa["fold_scores"])
        if c["mean_gain"] <= max(self.min_delta, 0.0):
            node.reason = f"скачок {rel:+.1%} не повторился на других фолдах (Δ={c['mean_gain']:+.4g})"
            return False
        return self._critic_ok(node, parent, rel, f"скачок повторился на других фолдах: Δ={c['mean_gain']:+.4g}")

    def _critic_ok(self, node, parent, rel, alt_text) -> bool:
        if not self.llm or not self.llm.available or not self.cfg.critic_enabled:
            return True
        code = json.dumps((node.hypothesis or {}).get("patch"), ensure_ascii=False)
        cid = node.config.get("model", {}).get("custom_id")
        ids = ([cid] if cid else []) + list(node.config.get("features", {}).get("chain", []))
        for fid in ids:
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
            t_cr = self._budget("search", self.cfg.llm_timeout_s)
            v = self._with_timeout(lambda: llm_roles.critic_jump(self.llm, ev, timeout=t_cr), t_cr)
        except Exception as e:  # noqa: BLE001
            log.warning("Критик не ответил: %s", str(e)[:200])
            return True
        if v["verdict"] == "reject":
            node.reason = "Критик: " + v.get("reason", "")[:300]
            return False
        return True

    # ------------------------------------------------------------------ Restart / Tune / Escape
    def _op_restart(self, desc, patch):
        """Другое семейство моделей на текущем решении (подэтап Model): данные решения остаются."""
        self.restart_done.add(self._restart_key(patch))
        self._evaluate_hypothesis({"kind": "restart", "stage": self.stage, "block": "model", "patch": patch,
                                   "rationale": desc, "source": "scheduler"})

    def _op_tune(self, node):
        budget = self._budget("search", min(900.0, 0.5 * self.scheduler.stage_left()))
        if not self.cfg.unlimited and budget < 6 * 0.45 * self._est_time(node.family):
            return
        if not self._can_start("search", 0.0):
            return
        upto = self._chain_len(node.config)
        job = {"kind": "tune", "config": node.config, "n_trials": self.cfg.tune_trials, "cache_upto": upto,
               "cache_max_gb": self.cfg.cache_max_gb}
        res = exp_runner.run_job(job, self.ctx, budget, f"{node.id}_tune")
        if not res.get("ok"):
            log.info("Tune %s не удался: %s", node.id, str(res.get("error"))[:200])
            return
        cfg = search_operators.apply_patch(node.config, {"model.params": res["best_params"]})
        log.info("Tune %s: %d проб Optuna", node.id, res["n_trials"])
        self._evaluate(cfg, node, "tune", f"Optuna ({res['n_trials']} проб) для {node.id}", node.branch,
                       {"kind": "tune", "stage": self.stage, "patch": {"model.params": res["best_params"]}})

    def _op_escape(self, hyp=None):
        if not self.llm or not self.llm.available:
            return
        reason = self.spec.out_of_library_reason or "предложено Исследователем"
        hyp = dict(hyp or {"kind": "fit_predict", "rationale": f"аварийный оператор: полный пайплайн ({reason})",
                           "spec": ("Напиши сильный пайплайн целиком: подготовка признаков (включая текст и даты, если "
                                    "есть), модель и предсказания. Учитывай метрику " + self.spec.metric + "."),
                           "expected": "работоспособное решение для нестандартной задачи"})
        hyp.setdefault("stage", self.stage)
        slot_id = self._slot_id("es")
        deadline = self._deadline("search", min(self.timer.deadline("search") - 60,
                                                time.time() + max(600.0, 0.2 * self.timer.left("search"))))
        impl = self._implement("escape", slot_id, hyp, deadline)
        if not impl:
            return
        cfg = search_operators.apply_patch(self.l1.config, {"model.name": "escape"})
        cfg["model"]["escape_id"] = slot_id
        # сравнивается с текущим решением, но цепочку не продолжает: это отдельный пайплайн для бленда
        self._evaluate(cfg, self._incumbent(), "escape", hyp.get("rationale", "fit_predict")[:200], self._new_branch(),
                       hyp)

    # ================================================================== этап 4: финалисты
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
        self.stage = None
        pool = self._final_pool()
        finalists = pool[: self.cfg.finalists]
        if self.l1 not in finalists:
            finalists.append(self.l1)
        for n in finalists:
            if n.op == "escape":
                continue
            est = self._est_time(n.family) * self.cfg.reseed_seeds
            if not self._can_start("reseed", est * 0.9):
                continue
            seeds = list(range(1, self.cfg.reseed_seeds + 1))
            res = self._exp(n.id, n.config, "full", self._budget("reseed", min(self.timer.left("reseed"), est * 2)),
                            seeds=seeds)
            if res.get("ok") and res.get("truncated"):
                log.info("[%s] сиды %s обрезаны по времени — проверка стабильности не засчитана", self.timer.stamp(), n.id)
            elif res.get("ok"):
                n.reseed = res
                means = [n.mean] + list(res.get("seed_means", []))
                log.info("[%s] сиды %s: %s", self.timer.stamp(), n.id, [round(m, 6) for m in means])
            else:
                log.info("[%s] сиды %s не удались: %s", self.timer.stamp(), n.id,
                         str(res.get("error")).splitlines()[0][:200] if res.get("error") else "")
        self.finalists = finalists

    def _combined(self, node):
        """Предсказания узла с усреднением по сидам (если перепрогон был)."""
        base = self._load_preds(node)
        if node.reseed and node.reseed.get("ok"):
            z = np.load(node.reseed["preds_path"])
            S = len(node.reseed.get("seed_means", [1]))
            return {k: (base[k] + S * z[k]) / (1 + S) for k in ("oof", "ho", "te")}
        return base

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
        spreads = {n.id: float(np.std([n.mean] + n.reseed.get("seed_means", []))) for n in cands
                   if n.reseed and n.reseed.get("ok")}
        if len(spreads) >= 3:
            med = float(np.median(list(spreads.values())))
            cands = [n for n in cands if n is self.l1 or spreads.get(n.id, 0) <= max(3 * med, 1e-12)]
        excluded = self._critic_final(cands)
        cands = [n for n in cands if n.id not in excluded or n is self.l1]
        preds = {n.id: self._combined(n) for n in cands}
        ys = self.y_search
        # строки, оценённые всеми кандидатами (в screen-режиме OOF не используется — тут только full)
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
            b_post = final_ensemble.fit_post(b_oof, ys, self.spec, post_cfg, self.metric)
            options.append({"name": "бленд " + "+".join(f"{cands[i].id}:{weights[i]:.2f}" for i in used),
                            "complexity": 2, "nodes": [cands[i] for i in used], "w": [weights[i] for i in used],
                            "ho": final_ensemble.blend([preds[n.id]["ho"] for n in cands], weights),
                            "te": final_ensemble.blend([preds[n.id]["te"] for n in cands], weights),
                            "post": b_post,
                            "oof_score": task_metrics.score(self.metric, ys, b_oof, self.spec.task_type, b_post)})
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
                            l1_cv=self.l1.mean, best_cv=self._incumbent().mean,
                            accepted_chain=[f"{n.id} [{n.stage}]: {n.change}" for n in self.pop.lineage()])
        self._refit(winner, post)

    def _critic_final(self, cands) -> set:
        if not self.llm or not self.llm.available or not self.cfg.critic_enabled or len(cands) < 2:
            return set()
        lines = []
        for n in cands:
            lines.append(f"- {n.id} [{n.family}] CV={n.mean:.6g} folds={[round(x, 5) for x in n.fold_scores]} "
                         f"time={n.result.get('time_s')}s op={n.op}: {n.desc[:150]}; "
                         f"changes vs L1: {search_operators.diff(self.l1.config, n.config)}")
        try:
            t_cr = self._budget("final", min(self.cfg.llm_timeout_s, max(30.0, self.timer.left("final") * 0.2)))
            out = self._with_timeout(lambda: llm_roles.critic_final(self.llm, self.task_text, "\n".join(lines),
                                                                    timeout=t_cr), t_cr)
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
        if not self._can_start("final", est * 1.5 + 20):
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
            budget = self._budget("final", max(30.0, self.timer.left("final") - 10))
            res = exp_runner.run_job(job, self.ctx, budget, f"{n.id}_refit")
            if not res.get("ok"):
                log.info("refit %s не удался: %s", n.id, str(res.get("error"))[:200])
                return
            if res.get("truncated"):
                log.info("refit %s обрезан по времени — не используется, остаётся сабмит по фолдам", n.id)
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
            self.summary["journal"] = self.journal.stats() if self.journal is not None else None
            self.summary["split_source"] = self.split_source
            self.summary["registry"] = {k: v["status"] for k, v in self.reg.items.items()}
            self.summary["elapsed_s"] = round(self.timer.elapsed(), 1)
            (self.workdir / "final.json").write_text(json.dumps(self.summary, ensure_ascii=False, indent=1,
                                                                default=str), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log.warning("итоговый отчёт не сохранён: %s", e)
