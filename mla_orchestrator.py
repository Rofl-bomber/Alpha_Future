"""End-to-end orchestration with time budgeting and anytime submissions."""
from __future__ import annotations

import concurrent.futures as cf
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import mla_metrics as MT
import mla_prompts as PR
from mla_config import Config, Deadline
from mla_cv import CVPlan, adversarial_validation, choose_cv, make_folds
from mla_features import FeatureState
from mla_io_data import load_task, parse_dictionary
from mla_modeling import ModelSearch
from mla_models import CVResult, ModelSpec, Problem, run_cv
from mla_postprocess import Decision, fit_decision
from mla_preprocess import build_base_features
from mla_profiling import dataset_overview_text, profile_dataset, target_summary
from mla_research import ProxyEvaluator, ResearchLoop
from mla_sandbox import Sandbox
from mla_submission import SubmissionWriter, emergency_submission
from mla_task_spec import TaskSpec, build_classes, heuristic_spec, reconcile_with_llm
from mla_utils import LOG, fmt_num, format_exc, save_json, timer, truncate


class Orchestrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        B = cfg.time_budget
        self.dl = Deadline(B, cfg.start_time, reserve=min(max(30.0, 0.02 * B), 300.0))
        self.report: dict = {"budget_s": B, "n_jobs": cfg.n_jobs, "memory_gb": round(cfg.memory_gb, 1), "phases": {}, "errors": []}
        self.llm = None
        self.pool = cf.ThreadPoolExecutor(max_workers=2)
        self.current_preds: dict = {}  # target -> (test_pred, decision, source, utility)
        self.writer: SubmissionWriter | None = None
        self.spec: TaskSpec | None = None

    # ==========================================================================================
    def run(self):
        try:
            self._run()
        except Exception as e:
            LOG.error(f"FATAL in orchestrator: {type(e).__name__}: {e}")
            LOG.error(format_exc(60))
            self.report["errors"].append({"fatal": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-4000:]})
        finally:
            self._ensure_submission()
            self.report["elapsed_s"] = round(self.dl.elapsed(), 1)
            if self.llm is not None:
                try:
                    self.report["llm_usage"] = self.llm.usage()
                except Exception:
                    pass
            save_json(self.cfg.work_dir / "agent_report.json", self.report)
            try:
                self._write_markdown_report()
            except Exception as e:
                LOG.debug(f"markdown report failed: {e}")
            try:
                self.pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            LOG.info(f"Finished in {self.dl.elapsed():.0f}s (budget {self.cfg.time_budget:.0f}s)")

    def _phase(self, name: str, **info):
        self.report["phases"][name] = {"t": round(self.dl.elapsed(), 1), **info}

    def _write_markdown_report(self):
        r = self.report
        L = ["# Agent run report", "", f"- elapsed: {r.get('elapsed_s')} s of {r.get('budget_s'):.0f} s; CPUs {r.get('n_jobs')}"]
        if self.spec is not None:
            L.append(f"- task: {self.spec.summary()}; id = `{self.spec.id_col}`")
        if r.get("cv"):
            L.append(f"- validation: {r['cv']}")
        ph = r.get("phases", {})
        if "baseline" in ph:
            L.append(f"- baseline (proxy LightGBM) score: {ph['baseline'].get('score')}")
        res = r.get("research")
        if res:
            L += ["", "## Feature research", f"proxy score {res.get('base_score')} → {res.get('final_score')}; "
                  f"researcher calls: {res.get('researcher_calls')}", "", "| id | type | title | status | Δ |", "|---|---|---|---|---|"]
            for e in res.get("experiments", []):
                d = "" if e.get("delta") is None else f"{e['delta']:+.5g}"
                L.append(f"| {e['id']} | {e['type']} | {truncate(e['title'], 80)} | {e['status']} | {d} |")
            if res.get("insights"):
                L += ["", "Lessons:"] + [f"- {i}" for i in res["insights"]]
        for tname, m in (r.get("models") or {}).items():
            L += ["", f"## Models — {tname}", "| model | OOF score | seconds |", "|---|---|---|"]
            for x in m.get("results", []):
                L.append(f"| {x['name']} | {x['score']:.6g} | {x['elapsed']} |")
            if m.get("best"):
                L.append(f"\nFinal: {m['best']['method']} → {m['best']['score']:.6g}; members {m['best']['members']}")
        if r.get("llm_usage"):
            u = r["llm_usage"]
            L += ["", f"LLM: {u.get('calls')} calls, {u.get('prompt_tokens')} prompt / {u.get('completion_tokens')} completion tokens"]
        if r.get("errors"):
            L += ["", "Errors:"] + [f"- {e.get('fatal')}" for e in r["errors"]]
        (self.cfg.work_dir / "agent_report.md").write_text("\n".join(L), encoding="utf-8")

    # ==========================================================================================
    def _run(self):
        cfg = self.cfg
        LOG.info(f"Task dir {cfg.task_dir}; budget {cfg.time_budget:.0f}s; cpus {cfg.n_jobs}; mem {cfg.memory_gb:.1f}GB; "
                 f"llm {'mock' if cfg.llm_mock else (cfg.model or 'auto') if cfg.llm_enabled else 'disabled'}")
        # ---------------- A. data
        with timer("load data"):
            raw = load_task(cfg.task_md, cfg.task_dir)
        self.raw = raw
        ddict = parse_dictionary(raw.ddict)
        self.ddict = ddict
        spec = heuristic_spec(raw, ddict)
        self.spec = spec
        LOG.info(f"Heuristic spec: {spec.summary()} | id={spec.id_col} | notes={spec.notes}")
        target_names = [t.name for t in spec.targets]
        feat_cols = [c for c in raw.train.columns if c in raw.test.columns and c != spec.id_col and c not in target_names]
        prof_cols = feat_cols + [c for c in target_names if c in raw.train.columns]
        with timer("profile"):
            self.profile = profile_dataset(raw.train, raw.test, feat_cols)
        # writer early so that we always have a valid file
        self.writer = SubmissionWriter(raw, spec, cfg.task_dir / "submission.csv")
        try:
            self.writer.write(self.writer.fallback())
            LOG.info("Wrote prior-based fallback submission")
        except Exception as e:
            LOG.warning(f"fallback submission failed: {e}")

        # ---------------- B. LLM bootstrap (health + analyst) in background
        analyst_future = None
        if cfg.llm_enabled:
            analyst_future = self.pool.submit(self._llm_bootstrap, raw, spec, feat_cols)

        # ---------------- C. base features + baseline
        self._prepare(spec, feat_cols, llm_cv=None)
        base_eval = self._baseline()
        self._phase("baseline", score=base_eval.score if base_eval else None)

        # adversarial validation while the analyst thinks
        self.adv = None
        try:
            if self.dl.remaining() > 120:
                self.adv = adversarial_validation(self.state.X, self.state.n_train, cfg.n_jobs)
                LOG.info(f"Adversarial validation AUC={self.adv['auc']:.4f}; top: {self.adv['top_features'][:5]}")
        except Exception as e:
            LOG.warning(f"adversarial validation failed: {e}")

        # analyst: block only when the task definition is genuinely ambiguous
        llm_obj = None
        self.late_analyst = None
        if analyst_future is not None:
            confident = self._spec_confident(spec)
            if confident and not analyst_future.done():
                LOG.info("Task definition is unambiguous; analyst insights will be merged into the research loop when ready")
                self.late_analyst = analyst_future
            else:
                wait_until = self.dl.start + max(0.08 * cfg.time_budget, 240.0)
                wait = max(5.0, min(wait_until - time.time(), self.dl.remaining() * 0.25))
                try:
                    llm_obj = analyst_future.result(timeout=wait)
                except cf.TimeoutError:
                    LOG.warning("Analyst did not answer in time; continuing with the heuristic task spec")
                    self.late_analyst = analyst_future
                except Exception as e:
                    LOG.warning(f"LLM bootstrap failed: {e}")
        if llm_obj:
            self.report["analyst"] = llm_obj
            old = json.dumps(self._spec_key(spec))
            spec = reconcile_with_llm(spec, llm_obj, raw)
            self.spec = spec
            LOG.info(f"Reconciled spec: {spec.summary()} | notes={spec.notes[-4:]}")
            changed = json.dumps(self._spec_key(spec)) != old
            cv_changed = self._cv_would_change(spec)
            if changed or cv_changed:
                LOG.info(f"Spec/CV changed after analysis (spec={changed}, cv={cv_changed}); rebuilding baseline")
                self.writer = SubmissionWriter(raw, spec, cfg.task_dir / "submission.csv")
                self._prepare(spec, feat_cols, llm_cv=spec.cv)
                base_eval = self._baseline()
        self.report["spec"] = spec.to_dict()
        self.report["cv"] = self.cv_plan.describe()

        # ---------------- D. feature research (primary target)
        primary = self.problems[0]
        llm_possible = cfg.llm_enabled and (self.llm is None or getattr(self.llm, "available", None) is not False)
        fe_frac = 0.40 if llm_possible else 0.22
        fe_deadline = min(self.dl.start + fe_frac * cfg.time_budget, self.dl.end - 0.45 * max(0.0, self.dl.remaining()))
        final_state = self.state
        # the endpoint may have been down only at start-up: retry once in the background
        if (cfg.llm_enabled and not cfg.llm_mock and self.llm is not None and getattr(self.llm, "available", None) is False
                and getattr(self, "late_analyst", None) is None and fe_deadline - time.time() > 300):
            def _retry_llm(raw=raw, spec=spec, feat_cols=feat_cols):
                self.llm.available = None
                self.llm.consecutive_failures = 0
                if self.llm.health_check(attempts=2):
                    return self._analyst(raw, spec, feat_cols)
                return None
            self.late_analyst = self.pool.submit(_retry_llm)
        if base_eval is not None and fe_deadline - time.time() > 2.5 * self.evaluator.est_time():
            final_state = self._research(primary, base_eval, fe_deadline)
        self._phase("research_done", n_features=final_state.X.shape[1], te=len(final_state.te_specs))

        # ---------------- E/F. models + ensemble per target
        self._model_phase(final_state, base_eval)
        self._phase("models_done")

    # ==========================================================================================
    def _spec_confident(self, spec: TaskSpec) -> bool:
        """True when the heuristic definition is certain enough that the analyst cannot change modelling."""
        from mla_task_spec import POS_PAIRS
        if len(spec.targets) != 1 or self.raw.sample is None:
            return False
        if not (spec.metric_source == "text" and spec.metric_text_priority <= 0):
            return False
        t = spec.targets[0]
        if t.task_type == "binary":
            low = {str(c).lower() for c in t.classes}
            return any({a.lower(), b.lower()} == low for a, b in POS_PAIRS)
        return True

    def _late_analyst(self, obj: dict) -> dict:
        """Merge a late analyst answer: hints only (the modelling set-up is already fixed)."""
        self.report["analyst"] = obj
        ideas =[str(i)[:400] for i in (obj.get("feature_ideas") or [])[:15]] if isinstance(obj, dict) else []
        hints = []
        for key, label in (("categorical_columns", "categorical_cast"), ("drop_columns", "drop_features")):
            cols = []
            for item in (obj.get(key) or []):
                name = item.get("column") if isinstance(item, dict) else item
                if isinstance(name, str) and name in self.ff.safe_of:
                    cols.append(self.ff.safe_of[name])
            if cols:
                hints.append(f"{label} of {cols} (suggested by the analysis)")
        return {"ideas": hints + ideas, "domain": str(obj.get("domain_summary", ""))[:3000]}

    def _spec_key(self, spec: TaskSpec):
        return {"id": spec.id_col, "metric": spec.metric,
                "targets": [(t.name, t.task_type, t.pred_format, t.positive, tuple(t.classes or [])) for t in spec.targets]}

    def _cv_would_change(self, spec: TaskSpec) -> bool:
        t = spec.targets[0]
        plan = choose_cv(t.task_type, self.raw.train, self.raw.test, self.profile, spec.cv, {spec.id_col, *[x.name for x in spec.targets]})
        return plan.kind != self.cv_plan.kind or plan.group_col != self.cv_plan.group_col

    def _encode_target(self, t) -> tuple[np.ndarray, np.ndarray]:
        """Returns (y_internal for labelled rows, mask of labelled rows)."""
        raw = self.raw
        if t.task_type == "regression":
            y = pd.to_numeric(raw.train[t.name], errors="coerce").values.astype(float)
            mask = np.isfinite(y)
            return y, mask
        classes, tokens, pos, canon = build_classes(raw, t.name, t.positive)
        mapping = {c: i for i, c in enumerate(t.classes)}
        y = canon.map(mapping)
        mask = y.notna().values
        return y.fillna(-1).values.astype(int), mask

    def _prepare(self, spec: TaskSpec, feat_cols: list, llm_cv):
        cfg = self.cfg
        raw = self.raw
        t0 = spec.targets[0]
        # labelled rows of the primary target define the training set
        y0, mask = self._encode_target(t0)
        train = raw.train.loc[mask].reset_index(drop=True) if not mask.all() else raw.train
        if not mask.all():
            LOG.warning(f"Dropping {int((~mask).sum())} train rows with missing target")
        self.train_mask = mask
        reserved = {c.lower() for c in ("te_", "cnt_")}
        ff = build_base_features(train, raw.test, feat_cols, self.profile, cat_hints=[], ddict_info=self.ddict, reserved_names=None)
        self.ff = ff
        self.state = FeatureState(ff.X, ff.n_train, list(ff.cat_cols))
        # descriptions per safe column (for prompts)
        self.desc_of = {}
        for safe, orig in ff.orig_of.items():
            d = self.ddict.get(orig, {})
            desc = d.get("description", "")
            if d.get("unit"):
                desc = f"{desc} [{d['unit']}]"
            if safe != orig:
                desc = f"(orig '{orig}') {desc}"
            self.desc_of[safe] = desc.strip()
        # problems
        self.problems = []
        for t in spec.targets:
            y, m = self._encode_target(t)
            y = y[mask] if not mask.all() else y
            if t.task_type != "regression":
                y = np.where(y < 0, 0, y)
            pb = Problem(t.name, t.task_type, t.n_classes, y, spec.metric,
                         (float(np.nanmin(y)), float(np.nanmax(y))) if t.task_type == "regression" else None,
                         nonneg=bool(t.task_type == "regression" and np.nanmin(y) >= 0))
            if t.task_type == "regression" and not np.all(np.isfinite(pb.y)):
                med = float(np.nanmedian(pb.y))
                pb.y = np.where(np.isfinite(pb.y), pb.y, med)
            self.problems.append(pb)
        # CV
        exclude = {spec.id_col, *[t.name for t in spec.targets]}
        tr_for_cv = train
        self.cv_plan = choose_cv(t0.task_type, tr_for_cv, raw.test, self.profile, llm_cv, exclude)
        LOG.info(f"CV plan: {self.cv_plan.describe()}")
        self.proxy_folds = make_folds(self.cv_plan, self.problems[0].y, t0.task_type, cfg.seed)
        self.final_folds = make_folds(self.cv_plan, self.problems[0].y, t0.task_type, cfg.seed + 1000)

    # ------------------------------------------------------------------------------------------
    def _proxy_params(self, n: int, lr: float) -> dict:
        p = {"learning_rate": lr, "num_leaves": 63 if n > 20000 else (31 if n > 3000 else 15),
             "min_child_samples": 20 if n > 3000 else 10, "feature_fraction": 0.8, "bagging_fraction": 0.8,
             "bagging_freq": 1, "lambda_l2": 1.0, "deterministic": True}
        return p

    def _baseline(self):
        cfg = self.cfg
        pb = self.problems[0]
        n = self.state.n_train
        params = self._proxy_params(n, 0.1)
        fold_ids = list(range(len(self.proxy_folds)))
        self.evaluator = ProxyEvaluator(pb, self.proxy_folds, fold_ids, params, cfg.n_jobs, f"proxy{cfg.seed}", seed=cfg.seed)
        with timer("baseline LightGBM CV"):
            ev = self.evaluator.evaluate(self.state, deadline=min(self.dl.end, self.dl.start + max(90.0, 0.3 * cfg.time_budget)))
        if ev is None:
            LOG.warning("baseline evaluation failed or ran out of time; fitting a quick full-data model instead")
            for pb_ in self.problems:
                self._quick_model(pb_)
            return None
        LOG.info(f"Baseline {pb.metric} = {fmt_num(ev.score, 6)} (folds {np.round(ev.utils, 5)}) in {ev.elapsed:.1f}s")
        self._push_pred(pb, ev.test, ev.oof, "baseline", self.proxy_folds)
        # other targets (multi-target tasks): baseline predictions so the file is complete
        for pb2 in self.problems[1:]:
            ev2 = ProxyEvaluator(pb2, self.proxy_folds, fold_ids, params, cfg.n_jobs, f"proxy{cfg.seed}", seed=cfg.seed).evaluate(self.state, deadline=self.dl.end)
            if ev2 is not None:
                self._push_pred(pb2, ev2.test, ev2.oof, "baseline", self.proxy_folds)
        self._calibrate_proxy(ev)
        return ev

    def _quick_model(self, pb: Problem):
        """Emergency predictor: one LightGBM on all training rows, no CV (seconds even on large data)."""
        try:
            import lightgbm as lgb
            from mla_models import as_proba, lgb_objective, model_view, cat_columns_in_view
            V = model_view(self.state, "lgbm")
            cats = cat_columns_in_view(self.state, "lgbm")
            n = self.state.n_train
            params = {"learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20, "feature_fraction": 0.8,
                      "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "n_jobs": self.cfg.n_jobs, "seed": self.cfg.seed}
            params.update(lgb_objective(pb, "none", {}))
            rounds = 300 if n < 200_000 else 150
            m = lgb.train(params, lgb.Dataset(V.iloc[:n], pb.y, categorical_feature=cats or "auto"), num_boost_round=rounds)
            pred = m.predict(V.iloc[n:])
            if pb.task != "regression":
                pred = as_proba(pred, pb.task)
            self.current_preds[pb.name] = (pred, Decision(), "quick_model")
            self._write_current()
            LOG.info(f"quick model for {pb.name} written")
        except Exception as e:
            LOG.warning(f"quick model failed: {e}")

    def _calibrate_proxy(self, ev):
        """Pick proxy learning rate / folds so that one experiment costs a sensible share of the FE budget."""
        cfg = self.cfg
        n_f = len(self.proxy_folds)
        per_fold = ev.elapsed / max(1, n_f)
        fe_budget = max(60.0, 0.38 * cfg.time_budget)
        target = float(np.clip(fe_budget / 30.0, 15.0, 150.0))
        lr, folds = 0.1, list(range(n_f))
        if per_fold * n_f * 1.6 < target:
            lr = 0.06
        elif per_fold * n_f > target:
            if per_fold * min(3, n_f) <= target:
                folds = list(range(min(3, n_f)))
            else:
                folds = list(range(min(3, n_f)))
                lr = 0.15
        self.proxy_per_fold = per_fold * (0.1 / lr)
        self.proxy_lr = lr
        if lr != 0.1 or len(folds) != n_f:
            self.evaluator.params["learning_rate"] = lr
            self.evaluator.fold_ids = folds
            self.evaluator.times = [per_fold * len(folds) * (0.1 / lr)]
            LOG.info(f"Proxy calibrated: lr={lr}, folds={folds} (~{self.evaluator.est_time():.0f}s per experiment)")
            # re-evaluate the base state under the calibrated proxy for fair comparisons
            ev2 = self.evaluator.evaluate(self.state, deadline=self.dl.end)
            if ev2 is not None:
                ev.__dict__.update(ev2.__dict__)
        self.proxy_fold_time = per_fold

    # ------------------------------------------------------------------------------------------
    def _push_pred(self, pb: Problem, test_pred, oof, source: str, folds, decision: Decision | None = None, force: bool = True):
        if test_pred is None:
            return
        if decision is None:
            decision = Decision()
            if oof is not None:
                o = np.asarray(oof)
                ok = np.isfinite(o) if o.ndim == 1 else np.isfinite(o).all(1)
                if ok.sum() >= 30:
                    rng = (float(np.min(pb.y)), float(np.max(pb.y))) if pb.task == "regression" else None
                    decision = fit_decision(pb.metric, pb.task, pb.y[ok], o[ok], pb.n_classes, y_range=rng, nonneg=pb.nonneg, seed=self.cfg.seed)
        self.current_preds[pb.name] = (test_pred, decision, source)
        self._write_current()

    def _write_current(self):
        if self.writer is None:
            return
        if not all(t.name in self.current_preds for t in self.spec.targets):
            return
        try:
            preds = {k: v[0] for k, v in self.current_preds.items()}
            decs = {k: v[1] for k, v in self.current_preds.items()}
            df = self.writer.build(preds, decs)
            if self.writer.write(df):
                srcs = {k: v[2] for k, v in self.current_preds.items()}
                LOG.info(f"submission.csv updated ({srcs})")
                self.report["last_submission_source"] = srcs
        except Exception as e:
            LOG.warning(f"writing submission failed: {e}")
            LOG.debug(format_exc())

    def _ensure_submission(self):
        out = self.cfg.task_dir / "submission.csv"
        ok = out.exists() and out.stat().st_size > 0
        if ok and self.writer is not None:
            try:
                df = pd.read_csv(out, dtype=str, keep_default_na=False)
                if self.writer.sample is not None and (list(df.columns) != list(self.writer.sample.columns) or len(df) != len(self.writer.sample)):
                    ok = False
            except Exception:
                ok = False
        if not ok:
            LOG.warning("No valid submission found at the end; writing fallback")
            try:
                if self.writer is not None:
                    self.writer.write(self.writer.fallback())
                    return
            except Exception as e:
                LOG.error(f"fallback failed: {e}")
            emergency_submission(self.cfg.task_dir)

    # ==========================================================================================
    def _llm_bootstrap(self, raw, spec, feat_cols):
        cfg = self.cfg
        if cfg.llm_mock:
            from mla_mock_llm import MockLLM
            self.llm = MockLLM(cfg, self.dl, latency=0.5)
        else:
            from mla_llm import LLMClient
            self.llm = LLMClient(cfg, self.dl, log_path=cfg.work_dir / "llm_calls.jsonl")
            if not self.llm.health_check():
                return None
        return self._analyst(raw, spec, feat_cols)

    def _analyst(self, raw, spec, feat_cols):
        t0 = spec.targets[0]
        tr, te = raw.train, raw.test
        only_tr = [c for c in tr.columns if c not in te.columns]
        only_te = [c for c in te.columns if c not in tr.columns]
        files = (f"- train.csv: {len(tr)} rows x {tr.shape[1]} columns\n- test.csv: {len(te)} rows x {te.shape[1]} columns\n"
                 f"- columns only in train (target candidates): {only_tr}\n- columns only in test: {only_te}\n"
                 f"- sample_submission.csv columns: {list(raw.sample.columns) if raw.sample is not None else 'not provided'}\n"
                 f"- other files: {raw.extra_files}")
        sample_head = raw.sample.head(5).to_csv(index=False) if raw.sample is not None else "(none)"
        dd_lines = []
        for c, d in list(self.ddict.items())[:400]:
            dd_lines.append(f"{c} | {d.get('role', '')} | {d.get('dtype', '')} | {d.get('unit', '')} | {truncate(d.get('description', ''), 160)}")
        ddict_text = truncate("\n".join(dd_lines), 9000, tail=500)
        prof = dict(self.profile)
        profile_text = dataset_overview_text(prof, {}, max_chars=9000, desc_len=0)
        tgt_lines = []
        for c in only_tr[:5]:
            try:
                kind = "regression" if (pd.api.types.is_numeric_dtype(tr[c]) and tr[c].nunique() > 30) else "classification"
                tgt_lines.append(f"- {c}: {target_summary(tr[c], kind)}")
            except Exception:
                pass
        heur = {"target_columns": [t.name for t in spec.targets], "id_column": spec.id_col, "task_type": t0.task_type,
                "metric": spec.metric, "positive_label": t0.positive, "prediction_format": t0.pred_format,
                "classes": (t0.classes[:30] if t0.classes else None)}
        msgs = [{"role": "system", "content": PR.ANALYST_SYSTEM},
                {"role": "user", "content": PR.analyst_user(raw.task_text, files, sample_head, ddict_text, profile_text, "\n".join(tgt_lines), heur)}]
        t = time.time()
        obj, r = self.llm.chat_json(msgs, purpose="analyst", temperature=0.6, max_tokens=9000)
        LOG.info(f"Analyst answered in {time.time() - t:.0f}s: metric={obj.get('metric')} type={obj.get('task_type')} "
                 f"targets={obj.get('target_columns')} cv={obj.get('cv_strategy')}")
        return obj

    # ==========================================================================================
    def _research(self, pb: Problem, base_eval, fe_deadline: float) -> FeatureState:
        cfg = self.cfg
        spec = self.spec
        t0 = spec.targets[0]
        m = MT.get(spec.metric)
        n_tr, n_te = self.state.n_train, len(self.state.X) - self.state.n_train
        class_names = [str(x) for x in (t0.out_tokens or [])]
        tsum = target_summary(self.raw.train[t0.name], "regression" if t0.task_type == "regression" else "classification")
        task_line = f"predict '{t0.name}' ({t0.task_type}); metric {spec.metric} ({'higher' if m.greater_is_better else 'lower'} is better); {n_tr} train rows, {n_te} test rows"
        task_block = (f"Predict `{t0.name}` — {t0.task_type}. Metric: {spec.metric} — {m.title} ({'higher' if m.greater_is_better else 'lower'} is better)."
                      f"\nTrain rows: {n_tr}; test rows: {n_te}. Validation: {self.cv_plan.describe()}.\nTarget: {tsum}")
        if spec.domain_notes:
            task_block += f"\nDomain notes from the initial analysis: {spec.domain_notes}"
        task_block += f"\nTask statement excerpt:\n{truncate(self.raw.task_text, 2500)}"
        dd = []
        for orig, d in self.ddict.items():
            if orig in (t0.name,):
                continue
            safe = self.ff.safe_of.get(orig, orig)
            label = safe if safe == orig else f"{safe} (orig '{orig}')"
            desc = d.get("description", "")
            if d.get("unit"):
                desc += f" [{d['unit']}]"
            dd.append(f"- {label}: {truncate(desc, 150)}")
        ddict_text = truncate("\n".join(dd), 7000, tail=300)
        shift = "not computed"
        if self.adv:
            shift = f"adversarial AUC train-vs-test = {self.adv['auc']:.3f} (0.5 = identical distributions). Most shifted features: " + \
                    ", ".join(f"{k} ({100 * v:.0f}%)" for k, v in self.adv["top_features"][:6])
            if self.adv["auc"] > 0.8:
                shift += ("\nSTRONG SHIFT: train and test come from different distributions, so cross-validation is optimistic and "
                          "small CV gains often do not transfer. Prefer robust, physically meaningful features (normalised ratios, "
                          "flags, smoothed priors of stable keys) over fine-grained features of the most shifted columns; "
                          "simpler is safer here.")
        ideas = list(spec.feature_ideas)
        # analyst hints become hypotheses for the researcher
        if spec.cat_hints:
            ideas.insert(0, f"categorical_cast of {[self.ff.safe_of.get(c, c) for c in spec.cat_hints]} (suggested by the analysis)")
        if spec.drop_hints:
            ideas.insert(0, f"drop_features {[self.ff.safe_of.get(c, c) for c in spec.drop_hints]} (suggested by the analysis)")
        ctx = {"task_line": task_line, "task_block": task_block, "ddict_text": ddict_text, "desc_of": self.desc_of,
               "shift_text": shift, "analyst_ideas": ideas, "class_names": class_names}

        def on_inc(state, res):
            self._push_pred(pb, res.test, res.oof, "research", self.proxy_folds)

        sandbox = Sandbox(cfg.work_dir, n_jobs=cfg.n_jobs)
        llm = self.llm if (self.llm is not None and getattr(self.llm, "available", False)) else None
        getter = (lambda: self.llm if (self.llm is not None and getattr(self.llm, "available", False)) else None)
        loop = ResearchLoop(problem=pb, state=self.state, evaluator=self.evaluator, llm=llm, sandbox=sandbox, ctx=ctx,
                            deadline_ts=fe_deadline, on_incumbent=on_inc, max_researcher_calls=cfg.extra.get("researcher_calls", 14), ideas_per_call=cfg.extra.get("ideas_per_call", 5))
        loop.llm_getter = getter
        if getattr(self, "late_analyst", None) is not None:
            loop.analyst_future = self.late_analyst
            loop.analyst_merge = self._late_analyst
            loop.analyst_wait_until = time.time() + max(120.0, 0.05 * cfg.time_budget)
        with timer("feature research"):
            try:
                state, inc = loop.run(base_eval)
            except Exception as e:
                LOG.error(f"research loop crashed: {e}\n{format_exc()}")
                state, inc = loop.state, loop.inc_eval
        self.report["research"] = loop.report()
        save_json(cfg.work_dir / "journal.json", self.report["research"])
        LOG.info(f"Feature research: {fmt_num(base_eval.score, 6)} -> {fmt_num(inc.score if inc else None, 6)}; "
                 f"{state.X.shape[1]} cols, TE {len(state.te_specs)}; accepted {self.report['research']['accepted']}")
        return state

    # ==========================================================================================
    def _model_phase(self, final_state: FeatureState, base_eval):
        cfg = self.cfg
        n_t = len(self.problems)
        self.report["models"] = {}
        for i, pb in enumerate(self.problems):
            remaining = self.dl.end - time.time()
            if remaining < 20:
                break
            t_deadline = time.time() + remaining / (n_t - i)
            folds = self.final_folds
            extra = []
            # confirmation of the feature set on fresh folds (anti-overfitting of the greedy search)
            state = final_state
            if final_state is not self.state and i == 0:
                state, extra = self._confirm_features(pb, final_state, folds, t_deadline)
            proxy_params = dict(self.evaluator.params)
            base_lgb = {k: v for k, v in proxy_params.items() if k not in ("learning_rate", "deterministic")}

            def on_update(pb_, best):
                self._push_pred(pb_, best.test, best.oof, f"models:{best.method}", folds, decision=best.decision)

            if not hasattr(self, "proxy_fold_time"):
                self.proxy_fold_time = 30.0
            ms = ModelSearch(pb, state, folds, cfg.n_jobs, t_deadline, cfg.seed, f"final{cfg.seed}", self.proxy_fold_time,
                             0.1, on_update=on_update, base_lgb_params=base_lgb, extra_results=extra, work_dir=cfg.work_dir,
                             use_autogluon=cfg.extra.get("autogluon", True))
            with timer(f"model search [{pb.name}]"):
                try:
                    ms.run()
                except Exception as e:
                    LOG.error(f"model search crashed: {e}\n{format_exc()}")
            best = ms.best
            self.report["models"][pb.name] = {
                "results": [{"name": r.name, "score": r.score, "elapsed": round(r.elapsed, 1), "iters": r.best_iters} for r in ms.results],
                "best": None if best is None else {"method": best.method, "score": best.score, "members": best.members,
                                                     "decision": best.decision.to_dict()},
                "log": ms.log}
            if best is not None:
                LOG.info(f"[{pb.name}] final OOF {pb.metric} = {fmt_num(best.score, 6)} via {best.method}")

    def _confirm_features(self, pb: Problem, final_state: FeatureState, folds: list, t_deadline: float):
        """Evaluate base vs final feature sets with the proxy model on fresh folds; keep the better one."""
        cfg = self.cfg
        params = dict(self.evaluator.params)
        params["learning_rate"] = 0.1 if self.proxy_fold_time * len(folds) < 0.1 * (t_deadline - time.time()) else 0.15
        spec = ModelSpec("lgbm", {k: v for k, v in params.items()}, seed=cfg.seed + 77, name="lgbm_confirm")
        est = self.proxy_fold_time * len(folds) * 2.2
        if t_deadline - time.time() < est * 2.5:
            LOG.info("Skipping feature confirmation (not enough time)")
            return final_state, []
        r_final = run_cv(spec, pb, final_state, folds, cfg.n_jobs, f"final{cfg.seed}", predict_test=True, deadline=t_deadline)
        r_base = run_cv(spec, pb, self.state, folds, cfg.n_jobs, f"final{cfg.seed}", predict_test=True, deadline=t_deadline)
        if r_final is None or r_base is None or not (r_final.complete and r_base.complete):
            return final_state, ([r_final] if r_final is not None and r_final.complete else [])

        def u(r):
            dec = fit_decision(pb.metric, pb.task, pb.y, r.oof, pb.n_classes, validate=False) if MT.get(pb.metric).needs == "label" else Decision()
            from mla_postprocess import metric_with_decision
            return MT.utility(pb.metric, metric_with_decision(pb.metric, pb.y, r.oof, dec, pb.n_classes))
        uf, ub = u(r_final), u(r_base)
        self.report["confirmation"] = {"final_features": uf, "base_features": ub}
        LOG.info(f"Feature confirmation on fresh folds: engineered {uf:.6g} vs base {ub:.6g}")
        r_final.name = "lgbm_proxy_final_feats"
        if uf >= ub:
            return final_state, [r_final]
        LOG.warning("Engineered features did not confirm on fresh folds; reverting to base features")
        r_base.name = "lgbm_proxy_base_feats"
        return self.state, [r_base]
