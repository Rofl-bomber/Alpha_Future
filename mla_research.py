"""AutoResearch loop over the feature space.

Researcher (LLM) proposes hypotheses -> Developer (LLM) writes a `transform(df)` block -> Sandbox executes it ->
ProxyEvaluator trains a fixed LightGBM on fixed folds -> paired fold test vs the incumbent -> accept/reject ->
journal + lessons go back to the Researcher.  LLM calls run in background threads while the CPU evaluates.
"""
from __future__ import annotations

import concurrent.futures as cf
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import mla_metrics as MT
import mla_prompts as PR
from mla_features import (FeatureState, apply_auto_op, apply_new_frame, auto_hypotheses, cast_categorical, drop_features,
                       with_te)
from mla_models import ModelSpec, Problem, oof_fold_utilities, run_cv
from mla_postprocess import Decision, fit_decision, metric_with_decision
from mla_sandbox import Sandbox
from mla_utils import LOG, fmt_num, format_exc, truncate


# =============================================================================================
@dataclass
class Experiment:
    id: str
    source: str  # auto | llm
    type: str  # auto | feature_code | target_encoding | drop_features | categorical_cast
    title: str
    rationale: str = ""
    spec: str = ""
    columns: list | None = None
    payload: dict = field(default_factory=dict)  # auto op params
    code: str | None = None
    status: str = "pending"  # pending|coding|ready|debugging|evaluating|accepted|rejected|error|skipped
    delta: float | None = None
    se: float | None = None
    score: float | None = None
    info: dict = field(default_factory=dict)
    error: str | None = None
    attempts: int = 0
    expected: str = ""
    t_created: float = field(default_factory=time.time)

    def result_text(self, metric: str) -> str:
        if self.status == "accepted":
            r = f"ACCEPTED Δ={self.delta:+.5g} (se {self.se:.2g}) → {fmt_num(self.score, 6)}"
        elif self.status == "rejected":
            r = f"rejected Δ={self.delta:+.5g} (se {self.se:.2g})" if self.delta is not None else "rejected"
        elif self.status == "error":
            r = f"ERROR: {truncate(self.error or '', 140)}"
        else:
            r = self.status
        added = self.info.get("added") or []
        if added and self.status in ("accepted", "rejected"):
            r += f" | new cols: {', '.join(added[:6])}{'…' if len(added) > 6 else ''}"
        return r


@dataclass
class EvalResult:
    utils: np.ndarray
    mean: float
    score: float
    oof: np.ndarray
    test: np.ndarray | None
    importance: pd.Series | None
    decision: Decision
    elapsed: float
    n_features: int


class ProxyEvaluator:
    """Fixed LightGBM + fixed folds; decision-aware per-fold utilities for paired comparisons."""

    def __init__(self, problem: Problem, folds: list, fold_ids: list, params: dict, n_jobs: int, tag: str, seed: int = 0):
        self.problem = problem
        self.folds = folds
        self.fold_ids = list(fold_ids)
        self.params = dict(params)
        self.n_jobs = n_jobs
        self.tag = tag
        self.seed = seed
        self.times = []

    def est_time(self) -> float:
        return float(np.median(self.times[-5:])) if self.times else 60.0

    def evaluate(self, state: FeatureState, deadline: float | None = None, predict_test: bool = True) -> EvalResult | None:
        t0 = time.time()
        spec = ModelSpec("lgbm", dict(self.params), seed=self.seed, name="proxy")
        r = run_cv(spec, self.problem, state, self.folds, self.n_jobs, self.tag, predict_test=predict_test,
                   deadline=deadline, fold_subset=self.fold_ids)
        if r is None or not r.complete:
            return None
        pb = self.problem
        idx = np.concatenate([self.folds[fi][1] for fi in self.fold_ids])
        m = MT.get(pb.metric)
        if m.kind == "classification" and m.needs == "label":
            dec = fit_decision(pb.metric, pb.task, pb.y[idx], r.oof[idx], pb.n_classes, validate=False)
        else:
            dec = Decision()
        utils = oof_fold_utilities(pb, r.oof, self.folds, self.fold_ids, dec)
        score = metric_with_decision(pb.metric, pb.y[idx], r.oof[idx], dec, pb.n_classes)
        el = time.time() - t0
        self.times.append(el)
        return EvalResult(utils, float(utils.mean()), float(score), r.oof, r.test, r.importance, dec, el, state.X.shape[1])


def paired_test(cand: EvalResult, inc: EvalResult, z: float = 1.0, rel_min: float = 1e-4) -> tuple[bool, float, float, bool]:
    d = cand.utils - inc.utils
    md = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else abs(md)
    min_gain = rel_min * max(abs(inc.mean), 1e-9)
    accept = md > max(min_gain, z * se)
    near = (not accept) and md > 0
    return accept, md, se, near


# =============================================================================================
class ResearchLoop:
    def __init__(self, *, problem: Problem, state: FeatureState, evaluator: ProxyEvaluator, llm, sandbox: Sandbox,
                 ctx: dict, deadline_ts: float, on_incumbent=None, max_researcher_calls: int = 12, ideas_per_call: int = 5,
                 z: float = 1.0):
        self.pb = problem
        self.base_state = state
        self.state = state
        self.ev = evaluator
        self.llm = llm
        self.sandbox = sandbox
        self.ctx = ctx  # task_line, task_block, ddict_text, desc_of, shift_text, analyst_ideas, domain_notes
        self.deadline = deadline_ts
        self.on_incumbent = on_incumbent
        self.max_researcher_calls = max_researcher_calls
        self.ideas_per_call = ideas_per_call
        self.z = z
        self.exps: list[Experiment] = []
        self.insights: list[str] = []
        self.analyses: list[str] = []
        self.inc_eval: EvalResult | None = None
        self.base_eval: EvalResult | None = None
        self.n_researcher_calls = 0
        self.pool = cf.ThreadPoolExecutor(max_workers=max(2, (getattr(getattr(llm, "cfg", None), "llm_concurrency", 2) or 2) + 1))
        self.futures: dict = {}  # future -> (kind, exp or None)
        self.counter = 0
        self.near_miss: list = []
        self._col_cache: dict = {}
        self.analyst_ideas = list(ctx.get("analyst_ideas") or [])
        self.events: list = []
        self.llm_getter = None  # picks up the LLM once its health check has finished
        self.analyst_future = None  # late analyst answer (merged when ready)
        self.analyst_merge = None
        self.analyst_wait_until = 0.0

    # ------------------------------------------------------------------ helpers
    def _new_id(self, prefix="E") -> str:
        self.counter += 1
        return f"{prefix}{self.counter:02d}"

    def _time_left(self) -> float:
        return self.deadline - time.time()

    def llm_ok(self) -> bool:
        if self.llm is None and self.llm_getter is not None:
            self.llm = self.llm_getter()
            if self.llm is not None:
                self.pool._max_workers = max(self.pool._max_workers, getattr(self.llm.cfg, "llm_concurrency", 2) + 1)
        return self.llm is not None and getattr(self.llm, "available", False) and self.llm.budget_left()

    def _running(self, kind: str) -> int:
        return sum(1 for f, (k, _) in self.futures.items() if k == kind and not f.done())

    # ------------------------------------------------------------------ context for prompts
    def columns_table(self, state: FeatureState, max_rows: int = 260) -> str:
        key = state.key()
        if key in self._col_cache:
            return self._col_cache[key]
        X = state.X
        desc_of = self.ctx.get("desc_of", {})
        lines = []
        for c in list(X.columns)[:max_rows]:
            s = X[c]
            miss = s.isna().mean()
            if c in state.cat_cols:
                vc = s.value_counts()
                tops = ", ".join(truncate(str(v), 20) for v in vc.index[:4])
                vals = f"{s.nunique()} levels; top: {tops}"
                kind = "cat"
            else:
                x = s.dropna()
                if len(x):
                    vals = f"[{fmt_num(x.min())} .. {fmt_num(x.max())}], median {fmt_num(x.median())}"
                    if np.all(np.mod(x.values[:3000], 1) == 0):
                        vals += ", integer"
                else:
                    vals = "all NaN"
                kind = "num"
            if miss > 0:
                vals += f", {100 * miss:.1f}% NaN"
            d = desc_of.get(c, "")
            lines.append(f"{c} | {kind} | {vals} | {truncate(d, 90)}")
        if X.shape[1] > max_rows:
            lines.append(f"... and {X.shape[1] - max_rows} more columns: {', '.join(list(X.columns)[max_rows:max_rows + 80])}")
        txt = "\n".join(lines)
        self._col_cache[key] = txt
        return txt

    def features_text(self) -> str:
        st = self.state
        imp = self.inc_eval.importance if self.inc_eval is not None else None
        base_cols = [c for c in st.X.columns if c in self.base_state.X.columns]
        eng = [c for c in st.X.columns if c not in self.base_state.X.columns]
        out = [f"{st.X.shape[1]} columns ({len(base_cols)} original/base, {len(eng)} engineered); categorical: {', '.join(st.cat_cols[:40]) or 'none'}"]
        if eng:
            out.append("Engineered columns: " + ", ".join(eng[:120]) + (" …" if len(eng) > 120 else ""))
        if st.te_specs:
            out.append("Target-encoded keys in use: " + "; ".join("+".join(t) for t in st.te_specs[:30]))
        out.append("Column table (name | kind | values | description):")
        out.append(self.columns_table(st, max_rows=160))
        return "\n".join(out)

    def error_analysis(self) -> str:
        ev = self.inc_eval
        if ev is None:
            return ""
        pb = self.pb
        idx = np.concatenate([self.ev.folds[fi][1] for fi in self.ev.fold_ids])
        y = pb.y[idx]
        try:
            if pb.task == "regression":
                p = ev.oof[idx]
                res = p - y
                lines = [f"Residuals (pred - true): mean {fmt_num(res.mean())}, RMSE {fmt_num(np.sqrt(np.mean(res ** 2)))}, MAE {fmt_num(np.mean(np.abs(res)))}"]
                q = pd.qcut(pd.Series(y).rank(method="first"), 5, labels=False).values
                parts = []
                for b in range(5):
                    mk = q == b
                    parts.append(f"Q{b + 1}: bias {fmt_num(res[mk].mean())}, rmse {fmt_num(np.sqrt(np.mean(res[mk] ** 2)))}")
                lines.append("By true-target quintile: " + "; ".join(parts))
                X = self.state.X.iloc[idx]
                imp = ev.importance if ev.importance is not None else pd.Series(dtype=float)
                cats = [c for c in imp.sort_values(ascending=False).index if c in self.state.cat_cols][:2] or self.state.cat_cols[:2]
                for c in cats:
                    df = pd.DataFrame({"g": X[c].astype(str).values, "e2": res ** 2, "r": res})
                    g = df.groupby("g").agg(n=("e2", "size"), sse=("e2", "sum"), bias=("r", "mean")).sort_values("sse", ascending=False).head(5)
                    tot = df["e2"].sum()
                    lines.append(f"Largest error share by {c}: " + "; ".join(
                        f"{truncate(k, 25)} (n={int(r.n)}, {100 * r.sse / tot:.1f}% of SSE, bias {fmt_num(r.bias)})" for k, r in g.iterrows()))
                return "\n".join(lines)
            P = ev.decision.transform_proba(ev.oof[idx])
            lab = MT.to_labels(P, ev.decision.decision_arg())
            K = pb.n_classes
            names = self.ctx.get("class_names") or [str(i) for i in range(K)]
            C = np.bincount(y * K + lab, minlength=K * K).reshape(K, K)
            rows = []
            for k in range(K):
                tp = C[k, k]
                prec = tp / max(1, C[:, k].sum())
                rec = tp / max(1, C[k].sum())
                f1 = 2 * prec * rec / max(1e-12, prec + rec)
                rows.append(f"{truncate(names[k], 22)}: support {C[k].sum()}, precision {prec:.3f}, recall {rec:.3f}, F1 {f1:.3f}")
            off = [(C[i, j], i, j) for i in range(K) for j in range(K) if i != j and C[i, j] > 0]
            off.sort(reverse=True)
            conf = "; ".join(f"{truncate(names[i], 18)}→{truncate(names[j], 18)}: {n}" for n, i, j in off[:8])
            return "Per-class OOF quality (after decision rule):\n" + "\n".join(rows) + f"\nMost frequent confusions (true→pred): {conf}"
        except Exception as e:
            LOG.debug(f"error analysis failed: {e}")
            return ""

    def model_block(self) -> str:
        ev = self.inc_eval
        m = MT.get(self.pb.metric)
        lines = [f"CV {self.pb.metric} ({'higher' if m.greater_is_better else 'lower'} is better): current {fmt_num(ev.score, 6)}; "
                 f"baseline (no engineered features) {fmt_num(self.base_eval.score, 6)}"]
        imp = ev.importance
        if imp is not None and len(imp):
            imp = imp.sort_values(ascending=False)
            tot = imp.sum() + 1e-12
            lines.append("Top features by gain: " + ", ".join(f"{k} {100 * v / tot:.1f}%" for k, v in imp.head(25).items()))
            zero = [k for k, v in imp.items() if v <= 0]
            if zero:
                lines.append("Zero-importance features: " + ", ".join(zero[:40]))
        ea = self.error_analysis()
        if ea:
            lines.append(ea)
        return "\n".join(lines)

    def journal_text(self, max_rows: int = 36) -> str:
        done = [e for e in self.exps if e.status in ("accepted", "rejected", "error")]
        if not done:
            return "(no experiments yet)"
        rows = []
        older = done[:-max_rows]
        if older:
            acc = sum(e.status == "accepted" for e in older)
            rows.append(f"[{len(older)} older experiments: {acc} accepted; accepted titles: " + "; ".join(e.title for e in older if e.status == "accepted")[:600] + "]")
        for e in done[-max_rows:]:
            extra = ""
            if e.type == "target_encoding" and e.columns:
                extra = " cols=" + str(e.columns)[:120]
            rows.append(f"{e.id} [{e.type}] {truncate(e.title, 90)}{extra} → {e.result_text(self.pb.metric)}")
        return "\n".join(rows)

    # ------------------------------------------------------------------ LLM jobs
    def _research_job(self, k: int, snapshot: dict) -> dict:
        msgs = [{"role": "system", "content": PR.RESEARCHER_SYSTEM},
                {"role": "user", "content": PR.researcher_user(snapshot["task_block"], snapshot["ddict_text"], snapshot["features_text"],
                                                               snapshot["model_block"], snapshot["shift_text"], snapshot["journal"],
                                                               snapshot["insights"], snapshot["ideas"], k)}]
        obj, r = self.llm.chat_json(msgs, purpose="researcher", temperature=0.7, max_tokens=8000, mock_ctx=snapshot.get("mock_ctx"))
        return obj

    def _avg_seconds(self, purpose: str) -> float:
        try:
            pp = self.llm.per_purpose.get(purpose) or {}
            return pp.get("seconds", 0.0) / max(1, pp.get("calls", 0)) if pp.get("calls") else 0.0
        except Exception:
            return 0.0

    def _dev_thinking(self):
        """Keep reasoning for code while the LLM is fast; drop it when it becomes the bottleneck."""
        return None if self._avg_seconds("developer") < 120 else False

    def _develop_job(self, exp: Experiment, columns_table: str, n_rows: int, n_cols: int) -> str:
        msgs = [{"role": "system", "content": PR.DEVELOPER_SYSTEM},
                {"role": "user", "content": PR.developer_user({"title": exp.title, "rationale": exp.rationale, "spec": exp.spec},
                                                              columns_table, n_rows, n_cols, self.ctx.get("task_line", ""))}]
        code, r = self.llm.chat_code(msgs, purpose="developer", temperature=0.3, max_tokens=6000, thinking=self._dev_thinking(), mock_ctx={"exp": exp.__dict__})
        return code

    def _debug_job(self, exp: Experiment, code: str, error: str, columns_table: str, n_rows: int) -> str:
        msgs = [{"role": "system", "content": PR.DEVELOPER_SYSTEM},
                {"role": "user", "content": PR.debugger_user({"title": exp.title, "spec": exp.spec}, code, error, columns_table, n_rows)}]
        code, r = self.llm.chat_code(msgs, purpose="debugger", temperature=0.2, max_tokens=6000, thinking=self._dev_thinking(), mock_ctx={"exp": exp.__dict__, "error": error})
        return code

    def _snapshot(self) -> dict:
        return {"task_block": self.ctx.get("task_block", ""), "ddict_text": self.ctx.get("ddict_text", ""),
                "features_text": self.features_text(), "model_block": self.model_block(), "shift_text": self.ctx.get("shift_text", ""),
                "journal": self.journal_text(), "insights": list(self.insights),
                "ideas": self.analyst_ideas[:12] if self.n_researcher_calls == 0 else self.analyst_ideas[12:18],
                "mock_ctx": {"columns": list(self.state.X.columns), "cats": list(self.state.cat_cols),
                             "importance": (self.inc_eval.importance.sort_values(ascending=False).index[:20].tolist()
                                            if self.inc_eval is not None and self.inc_eval.importance is not None else [])}}

    def poll_analyst(self):
        f = self.analyst_future
        if f is None or not f.done():
            return
        self.analyst_future = None
        try:
            obj = f.result()
        except Exception as e:
            LOG.warning(f"[research] late analyst failed: {e}")
            return
        if not obj:
            return
        try:
            upd = self.analyst_merge(obj) if self.analyst_merge else {"ideas": obj.get("feature_ideas") or [], "domain": ""}
            self.analyst_ideas = list(upd.get("ideas") or []) + self.analyst_ideas
            if upd.get("domain"):
                self.ctx["task_block"] = self.ctx.get("task_block", "") + f"\nDomain notes from the initial analysis: {upd['domain']}"
            LOG.info(f"[research] merged late analyst insights ({len(upd.get('ideas') or [])} ideas)")
        except Exception as e:
            LOG.warning(f"[research] merging analyst output failed: {e}")

    def submit_research(self):
        if not self.llm_ok() or self.n_researcher_calls >= self.max_researcher_calls or self._running("research"):
            return
        # give the analyst a chance to seed the first research round with domain knowledge
        if self.n_researcher_calls == 0 and self.analyst_future is not None and time.time() < self.analyst_wait_until:
            return
        snap = self._snapshot()  # before incrementing: the first call gets the analyst's ideas
        self.n_researcher_calls += 1
        f = self.pool.submit(self._research_job, self.ideas_per_call, snap)
        self.futures[f] = ("research", None)
        LOG.info(f"[research] Researcher call #{self.n_researcher_calls} submitted")

    def submit_develop(self, exp: Experiment):
        exp.status = "coding"
        f = self.pool.submit(self._develop_job, exp, self.columns_table(self.state), len(self.state.X), self.state.X.shape[1])
        self.futures[f] = ("develop", exp)

    def submit_debug(self, exp: Experiment, error: str):
        exp.status = "debugging"
        f = self.pool.submit(self._debug_job, exp, exp.code or "", error, self.columns_table(self.state), len(self.state.X))
        self.futures[f] = ("debug", exp)

    def _norm_title(self, t: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()

    def _ingest_research(self, obj: dict):
        if not isinstance(obj, dict):
            return
        ins = obj.get("insights")
        if isinstance(ins, list) and ins:
            self.insights = [truncate(str(i), 300) for i in ins[:8]]
        if obj.get("analysis"):
            self.analyses.append(truncate(str(obj["analysis"]), 1200))
            LOG.info(f"[research] analysis: {truncate(str(obj['analysis']), 400)}")
        exps = obj.get("experiments") or []
        seen = {self._norm_title(e.title) for e in self.exps}
        seen_te = {str(sorted(map(tuple, e.columns))) for e in self.exps if e.type == "target_encoding" and e.columns}
        cols = set(self.state.X.columns)
        n_added = 0
        for raw in exps[: self.ideas_per_call + 2]:
            if not isinstance(raw, dict):
                continue
            typ = str(raw.get("type", "feature_code")).strip().lower()
            title = truncate(str(raw.get("title", "")).strip() or typ, 160)
            if self._norm_title(title) in seen:
                continue
            e = Experiment(self._new_id(), "llm", typ, title, truncate(str(raw.get("rationale", "")), 800),
                           truncate(str(raw.get("spec", "")), 2500), expected=str(raw.get("expected_gain", "")))
            if typ == "target_encoding":
                if self.pb.task == "multiclass" and self.pb.n_classes > 20:
                    continue
                specs = []
                for item in raw.get("columns") or []:
                    item = [item] if isinstance(item, str) else item
                    if isinstance(item, (list, tuple)) and item and all(isinstance(c, str) and c in cols for c in item) and len(item) <= 4:
                        specs.append(list(item))
                specs = [s for s in specs if tuple(s) not in self.state.te_specs]
                if not specs:
                    continue
                key = str(sorted(map(tuple, specs)))
                if key in seen_te:
                    continue
                e.columns = specs
                e.status = "ready"
            elif typ in ("drop_features", "categorical_cast"):
                cl = [c for c in (raw.get("columns") or []) if isinstance(c, str) and c in cols]
                if typ == "categorical_cast":
                    cl = [c for c in cl if c not in self.state.cat_cols]
                if not cl:
                    continue
                e.columns = cl
                e.status = "ready"
            else:
                e.type = "feature_code"
                if not e.spec:
                    e.spec = title
                e.status = "pending"
            self.exps.append(e)
            seen.add(self._norm_title(title))
            n_added += 1
        LOG.info(f"[research] {n_added} new experiments queued")

    def poll_futures(self):
        for f in [f for f in self.futures if f.done()]:
            kind, exp = self.futures.pop(f)
            try:
                res = f.result()
            except Exception as e:
                LOG.warning(f"[research] LLM {kind} job failed: {type(e).__name__}: {truncate(str(e), 300)}")
                if exp is not None:
                    exp.status = "error"
                    exp.error = f"LLM {kind} failed: {truncate(str(e), 200)}"
                continue
            if kind == "research":
                self._ingest_research(res)
            elif kind in ("develop", "debug"):
                exp.code = res
                exp.status = "ready"

    # ------------------------------------------------------------------ evaluation of one experiment
    def build_candidate(self, exp: Experiment) -> FeatureState | None:
        block = {"id": exp.id, "type": exp.type, "title": exp.title}
        st = self.state
        if exp.type == "target_encoding":
            return with_te(st, exp.columns, block)
        if exp.type == "drop_features":
            return drop_features(st, exp.columns, block)
        if exp.type == "categorical_cast":
            return cast_categorical(st, exp.columns, block)
        if exp.type == "auto":
            newX = apply_auto_op(st, exp.payload)
            ns, info = apply_new_frame(st, newX, block)
            exp.info.update(info)
            key_col = exp.payload.get("key_col", "entity_key")
            if exp.payload.get("te_after") and key_col in ns.X.columns:
                ns = with_te(ns, [[key_col]], block)
            return ns
        # feature_code
        newX, meta = self.sandbox.run(st, exp.code or "", timeout=max(60.0, min(240.0, 60 + len(st.X) / 5000)))
        if newX is None:
            err = meta.get("error", "unknown error") + ("\n" + meta.get("traceback", "") if meta.get("traceback") else "")
            raise RuntimeError(err)
        ns, info = apply_new_frame(st, newX, block)
        info["sandbox_seconds"] = meta.get("seconds")
        if meta.get("warnings"):
            info["warnings"] = meta["warnings"]
        exp.info.update(info)
        if not info["added"] and not info["modified"] and not info["removed"]:
            raise RuntimeError("the function did not create any new non-constant column")
        return ns

    def temporal_check(self, cand, period_values):
        """Mean utility gain of `cand` over the incumbent on forward folds (latest two periods)."""
        try:
            per = np.asarray(period_values, float)[: self.state.n_train]
            ups = np.unique(per[np.isfinite(per)])
            if len(ups) < 3:
                return None, 0.0
            folds = []
            for p_ in ups[-2:]:
                tr = np.where(per < p_)[0]
                va = np.where(per == p_)[0]
                if len(tr) >= 50 and len(va) >= 20:
                    folds.append((tr, va))
            if not folds:
                return None, 0.0
            ev = ProxyEvaluator(self.pb, folds, list(range(len(folds))), self.ev.params, self.ev.n_jobs, "forward", self.ev.seed)
            a = ev.evaluate(cand, deadline=self.deadline, predict_test=False)
            b = ev.evaluate(self.state, deadline=self.deadline, predict_test=False)
            if a is None or b is None:
                return None, 0.0
            d = float(np.mean(a.utils - b.utils))
            return d > 0, d
        except Exception as e:
            LOG.warning(f"temporal check failed: {e}")
            return None, 0.0

    def evaluate_experiment(self, exp: Experiment):
        exp.status = "evaluating"
        exp.attempts += 1
        try:
            cand = self.build_candidate(exp)
        except Exception as e:
            msg = str(e)
            first = msg.strip().splitlines()[0] if msg.strip() else msg
            LOG.info(f"[{exp.id}] build failed: {truncate(first, 200)}")
            if exp.type == "feature_code" and exp.attempts < 3 and self.llm_ok():
                self.submit_debug(exp, msg)
            else:
                exp.status, exp.error = "error", truncate(msg, 400)
            return
        if cand is None:
            exp.status, exp.error = "error", "empty candidate"
            return
        res = self.ev.evaluate(cand, deadline=self.deadline)
        if res is None:
            exp.status, exp.error = "skipped", "evaluation incomplete (time)"
            return
        # multiple-comparison guard: the more hypotheses were tried, the more likely a 1-SE gain is luck
        n_eval = sum(1 for e in self.exps if e.status in ("accepted", "rejected"))
        z_eff = self.z + 0.5 * min(1.0, max(0.0, (n_eval - 10) / 20.0))
        accept, md, se, near = paired_test(res, self.inc_eval, z=z_eff)
        if exp.payload.get("temporal") is not None:
            # future-period hypotheses (entity history, dropping the period column): random K-fold surrounds every
            # validation row with rows of the same periods and cannot measure extrapolation in time, and single-period
            # forward folds proved too noisy. On replicated tasks both helped the hidden test consistently, both use
            # only past information, so they are accepted by rule unless CV shows clear harm.
            accept = md > -2.0 * se
            near = False
        if exp.type == "drop_features" and exp.payload.get("temporal") is None:
            accept = md > -0.1 * se and md > -1e-4 * abs(self.inc_eval.mean)
            if exp.payload.get("lenient"):
                # principled drop (e.g. a period column whose test values never occur in train): random CV cannot
                # see the benefit, so only refuse it when it clearly hurts
                accept = md > -1.0 * se
        exp.delta, exp.se, exp.score = md, se, res.score
        if accept:
            exp.status = "accepted"
            self.state = cand
            self.inc_eval = res
            LOG.info(f"[{exp.id}] ACCEPTED {exp.title!r}: Δ={md:+.5g} se={se:.2g} → {self.pb.metric}={res.score:.6g} "
                     f"({cand.X.shape[1]} cols, {len(cand.te_specs)} TE)")
            if self.on_incumbent:
                try:
                    self.on_incumbent(self.state, res)
                except Exception as e:
                    LOG.warning(f"on_incumbent callback failed: {e}")
        else:
            exp.status = "rejected"
            LOG.info(f"[{exp.id}] rejected {exp.title!r}: Δ={md:+.5g} se={se:.2g}")
            if near:
                self.near_miss.append(exp)

    # ------------------------------------------------------------------ main loop
    def run(self, base_eval: EvalResult) -> tuple[FeatureState, EvalResult]:
        self.base_eval = base_eval
        self.inc_eval = base_eval
        for h in auto_hypotheses(self.state, base_eval.importance, self.pb.task, self.pb.n_classes, y=self.pb.y):
            e = Experiment(self._new_id("A"), "auto", h["type"], h["title"], h.get("rationale", ""))
            e.payload = h
            if h["type"] != "auto":
                e.columns = h.get("columns")
            e.status = "ready"
            self.exps.append(e)
        self.submit_research()
        idle_since = None
        while True:
            self.poll_futures()
            self.poll_analyst()
            left = self._time_left()
            if left < self.ev.est_time() * 1.15 + 5:
                LOG.info("[research] feature phase time is up")
                break
            # keep the LLM pipeline busy
            pending_llm = [e for e in self.exps if e.source == "llm" and e.status in ("pending", "coding", "debugging", "ready")]
            if len(pending_llm) < 3:
                self.submit_research()
            if self.llm_ok():
                n_dev = self._running("develop") + self._running("debug")
                cap = max(1, getattr(self.llm.cfg, "llm_concurrency", 2))
                for e in [e for e in self.exps if e.status == "pending" and e.type == "feature_code"]:
                    if n_dev >= cap:
                        break
                    self.submit_develop(e)
                    n_dev += 1
            elif self.analyst_future is None:
                for e in self.exps:
                    if e.status == "pending":
                        e.status = "skipped"
                        e.error = "LLM unavailable"
            ready = [e for e in self.exps if e.status == "ready"]
            if ready:
                # LLM ideas with high expected gain first, but auto ideas interleave
                ready.sort(key=lambda e: (0 if e.source == "auto" and (e.id in ("A01", "A02") or e.payload.get("op") in ("entity_key", "row_key", "entity_history")) else 1,
                                          {"high": 0, "medium": 1, "low": 2}.get(e.expected, 1), e.t_created))
                self.evaluate_experiment(ready[0])
                idle_since = None
                continue
            busy = any(not f.done() for f in self.futures) or self.analyst_future is not None
            if not busy:
                if not self.llm_ok() or self.n_researcher_calls >= self.max_researcher_calls:
                    LOG.info("[research] no more ideas; stopping feature search")
                    break
                self.submit_research()
                if not self.futures:
                    break
            if idle_since is None:
                idle_since = time.time()
            time.sleep(1.0)
        self.finish()
        return self.state, self.inc_eval

    def finish(self):
        # best-effort: try a bundle of near-miss experiments together
        try:
            if len(self.near_miss) >= 2 and self._time_left() > self.ev.est_time() * 1.3 + 10:
                cand = self.state
                applied = []
                for e in self.near_miss[:6]:
                    try:
                        old = self.state
                        self.state = cand
                        c2 = self.build_candidate(Experiment(e.id + "b", e.source, e.type, e.title, payload=e.payload, columns=e.columns, code=e.code))
                        self.state = old
                        if c2 is not None:
                            cand = c2
                            applied.append(e.id)
                    except Exception:
                        self.state = old
                if len(applied) >= 2:
                    res = self.ev.evaluate(cand, deadline=self.deadline)
                    if res is not None:
                        accept, md, se, _ = paired_test(res, self.inc_eval, z=self.z)
                        bundle = Experiment(self._new_id("B"), "auto", "bundle", f"bundle of near-miss experiments {applied}")
                        bundle.delta, bundle.se, bundle.score = md, se, res.score
                        if accept:
                            bundle.status = "accepted"
                            self.state, self.inc_eval = cand, res
                            LOG.info(f"[research] near-miss bundle ACCEPTED Δ={md:+.5g}")
                            if self.on_incumbent:
                                self.on_incumbent(self.state, res)
                        else:
                            bundle.status = "rejected"
                            LOG.info(f"[research] near-miss bundle rejected Δ={md:+.5g}")
                        self.exps.append(bundle)
        except Exception as e:
            LOG.warning(f"near-miss bundle failed: {e}")
        for f in list(self.futures):
            f.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def report(self) -> dict:
        return {
            "experiments": [{"id": e.id, "source": e.source, "type": e.type, "title": e.title, "status": e.status,
                             "delta": e.delta, "se": e.se, "score": e.score, "error": e.error, "info": e.info,
                             "rationale": e.rationale, "spec": e.spec, "columns": e.columns,
                             "code": e.code} for e in self.exps],
            "insights": self.insights, "analyses": self.analyses,
            "base_score": self.base_eval.score if self.base_eval else None,
            "final_score": self.inc_eval.score if self.inc_eval else None,
            "accepted": [e.id for e in self.exps if e.status == "accepted"],
            "researcher_calls": self.n_researcher_calls,
        }
