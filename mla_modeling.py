"""Model phase: share-based time allocation over a queue of model candidates + incremental ensembling.

Time allocation: each candidate has a weight; when it starts it receives
    budget = time_left * weight / sum(weights of candidates not yet run)
so unused time automatically flows to later candidates.  Learning rates are chosen from measured
speed (proxy timing for LightGBM, a short probe run for XGBoost / CatBoost) so that models converge
inside their budget instead of being cut off.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import mla_metrics as MT
from mla_ensemble import blend_objective, hill_climb, stack_level2
from mla_features import FeatureState
from mla_hpo import tune
from mla_models import CVResult, ModelSpec, Problem, run_cv
from mla_postprocess import Decision, fit_decision, metric_with_decision
from mla_utils import LOG, fmt_num


@dataclass
class FinalPrediction:
    utility: float
    score: float
    test: np.ndarray
    oof: np.ndarray
    decision: Decision
    members: dict
    method: str
    sel: float = 0.0  # selection score (utility minus a noise penalty)


@dataclass
class Candidate:
    name: str
    family: str
    weight: float
    params: dict = field(default_factory=dict)
    transform: str = "none"
    class_weight: str | None = None
    tune: bool = False
    seeds: int = 1
    bag_of: str | None = None  # refill: average new seeds into this existing result


# iterations a model typically needs to converge at learning rate lr (rule of thumb)
NEEDED_ITERS = {"lgbm": 40.0, "xgb": 35.0, "cat": 60.0}


class ModelSearch:
    def __init__(self, problem: Problem, state: FeatureState, folds: list, n_jobs: int, deadline_ts: float, seed: int,
                 fold_tag: str, lgb_proxy_fold_time: float, proxy_lr: float, on_update=None, base_lgb_params: dict | None = None,
                 extra_results: list | None = None, work_dir=None, use_autogluon: bool = True):
        self.pb = problem
        self.state = state
        self.folds = folds
        self.n_jobs = n_jobs
        self.deadline = deadline_ts
        self.seed = seed
        self.tag = fold_tag
        self.fold_time_proxy = max(0.2, lgb_proxy_fold_time)
        self.proxy_lr = proxy_lr
        self.on_update = on_update
        self.base_lgb_params = dict(base_lgb_params or {})
        self.tuned = {}
        self.results: list[CVResult] = list(extra_results or [])
        self.best: FinalPrediction | None = None
        self.log: list = []
        self.iter_time: dict = {}
        self.work_dir = work_dir
        self.use_autogluon = use_autogluon
        if self.results:
            self.update_ensemble()

    # ------------------------------------------------------------------ timing helpers
    def left(self) -> float:
        return self.deadline - time.time()

    def est_lgb_time(self, lr: float) -> float:
        """Rough wall time of a full CV run of LightGBM at learning rate lr (features may have grown -> 1.3x)."""
        return self.fold_time_proxy * (self.proxy_lr / lr) * len(self.folds) * 1.3

    def lgb_lr_for(self, budget: float) -> float:
        for lr in (0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15):
            if self.est_lgb_time(lr) <= budget * 0.8:
                return lr
        return 0.2

    def probe(self, family: str, params: dict, iters: int = 30) -> float:
        """Seconds per boosting iteration for one fold (measured)."""
        key = (family, tuple(sorted((k, str(v)) for k, v in params.items() if k in ("depth", "max_depth"))))
        if key in self.iter_time:
            return self.iter_time[key]
        # two runs (n and 3n iterations): the difference isolates the per-iteration cost from fixed set-up
        # costs (target encoding, Pool/DMatrix construction) that previously inflated the estimate 10-30x
        per = None
        times = []
        for n_it in (iters, 3 * iters):
            spec = ModelSpec(family, {**params, "n_estimators": n_it, "early_stopping": 10 ** 6}, seed=self.seed, name=f"{family}_probe")
            r = run_cv(spec, self.pb, self.state, self.folds, self.n_jobs, self.tag, predict_test=False, deadline=time.time() + 300, fold_subset=[0])
            if r is None:
                break
            times.append(sum(r.extra.get("fit_secs", [0.0])))
        if len(times) == 2:
            per = max(1e-4, (times[1] - times[0]) / (2 * iters))
        elif times:
            per = times[0] / iters
        if per is None:
            per = 1.0
        self.iter_time[key] = per
        LOG.info(f"[models] probe {family}: {per * 1000:.1f} ms/iter/fold")
        return per

    def affordable_iters(self, per_iter: float, budget: float) -> float:
        return budget * 0.75 / max(1e-6, per_iter * len(self.folds) * 1.15)

    def lr_from_probe(self, family: str, per_iter: float, budget: float, lo: float, hi: float) -> float:
        iters_affordable = self.affordable_iters(per_iter, budget)
        lr = NEEDED_ITERS[family] / max(1.0, iters_affordable)
        return float(np.clip(lr, lo, hi))

    def _imbalanced(self) -> bool:
        if self.pb.task == "regression":
            return False
        cnt = np.bincount(self.pb.y.astype(int), minlength=self.pb.n_classes)
        cnt = cnt[cnt > 0]
        return cnt.max() / max(1, cnt.min()) > 5

    # ------------------------------------------------------------------ plan
    def plan(self) -> list[Candidate]:
        pb = self.pb
        y = pb.y
        c = [Candidate("lgbm_tuned", "lgbm", 30, tune=True),
             Candidate("catboost", "cat", 22),
             Candidate("xgboost", "xgb", 12)]
        if pb.task == "regression":
            skewed = float(np.min(y)) >= 0 and abs(float(np.mean(y) - np.median(y))) > 0.1 * float(np.std(y) + 1e-9)
            if pb.metric in ("rmsle", "msle"):
                c.insert(1, Candidate("lgbm_log1p", "lgbm", 10, transform="log1p"))
            elif skewed:
                c.append(Candidate("lgbm_log1p", "lgbm", 7, transform="log1p"))
            if pb.metric in ("rmse", "mse", "r2"):
                c.append(Candidate("lgbm_huber", "lgbm", 5, params={"objective": "huber", "alpha": float(np.std(y))}))
            if pb.metric in ("mae", "medae", "mape", "smape"):
                c.append(Candidate("lgbm_l2", "lgbm", 5, params={"objective": "l2"}))
        else:
            m = MT.get(pb.metric)
            if self._imbalanced() and m.needs == "label":
                c.append(Candidate("lgbm_cw_sqrt", "lgbm", 7, class_weight="sqrt"))
        if self.use_autogluon and self.work_dir is not None and self.state.n_train >= 500:
            from mla_models_ag import ag_available
            if ag_available():
                c.append(Candidate("autogluon", "ag", 20))
        if self.state.n_train >= 800:
            c.append(Candidate("mlp", "mlp", 7))
        c.append(Candidate("lgbm_seeds", "lgbm", 9, seeds=3))
        c.append(Candidate("lgbm_regularized", "lgbm", 5, params={"feature_fraction": 0.5, "bagging_fraction": 0.7, "lambda_l2": 10.0,
                                                                  "num_leaves": 31, "min_child_samples": 40}))
        c.append(Candidate("catboost_d8", "cat", 10, params={"depth": 8, "l2_leaf_reg": 6.0}))
        c.append(Candidate("xgboost_tuned", "xgb", 9, tune=True))
        if pb.task != "regression" and self._imbalanced() and MT.get(pb.metric).needs == "label":
            c.append(Candidate("catboost_cw_sqrt", "cat", 7, class_weight="sqrt"))
        if self.state.n_train <= 200_000:
            c.append(Candidate("extratrees", "et", 3, params={"n_estimators": 400}))
        from mla_models import family_available
        c = [x for x in c if family_available(x.family)]
        return c

    # ------------------------------------------------------------------ execution
    def _run_spec(self, spec: ModelSpec, budget: float) -> CVResult | None:
        per_fold = max(5.0, budget / max(1, len(self.folds)))
        dl = min(self.deadline, time.time() + budget * 1.5 + 30)
        t0 = time.time()
        r = run_cv(spec, self.pb, self.state, self.folds, self.n_jobs, self.tag, predict_test=True, deadline=dl, per_fold_budget=per_fold)
        el = time.time() - t0
        if r is None or not r.complete:
            LOG.info(f"[models] {spec.label()} incomplete ({el:.0f}s)")
            return None
        LOG.info(f"[models] {spec.label()}: {self.pb.metric}={fmt_num(r.score, 6)} in {el:.0f}s (budget {budget:.0f}s, iters {r.best_iters})")
        return r

    def run_candidate(self, c: Candidate, budget: float) -> CVResult | None:
        fam = c.family
        if fam == "lgbm":
            base = dict(self.tuned.get("lgbm", self.base_lgb_params))
            if c.tune:
                hpo_budget = budget * 0.6
                if hpo_budget > max(45.0, self.fold_time_proxy * 8):
                    hpo_lr = 0.1 if self.fold_time_proxy * 3 * 15 < hpo_budget else 0.15
                    fold_ids = list(range(min(3, len(self.folds))))
                    best, hist = tune("lgbm", self.pb, self.state, self.folds, fold_ids, self.n_jobs, hpo_budget,
                                      {**self.base_lgb_params, "learning_rate": hpo_lr}, self.seed, self.tag)
                    base.update({k: v for k, v in best.items() if k != "learning_rate"})
                    self.log.append({"hpo": "lgbm", "trials": len(hist), "best": base})
                self.tuned["lgbm"] = base
                budget = max(10.0, budget - (time.time() - self._t_start_c))
            base.update(c.params)
            if c.seeds > 1:
                per = budget / c.seeds
                lr = self.lgb_lr_for(per)
                outs = []
                for s in range(c.seeds):
                    if self.left() < self.est_lgb_time(lr) * 0.8:
                        break
                    spec = ModelSpec("lgbm", {**base, "learning_rate": lr}, target_transform=c.transform, class_weight=c.class_weight,
                                     seed=self.seed + 101 * (s + 1), name=f"{c.name}_s{s}")
                    r = self._run_spec(spec, per)
                    if r is not None:
                        outs.append(r)
                if not outs:
                    return None
                r = outs[0]
                if len(outs) > 1:
                    r.oof = np.mean([o.oof for o in outs], 0)
                    r.test = np.mean([o.test for o in outs], 0)
                    r.score = MT.safe_score(self.pb.metric, self.pb.y, r.oof)
                    LOG.info(f"[models] {c.name} ({len(outs)} seeds averaged): {self.pb.metric}={fmt_num(r.score, 6)}")
                return r
            lr = self.lgb_lr_for(budget)
            spec = ModelSpec("lgbm", {**base, "learning_rate": lr}, target_transform=c.transform, class_weight=c.class_weight,
                             seed=self.seed + 7, name=c.name)
            return self._run_spec(spec, budget)
        if fam == "cat":
            p = dict(c.params)
            if self.state.n_train < 5000:
                p.setdefault("depth", 6)
            per_iter = self.probe("cat", {**p, "learning_rate": 0.1}, iters=12 if self.pb.task == "multiclass" else 25)
            budget = max(10.0, budget - (time.time() - self._t_start_c))
            if self.affordable_iters(per_iter, budget) < 120:
                LOG.info(f"[models] skip {c.name}: only ~{self.affordable_iters(per_iter, budget):.0f} iterations affordable")
                return None
            lr = self.lr_from_probe("cat", per_iter, budget, 0.03, 0.3)
            spec = ModelSpec("cat", {**p, "learning_rate": lr, "early_stopping": int(max(50, 8 / lr))},
                             target_transform=c.transform, class_weight=c.class_weight, seed=self.seed + 3, name=c.name)
            return self._run_spec(spec, budget)
        if fam == "xgb":
            params = dict(self.tuned.get("xgb", {}))
            if c.tune:
                hpo_budget = budget * 0.6
                if hpo_budget < max(60.0, self.fold_time_proxy * 10):
                    return None
                best, hist = tune("xgb", self.pb, self.state, self.folds, list(range(min(3, len(self.folds)))), self.n_jobs,
                                  hpo_budget, {"eta": 0.1}, self.seed, self.tag)
                params = {k: v for k, v in best.items() if k != "eta"}
                self.tuned["xgb"] = params
                self.log.append({"hpo": "xgb", "trials": len(hist), "best": params})
            per_iter = self.probe("xgb", {**params, "eta": 0.1}, iters=10 if self.pb.task == "multiclass" else 20)
            budget = max(10.0, budget - (time.time() - self._t_start_c))
            if self.affordable_iters(per_iter, budget) < 100:
                LOG.info(f"[models] skip {c.name}: only ~{self.affordable_iters(per_iter, budget):.0f} iterations affordable")
                return None
            eta = self.lr_from_probe("xgb", per_iter, budget, 0.02, 0.3)
            spec = ModelSpec("xgb", {**params, "eta": eta}, target_transform=c.transform, seed=self.seed + 5, name=c.name)
            return self._run_spec(spec, budget)
        if fam == "ag":
            if budget < 60 * len(self.folds):
                LOG.info(f"[models] skip autogluon: budget {budget:.0f}s too small")
                return None
            # one AutoGluon fit per outer fold (honest OOF w.r.t. our fold-wise target encoding)
            spec = ModelSpec("ag", {"work_dir": str(self.work_dir), "bag_folds": 3}, seed=self.seed + 17, name=c.name)
            return self._run_spec(spec, min(budget, self.left() - 20))
        if fam == "mlp":
            spec = ModelSpec("mlp", dict(c.params), seed=self.seed + 11, name=c.name)
            return self._run_spec(spec, budget)
        if fam in ("et", "linear"):
            spec = ModelSpec(fam, dict(c.params), seed=self.seed + 13, name=c.name)
            return self._run_spec(spec, budget)
        return None

    # ------------------------------------------------------------------ ensembling
    def update_ensemble(self, final: bool = False):
        pb = self.pb
        res = [r for r in self.results if r is not None and r.complete and r.test is not None and np.all(np.isfinite(r.oof))]
        if not res:
            return
        y = pb.y
        obj = blend_objective(pb.metric, pb.task)
        options = []
        for r in res:
            if "single_option" not in r.extra:
                dec = self._decision(r.oof)
                sc = metric_with_decision(pb.metric, y, r.oof, dec, pb.n_classes)
                r.extra["single_option"] = FinalPrediction(MT.utility(pb.metric, sc), sc, r.test, r.oof, dec, {r.name: 1.0}, "single")
            options.append(r.extra["single_option"])
        if len(res) >= 2:
            preds = [r.oof for r in res]
            w, _ = hill_climb(preds, y, obj)
            oof = sum(wi * p for wi, p in zip(w, preds))
            test = sum(wi * r.test for wi, r in zip(w, res))
            dec = self._decision(oof)
            sc = metric_with_decision(pb.metric, y, oof, dec, pb.n_classes)
            members = {r.name: round(float(wi), 4) for r, wi in zip(res, w) if wi > 0}
            options.append(FinalPrediction(MT.utility(pb.metric, sc), sc, test, oof, dec, members, "hill_climb"))
            # label metrics (F1, accuracy, kappa...): optimise the blend for the metric itself, decision rule in the loop
            if final and pb.task != "regression" and MT.get(pb.metric).needs == "label" and self.left() > 60:
                opt = self._metric_hill_climb(res, min(300.0, self.left() * 0.3))
                if opt is not None:
                    options.append(opt)
            # stacking is costlier: run it every third model and at the end
            do_stack = final or (len(res) >= 3 and len(res) % 3 == 0)
            if do_stack and len(res) >= 3 and (pb.task != "multiclass" or pb.n_classes * len(res) <= 400):
                st = None
                if self.left() > 90:
                    st = stack_level2([r.oof for r in res], [r.test for r in res], y, pb.task, pb.n_classes, self.folds, obj, self.seed,
                                      time_budget=min(120.0, self.left() * 0.15))
                if st is not None:
                    oof2, test2, info = st
                    dec = self._decision(oof2)
                    sc = metric_with_decision(pb.metric, y, oof2, dec, pb.n_classes)
                    u = MT.utility(pb.metric, sc)
                    # the stacker has more freedom: require a margin over blending
                    options.append(FinalPrediction(u - 2e-4 * max(abs(u), 1e-9), sc, test2, oof2, dec, {"stack": info}, "stacking"))
        # winner's curse: a single model picked as the max of many noisy OOF scores (and a free-form stacker)
        # is optimistic; penalise by a fraction of the fold-level standard error
        se = self._noise()
        for o in options:
            if o.method == "single" and len(res) > 1:
                o.sel = o.utility - 0.5 * se
            elif o.method == "stacking":
                o.sel = o.utility - 0.25 * se
            else:
                o.sel = o.utility
        best = max(options, key=lambda o: o.sel)
        if self.best is None or best.sel > self.best.sel + 1e-12:
            self.best = best
            mem = {k: v for k, v in best.members.items() if isinstance(v, float)}
            LOG.info(f"[ensemble] new best {best.method}: {pb.metric}={fmt_num(best.score, 6)} members={mem or '(stack)'} decision={best.decision.kind}")
            if self.on_update:
                try:
                    self.on_update(self.pb, best)
                except Exception as e:
                    LOG.warning(f"on_update failed: {e}")

    def _metric_hill_climb(self, res, budget: float):
        """Caruana selection where every candidate blend is scored by the target metric after re-fitting the
        decision rule (class weights / threshold) - the quantity that is actually submitted."""
        from mla_postprocess import _fit_threshold, _fit_weights
        pb = self.pb
        y = pb.y
        K = pb.n_classes
        t_end = time.time() + budget

        def score(P):
            if pb.task == "binary":
                prm = _fit_threshold(pb.metric, y, P)
                d = Decision("threshold", prm)
            else:
                prm = _fit_weights(pb.metric, y, P, n_pass=1)
                d = Decision("weights", prm)
            return MT.utility(pb.metric, metric_with_decision(pb.metric, y, P, d, K))

        preds = [r.oof for r in res]
        base = [r.extra["single_option"].utility if "single_option" in r.extra else score(p) for r, p in zip(res, preds)]
        counts = np.zeros(len(res))
        counts[int(np.argmax(base))] = 1
        cur = preds[int(np.argmax(base))].copy()
        best = max(base)
        for _ in range(25):
            n = counts.sum()
            cand_best, cand_i = best, -1
            for i, p in enumerate(preds):
                if time.time() > t_end:
                    break
                sc = score((cur * n + p) / (n + 1))
                if sc > cand_best + 1e-9:
                    cand_best, cand_i = sc, i
            if cand_i < 0 or time.time() > t_end:
                break
            counts[cand_i] += 1
            cur = (cur * n + preds[cand_i]) / (n + 1)
            best = cand_best
        if counts.sum() < 2:
            return None
        w = counts / counts.sum()
        oof = sum(wi * p for wi, p in zip(w, preds))
        test = sum(wi * r.test for wi, r in zip(w, res))
        dec = self._decision(oof)
        sc = metric_with_decision(pb.metric, y, oof, dec, K)
        members = {r.name: round(float(wi), 4) for r, wi in zip(res, w) if wi > 0}
        LOG.info(f"[ensemble] metric-direct blend: {pb.metric}={fmt_num(sc, 6)} members={members}")
        return FinalPrediction(MT.utility(pb.metric, sc), sc, test, oof, dec, members, "metric_blend")

    def _noise(self) -> float:
        """Standard error of the metric across folds for the strongest single model."""
        vals = []
        for r in self.results:
            fs = [MT.utility(self.pb.metric, v) for v in (r.fold_scores or []) if np.isfinite(v)]
            if len(fs) >= 3:
                vals.append(np.std(fs, ddof=1) / np.sqrt(len(fs)))
        return float(np.median(vals)) if vals else 0.0

    def _decision(self, oof) -> Decision:
        pb = self.pb
        rng = (float(np.min(pb.y)), float(np.max(pb.y))) if pb.task == "regression" else None
        return fit_decision(pb.metric, pb.task, pb.y, oof, pb.n_classes, y_range=rng, nonneg=pb.nonneg, seed=self.seed)

    # ------------------------------------------------------------------ main
    def run(self):
        queue = self.plan()
        LOG.info(f"[models] plan: {[c.name for c in queue]}; {self.left():.0f}s available")
        while queue:
            left = self.left()
            if left < 15:
                break
            c = queue.pop(0)
            wsum = c.weight + sum(q.weight for q in queue)
            budget = left * c.weight / wsum
            # skip heavy candidates that cannot reasonably finish
            if c.family in ("lgbm",) and self.est_lgb_time(0.2) > budget * 1.5:
                LOG.info(f"[models] skip {c.name}: budget {budget:.0f}s too small")
                continue
            if c.family in ("cat", "xgb") and self.est_lgb_time(0.2) * 1.5 > budget * 1.5:
                LOG.info(f"[models] skip {c.name}: budget {budget:.0f}s too small")
                continue
            self._t_start_c = time.time()
            try:
                r = self.run_candidate(c, budget)
            except Exception as e:
                LOG.warning(f"[models] candidate {c.name} failed: {type(e).__name__}: {e}")
                r = None
            if r is not None and c.bag_of:
                tgt = next((x for x in self.results if x.name == c.bag_of), None)
                if tgt is not None and tgt.oof.shape == r.oof.shape and tgt.test is not None and r.test is not None:
                    nb = tgt.extra.get("n_bag", 1)
                    tgt.oof = (tgt.oof * nb + r.oof) / (nb + 1)
                    tgt.test = (tgt.test * nb + r.test) / (nb + 1)
                    tgt.extra["n_bag"] = nb + 1
                    tgt.extra.pop("single_option", None)
                    tgt.score = MT.safe_score(self.pb.metric, self.pb.y, tgt.oof)
                    LOG.info(f"[models] {tgt.name}: bagged {nb + 1} seeds -> {self.pb.metric}={fmt_num(tgt.score, 6)}")
                    self.log.append({"model": tgt.name, "bagged": nb + 1, "score": tgt.score})
                    self.update_ensemble()
            elif r is not None:
                r.name = c.name
                self.results.append(r)
                self.log.append({"model": c.name, "score": r.score, "elapsed": round(r.elapsed, 1), "iters": r.best_iters})
                self.update_ensemble()
            if not queue and self.left() > 60 and self.best is not None and len(self.results) < 25:
                queue = self.refill()
        try:
            self.update_ensemble(final=True)
        except Exception as e:
            LOG.warning(f"final ensemble failed: {e}")
        return self.best

    def refill(self) -> list[Candidate]:
        """When the plan is exhausted early: more seeds of the families that carry the ensemble."""
        fams = {}
        for r in self.results:
            fams.setdefault(r.spec.family, []).append(r)
        mem = self.best.members if self.best else {}
        top = sorted(((v, k) for k, v in mem.items() if isinstance(v, float)), reverse=True)
        if not top:  # stacking (or anything without weights): use the best single models
            singles = [(r.extra["single_option"].utility, r.name) for r in self.results if "single_option" in r.extra]
            top = sorted(singles, reverse=True)
        out = []
        for _, name in top[:3]:
            r = next((x for x in self.results if x.name == name), None)
            if r is None or r.spec.family not in ("lgbm", "cat", "xgb", "mlp") or r.extra.get("n_bag", 1) >= 6:
                continue
            out.append(Candidate(f"{name}_bag", r.spec.family, 10, params={k: v for k, v in r.spec.params.items()
                                                                          if k not in ("learning_rate", "eta", "n_estimators", "early_stopping")},
                                 transform=r.spec.target_transform, class_weight=r.spec.class_weight, bag_of=name))
            if len(out) >= 2:
                break
        if out:
            LOG.info(f"[models] refill with {[c.name for c in out]}")
            self.seed += 1
        return out
