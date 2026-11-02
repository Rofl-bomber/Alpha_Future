"""Feature state, fold-aware target encoding, deterministic ("auto") feature hypotheses."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from mla_preprocess import finalize_types, make_safe_names, to_category
from mla_utils import LOG, short_hash

MAX_FEATURES = 600


@dataclass
class FeatureState:
    X: pd.DataFrame  # train rows first, then test rows
    n_train: int
    cat_cols: list
    te_specs: list = field(default_factory=list)  # list[tuple[str, ...]]
    blocks: list = field(default_factory=list)  # accepted block descriptions
    version: str = "base"

    def copy(self) -> "FeatureState":
        return FeatureState(self.X.copy(), self.n_train, list(self.cat_cols), list(self.te_specs), list(self.blocks), self.version)

    @property
    def n_features(self) -> int:
        return self.X.shape[1] + sum(1 for _ in self.te_specs)

    def key(self) -> str:
        return short_hash(self.version, tuple(self.X.columns), tuple(self.te_specs))

    def describe_columns(self, max_items: int = 400) -> list[str]:
        out = []
        for c in list(self.X.columns)[:max_items]:
            out.append(f"{c}:{'cat' if c in self.cat_cols else 'num'}")
        return out


def apply_new_frame(state: FeatureState, new_X: pd.DataFrame, block: dict) -> tuple[FeatureState, dict]:
    """Install a transformed frame as a new state (types normalized, names sanitized)."""
    new_X = new_X.reset_index(drop=True)
    old_cols = set(state.X.columns)
    # sanitize new column names
    rename = {}
    new_cols = [c for c in new_X.columns if c not in old_cols]
    if new_cols:
        safe = make_safe_names([str(c) for c in new_cols], reserved={c.lower() for c in old_cols})
        rename = {c: safe[str(c)] for c in new_cols if safe[str(c)] != c}
        if rename:
            new_X = new_X.rename(columns=rename)
    new_X = new_X.loc[:, ~new_X.columns.duplicated()]
    new_X, cats = finalize_types(new_X, known_cats=set(state.cat_cols))
    # drop useless new columns (constant / all NaN)
    info = {"added": [], "removed": [c for c in state.X.columns if c not in new_X.columns], "modified": [], "dropped_useless": []}
    for c in list(new_X.columns):
        if c not in old_cols:
            s = new_X[c]
            if s.nunique(dropna=True) <= 1:
                new_X = new_X.drop(columns=[c])
                info["dropped_useless"].append(c)
                continue
            info["added"].append(c)
        else:
            try:
                a, b = state.X[c], new_X[c]
                if a.dtype != b.dtype or not a.equals(b):
                    info["modified"].append(c)
            except Exception:
                info["modified"].append(c)
    if new_X.shape[1] > MAX_FEATURES:
        raise ValueError(f"too many features after transform: {new_X.shape[1]} > {MAX_FEATURES}")
    cat_cols = [c for c in new_X.columns if c in cats]
    te_specs = [t for t in state.te_specs if all(c in new_X.columns for c in t)]
    ns = FeatureState(new_X, state.n_train, cat_cols, te_specs, state.blocks + [block], short_hash(state.version, block.get("id")))
    return ns, info


def with_te(state: FeatureState, specs: list, block: dict) -> FeatureState:
    # X is not modified: share it instead of copying (memory on large data)
    ns = FeatureState(state.X, state.n_train, list(state.cat_cols), list(state.te_specs), list(state.blocks), state.version)
    for s in specs:
        t = tuple(s)
        if all(c in ns.X.columns for c in t) and t not in ns.te_specs:
            ns.te_specs.append(t)
    ns.blocks.append(block)
    ns.version = short_hash(state.version, block.get("id"))
    return ns


def drop_features(state: FeatureState, cols: list, block: dict) -> FeatureState:
    ns = state.copy()
    cols = [c for c in cols if c in ns.X.columns]
    if len(cols) >= ns.X.shape[1]:
        raise ValueError("cannot drop all features")
    ns.X = ns.X.drop(columns=cols)
    ns.cat_cols = [c for c in ns.cat_cols if c not in cols]
    ns.te_specs = [t for t in ns.te_specs if not any(c in cols for c in t)]
    ns.blocks.append(block)
    ns.version = short_hash(state.version, block.get("id"))
    return ns


def cast_categorical(state: FeatureState, cols: list, block: dict) -> FeatureState:
    ns = state.copy()
    for c in cols:
        if c in ns.X.columns and c not in ns.cat_cols:
            s = ns.X[c]
            ns.X[c] = to_category(s.map(lambda z: np.nan if pd.isna(z) else (str(int(z)) if float(z).is_integer() else str(z))))
            ns.cat_cols.append(c)
    ns.blocks.append(block)
    ns.version = short_hash(state.version, block.get("id"))
    return ns


# ---------------------------------------------------------------------------------------------
# Fold-aware target encoding
# ---------------------------------------------------------------------------------------------
def key_codes(X: pd.DataFrame, cols: tuple) -> np.ndarray:
    if len(cols) == 1:
        s = X[cols[0]]
        if isinstance(s.dtype, pd.CategoricalDtype):
            codes = s.cat.codes.values.astype(np.int64)
            return codes  # -1 = NaN, treated as its own level below
        return pd.factorize(s, use_na_sentinel=True)[0].astype(np.int64)
    parts = []
    for c in cols:
        s = X[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            parts.append(s.cat.codes.values.astype(np.int64))
        else:
            parts.append(pd.factorize(s, use_na_sentinel=True)[0].astype(np.int64))
    df = pd.DataFrame(np.stack(parts, 1))
    return df.groupby(list(df.columns), sort=False).ngroup().values.astype(np.int64)


def _te_stats(codes_fit, y_fit, n_levels, task, n_classes):
    """Return (count[level], sum[level, out_dims])."""
    cnt = np.bincount(codes_fit, minlength=n_levels).astype(float)
    if task == "regression":
        s = np.bincount(codes_fit, weights=y_fit, minlength=n_levels)[:, None]
    elif task == "binary":
        s = np.bincount(codes_fit, weights=(y_fit == 1).astype(float), minlength=n_levels)[:, None]
    else:
        s = np.stack([np.bincount(codes_fit, weights=(y_fit == k).astype(float), minlength=n_levels) for k in range(n_classes)], 1)
    return cnt, s


def _te_apply(codes, cnt, s, prior, m):
    c = cnt[codes]
    out = (s[codes] + prior[None, :] * m) / (c[:, None] + m)
    return out, c


def te_for_fold(codes_all: np.ndarray, y_tr: np.ndarray, tr_idx: np.ndarray, apply_idx: list, task: str, n_classes: int,
                smoothing: float = 10.0, inner_splits: int = 5, seed: int = 0):
    """Encode one key: OOF inside tr_idx, full-train-part stats for each array in apply_idx."""
    codes = codes_all.copy()
    codes[codes < 0] = codes.max() + 1  # NaN level
    n_levels = int(codes.max()) + 1
    if task == "regression":
        prior = np.array([float(np.mean(y_tr))])
    elif task == "binary":
        prior = np.array([float(np.mean(y_tr == 1))])
    else:
        prior = np.bincount(y_tr.astype(int), minlength=n_classes).astype(float) / len(y_tr)
    ctr = codes[tr_idx]
    enc_tr = np.zeros((len(tr_idx), len(prior)))
    cnt_tr = np.zeros(len(tr_idx))
    kf = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    for a, b in kf.split(tr_idx):
        cnt, s = _te_stats(ctr[a], y_tr[a], n_levels, task, n_classes)
        enc_tr[b], cnt_tr[b] = _te_apply(ctr[b], cnt, s, prior, smoothing)
    cnt, s = _te_stats(ctr, y_tr, n_levels, task, n_classes)
    outs = []
    for idx in apply_idx:
        e, c = _te_apply(codes[idx], cnt, s, prior, smoothing)
        outs.append((e, c))
    return (enc_tr, cnt_tr), outs


class TECache:
    """Caches per-fold TE matrices for a given (state key, folds id)."""

    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def put(self, key, val):
        if len(self.store) > 64:
            self.store.pop(next(iter(self.store)))
        self.store[key] = val


TE_CACHE = TECache()


def te_names(spec: tuple, task: str, n_classes: int) -> list[str]:
    base = "te_" + "__".join(spec)
    if len(base) > 60:
        base = "te_" + short_hash(*spec)
    if task in ("regression", "binary"):
        return [base, base + "_n"]
    return [f"{base}_c{k}" for k in range(n_classes)] + [base + "_n"]


def build_te_block(state: FeatureState, y: np.ndarray, task: str, n_classes: int, tr_idx: np.ndarray, va_idx: np.ndarray,
                   test_idx: np.ndarray | None, fold_id: str, seed: int = 0):
    """Return (tr_df, va_df, te_df) of target-encoded features for one fold (empty frames if no specs)."""
    if not state.te_specs:
        return None
    ck = (state.key(), id(state.X), state.X.shape, fold_id, len(tr_idx), int(tr_idx[:50].sum()), task)
    hit = TE_CACHE.get(ck)
    if hit is not None:
        return hit
    tr_parts, va_parts, te_parts = {}, {}, {}
    y_tr = y[tr_idx]
    for spec in state.te_specs:
        codes = key_codes(state.X, spec)
        apply = [va_idx] + ([test_idx] if test_idx is not None else [])
        (etr, ctr), outs = te_for_fold(codes, y_tr, tr_idx, apply, task, n_classes, seed=seed)
        names = te_names(spec, task, n_classes)
        for j, nm in enumerate(names[:-1]):
            tr_parts[nm] = etr[:, j]
            va_parts[nm] = outs[0][0][:, j]
            if test_idx is not None:
                te_parts[nm] = outs[1][0][:, j]
        tr_parts[names[-1]] = ctr
        va_parts[names[-1]] = outs[0][1]
        if test_idx is not None:
            te_parts[names[-1]] = outs[1][1]
    res = (pd.DataFrame(tr_parts), pd.DataFrame(va_parts), pd.DataFrame(te_parts) if test_idx is not None else None)
    TE_CACHE.put(ck, res)
    return res


# ---------------------------------------------------------------------------------------------
# Deterministic hypotheses
# ---------------------------------------------------------------------------------------------
def _cats(state, min_card=3):
    return [c for c in state.cat_cols if state.X[c].nunique(dropna=True) >= min_card]


MAX_TE_CLASSES = 20


def _codes(s: pd.Series) -> np.ndarray:
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s.cat.codes.values.astype(np.int64)
    return pd.factorize(s, use_na_sentinel=True)[0].astype(np.int64)


def discover_entity_key(state: FeatureState, max_cols: int = 6) -> tuple[list, dict] | None:
    """Find columns that jointly identify a recurring real-world entity (e.g. the same building in different
    years): high-cardinality attributes whose values repeat across train and test and stay constant within
    the entity.  Greedy: start from the most specific column, add columns that barely split the groups."""
    X = state.X
    n = len(X)
    ntr = state.n_train
    if n < 200 or ntr < 100 or n - ntr < 10:
        return None
    cands = []
    for c in X.columns:
        s = X[c]
        nun = s.nunique(dropna=True)
        if nun < max(30, 0.002 * n) or nun > 0.9 * n:
            continue
        codes = _codes(s)
        tr_set = set(codes[:ntr][codes[:ntr] >= 0])
        te = codes[ntr:]
        te = te[te >= 0]
        if len(te) == 0:
            continue
        overlap = np.isin(te, list(tr_set)).mean()
        if overlap < 0.5:
            continue
        cands.append((nun, c, codes))
    if not cands:
        return None
    cands.sort(key=lambda t: -t[0])
    _, first, key = cands[0]
    key = key.copy()
    cols = [first]

    def group_size(k):
        return n / max(1, len(np.unique(k)))

    g = group_size(key)
    if g < 1.3:
        return None
    others = [(c, codes) for _, c, codes in cands[1:]]
    # low-cardinality categoricals can also be entity attributes (type, region)
    for c in state.cat_cols:
        if c not in cols and c not in [o[0] for o in others] and 2 <= X[c].nunique() < 30:
            others.append((c, _codes(X[c])))
    for c, codes in others:
        if len(cols) >= max_cols:
            break
        combo = pd.DataFrame({"a": key, "b": codes}).groupby(["a", "b"], sort=False).ngroup().values
        g2 = group_size(combo)
        if g2 >= 0.95 * g and g2 >= 1.3:
            key, g = combo, g2
            cols.append(c)
    if len(cols) < 2 and g > 50:
        return None
    te_overlap = np.isin(key[ntr:], np.unique(key[:ntr])).mean()
    if g > 200 or te_overlap < 0.2:
        return None
    return cols, {"mean_group_size": round(g, 2), "test_overlap": round(float(te_overlap), 3)}


def detect_period(state: FeatureState) -> str | None:
    """A low-cardinality integer column whose test values are new and later than every train value
    (e.g. Year_Factor 1..5 in train, 6 in test): the test is a future period."""
    X, n = state.X, state.n_train
    best = None
    for c in X.columns:
        if c in state.cat_cols:
            continue
        s = X[c]
        tr, te = s.iloc[:n].dropna(), s.iloc[n:].dropna()
        if len(tr) < 50 or len(te) < 10:
            continue
        nun = tr.nunique()
        if not (2 <= nun <= 40) or not np.all(np.mod(tr.values[:5000], 1) == 0):
            continue
        new_frac = float((~te.isin(set(tr.unique()))).mean())
        later = float((te > tr.max()).mean())
        if new_frac >= 0.9 and later >= 0.9:
            score = new_frac + later
            if best is None or score > best[0]:
                best = (score, c)
    return best[1] if best else None


def entity_history(key: pd.Series, period: np.ndarray, y_train: np.ndarray, n_train: int) -> pd.DataFrame:
    """Per row: target statistics of the same entity in EARLIER periods only (never the row itself).
    Test rows (a later period) see the full train history - exactly what is known at prediction time."""
    n = len(key)
    rows = pd.DataFrame({"key": key.values, "period": period.astype(float), "pos": np.arange(n)})
    tr = pd.DataFrame({"key": key.values[:n_train], "period": period[:n_train].astype(float), "y": y_train})
    agg = tr.groupby(["key", "period"], sort=False)["y"].agg(["sum", "count", "mean"]).reset_index().sort_values(["key", "period"])
    agg["cum_sum"] = agg.groupby("key")["sum"].cumsum()
    agg["cum_cnt"] = agg.groupby("key")["count"].cumsum()
    agg = agg.sort_values("period")
    r = rows.dropna(subset=["period"]).sort_values("period")
    m = pd.merge_asof(r, agg[["key", "period", "mean", "cum_sum", "cum_cnt"]].rename(columns={"period": "p_prev"}),
                      left_on="period", right_on="p_prev", by="key", allow_exact_matches=False, direction="backward")
    out = pd.DataFrame(index=np.arange(n))
    out["hist_last"] = np.nan
    out["hist_mean"] = np.nan
    out["hist_n"] = 0.0
    out["hist_gap"] = np.nan
    pos = m["pos"].values
    out.loc[pos, "hist_last"] = m["mean"].values
    out.loc[pos, "hist_mean"] = (m["cum_sum"] / m["cum_cnt"]).values
    out.loc[pos, "hist_n"] = m["cum_cnt"].fillna(0).values
    out.loc[pos, "hist_gap"] = (m["period"] - m["p_prev"]).values
    return out


def auto_hypotheses(state: FeatureState, importance: pd.Series | None, task: str, n_classes: int = 0,
                    y: np.ndarray | None = None) -> list[dict]:
    """Ordered list of cheap, generally-useful hypotheses (no LLM needed)."""
    X = state.X
    allow_te = not (task == "multiclass" and n_classes > MAX_TE_CLASSES)
    n = len(X)
    hyps = []
    cats = _cats(state)
    imp = importance if importance is not None else pd.Series(dtype=float)
    top_feats = [c for c in imp.sort_values(ascending=False).index if c in X.columns] if len(imp) else list(X.columns)
    top_cats = [c for c in top_feats if c in cats][:4] or cats[:4]
    num_cols = [c for c in X.columns if c not in state.cat_cols]
    top_nums = [c for c in top_feats if c in num_cols][:6] or num_cols[:6]

    try:
        ek = discover_entity_key(state)
    except Exception as e:
        LOG.debug(f"entity key discovery failed: {e}")
        ek = None
    if ek is not None:
        cols, info = ek
        hyps.append({"id": "auto_entity_key", "type": "auto", "op": "entity_key", "cols": cols, "te_after": allow_te,
                     "title": f"Entity key {'+'.join(cols)} (mean {info['mean_group_size']} rows/entity, {100 * info['test_overlap']:.0f}% of test entities seen in train) with count + target encoding",
                     "rationale": "Rows sharing these attributes look like the same real-world entity; its other rows reveal its typical target."})
    period = None
    try:
        period = detect_period(state)
    except Exception as e:
        LOG.debug(f"period detection failed: {e}")
    if period is not None:
        LOG.info(f"Test rows lie in a later, unseen period of '{period}'")
        if ek is not None and y is not None and task in ("regression", "binary"):
            hyps.append({"id": "auto_entity_history", "type": "auto", "op": "entity_history", "cols": ek[0], "period": period, "temporal": True,
                         "period_values": X[period].values.astype(float).copy(), "y": np.asarray(y, float).copy(),
                         "title": f"Entity history: target of the same {'+'.join(ek[0])} in earlier '{period}' periods (last, mean, count, gap)",
                         "rationale": "The test is a later period; the strongest predictor of an entity's next value is its own past values."})
        hyps.append({"id": "auto_drop_period", "type": "drop_features", "columns": [period], "temporal": True,
                     "period_values": X[period].values.astype(float).copy(),
                     "title": f"Drop '{period}': test periods never occur in train",
                     "rationale": "Trees cannot learn anything about an unseen future period; the column only extrapolates as the last one."})
    if cats:
        hyps.append({"id": "auto_count_enc", "type": "auto", "title": "Frequency (count) encoding of categorical features",
                     "rationale": "Category frequency is a strong, leak-free signal for trees.", "op": "count_encoding", "cols": cats[:30]})
    hc = [c for c in cats if X[c].nunique() > 30]
    if hc and allow_te:
        hyps.append({"id": "auto_te_highcard", "type": "target_encoding", "title": "Target encoding of high-cardinality categoricals",
                     "rationale": "Native categorical splits overfit on many levels; smoothed out-of-fold target means generalize better.",
                     "columns": [[c] for c in hc[:12]]})
    if cats and allow_te:
        hyps.append({"id": "auto_te_all", "type": "target_encoding", "title": "Target encoding of all categoricals",
                     "rationale": "Adds smoothed target statistics per category.", "columns": [[c] for c in cats[:15] if c not in hc[:12]]})
    if len(top_cats) >= 2 and allow_te:
        pairs = [list(p) for p in itertools.combinations(top_cats[:4], 2)]
        hyps.append({"id": "auto_te_pairs", "type": "target_encoding", "title": "Target encoding of pairs of top categoricals",
                     "rationale": "Captures interactions between the most important categorical features.", "columns": pairs})
    miss = X.iloc[: state.n_train].isna().mean()
    if (miss > 0.01).sum() >= 2:
        hyps.append({"id": "auto_nan_count", "type": "auto", "title": "Row-wise missing-value count", "rationale": "Missingness patterns often carry signal.",
                     "op": "nan_count"})
    hyps.append({"id": "auto_dup_count", "type": "auto", "title": "Count of identical feature rows",
                 "rationale": "Repeated identical records (e.g. scans, duplicated entities) are informative.", "op": "dup_count"})
    try:
        h_ = pd.util.hash_pandas_object(X, index=False).values
        tr_h, te_h = h_[: state.n_train], h_[state.n_train:]
        dup_frac = 1.0 - len(np.unique(tr_h)) / max(1, len(tr_h))
        te_seen = float(np.isin(te_h, tr_h).mean()) if len(te_h) else 0.0
        if dup_frac > 0.05 and te_seen > 0.05 and allow_te:
            hyps.append({"id": "auto_row_key", "type": "auto", "op": "row_key", "te_after": True, "key_col": "row_key",
                         "title": f"Target encoding of exact duplicate feature vectors ({100 * dup_frac:.0f}% duplicated train rows, {100 * te_seen:.0f}% of test rows seen in train)",
                         "rationale": "Identical records share their label distribution; a direct lookup beats tree approximation."})
    except Exception:
        pass
    if top_cats and top_nums:
        hyps.append({"id": "auto_group_stats", "type": "auto", "title": "Group statistics of top numeric features by top categorical features",
                     "rationale": "Deviation of a value from its group norm is a classic strong feature.", "op": "group_stats",
                     "cats": top_cats[:2], "nums": top_nums[:4]})
    int_low = [c for c in num_cols if 3 <= X[c].nunique() <= 30 and np.all(np.mod(X[c].dropna().values[:5000], 1) == 0)]
    if int_low:
        hyps.append({"id": "auto_num_as_cat", "type": "categorical_cast", "title": "Treat low-cardinality integer columns as categorical",
                     "rationale": "Integer codes may be nominal categories.", "columns": int_low[:15]})
    if len(top_nums) >= 3:
        hyps.append({"id": "auto_num_pairs", "type": "auto", "title": "Ratios and differences of top numeric features",
                     "rationale": "Trees approximate ratios poorly; explicit ratios of strong features help.", "op": "num_pairs", "nums": top_nums[:5]})
    if len(imp) and (imp <= 0).sum() >= 3:
        zero = [c for c in imp.index if imp[c] <= 0 and c in X.columns]
        hyps.append({"id": "auto_drop_zero_imp", "type": "drop_features", "title": "Drop features with zero importance",
                     "rationale": "Removes noise and speeds up training.", "columns": zero})
    return hyps


def apply_auto_op(state: FeatureState, h: dict) -> pd.DataFrame:
    """Compute the frame for an 'auto' operation (returns full new X)."""
    X = state.X.copy()
    op = h.get("op")
    if op == "count_encoding":
        for c in h["cols"]:
            if c in X.columns:
                vc = X[c].astype("object").value_counts(dropna=False)
                X[f"cnt_{c}"] = X[c].astype("object").map(vc).astype(float).values
    elif op == "nan_count":
        X["row_nan_count"] = X.isna().sum(1).astype(float)
    elif op == "dup_count":
        h_ = pd.util.hash_pandas_object(X, index=False)
        X["dup_row_count"] = h_.map(h_.value_counts()).astype(float).values
    elif op == "group_stats":
        for g in h["cats"]:
            for v in h["nums"]:
                if g in X.columns and v in X.columns:
                    grp = X.groupby(g, observed=True)[v]
                    mean = grp.transform("mean")
                    std = grp.transform("std")
                    X[f"{v}_mean_by_{g}"] = mean.astype(float)
                    X[f"{v}_dev_by_{g}"] = (X[v] - mean).astype(float)
                    X[f"{v}_z_by_{g}"] = ((X[v] - mean) / std.replace(0, np.nan)).astype(float)
    elif op == "row_key":
        h_ = pd.util.hash_pandas_object(state.X, index=False)
        X["row_key"] = h_.astype(str).values
        X["row_key_count"] = h_.map(h_.value_counts()).astype(float).values
    elif op == "entity_key":
        cols = [c for c in h["cols"] if c in X.columns]
        parts = [X[c].astype("object").where(X[c].notna(), "nan").astype(str) for c in cols]
        key = parts[0]
        for p_ in parts[1:]:
            key = key + "|" + p_
        X["entity_key"] = key.values
        X["entity_count"] = key.map(key.value_counts()).astype(float).values
    elif op == "entity_history":
        cols = [c for c in h["cols"] if c in X.columns]
        parts = [X[c].astype("object").where(X[c].notna(), "nan").astype(str) for c in cols]
        key = parts[0]
        for p_ in parts[1:]:
            key = key + "|" + p_
        hist = entity_history(key.reset_index(drop=True), h["period_values"], h["y"], state.n_train)
        for c in hist.columns:
            X[c] = hist[c].values
        X["hist_last_minus_mean"] = X["hist_last"] - X["hist_mean"]
    elif op == "num_pairs":
        nums = [c for c in h["nums"] if c in X.columns]
        for a, b in itertools.combinations(nums, 2):
            X[f"{a}_div_{b}"] = (X[a] / X[b].replace(0, np.nan)).astype(float)
            X[f"{a}_minus_{b}"] = (X[a] - X[b]).astype(float)
    else:
        raise ValueError(f"unknown auto op {op}")
    return X
