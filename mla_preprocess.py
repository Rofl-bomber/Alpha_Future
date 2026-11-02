"""Base (target-free) preprocessing: typing, dates, text, categoricals, safe column names."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mla_profiling import looks_like_date, looks_like_text
from mla_utils import LOG

_TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i", "й": "y",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
    "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
})

MAX_NATIVE_CAT = 200  # categoricals above this cardinality are "high-card"


def safe_name(name: str) -> str:
    s = str(name).strip()
    out = []
    for ch in s:
        lo = ch.lower()
        t = lo.translate(_TRANSLIT)
        if t != lo:
            out.append(t.capitalize() if ch != lo else t)
        else:
            out.append(ch)
    s = "".join(out)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s).strip("_")
    if not s:
        s = "f"
    if s[0].isdigit():
        s = "f_" + s
    return s


def make_safe_names(cols: list[str], reserved: set | None = None) -> dict:
    used = set(reserved or ())
    out = {}
    for c in cols:
        base = safe_name(c)
        n, k = base, 1
        while n.lower() in used:
            k += 1
            n = f"{base}_{k}"
        used.add(n.lower())
        out[c] = n
    return out


def to_category(s: pd.Series) -> pd.Series:
    """Object-like series -> pandas category with string categories (NaN kept as NaN)."""
    x = s.astype("object")
    mask = x.notna()
    x = x.where(~mask, x.astype(str).str.strip())
    x = x.where(x != "", np.nan)
    return x.astype("category")


def _try_numeric(s: pd.Series) -> pd.Series | None:
    x = s.dropna().astype(str).str.strip()
    if len(x) == 0:
        return None
    sample = x.sample(min(len(x), 2000), random_state=0)
    cleaned = sample.str.replace(r"[\s ]", "", regex=True)
    num = pd.to_numeric(cleaned, errors="coerce")
    if num.notna().mean() < 0.97:
        cleaned2 = cleaned.str.replace(",", ".", regex=False)
        num2 = pd.to_numeric(cleaned2, errors="coerce")
        if num2.notna().mean() < 0.97:
            return None
        full = pd.to_numeric(s.astype(str).str.strip().str.replace(r"[\s ]", "", regex=True).str.replace(",", ".", regex=False), errors="coerce")
    else:
        full = pd.to_numeric(s.astype(str).str.strip().str.replace(r"[\s ]", "", regex=True), errors="coerce")
    full[s.isna()] = np.nan
    return full.astype(float)


@dataclass
class FeatureFrame:
    X: pd.DataFrame  # train rows first, then test rows; RangeIndex
    n_train: int
    cat_cols: list
    orig_of: dict  # safe -> original column (for base features)
    safe_of: dict  # original -> safe
    dropped: dict = field(default_factory=dict)  # original -> reason
    text_cols: list = field(default_factory=list)
    date_cols: list = field(default_factory=list)
    log: list = field(default_factory=list)

    @property
    def n_test(self) -> int:
        return len(self.X) - self.n_train

    def copy(self) -> "FeatureFrame":
        return FeatureFrame(self.X.copy(), self.n_train, list(self.cat_cols), dict(self.orig_of), dict(self.safe_of),
                            dict(self.dropped), list(self.text_cols), list(self.date_cols), list(self.log))


def _expand_dates(s: pd.Series, name: str) -> pd.DataFrame:
    d = pd.to_datetime(s, errors="coerce", utc=False)
    out = pd.DataFrame(index=s.index)
    out[f"{name}_year"] = d.dt.year.astype(float)
    out[f"{name}_month"] = d.dt.month.astype(float)
    out[f"{name}_day"] = d.dt.day.astype(float)
    out[f"{name}_dow"] = d.dt.dayofweek.astype(float)
    out[f"{name}_doy"] = d.dt.dayofyear.astype(float)
    if (d.dt.hour.fillna(0) != 0).mean() > 0.05:
        out[f"{name}_hour"] = d.dt.hour.astype(float)
    vals = d.values.astype("datetime64[s]").astype("int64").astype(float) / 86400.0
    out[f"{name}_days"] = pd.Series(np.where(d.isna().values, np.nan, vals), index=s.index)
    return out


def _text_features(s: pd.Series, name: str, n_comp: int = 16, seed: int = 0) -> pd.DataFrame:
    x = s.fillna("").astype(str)
    out = pd.DataFrame(index=s.index)
    out[f"{name}_len"] = x.str.len().astype(float)
    out[f"{name}_nwords"] = x.str.split().map(len).astype(float)
    out[f"{name}_ndigits"] = x.str.count(r"\d").astype(float)
    out[f"{name}_nupper"] = x.str.count(r"[A-ZА-Я]").astype(float)
    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        M = vec.fit_transform(x.values)
        k = int(min(n_comp, max(1, M.shape[1] - 1)))
        if M.shape[1] > 2 and k >= 1:
            Z = TruncatedSVD(n_components=k, random_state=seed).fit_transform(M)
            for i in range(Z.shape[1]):
                out[f"{name}_svd{i}"] = Z[:, i]
    except Exception as e:
        LOG.debug(f"tfidf failed for {name}: {e}")
    return out


def build_base_features(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], profile: dict,
                        cat_hints: list | None = None, ddict_info: dict | None = None,
                        reserved_names: set | None = None) -> FeatureFrame:
    cat_hints = set(cat_hints or [])
    ddict_info = ddict_info or {}
    n_train = len(train)
    both = pd.concat([train[feature_cols], test[feature_cols]], axis=0, ignore_index=True)
    safe_of = make_safe_names(feature_cols, reserved_names)
    cols, orig_of, dropped, cat_cols, text_cols, date_cols, log = {}, {}, {}, [], [], [], []
    for c in feature_cols:
        s = both[c]
        sn = safe_of[c]
        p = profile.get(c, {})
        dict_dtype = str(ddict_info.get(c, {}).get("dtype", "")).lower()
        try:
            nun = s.nunique(dropna=True)
            if nun == 0:
                dropped[c] = "all missing"
                continue
            if nun == 1:
                if s.isna().any():
                    cols[sn + "_isna"] = s.isna().astype(float)
                    orig_of[sn + "_isna"] = c
                    log.append(f"{c}: single value + NaN -> indicator")
                else:
                    dropped[c] = "constant"
                continue
            if p.get("is_id_like") and c not in cat_hints and p.get("kind") not in ("text", "datetime"):
                # unique per row: an identifier when it is a string, has an id-like name, or is a dense integer sequence
                is_num = pd.api.types.is_numeric_dtype(s)
                dense = False
                if is_num:
                    x = s.dropna()
                    dense = len(x) > 0 and (float(x.max()) - float(x.min()) + 1) <= 1.5 * len(x)
                if (not is_num) or dense or re.match(r"^(id|index|idx|row_?id|.*_id|id_.*|.*id)$", str(c), re.IGNORECASE):
                    dropped[c] = "id-like (unique per row)"
                    continue
            if pd.api.types.is_bool_dtype(s):
                cols[sn] = s.astype(float)
                orig_of[sn] = c
                continue
            if pd.api.types.is_numeric_dtype(s):
                as_cat = c in cat_hints or any(k in dict_dtype for k in ("categor", "категор", "nominal"))
                if as_cat and nun <= 5000:
                    v = s.map(lambda z: np.nan if pd.isna(z) else (str(int(z)) if float(z).is_integer() else str(z)))
                    cols[sn] = to_category(v)
                    cat_cols.append(sn)
                else:
                    x = s.astype(float).replace([np.inf, -np.inf], np.nan)
                    cols[sn] = x
                orig_of[sn] = c
                continue
            # object-like
            num = _try_numeric(s)
            if num is not None and num.nunique() > 2 and c not in cat_hints:
                cols[sn] = num
                orig_of[sn] = c
                log.append(f"{c}: parsed as numeric")
                continue
            kind = p.get("kind")
            if kind == "datetime" or (kind is None and looks_like_date(s)):
                df = _expand_dates(s, sn)
                for k in df.columns:
                    if df[k].nunique(dropna=True) > 1:
                        cols[k] = df[k]
                        orig_of[k] = c
                date_cols.append(c)
                log.append(f"{c}: datetime expanded")
                continue
            if kind == "text" or (kind is None and looks_like_text(s)):
                df = _text_features(s, sn)
                for k in df.columns:
                    cols[k] = df[k]
                    orig_of[k] = c
                text_cols.append(c)
                log.append(f"{c}: text -> {df.shape[1]} features")
                # keep the text also as a categorical when its cardinality is moderate (e.g. repeated strings)
                if s.nunique() < 0.5 * len(s):
                    cols[sn] = to_category(s)
                    cat_cols.append(sn)
                    orig_of[sn] = c
                continue
            cols[sn] = to_category(s)
            cat_cols.append(sn)
            orig_of[sn] = c
        except Exception as e:
            LOG.warning(f"preprocess failed for column {c}: {e}; keeping as category")
            try:
                cols[sn] = to_category(s.astype(str))
                cat_cols.append(sn)
                orig_of[sn] = c
            except Exception:
                dropped[c] = f"unprocessable: {e}"
    X = pd.DataFrame(cols)
    X.index = pd.RangeIndex(len(X))
    for k in X.columns:
        if k not in cat_cols and X[k].dtype != float:
            X[k] = X[k].astype(float)
    ff = FeatureFrame(X, n_train, cat_cols, orig_of, safe_of, dropped, text_cols, date_cols, log)
    if dropped:
        LOG.info(f"Dropped base columns: {dropped}")
    return ff


def finalize_types(X: pd.DataFrame, known_cats: set | None = None) -> tuple[pd.DataFrame, list]:
    """Normalize dtypes after arbitrary (LLM) transformations. Returns (X, cat_cols)."""
    known_cats = set(known_cats or ())
    cats = []
    for c in list(X.columns):
        s = X[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            # ensure string categories
            if not all(isinstance(v, str) for v in s.cat.categories[:50]):
                s = to_category(s.astype("object").where(s.notna(), np.nan).map(lambda z: z if pd.isna(z) else str(z)))
            X[c] = s
            cats.append(c)
        elif pd.api.types.is_bool_dtype(s):
            X[c] = s.astype(float)
        elif pd.api.types.is_numeric_dtype(s):
            x = s.astype(float)
            X[c] = x.replace([np.inf, -np.inf], np.nan)
        elif pd.api.types.is_datetime64_any_dtype(s):
            X[c] = (s.astype("int64") / 86400e9).where(s.notna(), np.nan)
        else:
            num = _try_numeric(s) if c not in known_cats else None
            if num is not None and num.nunique() > 2:
                X[c] = num
            else:
                X[c] = to_category(s.map(lambda z: z if pd.isna(z) else str(z)))
                cats.append(c)
    return X, cats
