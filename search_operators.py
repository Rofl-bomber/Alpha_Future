"""Конфигурация пайплайна и операторы над ней: патчи, проверка патчей, слияние, рестарт, библиотека.

Конфигурация — один JSON, полностью описывающий эксперимент:
  preprocess  {cat_encoding, missing_flags, rare_min_count}
  features    {builtin: [...], custom: [...], drop: [...]}
  target      {transform, class_weight, clip_quantile}
  model       {name, params, custom_id, escape_id}
  postprocess {class_weights_opt, threshold_opt, clip_to_train_range}
  training    {pseudo_label, pseudo_threshold}

Патч — словарь «путь.через.точку → значение». Для списков: "+x" — добавить, "-x" — убрать.
"""

import copy
import hashlib
import json

import feat_builtin
import task_metrics
from exp_runner import MODEL_MODULES, model_module

ENUMS = {
    "preprocess.cat_encoding": feat_builtin.ENCODINGS,
    "target.transform": ("none", "log1p", "sqrt"),
    "target.class_weight": ("none", "balanced", "sqrt_balanced"),
    "model.name": tuple(MODEL_MODULES) + ("custom", "escape"),
}
BOOLS = ("preprocess.missing_flags", "postprocess.class_weights_opt", "postprocess.threshold_opt",
         "postprocess.clip_to_train_range", "training.pseudo_label")
NUMS = {"preprocess.rare_min_count": (int, 0, 1000), "target.clip_quantile": (float, 0.0, 0.05),
        "training.pseudo_threshold": (float, 0.8, 0.999)}
LISTS = ("features.builtin", "features.custom", "features.drop")


def default_config(spec) -> dict:
    metric = task_metrics.metric_for(spec)
    return {
        "preprocess": {"cat_encoding": "native", "missing_flags": False, "rare_min_count": 0},
        "features": {"builtin": [], "custom": [], "drop": []},
        "target": {"transform": metric.target_transform if spec.task_type == "regression" else "none",
                   "class_weight": "none", "clip_quantile": 0.0},
        "model": {"name": "lgbm", "params": l1_params(spec), "custom_id": None, "escape_id": None},
        "postprocess": {"class_weights_opt": True, "threshold_opt": True, "clip_to_train_range": True},
        "training": {"pseudo_label": False, "pseudo_threshold": 0.95},
    }


def l1_params(spec) -> dict:
    """Надёжные дефолты L1: больше классов — выше learning rate, чтобы уложиться во время."""
    if spec.task_type == "multiclass" and spec.n_classes > 4:
        return {"learning_rate": 0.08}
    return {}


def family(config: dict) -> str:
    return config.get("model", {}).get("name", "lgbm")


def config_hash(config: dict) -> str:
    return hashlib.md5(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


def flatten(d: dict, prefix="") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and not (key == "model.params" and not v):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def _set(cfg: dict, path: str, value):
    parts = path.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def _get(cfg: dict, path: str, default=None):
    d = cfg
    for p in path.split("."):
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def apply_patch(config: dict, patch: dict) -> dict:
    cfg = copy.deepcopy(config)
    keys = list(patch)
    if "model.name" in patch and patch["model.name"] != _get(cfg, "model.name"):
        _set(cfg, "model.params", {})  # параметры старой модели новой не подходят
    for k in keys:
        v = patch[k]
        if k in LISTS:
            cur = list(_get(cfg, k, []) or [])
            if isinstance(v, str) and v[:1] in "+-":
                item = v[1:]
                if v[0] == "+" and item not in cur:
                    cur.append(item)
                elif v[0] == "-" and item in cur:
                    cur.remove(item)
                _set(cfg, k, cur)
            elif isinstance(v, list):
                # непустой список добавляется к текущему (LLM часто пишет список вместо "+x"); [] — очистка
                _set(cfg, k, list(dict.fromkeys(cur + list(v))) if v else [])
            continue
        if k == "model.params" and isinstance(v, dict):
            _set(cfg, k, dict(_get(cfg, k, {}) or {}, **v))
            continue
        _set(cfg, k, v)
    return cfg


def _coerce_num(v, typ, lo, hi):
    if typ is bool:
        return bool(v) if not isinstance(v, str) else v.strip().lower() in ("1", "true", "yes")
    x = typ(float(v))
    return max(typ(lo), min(typ(hi), x))


def validate_patch(patch: dict, config: dict, spec, known_custom=(), hints=None, allow_params: bool = True) -> tuple:
    """Проверить и нормализовать патч LLM. Возвращает (чистый патч, список замечаний).

    allow_params=False — гиперпараметры модели отбрасываются (Исследователь их не предлагает,
    их подбирает Optuna в харнессе).
    """
    clean, notes = {}, []
    if not isinstance(patch, dict):
        return {}, ["патч должен быть объектом"]
    ops = set(feat_builtin.applicable_ops(spec, hints))
    flat = {}

    def walk(prefix, obj):  # вложенные словари LLM → ключи через точку
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict) and key not in LISTS:
                walk(key, v)
            else:
                flat[key] = v

    walk("", patch)
    new_model = flat.get("model.name", _get(config, "model.name"))
    for k, v in flat.items():
        try:
            if k in ENUMS:
                if v not in ENUMS[k]:
                    notes.append(f"{k}={v} — недопустимое значение")
                    continue
                if k == "model.name" and v in ("custom", "escape"):
                    notes.append("model.name=custom/escape задаётся только через код Инженера")
                    continue
                if k == "preprocess.cat_encoding" and not spec.cat_cols:
                    notes.append("кодирование категорий без категориальных колонок — пропущено")
                    continue
                if k.startswith("target.") and k != "target.class_weight" and spec.task_type != "regression":
                    continue
                clean[k] = v
            elif k in BOOLS:
                clean[k] = _coerce_num(v, bool, 0, 1)
            elif k in NUMS:
                clean[k] = _coerce_num(v, *NUMS[k])
            elif k in LISTS:
                if isinstance(v, list) and not v:
                    clean[k] = []  # явная очистка списка
                    continue
                items = v if isinstance(v, list) else [v]
                ok = []
                for it in items:
                    it = str(it)
                    sign = it[0] if it[:1] in "+-" else ""
                    name = it[1:] if sign else it
                    if k == "features.builtin" and name not in ops:
                        notes.append(f"операции {name} нет в библиотеке для этой задачи")
                        continue
                    if k == "features.custom" and name not in known_custom:
                        notes.append(f"признаков {name} нет")
                        continue
                    if k == "features.drop" and name not in spec.feature_cols:
                        notes.append(f"колонки {name} нет среди признаков")
                        continue
                    ok.append(sign + name)
                if not ok:
                    continue
                clean[k] = ok[0] if not isinstance(v, list) else [x.lstrip("+") for x in ok if not x.startswith("-")]
            elif k.startswith("model.params."):
                if not allow_params:
                    notes.append(f"{k}: гиперпараметры подбирает харнесс — пропущено")
                    continue
                pk = k.split(".", 2)[2]
                if new_model not in MODEL_MODULES:
                    notes.append(f"параметры для модели {new_model} не настраиваются")
                    continue
                ranges = model_module(new_model).RANGES
                if pk not in ranges:
                    notes.append(f"параметра {pk} нет у {new_model}")
                    continue
                clean[k] = _coerce_num(v, *ranges[pk])
            else:
                notes.append(f"неизвестный ключ {k}")
        except (TypeError, ValueError):
            notes.append(f"{k}={v!r} — неверный тип")
    # собрать параметры модели обратно в словарь
    params = {k.split(".", 2)[2]: v for k, v in clean.items() if k.startswith("model.params.")}
    clean = {k: v for k, v in clean.items() if not k.startswith("model.params.")}
    if params:
        clean["model.params"] = params
    return clean, notes


def diff(base: dict, other: dict) -> dict:
    fb, fo = flatten(base), flatten(other)
    return {k: v for k, v in fo.items() if fb.get(k) != v}


def _model_defaults(name: str) -> dict:
    try:
        return dict(getattr(model_module(name), "DEFAULTS", {}) or {})
    except Exception:
        return {}


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def describe_change(parent: dict, child: dict, custom_desc: dict | None = None) -> str:
    """Что изменено в конфигурации относительно родителя — коротко и по делу, для промпта и журнала."""
    custom_desc = custom_desc or {}
    parts = []
    for key in LISTS:
        a, b = list(_get(parent, key, []) or []), list(_get(child, key, []) or [])
        added = [x for x in b if x not in a]
        removed = [x for x in a if x not in b]
        if added:
            items = [f"{x} ({custom_desc[x][:120]})" if key == "features.custom" and x in custom_desc else str(x)
                     for x in added]
            parts.append(f"{key} += " + ", ".join(items))
        if removed:
            parts.append(f"{key} -= " + ", ".join(map(str, removed)))
    for sec in ("preprocess", "target", "postprocess", "training"):
        pa, pb = parent.get(sec, {}) or {}, child.get(sec, {}) or {}
        for k in sorted(set(pa) | set(pb)):
            if pa.get(k) != pb.get(k):
                parts.append(f"{sec}.{k} {_fmt(pa.get(k))}→{_fmt(pb.get(k))}")
    fa, fb = family(parent), family(child)
    ma, mb = parent.get("model", {}) or {}, child.get("model", {}) or {}
    if fa != fb:
        extra = mb.get("custom_id") or mb.get("escape_id")
        parts.append(f"модель {fa}→{fb}" + (f" ({extra})" if extra else ""))
    else:
        da = dict(_model_defaults(fa), **(ma.get("params") or {}))
        db = dict(_model_defaults(fb), **(mb.get("params") or {}))
        ch = [f"{k} {_fmt(da.get(k))}→{_fmt(db.get(k))}" for k in sorted(set(da) | set(db)) if da.get(k) != db.get(k)]
        if ch:
            parts.append(f"{fb}: " + ", ".join(ch))
    return "; ".join(parts) or "без изменений конфигурации"


def state_text(config: dict, custom_desc: dict | None = None, n_features=None) -> str:
    """Состояние пайплайна текущего решения: что в нём включено (для промпта Исследователя)."""
    custom_desc = custom_desc or {}
    f = config.get("features", {}) or {}
    pre = config.get("preprocess", {}) or {}
    tg = config.get("target", {}) or {}
    tr = config.get("training", {}) or {}
    lines = []
    if n_features:
        lines.append(f"- признаков в модели: {n_features}")
    lines.append("- features.builtin: " + (", ".join(f.get("builtin") or []) or "нет"))
    custom = f.get("custom") or []
    if custom:
        lines.append("- features.custom: " + "; ".join(f"{c} ({custom_desc.get(c, '')[:120]})" for c in custom))
    else:
        lines.append("- features.custom: нет")
    lines.append("- features.drop: " + (", ".join(f.get("drop") or []) or "нет"))
    lines.append("- preprocess: " + ", ".join(f"{k}={_fmt(v)}" for k, v in pre.items()))
    tgt = {k: v for k, v in tg.items() if v not in (None, "none", 0, 0.0)}
    lines.append("- target: " + (", ".join(f"{k}={_fmt(v)}" for k, v in tgt.items()) or "без преобразований"))
    if tr.get("pseudo_label"):
        lines.append(f"- training: pseudo_label, порог {tr.get('pseudo_threshold')}")
    lines.append(f"- модель: {family(config)} (гиперпараметры подбирает харнесс)")
    return "\n".join(lines)


def merge(cfg_a: dict, cfg_b: dict, anchor: dict) -> tuple:
    """Добавить к A изменения B относительно общего предка. Конфликт — значение A (лучшего узла)."""
    da, db = diff(anchor, cfg_a), diff(anchor, cfg_b)
    merged = copy.deepcopy(cfg_a)
    conflicts = []
    for k, v in db.items():
        if k in LISTS:
            cur = list(_get(merged, k, []) or [])
            for it in v or []:
                if it not in cur:
                    cur.append(it)
            _set(merged, k, cur)
        elif k.startswith("model.") and family(cfg_a) != family(cfg_b):
            conflicts.append(k)
        elif k in da and da[k] != v:
            conflicts.append(k)
        else:
            _set(merged, k, v)
    return merged, conflicts


def restart_options(spec, tried_families: set) -> list:
    """Рестарт на текущем решении: другое семейство моделей или кодирование (признаки сохраняются)."""
    out = []
    for fam in ("catboost", "xgb", "extratrees", "mlp", "linear"):
        if fam not in tried_families:
            out.append((f"рестарт: модель {fam}", {"model.name": fam}))
    if spec.cat_cols:
        out.append(("рестарт: target encoding категорий", {"preprocess.cat_encoding": "target_oof"}))
    return out


def library(spec, prof: dict) -> list:
    """Детерминированная очередь гипотез по приоритету (используется, когда LLM не ответила)."""
    hyps = []
    t = prof.get("target", {})

    def add(block, why, patch, cost="low"):
        hyps.append({"kind": "config", "block": block, "rationale": why, "patch": patch, "cost": cost,
                     "source": "library"})

    high_card = [c for c in spec.cat_cols if prof["columns"].get(c, {}).get("nunique", 0) > 30]
    if high_card:
        add("preprocess", "target encoding высококардинальных категорий", {"preprocess.cat_encoding": "target_oof"})
    if spec.cat_cols:
        add("features", "частоты категорий", {"features.builtin": "+freq_enc"})
    if spec.task_type == "regression" and t.get("nonneg") and abs(t.get("skew", 0)) > 1:
        add("target", "логарифм скошенного таргета", {"target.transform": "log1p"})
    add("model", "CatBoost вместо LightGBM", {"model.name": "catboost"}, "medium")
    if spec.cat_cols and spec.num_cols:
        add("features", "средние числовых признаков по группам категорий", {"features.builtin": "+group_stats"})
    if any(v.get("miss_train", 0) > 0.01 for v in prof["columns"].values()):
        add("preprocess", "флаги пропусков", {"preprocess.missing_flags": True})
    if spec.task_type in ("binary", "multiclass") and t.get("imbalance", 1) > 5:
        add("target", "веса редких классов", {"target.class_weight": "sqrt_balanced"})
    add("model", "XGBoost", {"model.name": "xgb"}, "medium")
    if len(spec.num_cols) >= 3:
        add("features", "отношения важных числовых признаков", {"features.builtin": "+num_ratios"})
    adv = prof.get("adversarial") or {}
    if adv.get("auc") and adv["auc"] > 0.7:
        drift = [f for f, _ in adv.get("top", [])[:2] if f in spec.feature_cols]
        if drift:
            add("features", f"убрать дрейфующие признаки {drift}", {"features.drop": drift})
    if spec.task_type in ("binary", "multiclass"):
        add("training", "псевдоразметка уверенных строк test", {"training.pseudo_label": True}, "high")
    add("model", "более глубокие деревья и медленнее обучение",
        {"model.params": {"num_leaves": 127, "learning_rate": 0.03, "min_child_samples": 30}}, "high")
    add("model", "сильнее регуляризация", {"model.params": {"feature_fraction": 0.5, "lambda_l2": 10.0}})
    if spec.cat_cols and len(spec.cat_cols) >= 2:
        add("features", "сочетания категорий", {"features.builtin": "+cat_combos"})
    if spec.text_cols:
        add("features", "TF-IDF + SVD текста", {"features.builtin": "+text_tfidf_svd"}, "medium")
    add("model", "ExtraTrees для разнообразия бленда", {"model.name": "extratrees"}, "medium")
    return hyps


def schema_text(spec, hints=None) -> str:
    """Описание допустимых патчей для промпта Исследователя (без гиперпараметров — их подбирает харнесс)."""
    ops = feat_builtin.applicable_ops(spec, hints)
    lines = [
        "Допустимые ключи патча к текущему решению (значение заменяет текущее; для списков \"+x\" добавляет, "
        "\"-x\" убирает):",
        f"- preprocess.cat_encoding: {list(feat_builtin.ENCODINGS)}" if spec.cat_cols else "",
        "- preprocess.missing_flags: true|false; preprocess.rare_min_count: 0..1000",
        "- features.builtin: операции библиотеки: " + "; ".join(f"{o} — {feat_builtin.OPS_DOC[o]}" for o in ops),
        "- features.drop: имена исходных колонок, которые убрать",
        "- features.custom: id готовых LLM-признаков (новые создаются гипотезой kind=feature_code)",
    ]
    if spec.task_type == "regression":
        lines.append("- target.transform: none|log1p|sqrt (метрика всегда считается в исходной шкале); "
                     "target.clip_quantile: 0..0.05 (обрезка выбросов таргета в обучении)")
    if spec.task_type in ("binary", "multiclass"):
        lines.append("- target.class_weight: none|balanced|sqrt_balanced")
        lines.append("- training.pseudo_label: true|false; training.pseudo_threshold: 0.8..0.999")
    lines.append("- model.name: " + "|".join(MODEL_MODULES)
                 + " — другое семейство моделей на тех же признаках; гиперпараметры подбирает харнесс, "
                   "ключи model.params не принимаются")
    lines.append("- postprocess.class_weights_opt / threshold_opt / clip_to_train_range: true|false")
    return "\n".join(x for x in lines if x)
