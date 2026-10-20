"""Конфигурация пайплайна: схема, значения по умолчанию (L1), применение патчей."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MODEL_NAMES = ("lightgbm", "catboost", "xgboost", "extratrees", "linear", "mlp", "resnet")
BUILTIN_OPS = ("freq_enc", "target_enc", "group_stats")
CAT_ENCODINGS = ("native", "ordinal", "freq", "target")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Preprocess(_Strict):
    cat_encoding: Literal["native", "ordinal", "freq", "target"] = "native"
    missing_flags: bool = False
    rare_min_count: int = Field(0, ge=0)


class Features(_Strict):
    builtin: list[Literal["freq_enc", "target_enc", "group_stats"]] = []
    custom: list[str] = []
    drop: list[str] = []
    group_by: list[str] = []      # для group_stats; пусто -> автоматически
    group_cols: list[str] = []


class Target(_Strict):
    transform: Literal["none", "log1p"] = "none"


class Model(_Strict):
    name: Literal["lightgbm", "catboost", "xgboost", "extratrees", "linear", "mlp", "resnet"] = "lightgbm"
    params: dict[str, Any] = {}


class Postprocess(_Strict):
    class_weights_opt: bool = True
    clip_to_train_range: bool = True


class PipelineConfig(_Strict):
    preprocess: Preprocess = Preprocess()
    features: Features = Features()
    target: Target = Target()
    model: Model = Model()
    postprocess: Postprocess = Postprocess()


def default_config(spec) -> dict:
    cfg = PipelineConfig()
    cfg.postprocess.class_weights_opt = spec.is_clf
    cfg.postprocess.clip_to_train_range = not spec.is_clf
    return validate_config(cfg.model_dump())


def validate_config(cfg: dict) -> dict:
    out = PipelineConfig.model_validate(cfg).model_dump()
    f = out["features"]
    for k in ("builtin", "custom", "drop", "group_by", "group_cols"):
        f[k] = sorted(dict.fromkeys(f[k]))
    return out


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def _is_delta_list(val) -> bool:
    items = val if isinstance(val, list) else [val]
    return all(isinstance(x, str) and x[:1] in "+-" for x in items) and len(items) > 0


def apply_patch(cfg: dict, patch: dict) -> dict:
    """Патч — плоский словарь с ключами через точку.
    Списки: ["+x", "-y"] добавляют/удаляют элементы, обычный список заменяет.
    Смена model.name сбрасывает model.params (если они не заданы в том же патче).
    """
    new = copy.deepcopy(cfg)
    keys = sorted(patch.keys(), key=lambda k: (k != "model.name", k != "model", k))
    for key in keys:
        val = patch[key]
        path = [p for p in str(key).split(".") if p]
        if not path:
            raise ValueError(f"пустой ключ патча: {key!r}")
        if path == ["model", "name"] or (path == ["model"] and isinstance(val, dict) and "name" in val):
            name = val if path == ["model", "name"] else val["name"]
            if name != new["model"]["name"]:
                new["model"]["params"] = {}
        node = new
        for p in path[:-1]:
            if p not in node or not isinstance(node[p], dict):
                if path[0] == "model" and p == "params":
                    node[p] = {}
                else:
                    raise ValueError(f"неизвестный раздел в ключе {key!r}")
            node = node[p]
        last = path[-1]
        cur = node.get(last, None)
        if last not in node and not (len(path) >= 3 and path[1] == "params"):
            raise ValueError(f"неизвестный ключ {key!r}")
        if isinstance(cur, list):
            if _is_delta_list(val):
                items = val if isinstance(val, list) else [val]
                lst = list(cur)
                for it in items:
                    name = it[1:]
                    if it[0] == "+" and name not in lst:
                        lst.append(name)
                    elif it[0] == "-" and name in lst:
                        lst.remove(name)
                node[last] = lst
            else:
                node[last] = list(val) if isinstance(val, (list, tuple)) else [val]
        elif isinstance(cur, dict) and isinstance(val, dict):
            merged = dict(cur)
            merged.update(val)
            node[last] = merged
        else:
            node[last] = val
    return validate_config(new)


def diff_sections(a: dict, b: dict) -> set[str]:
    return {k for k in a if json.dumps(a[k], sort_keys=True, default=str) != json.dumps(b.get(k), sort_keys=True, default=str)}
