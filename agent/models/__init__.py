"""Реестр моделей раннера. Импорт ленивый: torch грузится только при обучении mlp/resnet."""

from __future__ import annotations

import importlib

_REGISTRY = {
    "lightgbm": ("lgbm", "LightGBMModel"),
    "catboost": ("catboost_model", "CatBoostModel"),
    "xgboost": ("xgb", "XGBoostModel"),
    "extratrees": ("extratrees", "ExtraTreesModel"),
    "linear": ("linear", "LinearModel"),
    "mlp": ("mlp", "MLPModel"),
    "resnet": ("resnet", "ResNetModel"),
}

MODEL_NAMES = tuple(_REGISTRY)

_NOTES = {
    "lightgbm": "early stopping по внутреннему сплиту; alpha — параметр objective=huber",
    "catboost": "early stopping по внутреннему сплиту",
    "xgboost": "early stopping по внутреннему сплиту; huber_slope — параметр reg:pseudohubererror",
    "extratrees": "без early stopping",
    "linear": "Ridge для регрессии, LogisticRegression для классификации; one-hot топ-уровней категорий",
    "mlp": "quantile transform, эмбеддинги категорий, early stopping по эпохам",
    "resnet": "ResNet-MLP, остальное как у mlp",
}


def get_model_class(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"неизвестная модель {name}; доступные: {', '.join(MODEL_NAMES)}")
    mod_name, cls_name = _REGISTRY[name]
    mod = importlib.import_module(f"{__name__}.{mod_name}")
    return getattr(mod, cls_name)


def get_model(name: str, params: dict, spec, n_threads: int, seed: int):
    return get_model_class(name)(params, spec, n_threads=n_threads, seed=seed)


def validate_params(name: str, params: dict | None, spec) -> tuple[dict, list[str], list[str]]:
    """(канонические параметры, ошибки, исправления) по схеме модели. Ошибка — гипотеза отфильтровывается."""
    try:
        cls = get_model_class(name)
    except ValueError as e:
        return {}, [str(e)], []
    return cls.validate(params, spec)


def model_menu(spec) -> dict[str, str]:
    """Меню для Исследователя, сгенерированное из param_schema: ровно те имена, что пройдут проверку."""
    out = {}
    for name in MODEL_NAMES:
        cls = get_model_class(name)
        out[name] = f"{cls.menu(spec)}. {_NOTES.get(name, '')}".strip()
    return out
