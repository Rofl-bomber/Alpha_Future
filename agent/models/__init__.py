"""Реестр моделей раннера. Импорт ленивый: torch грузится только для mlp/resnet."""

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

# Подсказка Исследователю: основные параметры каждой модели и их значения по умолчанию.
MODEL_MENU = {
    "lightgbm": "learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, "
                "reg_lambda=1, n_estimators=3000 (early stopping 100), objective (напр. huber, regression_l1), class_weight",
    "catboost": "learning_rate=0.08, depth=6, l2_leaf_reg=3, iterations=2000 (early stopping 100), auto_class_weights",
    "xgboost": "learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, min_child_weight=1, reg_lambda=1",
    "extratrees": "n_estimators=300, min_samples_leaf=3, max_features=0.5, class_weight",
    "linear": "Ridge(alpha=1) / LogisticRegression(C=1, class_weight); one-hot топ-30 уровней категорий (max_levels)",
    "mlp": "hidden=256, n_layers=2, dropout=0.1, lr=1e-3, weight_decay=1e-5, batch_size=512, max_epochs=100, patience=8",
    "resnet": "d_main=192, n_blocks=3, d_hidden_factor=2, dropout=0.15, lr=1e-3, weight_decay=1e-5, batch_size=512",
}


def get_model(name: str, params: dict, spec, n_threads: int, seed: int):
    if name not in _REGISTRY:
        raise ValueError(f"неизвестная модель {name}")
    mod_name, cls_name = _REGISTRY[name]
    mod = importlib.import_module(f"{__name__}.{mod_name}")
    return getattr(mod, cls_name)(params, spec, n_threads=n_threads, seed=seed)
