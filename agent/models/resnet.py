"""ResNet-MLP для табличных данных (Gorishniy et al., 2021), короткая реализация на torch."""

from __future__ import annotations

from .mlp import MLPModel


class ResNetModel(MLPModel):
    name = "resnet"
    kind = "resnet"
    defaults = dict(lr=1e-3, weight_decay=1e-5, batch_size=512, max_epochs=100, patience=8,
                    d_main=192, n_blocks=3, dropout=0.15)
    param_schema = {
        "lr": ("float", 1e-5, 0.1),
        "weight_decay": ("float", 0.0, 0.1),
        "batch_size": ("int", 32, 8192),
        "max_epochs": ("int", 1, 500),
        "patience": ("int", 1, 50),
        "d_main": ("int", 16, 1024),
        "n_blocks": ("int", 1, 16),
        "d_hidden_factor": ("float", 0.5, 8.0),
        "dropout": ("float", 0.0, 0.9),
    }
    param_aliases = {
        "learning_rate": "lr", "epochs": "max_epochs", "n_epochs": "max_epochs", "num_epochs": "max_epochs",
        "hidden": "d_main", "hidden_size": "d_main", "hidden_dim": "d_main", "width": "d_main",
        "num_blocks": "n_blocks", "blocks": "n_blocks", "n_layers": "n_blocks", "depth": "n_blocks",
        "early_stopping_rounds": "patience", "wd": "weight_decay",
    }
    doc_defaults = {"d_hidden_factor": 2.0}
