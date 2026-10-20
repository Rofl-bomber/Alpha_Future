"""ResNet-MLP для табличных данных (Gorishniy et al., 2021), короткая реализация на torch."""

from __future__ import annotations

from .mlp import MLPModel


class ResNetModel(MLPModel):
    name = "resnet"
    kind = "resnet"
    defaults = dict(lr=1e-3, weight_decay=1e-5, batch_size=512, max_epochs=100, patience=8,
                    d_main=192, n_blocks=3, dropout=0.15)
