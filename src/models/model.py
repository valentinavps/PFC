"""Construção das redes de classificação (ResNet do torchvision)."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

# Nome -> (construtor, pesos ImageNet)
_ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


def build_model(name: str, n_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """ResNet com a camada `fc` trocada por uma saída de `n_classes` classes.

    Se `freeze_backbone`, apenas a nova `fc` é treinada."""
    if name not in _ARCHS:
        raise ValueError(f"Modelo não suportado: {name}. Opções: {list(_ARCHS)}")
    builder, weights = _ARCHS[name]
    model = builder(weights=weights if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    if freeze_backbone:
        for pname, p in model.named_parameters():
            p.requires_grad = pname.startswith("fc.")
    return model
