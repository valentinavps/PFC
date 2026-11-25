"""Funções auxiliares: configuração, sementes e caminhos."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

# Pasta raiz do projeto (dois níveis acima de src/utils.py)
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, key: str) -> Path:
    """Caminho absoluto de uma entrada de cfg['paths'], relativo à raiz do projeto."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def get_classes(cfg: dict) -> list[str]:
    """Lista de classes do cenário escolhido (A ou B)."""
    return list(cfg["classes"][cfg["scenario"]])


def set_seed(seed: int) -> None:
    """Fixa as sementes aleatórias para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # PyTorch não instalado (ex.: no computador local)