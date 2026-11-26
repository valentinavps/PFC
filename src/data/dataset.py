"""Dataset PyTorch, transformações e DataLoaders para as dobras da validação cruzada."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from src.utils import resolve

# Média e desvio-padrão do ImageNet (os modelos pré-treinados esperam essa normalização)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(cfg: dict, train: bool) -> T.Compose:
    """Transformações das imagens. Aumento de dados apenas no treino."""
    size = cfg["image_size"]
    aug = cfg["augmentation"]
    steps: list = [T.Resize((size, size))]
    if train:
        if aug.get("hflip", False):
            steps.append(T.RandomHorizontalFlip())
        if aug.get("vflip", False):
            steps.append(T.RandomVerticalFlip())
        steps.append(T.RandomRotation(aug.get("rotation_deg", 0)))
        steps.append(T.ColorJitter(brightness=aug.get("brightness", 0), contrast=aug.get("contrast", 0)))
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


class RailDataset(Dataset):
    """Imagens de trilhos listadas em um DataFrame (colunas `path` e `label`)."""

    def __init__(self, df: pd.DataFrame, raw_dir: Path, classes: list[str], transform=None):
        self.raw_dir = Path(raw_dir)
        # O índice pode ter sido gerado no Windows (barras invertidas); normaliza para "/"
        self.paths = [p.replace("\\", "/") for p in df["path"]]
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.labels = [self.class_to_idx[l] for l in df["label"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.raw_dir / self.paths[i]) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]


def class_weights(labels_idx, n_classes: int) -> torch.Tensor:
    """Pesos w_k = n / (K * n_k). Classes sem amostras recebem n_k = 1 para evitar divisão por zero."""
    counts = np.bincount(np.asarray(labels_idx), minlength=n_classes).astype(float)
    n = counts.sum()
    w = n / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def _loader(cfg: dict, ds: Dataset, shuffle: bool = False, sampler=None) -> DataLoader:
    """DataLoader com batch_size e num_workers do config."""
    tr = cfg["training"]
    return DataLoader(
        ds,
        batch_size=tr["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=tr["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )


def get_loaders(cfg: dict, df: pd.DataFrame, classes: list[str], fold: int
                ) -> tuple[DataLoader, DataLoader, torch.Tensor | None]:
    """Loaders de treino e validação da dobra `fold` e os pesos das classes.

    Treino = split "dev" com fold != k; validação = split "dev" com fold == k.
    Com oversample=true, o treino usa WeightedRandomSampler e os pesos devolvidos
    são None (para não compensar o desbalanceamento duas vezes)."""
    raw_dir = resolve(cfg, "raw_dir")
    dev = df[df["split"] == "dev"]
    train_df = dev[dev["fold"] != fold]
    val_df = dev[dev["fold"] == fold]

    train_ds = RailDataset(train_df, raw_dir, classes, get_transforms(cfg, train=True))
    val_ds = RailDataset(val_df, raw_dir, classes, get_transforms(cfg, train=False))
    weights = class_weights(train_ds.labels, len(classes))

    if cfg["training"].get("oversample", False):
        # Probabilidade de sorteio de cada amostra proporcional ao peso da sua classe
        sample_w = weights[torch.tensor(train_ds.labels)]
        sampler = WeightedRandomSampler(sample_w.double(), num_samples=len(train_ds), replacement=True)
        train_loader = _loader(cfg, train_ds, sampler=sampler)
        weights = None
    else:
        train_loader = _loader(cfg, train_ds, shuffle=True)

    val_loader = _loader(cfg, val_ds)
    return train_loader, val_loader, weights


def get_test_loader(cfg: dict, df: pd.DataFrame, classes: list[str]) -> DataLoader:
    """Loader do teste isolado (split "test"), sem embaralhar."""
    test_df = df[df["split"] == "test"]
    ds = RailDataset(test_df, resolve(cfg, "raw_dir"), classes, get_transforms(cfg, train=False))
    return _loader(cfg, ds)
