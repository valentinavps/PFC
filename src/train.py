"""Treino de uma ou de todas as dobras da validação cruzada.

Uso (a partir da raiz do projeto):
    python -m src.train --fold 0 [--seed 42] [--epochs 1]
    python -m src.train --all-folds
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.data.dataset import get_loaders
from src.data.preprocess import index_path
from src.models.model import build_model
from src.utils import get_classes, load_config, resolve, set_seed


def get_device() -> torch.device:
    """GPU se disponível (Colab), senão CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_name(cfg: dict, seed: int, fold: int) -> str:
    """Nome base dos arquivos de uma dobra, ex.: A_resnet18_s42_f0."""
    return f"{cfg['scenario']}_{cfg['training']['model']}_s{seed}_f{fold}"


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device: torch.device, desc: str) -> float:
    """Uma época de treino. Devolve a loss média por amostra."""
    model.train()
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def validate(model: nn.Module, loader, criterion, device: torch.device, n_classes: int) -> tuple[float, float]:
    """Loss média e F1 macro na validação."""
    model.eval()
    total, n = 0.0, 0
    y_true, y_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += criterion(logits, y).item() * len(y)
        n += len(y)
        y_true.append(y.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
    f1 = f1_score(np.concatenate(y_true), np.concatenate(y_pred),
                  average="macro", labels=list(range(n_classes)), zero_division=0)
    return total / max(n, 1), float(f1)


def run_fold(cfg: dict, df: pd.DataFrame, classes: list[str], fold: int,
             device: torch.device, seed: int, epochs: int | None = None) -> dict:
    """Treina um modelo na dobra `fold` com early stopping pelo F1 macro de validação.

    Salva o melhor checkpoint e o histórico por época. Devolve um resumo do treino."""
    tr = cfg["training"]
    epochs = tr["epochs"] if epochs is None else epochs
    K = len(classes)

    train_loader, val_loader, weights = get_loaders(cfg, df, classes, fold)
    model = build_model(tr["model"], K, tr["pretrained"], tr["freeze_backbone"]).to(device)

    use_w = tr.get("class_weights", False) and weights is not None
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if use_w else None)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tr["lr"], weight_decay=tr["weight_decay"])

    ckpt_dir = resolve(cfg, "checkpoints_dir")
    res_dir = resolve(cfg, "results_dir")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    name = run_name(cfg, seed, fold)
    ckpt_path = ckpt_dir / f"{name}.pt"
    hist_path = res_dir / f"history_{name}.csv"

    print(f"\n=== dobra {fold} | treino: {len(train_loader.dataset)} | val: {len(val_loader.dataset)} "
          f"| device: {device} | pesos na perda: {use_w} ===")

    history = []
    best_f1, best_epoch, bad_epochs = -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                     desc=f"f{fold} ep{epoch}/{epochs}")
        val_loss, val_f1 = validate(model, val_loader, criterion, device, K)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_macro_f1": val_f1})
        pd.DataFrame(history).to_csv(hist_path, index=False)  # salva a cada época

        improved = val_f1 > best_f1
        if improved:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "model": tr["model"],
                "seed": seed,
                "fold": fold,
                "scenario": cfg["scenario"],
                "epoch": epoch,
            }, ckpt_path)
        else:
            bad_epochs += 1

        print(f"[f{fold}] época {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| val_macro_f1 {val_f1:.4f}{' *' if improved else ''}")

        if bad_epochs >= tr["patience"]:
            print(f"[f{fold}] early stopping: {tr['patience']} épocas sem melhora")
            break

    print(f"[f{fold}] melhor F1 macro {best_f1:.4f} na época {best_epoch} -> {ckpt_path}")
    return {"fold": fold, "best_val_macro_f1": best_f1, "best_epoch": best_epoch}


def main() -> None:
    ap = argparse.ArgumentParser(description="Treino por dobra da validação cruzada")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--seed", type=int, default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fold", type=int, help="treina apenas esta dobra")
    g.add_argument("--all-folds", action="store_true", help="treina as dobras 0..n_folds-1")
    ap.add_argument("--epochs", type=int, default=None, help="sobrescreve training.epochs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    classes = get_classes(cfg)
    df = pd.read_csv(index_path(cfg, seed))
    device = get_device()

    n_folds = cfg["split"]["n_folds"]
    folds = list(range(n_folds)) if args.all_folds else [args.fold]
    for k in folds:
        if not 0 <= k < n_folds:
            raise ValueError(f"dobra {k} fora do intervalo 0..{n_folds - 1}")

    results = []
    for k in folds:
        set_seed(seed)  # mesma semente no início de cada dobra
        results.append(run_fold(cfg, df, classes, k, device, seed, args.epochs))

    print("\nResumo:")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
