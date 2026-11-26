"""Avaliação dos checkpoints de cada dobra: validação (fora da amostra) e teste isolado.

Uso (a partir da raiz do projeto):
    python -m src.evaluate [--seed 42] [--split val|test]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janela (Colab / servidor)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support)

from src.data.dataset import get_loaders, get_test_loader
from src.data.preprocess import index_path
from src.models.model import build_model
from src.train import get_device, run_name
from src.utils import get_classes, load_config, resolve


@torch.no_grad()
def predict(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rótulos verdadeiros, preditos e probabilidades (softmax) para todo o loader."""
    model.eval()
    y_true, y_prob = [], []
    for x, y in loader:
        probs = torch.softmax(model(x.to(device)), dim=1)
        y_true.append(y.numpy())
        y_prob.append(probs.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    return y_true, y_prob.argmax(1), y_prob


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, classes: list[str]) -> dict:
    """Métricas globais e por classe.

    Classes ausentes em y_true ficam com recall/F1 = 0 (zero_division=0) e PR-AUC = NaN,
    e a PR-AUC macro é a média apenas das classes presentes."""
    labels = list(range(len(classes)))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)

    m = {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    pr_aucs = []
    for k, c in enumerate(classes):
        pos = (y_true == k).astype(int)
        # Sem positivos a PR-AUC não é definida
        ap = float(average_precision_score(pos, y_prob[:, k])) if pos.any() else float("nan")
        pr_aucs.append(ap)
        m[f"precision_{c}"] = float(prec[k])
        m[f"recall_{c}"] = float(rec[k])
        m[f"f1_{c}"] = float(f1[k])
        m[f"pr_auc_{c}"] = ap
        m[f"support_{c}"] = int(support[k])
    m["pr_auc_macro"] = float(np.nanmean(pr_aucs)) if not np.all(np.isnan(pr_aucs)) else float("nan")
    return m


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], path: Path) -> None:
    """Salva em PNG a matriz de confusão absoluta e a normalizada por linha (recall)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, mat, title, fmt in [(axes[0], cm, "Absoluta", "d"), (axes[1], cm_norm, "Normalizada por linha", ".2f")]:
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1 if fmt == ".2f" else None)
        ax.set_title(title)
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Verdadeiro")
        limiar = mat.max() / 2 if mat.max() > 0 else 0.5
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] > limiar else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _nan_to_none(d: dict) -> dict:
    """JSON padrão não aceita NaN; troca por null."""
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def save_results(rows: list[dict], res_dir: Path, base: str) -> None:
    """Salva métricas por dobra (CSV/JSON) e o resumo média ± desvio-padrão entre dobras."""
    per_fold = pd.DataFrame(rows)
    num = per_fold.drop(columns=["fold"]).select_dtypes("number")
    summary = pd.DataFrame({"mean": num.mean(), "std": num.std()})

    per_fold.to_csv(res_dir / f"metrics_{base}.csv", index=False)
    summary.to_csv(res_dir / f"summary_{base}.csv", index_label="metric")
    with open(res_dir / f"metrics_{base}.json", "w", encoding="utf-8") as f:
        json.dump({
            "folds": [_nan_to_none(r) for r in rows],
            "summary": {"mean": _nan_to_none(summary["mean"].to_dict()),
                        "std": _nan_to_none(summary["std"].to_dict())},
        }, f, indent=2, ensure_ascii=False)

    print(f"\n--- {base} ---")
    for k in ["macro_f1", "balanced_accuracy", "accuracy", "pr_auc_macro"]:
        print(f"{k:18s} {summary.loc[k, 'mean']:.4f} ± {summary.loc[k, 'std']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliação dos checkpoints por dobra")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--split", choices=["val", "test"], default=None,
                    help="padrão: val e, se existir, test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    classes = get_classes(cfg)
    df = pd.read_csv(index_path(cfg, seed))
    device = get_device()

    has_test = (df["split"] == "test").any()
    splits = [args.split] if args.split else (["val", "test"] if has_test else ["val"])
    if "test" in splits and not has_test:
        raise ValueError(f"o cenário {cfg['scenario']} não tem teste isolado")

    ckpt_dir = resolve(cfg, "checkpoints_dir")
    res_dir = resolve(cfg, "results_dir")
    fig_dir = res_dir / "confusion"
    res_dir.mkdir(parents=True, exist_ok=True)
    test_loader = get_test_loader(cfg, df, classes) if "test" in splits else None

    rows = {s: [] for s in splits}
    oof_true, oof_pred = [], []  # predições fora da amostra de todas as dobras juntas
    for k in range(cfg["split"]["n_folds"]):
        name = run_name(cfg, seed, k)
        ckpt_path = ckpt_dir / f"{name}.pt"
        if not ckpt_path.exists():
            print(f"[aviso] checkpoint ausente, pulando dobra {k}: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if ckpt["classes"] != classes:
            raise ValueError(f"classes do checkpoint {ckpt['classes']} diferem do config {classes}")
        model = build_model(ckpt["model"], len(classes), pretrained=False).to(device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"dobra {k}: checkpoint da época {ckpt['epoch']}")

        for s in splits:
            loader = get_loaders(cfg, df, classes, k)[1] if s == "val" else test_loader
            y_true, y_pred, y_prob = predict(model, loader, device)
            rows[s].append({"fold": k, **compute_metrics(y_true, y_pred, y_prob, classes)})
            save_confusion(y_true, y_pred, classes, fig_dir / f"cm_{name}_{s}.png")
            if s == "val":
                oof_true.append(y_true)
                oof_pred.append(y_pred)

    if not any(rows.values()):
        raise FileNotFoundError(f"nenhum checkpoint encontrado em {ckpt_dir}")

    base = f"{cfg['scenario']}_{cfg['training']['model']}_s{seed}"
    for s in splits:
        save_results(rows[s], res_dir, f"{base}_{s}")
    if oof_true:
        save_confusion(np.concatenate(oof_true), np.concatenate(oof_pred), classes,
                       fig_dir / f"cm_{base}_val_oof.png")
    print(f"\nResultados em {res_dir}")


if __name__ == "__main__":
    main()
