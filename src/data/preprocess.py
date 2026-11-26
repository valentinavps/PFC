"""Indexação da base, agrupamento de quadros quase idênticos e divisão dos dados.

Uso (a partir da raiz do projeto):
    python -m src.data.preprocess --config configs/config.yaml [--seed 42]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import StratifiedGroupKFold

from src.utils import get_classes, load_config, resolve

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_index(raw_dir: Path, aliases: dict | None = None) -> pd.DataFrame:
    """Cria o DataFrame (path, label, width, height).

    A classe é o nome da pasta imediatamente acima do arquivo, em minúsculas.
    `aliases` permite renomear pastas (ex.: {"flaking": "flakings"})."""
    aliases = {k.lower(): v.lower() for k, v in (aliases or {}).items()}
    rows = []
    for p in sorted(raw_dir.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or not p.is_file():
            continue
        label = p.parent.name.lower()
        label = aliases.get(label, label)
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception as e:  # arquivo corrompido
            print(f"[aviso] ignorando {p}: {e}")
            continue
        rows.append({"path": str(p.relative_to(raw_dir)), "label": label, "width": w, "height": h})
    if not rows:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {raw_dir}")
    return pd.DataFrame(rows)


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path: Path, hash_size: int = 8) -> np.ndarray:
    """Hash de diferença: reduz a imagem para (hash_size+1) x hash_size em cinza
    e compara pixels vizinhos na horizontal. Devolve os bits empacotados."""
    with Image.open(path) as im:
        g = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = a[:, 1:] > a[:, :-1]
    return np.packbits(bits.flatten())


class _UnionFind:
    """Estrutura para juntar imagens parecidas em grupos."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[rj] = ri


def add_groups(df: pd.DataFrame, raw_dir: Path, threshold: int = 1, hash_size: int = 8) -> pd.DataFrame:
    """Adiciona a coluna `group`. Duas imagens ficam no mesmo grupo se a distância
    de Hamming entre os hashes for <= threshold (e o grupo é transitivo).
    Conta em df.attrs["dup_conflicts"] os pares parecidos com rótulos diferentes."""
    hashes = np.stack([dhash(raw_dir / p, hash_size) for p in tqdm(df["path"], desc="dHash")])
    labels = df["label"].to_numpy()
    n = len(df)
    uf = _UnionFind(n)
    conflicts = 0
    for i in range(n - 1):
        dist = _POPCOUNT[hashes[i] ^ hashes[i + 1:]].sum(axis=1)
        for k in np.where(dist <= threshold)[0]:
            j = i + 1 + int(k)
            uf.union(i, j)
            if labels[i] != labels[j]:
                conflicts += 1
    out = df.copy()
    out["group"] = [uf.find(i) for i in range(n)]
    out.attrs["dup_conflicts"] = conflicts
    return out


def group_report(df: pd.DataFrame) -> None:
    """Resumo dos grupos, para decidir se o limiar está bom."""
    sizes = df.groupby("group").size().sort_values(ascending=False)
    mixed = (df.groupby("group")["label"].nunique() > 1).sum()
    print(f"imagens: {len(df)} | grupos: {len(sizes)}")
    print(f"maior grupo: {sizes.iloc[0]} imagens ({100 * sizes.iloc[0] / len(df):.1f}%)")
    print(f"5 maiores grupos: {sizes.head(5).tolist()}")
    print(f"grupos com mais de uma classe: {mixed}")
    print(f"pares parecidos com rótulos diferentes: {df.attrs.get('dup_conflicts', 'n/d')}")



def make_splits(df: pd.DataFrame, classes: list, scenario: str,
                test_size: float, n_folds: int, seed: int) -> pd.DataFrame:
    """Divide por grupo, mantendo a proporção das classes.

    Cenário A: separa ~test_size das imagens como teste isolado (split="test")
    e divide o resto em n_folds dobras (split="dev", fold = 0..n_folds-1).
    Cenário B: não há teste; tudo é "dev" com validação cruzada em n_folds dobras.
    """
    df = df[df["label"].isin(classes)].reset_index(drop=True).copy()
    df["split"] = "dev"
    df["fold"] = -1

    if scenario == "A":
        n_outer = round(1 / test_size)  # 0.20 -> 5 partes; uma vira o teste
        outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
        dev_idx, test_idx = next(outer.split(df, df["label"], df["group"]))
        df.loc[test_idx, "split"] = "test"
    else:
        dev_idx = np.arange(len(df))

    dev = df.loc[dev_idx]
    inner = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(inner.split(dev, dev["label"], dev["group"])):
        df.loc[dev.index[val_idx], "fold"] = k
    return df


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Garante que nenhum grupo aparece em dois lados da divisão."""
    dev = df[df["split"] == "dev"]
    test_groups = set(df.loc[df["split"] == "test", "group"])
    assert not (test_groups & set(dev["group"])), "grupo no teste e no desenvolvimento"
    for k in sorted(dev["fold"].unique()):
        val_g = set(dev.loc[dev["fold"] == k, "group"])
        train_g = set(dev.loc[dev["fold"] != k, "group"])
        assert not (val_g & train_g), f"vazamento na dobra {k}"


def index_path(cfg: dict, seed: int) -> Path:
    return resolve(cfg, "processed_dir") / f"index_{cfg['scenario']}_s{seed}.csv"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    sp = cfg["split"]
    raw_dir = resolve(cfg, "raw_dir")

    df = build_index(raw_dir)
    df = add_groups(df, raw_dir, sp["dedup_threshold"], sp["hash_size"])
    group_report(df)
    df = make_splits(df, get_classes(cfg), cfg["scenario"], sp["test_size"], sp["n_folds"], seed)
    assert_no_leakage(df)

    out = index_path(cfg, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nCenário {cfg['scenario']} | seed {seed} | salvo em {out}")
    print(pd.crosstab(df["label"], df["split"], margins=True))
    print(pd.crosstab(df["label"], df["fold"], margins=True))


if __name__ == "__main__":
    main()