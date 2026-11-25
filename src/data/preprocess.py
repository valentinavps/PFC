"""Indexação da base, agrupamento de quadros quase idênticos e divisão dos dados."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

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
        g = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
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