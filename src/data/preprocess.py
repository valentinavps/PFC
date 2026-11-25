"""Indexação da base, agrupamento de quadros quase idênticos e divisão dos dados."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

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