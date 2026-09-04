import hashlib
import os

import torch

from acts import MODEL_NAME, get_acts, load

CACHE_DIR = "cache"

_MODEL = None  # (model, tok, device), populated on the first cache miss


def _handle():
    global _MODEL
    if _MODEL is None:
        print("  [cache miss] loading model...")
        _MODEL = load()
    return _MODEL


def _key(tag: str, mode: str, M: int, seed: int) -> str:

    raw = f"{MODEL_NAME}|{tag}|{mode}|{M}|{seed}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:8]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)[:40]
    return f"{safe}_{mode}_M{M}_s{seed}_{digest}.pt"


def cached_acts(tag: str, mode: str, loader, M: int, seed: int = 0):

    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, _key(tag, mode, M, seed))

    if os.path.exists(path):
        blob = torch.load(path, weights_only=False)
        return blob["acts"], blob["meta"]

    statements, meta = loader()
    if len(statements) != M:
        raise ValueError(
            f"{tag}: loader returned {len(statements)} statements, expected M={M}. "
            "Equal M across manifolds is a hard requirement -- see standard 2."
        )

    model, tok, device = _handle()
    print(f"  computing {tag}/{mode}  M={len(statements)}")
    acts = get_acts(statements, model, tok, device, mode=mode)

    torch.save(
        {"acts": acts, "meta": meta, "statements": statements,
         "model": MODEL_NAME, "tag": tag, "mode": mode, "M": M, "seed": seed},
        path,
    )
    return acts, meta


def cache_status(tags, modes, M: int, seed: int = 0):
    return [(t, m, os.path.exists(os.path.join(CACHE_DIR, _key(t, m, M, seed))))
            for t in tags for m in modes]
