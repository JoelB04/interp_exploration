"""Disk cache for activations.

This is the highest-leverage file in the repo. A full sweep of forward passes is
tens of minutes of CPU; loading the same thing from disk is a couple of seconds.
Compute once, then iterate on the analysis as many times as you like.

Dataset-agnostic by design. The caller supplies a `loader` callable that returns
the prompts and whatever metadata belongs with them; the cache never knows which
dataset it is holding. This is a change from the session-3 version, which
imported a geometry-of-truth loader directly and could only cache that.

The model is loaded LAZILY, and the loader is only CALLED on a miss -- if
everything you asked for is cached, no model is constructed and no dataset is
fetched, so a re-run starts instantly.
"""

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
    """Cache filename. Includes the model name -- switching models must not
    silently reuse another model's activations."""
    raw = f"{MODEL_NAME}|{tag}|{mode}|{M}|{seed}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:8]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)[:40]
    return f"{safe}_{mode}_M{M}_s{seed}_{digest}.pt"


def cached_acts(tag: str, mode: str, loader, M: int, seed: int = 0):
    """Last-token activations at every layer, cached to disk.

    tag     cache key. Any string that uniquely names this slice of data.
    mode    'raw' or 'chat' -- readout position, see acts.format_prompts.
    loader  zero-argument callable returning (statements, meta_dict).
            Called ONLY on a cache miss. meta_dict is stored alongside the
            activations and returned unchanged; put labels, parent/child ids,
            or anything else you need there.
    M       points in this slice. Part of the cache key, so changing it is a
            miss rather than a silent reuse.

    Returns (acts, meta) where acts is float32 (M, n_layers + 1, n) on CPU.
    Index 0 of the middle axis is the embedding; index i is the output of
    block i-1.
    """
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
    """What is already on disk. Cheap -- call before a long sweep."""
    return [(t, m, os.path.exists(os.path.join(CACHE_DIR, _key(t, m, M, seed))))
            for t in tags for m in modes]
