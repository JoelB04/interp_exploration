"""Disk cache for activations.

This is the highest-leverage file in the repo. A full sweep over 8 datasets in 2
readout modes is 30-60 min of CPU forward passes; loading the same thing from
disk is a couple of seconds. Compute once, then iterate on the analysis as many
times as you like.

The model is loaded LAZILY -- if every dataset you asked for is already cached,
no model is ever constructed and the script starts instantly.
"""

import hashlib
import os

import numpy as np
import torch

from acts import MODEL_NAME, get_acts, load
from data import prepare

CACHE_DIR = "cache"

_MODEL = None  # (model, tok, device), populated on first cache miss


def _handle():
    global _MODEL
    if _MODEL is None:
        print("  [cache miss] loading model...")
        _MODEL = load()
    return _MODEL


def _key(dataset: str, mode: str, max_n: int, seed: int) -> str:
    """Cache filename. Includes the model name -- switching models must not
    silently reuse another model's activations."""
    raw = f"{MODEL_NAME}|{dataset}|{mode}|{max_n}|{seed}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"{dataset}_{mode}_n{max_n}_s{seed}_{digest}.pt"


def cached_acts(dataset: str, mode: str, max_n: int = 400, seed: int = 0):
    """Return (acts, labels, groups, statements) for one dataset/mode.

    acts is float32 (n_examples, n_layers + 1, d_model), on CPU.
    Remember index 0 is the embedding, index i is the output of block i-1.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, _key(dataset, mode, max_n, seed))

    if os.path.exists(path):
        blob = torch.load(path, weights_only=False)
        return blob["acts"], blob["labels"], blob["groups"], blob["statements"]

    statements, labels, groups = prepare(dataset, max_n=max_n, seed=seed)
    model, tok, device = _handle()

    print(f"  computing {dataset}/{mode}  n={len(statements)}")
    acts = get_acts(statements, model, tok, device, mode=mode)

    torch.save(
        {
            "acts": acts,
            "labels": labels,
            "groups": groups,
            "statements": statements,
            "model": MODEL_NAME,
            "dataset": dataset,
            "mode": mode,
            "seed": seed,
        },
        path,
    )
    return acts, labels, groups, statements


def cache_status(datasets, modes, max_n: int = 400, seed: int = 0):
    """What is already on disk. Cheap -- call before a long sweep."""
    rows = []
    for d in datasets:
        for m in modes:
            path = os.path.join(CACHE_DIR, _key(d, m, max_n, seed))
            rows.append((d, m, os.path.exists(path)))
    return rows
