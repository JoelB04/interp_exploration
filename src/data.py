"""Dataset loading with group-aware splitting.

The geometry-of-truth CSVs are built from *pairs*: the same entity appears in a
true statement and a false one. A random row split therefore puts both halves of
a pair across the train/test boundary, and the probe can score well by memorising
the entity rather than learning truth. That inflates the within-dataset diagonal
of the transfer matrix, which is the reference every off-diagonal number is
compared against.

`group_key` below extracts the entity so `split()` can keep pairs together.
"""

import os
import re
import urllib.request

import numpy as np
import pandas as pd

# NOTE: hyphens, not underscores. The underscore spelling 404s -- that bug sat in
# 02_first_probe.py and meant the script never got past the fetch.
BASE = "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets"
DATA_DIR = "data"

# The paired negation datasets are the point of the study: neg_* is where probes
# are known to anti-generalise (AUROC < 0.5). Keep them adjacent to their base.
DATASETS = [
    "cities",
    "neg_cities",
    "sp_en_trans",
    "neg_sp_en_trans",
    "larger_than",
    "smaller_than",
    "companies_true_false",
    "common_claim_true_false",
]


def fetch(name: str) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        url = f"{BASE}/{name}.csv"
        print(f"  downloading {name}")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            raise SystemExit(
                f"Could not fetch {url} ({e}).\n"
                "Clone https://github.com/saprmarks/geometry-of-truth and copy "
                f"datasets/{name}.csv into ./data/ manually."
            )
    return pd.read_csv(path)


def group_key(df: pd.DataFrame, name: str) -> np.ndarray:
    """The entity a statement is *about*, so pairs stay on one side of a split.

    Returns one label per row. Falls back to a unique id per row when a dataset
    has no pair structure -- that degrades gracefully to a plain random split.
    """
    # cities / neg_cities: the same city appears with a true and a false country.
    if "city" in df.columns:
        return df["city"].astype(str).to_numpy()

    # counterfact: one subject, a true target and a false one.
    if "subject" in df.columns:
        return df["subject"].astype(str).to_numpy()

    # larger_than / smaller_than: "Fifty-one is larger than fifty-two" and its
    # reverse are the same unordered pair. Sort so both map to one key.
    if {"n1", "n2"}.issubset(df.columns):
        lo = np.minimum(df["n1"].to_numpy(), df["n2"].to_numpy())
        hi = np.maximum(df["n1"].to_numpy(), df["n2"].to_numpy())
        return np.array([f"{a}_{b}" for a, b in zip(lo, hi)])

    # sp_en_trans has no metadata columns, but the Spanish word is quoted in the
    # statement and is what repeats across the true/false pair.
    if name.endswith("sp_en_trans"):
        keys = []
        for s in df["statement"]:
            m = re.search(r"'([^']+)'", str(s))
            keys.append(m.group(1) if m else str(s))
        return np.array(keys)

    # No known pair structure. Every row is its own group.
    return np.arange(len(df)).astype(str)


def prepare(name: str, max_n: int = 400, seed: int = 0):
    """Shuffle, subsample, and return (statements, labels, groups).

    Subsampling happens at the GROUP level, not the row level, so a truncated
    dataset still contains whole pairs.
    """
    df = fetch(name)
    assert {"statement", "label"}.issubset(df.columns), \
        f"{name}: expected 'statement' and 'label', got {list(df.columns)}"

    groups = group_key(df, name)
    rng = np.random.default_rng(seed)

    uniq = np.unique(groups)
    rng.shuffle(uniq)

    keep, n_kept = set(), 0
    for g in uniq:
        size = int((groups == g).sum())
        if n_kept + size > max_n:
            continue
        keep.add(g)
        n_kept += size
        if n_kept >= max_n:
            break

    mask = np.array([g in keep for g in groups])
    df, groups = df[mask].reset_index(drop=True), groups[mask]

    order = rng.permutation(len(df))
    df, groups = df.iloc[order].reset_index(drop=True), groups[order]

    return df["statement"].tolist(), df["label"].to_numpy().astype(int), groups


def split(groups: np.ndarray, frac_train: float = 0.7, seed: int = 0):
    """Group-aware train/test split. No group spans the boundary."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)

    n_train_groups = max(1, int(round(frac_train * len(uniq))))
    train_groups = set(uniq[:n_train_groups])

    is_train = np.array([g in train_groups for g in groups])
    return np.where(is_train)[0], np.where(~is_train)[0]
