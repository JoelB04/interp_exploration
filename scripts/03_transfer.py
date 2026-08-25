"""Session 3: the transfer matrix.

Train a truth probe on each dataset, test it on all of them. The resulting grid
is the substrate for the whole project -- every predictor you write later is
trying to explain these numbers.

What to predict before you run this (write it in logs/research_log.md first):
  - Which off-diagonal cells sit near 0.5, and which go BELOW it?
  - Is cities -> neg_cities symmetric with neg_cities -> cities?
  - Does the diagonal drop now that the split is group-aware? By how much?
  - Which readout mode transfers better, and is that the same mode that has the
    higher diagonal?

Run from the repo root:  python scripts/03_transfer.py
First run computes activations (slow, ~30-60 min on CPU). Every later run reads
the cache and takes seconds.
"""

import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cache import cached_acts, cache_status  # noqa: E402
from data import DATASETS, split  # noqa: E402

MAX_N = 400          # per dataset. CPU-bound; raise when you have a GPU.
SEED = 0
MODES = ["raw", "chat"]
N_PERM = 20          # permutation null replicates
OUT = "results"


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
def diff_of_means_all_layers(acts: torch.Tensor, labels: np.ndarray) -> np.ndarray:
    """Unit-norm diff-of-means direction at every layer at once.

    acts   (n, L, d)  ->  directions (L, d)

    Diff-of-means cancels confounds that are mean-independent of the label, and
    only those. That proviso does real work: it does NOT cancel a confound that
    is itself correlated with truth.
    """
    a = acts.numpy() if isinstance(acts, torch.Tensor) else acts
    d = a[labels == 1].mean(axis=0) - a[labels == 0].mean(axis=0)   # (L, d)
    norm = np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12
    return d / norm


def project(acts: torch.Tensor, directions: np.ndarray) -> np.ndarray:
    """Score every example at every layer. -> (n, L)

    No centering. AUROC is rank-only, and subtracting any constant vector c
    shifts every score by the same c . d, which cannot change a ranking. So the
    train-mean-vs-test-mean centering question is genuinely moot HERE -- but it
    will matter for the geometric predictors, where distances are not rank-only.
    """
    a = acts.numpy() if isinstance(acts, torch.Tensor) else acts
    return np.einsum("nld,ld->nl", a, directions)


def auroc_by_layer(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """(n, L) scores -> (L,) AUROC. Returns NaN for degenerate label sets."""
    if len(np.unique(labels)) < 2:
        return np.full(scores.shape[1], np.nan)
    return np.array([roc_auc_score(labels, scores[:, l]) for l in range(scores.shape[1])])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)

    print("cache status:")
    for d, m, hit in cache_status(DATASETS, MODES, MAX_N, SEED):
        print(f"  {d:26s} {m:5s} {'cached' if hit else 'MISSING'}")

    results = {}

    for mode in MODES:
        print(f"\n{'=' * 70}\nreadout mode: {mode}\n{'=' * 70}")

        # --- load / compute activations, and carve a 60/20/20 group-aware split
        store = {}
        for name in DATASETS:
            acts, labels, groups, _ = cached_acts(name, mode, MAX_N, SEED)

            # Three-way split, and the reason matters. If you select the best
            # layer on the same data you report, the reported number is
            # optimistically biased -- you have fitted one hyperparameter (layer)
            # to the test set. val exists purely to choose the layer; test is
            # untouched until the final number.
            trval, test_i = split(groups, frac_train=0.8, seed=SEED)
            sub_groups = groups[trval]
            tr_rel, val_rel = split(sub_groups, frac_train=0.75, seed=SEED + 1)
            train_i, val_i = trval[tr_rel], trval[val_rel]

            store[name] = dict(
                acts=acts, labels=labels,
                train=train_i, val=val_i, test=test_i,
                n_layers=acts.shape[1] - 1,
            )
            print(f"  {name:26s} n={len(labels):4d}  "
                  f"train={len(train_i):3d} val={len(val_i):3d} test={len(test_i):3d}  "
                  f"groups={len(np.unique(groups)):4d}")

        n_layers_p1 = store[DATASETS[0]]["acts"].shape[1]
        n_ds = len(DATASETS)

        grid = np.full((n_ds, n_ds, n_layers_p1), np.nan)      # [train, test, layer]
        null = np.full((n_ds, n_ds, n_layers_p1), np.nan)      # permutation mean
        null_hi = np.full((n_ds, n_ds, n_layers_p1), np.nan)   # 95th pct of null
        chosen = np.zeros(n_ds, dtype=int)

        for i, tr_name in enumerate(DATASETS):
            S = store[tr_name]
            tr_acts = S["acts"][S["train"]]
            tr_y = S["labels"][S["train"]]

            directions = diff_of_means_all_layers(tr_acts, tr_y)

            # --- layer selection on the TRAIN dataset's val split only
            val_auroc = auroc_by_layer(project(S["acts"][S["val"]], directions),
                                       S["labels"][S["val"]])
            chosen[i] = int(np.nanargmax(val_auroc))

            # --- permutation null: same pipeline, labels shuffled within train
            rng = np.random.default_rng(SEED + 1000 + i)
            perm_dirs = [
                diff_of_means_all_layers(tr_acts, rng.permutation(tr_y))
                for _ in range(N_PERM)
            ]

            for j, te_name in enumerate(DATASETS):
                T = store[te_name]
                te_acts, te_y = T["acts"][T["test"]], T["labels"][T["test"]]

                grid[i, j] = auroc_by_layer(project(te_acts, directions), te_y)

                perm_auroc = np.stack([
                    auroc_by_layer(project(te_acts, pd_), te_y) for pd_ in perm_dirs
                ])
                null[i, j] = perm_auroc.mean(axis=0)
                null_hi[i, j] = np.nanpercentile(perm_auroc, 95, axis=0)

            print(f"  probe from {tr_name:26s} layer {chosen[i]:2d} "
                  f"(val AUROC {val_auroc[chosen[i]]:.3f})")

        results[mode] = dict(
            grid=grid, null=null, null_hi=null_hi, chosen=chosen,
            datasets=DATASETS,
            n_test={d: int(len(store[d]["test"])) for d in DATASETS},
        )

        # --- readable table at each probe's selected layer
        print(f"\ntransfer matrix ({mode}), row = train, col = test, "
              f"AUROC at each row's val-selected layer")
        short = [d[:11] for d in DATASETS]
        print(f"{'':>26s} " + " ".join(f"{s:>11s}" for s in short))
        for i, tr_name in enumerate(DATASETS):
            cells = []
            for j in range(n_ds):
                v, nh = grid[i, j, chosen[i]], null_hi[i, j, chosen[i]]
                mark = "*" if (v > nh or v < 1 - nh) else " "   # signed: both tails
                cells.append(f"{v:>10.3f}{mark}")
            print(f"{tr_name:>26s} " + " ".join(cells))

        print("\n  * = outside the two-sided 95% permutation null at that cell.")
        print("  n per test column:", {d: results[mode]["n_test"][d] for d in DATASETS})
        se = {d: 0.5 / np.sqrt(results[mode]["n_test"][d]) for d in DATASETS}
        print("  rough SE on a 0.5 AUROC:",
              {d: round(v, 3) for d, v in se.items()})

    path = os.path.join(OUT, f"transfer_n{MAX_N}_s{SEED}.pt")
    torch.save(results, path)
    print(f"\nsaved -> {path}")
    print("Full [train, test, layer] grids are in there. The printed table is one "
          "slice; look at the layer profiles before believing any single cell.")


if __name__ == "__main__":
    main()
