"""Session 3d: fix the layer instead of selecting it.

Session 3c found that val-based layer selection is the dominant variance source:
`cities` picks anywhere in layers 16-28 across split repeats, and different
layers give qualitatively different transfer, which is why
chat cities -> neg_cities came out 0.587 +/- 0.44.

So stop selecting. Compute the whole transfer matrix at EVERY layer, holding the
layer fixed across all eight probes, and resample the split 20 times at each.

Three questions:
  1. How much variance does fixing the layer remove?
  2. Which layer transfers best -- and is it the one selection kept picking?
  3. Do the negation cells stabilise? If cities -> neg_cities is tight at a fixed
     layer, the session-3 effect was real and selection was hiding it. If it is
     still wide, the effect was never there.

Note what this is NOT. Choosing the best fixed layer by looking at these curves
is still selection, just done by hand on the test set. Any layer picked this way
must be reported as such -- it is a descriptive sweep, not a held-out number.

Run from the repo root:  python scripts/07_fixed_layer.py
Reads the cache. No model, no forward passes.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cache import cached_acts  # noqa: E402
from data import DATASETS, split  # noqa: E402

MAX_N, SEED = 400, 0
MODES = ["raw", "chat"]
N_REPEATS = 20
OUT = "results"
PAIRS = [("cities", "neg_cities"),
         ("sp_en_trans", "neg_sp_en_trans"),
         ("larger_than", "smaller_than")]


def dom_all_layers(acts, labels):
    d = acts[labels == 1].mean(axis=0) - acts[labels == 0].mean(axis=0)
    return d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12)


def run_mode(mode):
    store = {}
    for name in DATASETS:
        a, y, g, _ = cached_acts(name, mode, MAX_N, SEED)
        store[name] = dict(acts=a.numpy(), labels=y, groups=g)

    n_ds = len(DATASETS)
    n_lay = store[DATASETS[0]]["acts"].shape[1]
    grids = np.zeros((N_REPEATS, n_ds, n_ds, n_lay))

    for rep in range(N_REPEATS):
        idx = {}
        for name in DATASETS:
            g = store[name]["groups"]
            trval, test_i = split(g, frac_train=0.8, seed=1000 * rep + 7)
            tr_rel, _ = split(g[trval], frac_train=0.75, seed=1000 * rep + 13)
            idx[name] = (trval[tr_rel], test_i)

        for i, tr in enumerate(DATASETS):
            A, y = store[tr]["acts"], store[tr]["labels"]
            train_i, _ = idx[tr]
            D = dom_all_layers(A[train_i], y[train_i])       # (n_lay, d)

            for j, te in enumerate(DATASETS):
                B, yb = store[te]["acts"], store[te]["labels"]
                _, test_j = idx[te]
                s = np.einsum("nld,ld->nl", B[test_j], D)    # (n_test, n_lay)
                grids[rep, i, j] = [roc_auc_score(yb[test_j], s[:, l])
                                    for l in range(n_lay)]
        print(f"\r  {mode}: repeat {rep + 1}/{N_REPEATS}", end="", flush=True)
    print()
    return grids


def main():
    os.makedirs(OUT, exist_ok=True)
    ix = {d: i for i, d in enumerate(DATASETS)}
    all_grids = {}

    for mode in MODES:
        print(f"\n{'=' * 70}\nmode: {mode}\n{'=' * 70}")
        G = run_mode(mode)                      # (rep, train, test, layer)
        all_grids[mode] = G

        mu, sd = G.mean(0), G.std(0, ddof=1)
        n_ds, n_lay = len(DATASETS), G.shape[-1]
        off = ~np.eye(n_ds, dtype=bool)

        # Q1/Q2: stability and quality by layer, off-diagonal only.
        print(f"\n{'layer':>6} {'mean off-diag':>14} {'mean sd':>9} {'max sd':>8}")
        rows = []
        for l in range(n_lay):
            rows.append((l, mu[:, :, l][off].mean(), sd[:, :, l][off].mean(),
                         sd[:, :, l][off].max()))
        for r in rows[::2]:
            print(f"{r[0]:>6} {r[1]:>14.3f} {r[2]:>9.3f} {r[3]:>8.3f}")

        best_stable = min(rows[1:], key=lambda r: r[2])
        best_transfer = max(rows[1:], key=lambda r: r[1])
        print(f"\n  most STABLE layer   : {best_stable[0]}  "
              f"(mean sd {best_stable[2]:.3f}, mean transfer {best_stable[1]:.3f})")
        print(f"  best TRANSFER layer : {best_transfer[0]}  "
              f"(mean transfer {best_transfer[1]:.3f}, mean sd {best_transfer[2]:.3f})")

        # Q3: do the negation cells stabilise?
        print(f"\n  negation cells at a few fixed layers (mean +/- sd over "
              f"{N_REPEATS} splits)")
        probe_layers = sorted({16, 17, best_stable[0], best_transfer[0], n_lay - 1})
        hdr = "  ".join(f"L{l}" .center(15) for l in probe_layers)
        print(f"    {'cell':<34}{hdr}")
        for a, b in PAIRS:
            for i, j, lbl in [(ix[a], ix[b], f"{a}->{b}"), (ix[b], ix[a], f"{b}->{a}")]:
                cells = "  ".join(f"{mu[i,j,l]:.3f}+-{sd[i,j,l]:.2f}".center(15)
                                  for l in probe_layers)
                print(f"    {lbl:<34}{cells}")

        # Compare against session 3c's selected-layer spread on the same cells.
        print("\n  for reference, 3c val-SELECTED layer gave "
              "cities->neg_cities 0.587+-0.44 (chat)")

    torch.save(all_grids, os.path.join(OUT, "fixed_layer.pt"))

    # ---- plot: stability and negation behaviour by layer
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8))
    for col, mode in enumerate(MODES):
        G = all_grids[mode]
        mu, sd = G.mean(0), G.std(0, ddof=1)
        off = ~np.eye(len(DATASETS), dtype=bool)
        L = np.arange(G.shape[-1])

        ax = axes[0, col]
        ax.plot(L, [mu[:, :, l][off].mean() for l in L], color="#2471a3",
                lw=2, label="mean off-diagonal AUROC")
        ax.fill_between(L,
                        [mu[:, :, l][off].mean() - sd[:, :, l][off].mean() for l in L],
                        [mu[:, :, l][off].mean() + sd[:, :, l][off].mean() for l in L],
                        color="#2471a3", alpha=0.2, label="+/- mean sd over splits")
        ax.axhline(0.5, color="0.35", ls="--", lw=1)
        ax.set_ylim(0.3, 1); ax.set_title(f"{mode}: transfer by fixed layer",
                                          fontweight="bold")
        ax.set_xlabel("layer"); ax.set_ylabel("mean off-diagonal AUROC")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)

        ax = axes[1, col]
        for (a, b), c in zip(PAIRS, ["#c0392b", "#8e44ad", "#16a085"]):
            i, j = ix[a], ix[b]
            ax.plot(L, mu[i, j], color=c, lw=1.8, label=f"{a[:9]}->{b[:9]}")
            ax.fill_between(L, mu[i, j] - sd[i, j], mu[i, j] + sd[i, j],
                            color=c, alpha=0.18)
        ax.axhline(0.5, color="0.35", ls="--", lw=1)
        ax.axhspan(0, 0.5, color="#c0392b", alpha=0.05)
        ax.set_ylim(0, 1); ax.set_xlabel("layer")
        ax.set_ylabel("transfer AUROC")
        ax.set_title(f"{mode}: negation pairs, +/- 1 sd", fontweight="bold")
        ax.legend(fontsize=7.5); ax.grid(alpha=0.25)

    axes[0, 2].axis("off"); axes[1, 2].axis("off")
    axes[0, 2].text(0, 0.5,
                    "Wide bands = the cell is\nsplit-dependent and should\n"
                    "not be reported as a\nfinding.\n\n"
                    "Narrow bands away from 0.5\n= a real effect.\n\n"
                    "Session 3 selected the layer\nper split, which mixed\n"
                    "layers with opposite\nbehaviour and produced\n"
                    "bimodal cells.",
                    fontsize=9, va="center", family="monospace")
    fig.suptitle("Transfer with the layer held FIXED across all probes, "
                 f"{N_REPEATS} resampled splits", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "fixed_layer.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\nwrote {p}")
    print(f"saved -> {OUT}/fixed_layer.pt")


if __name__ == "__main__":
    main()
