import os
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np  

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import salad  
from cache import cached_acts 

OUT = "results"
MODES = ["raw", "request"]
M_EQUAL = 640
N_PERM = 2000
N_TASKSHUF = 200
SEED = 0
N_LAYERS = 29


def _no_loader():
    raise RuntimeError("cache miss run scripts/04_extract.py first")


def task_centroids(mode, rng):
    """Centroid of every task at every layer. -> (T, L, n), sizes, parents, names"""
    tasks = salad.design_tasks()
    cen = np.zeros((len(tasks), N_LAYERS, 1536), dtype=np.float32)
    for ti, (t, _) in enumerate(tasks):
        cap = min(len(salad.fetch_task(t, cap=10 ** 9)), salad.EXTRACT_CAP)
        A, _ = cached_acts(t, mode, _no_loader, M=cap, seed=SEED)
        a = A.numpy()[rng.choice(cap, M_EQUAL, replace=False)]
        cen[ti] = a.mean(axis=0)
    par = np.array([d for _, d in tasks])
    return cen, par, [t for t, _ in tasks]


def pooled_fake_tasks(mode, rng, sizes):
    """Pool every prompt, cut into fake tasks of the given sizes. -> (T, L, n)"""
    tasks = salad.design_tasks()
    pool = []
    for t, _ in tasks:
        cap = min(len(salad.fetch_task(t, cap=10 ** 9)), salad.EXTRACT_CAP)
        A, _ = cached_acts(t, mode, _no_loader, M=cap, seed=SEED)
        pool.append(A.numpy()[rng.choice(cap, M_EQUAL, replace=False)])
    pool = np.concatenate(pool)                       # (T*M, L, n)
    order = rng.permutation(len(pool))
    out, start = [], 0
    for s in sizes:
        out.append(pool[order[start:start + s]].mean(axis=0)); start += s
    return np.stack(out)


def ratio_from_dists(dists, groups):
    """dists: dict (i,j)->distance. groups: array of group id per manifold."""
    w = [d for (i, j), d in dists.items() if groups[i] == groups[j]]
    b = [d for (i, j), d in dists.items() if groups[i] != groups[j]]
    return float(np.mean(w) / np.mean(b))


def group_sizes(par):
    _, counts = np.unique(par, return_counts=True)
    return sorted(counts, reverse=True)


def random_grouping(n_items, sizes, rng):
    """A grouping with EXACTLY these sizes. Identical sizes keep the pair split
    at 12/66, so the null differs only in which tasks are siblings."""
    perm = rng.permutation(n_items)
    g, start = np.zeros(n_items, int), 0
    for gi, s in enumerate(sizes):
        g[perm[start:start + s]] = gi; start += s
    return g


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    store = {}

    for mi, mode in enumerate(MODES):
        rng = np.random.default_rng(SEED)
        cen, par, names = task_centroids(mode, rng)
        T = len(names)
        sizes = group_sizes(par)
        true_g = np.unique(par, return_inverse=True)[1]

        real = np.full(N_LAYERS, np.nan)
        null = np.full((N_PERM, N_LAYERS), np.nan)
        pvals = np.full(N_LAYERS, np.nan)

        for l in range(N_LAYERS):
            dists = {(i, j): float(np.linalg.norm(cen[i, l] - cen[j, l]))
                     for i, j in combinations(range(T), 2)}
            if not np.isfinite(list(dists.values())).all() or \
               np.allclose(list(dists.values()), 0):
                continue      # templated layer 0: all centroids equal
            real[l] = ratio_from_dists(dists, true_g)
            prng = np.random.default_rng(1000 + l)
            null[:, l] = [ratio_from_dists(dists, random_grouping(T, sizes, prng))
                          for _ in range(N_PERM)]
            pvals[l] = (null[:, l] <= real[l]).mean()

        store[mode] = dict(real=real, null=null, p=pvals, sizes=sizes)

        print(f"\n=== {mode}   group sizes {sizes}, {N_PERM} permutations")
        print(f"{'layer':>6} {'real':>8} {'null mean':>10} {'null sd':>8} "
              f"{'null 5th':>9} {'p':>7}")
        for l in range(0, N_LAYERS, 2):
            if np.isnan(real[l]):
                print(f"{l:>6} {'--':>8} {'degenerate':>10}")
                continue
            n = null[:, l]
            print(f"{l:>6} {real[l]:>8.4f} {n.mean():>10.4f} {n.std(ddof=1):>8.4f} "
                  f"{np.percentile(n, 5):>9.4f} {pvals[l]:>7.3f}")

        #task shuffle: pipeline check, should sit around 1
        ts = []
        trng = np.random.default_rng(SEED + 7)
        fake = pooled_fake_tasks(mode, trng, [M_EQUAL] * T)
        for l in [4, 14, 26]:
            d = {(i, j): float(np.linalg.norm(fake[i, l] - fake[j, l]))
                 for i, j in combinations(range(T), 2)}
            ts.append((l, ratio_from_dists(d, true_g)))
        print("  task-shuffle check (should be about 1): " +
              ", ".join(f"L{l}:{v:.3f}" for l, v in ts))

        ax = axes[mi]
        lo = np.nanpercentile(null, 5, axis=0)
        hi = np.nanpercentile(null, 95, axis=0)
        med = np.nanmedian(null, axis=0)
        ax.fill_between(range(N_LAYERS), lo, hi, color="0.6", alpha=.35,
                        label="null, 5th-95th pct")
        ax.plot(med, color="0.35", lw=1.4, ls="--", label="null median")
        ax.plot(real, color="#c0392b", lw=2.2, label="true taxonomy")
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_xlabel("layer"); ax.set_ylabel("nesting ratio")
        ax.set_title(f"{mode}", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=.25)

    fig.suptitle("Random-partition null: does SALAD's parent structure beat a "
                 f"shuffle with identical group sizes? ({N_PERM} permutations)",
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "random_partition.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\nwrote {p}")
    np.savez(os.path.join(OUT, "random_partition.npz"),
             **{f"{k}_{m}": store[m][k] for m in MODES for k in ("real", "p")})


if __name__ == "__main__":
    main()
