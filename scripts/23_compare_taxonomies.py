"""Session 6b: SALAD against MMLU, matched.

Does representational geometry reflect an externally-given category hierarchy in
general, or is the SALAD result specific to harm?

Everything is held equal except the taxonomy:

  M = 200 for both. Forced by MMLU -- only 3 of 57 subjects have >=640 rows. So
      SALAD is subsampled from its cached 640-800 down to 200. Free, no new
      forward passes. SALAD@640 remains the primary result reported elsewhere;
      this is the matched comparison.

  Same statistic, same permutation machinery, same lexical control.

  Z-SCORES, not raw nesting ratios. Branching differs -- SALAD [4,3,2,2,2] with
  12 within-parent pairs, MMLU [4,3,3,3] with 15 -- and four official MMLU
  categories cannot be made into five without inventing structure. Each dataset
  is therefore compared against ITS OWN matched permutation null, and
  (real - null_mean)/null_sd is comparable across the two where the raw ratio
  is not.

READING THE OUTCOME
  both strongly negative  -> geometry reflects externally-given taxonomies in
                             general. The SALAD finding is real but not about
                             harm, which is a broader and cleaner claim.
  SALAD only              -> something specific to harm categories, but the
                             register difference (imperative requests vs exam
                             stems) is an alternative explanation that this
                             design cannot exclude.
  MMLU only               -> the SALAD effect is weak, and the harm framing was
                             the wrong place to look.
  neither                 -> the M=640 result does not survive at M=200, which
                             would point at power rather than at taxonomy.

Run from the repo root:  python scripts/23_compare_taxonomies.py
Cache only.
"""

import os
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mmlu  # noqa: E402
import salad  # noqa: E402
from cache import cached_acts  # noqa: E402

OUT = "results"
M_EQUAL = 200
MODES = ["raw", "request"]
N_PERM = 4000
LAYERS = list(range(0, 29))
SEED = 0

DATASETS = {"SALAD": (salad, ""), "MMLU": (mmlu, "mmlu:")}


def _no_loader():
    raise RuntimeError("cache miss -- run the extraction scripts first")


def centroids(mod, tag, mode):
    """-> (T, L, n), parents, names"""
    tasks = mod.design_tasks()
    rng = np.random.default_rng(SEED)
    cen = np.zeros((len(tasks), 29, 1536), dtype=np.float32)
    for ti, (t, _) in enumerate(tasks):
        cap = min(len(mod.fetch_task(t, cap=10 ** 9)), mod.EXTRACT_CAP)
        A, _ = cached_acts(tag + t, mode, _no_loader, M=cap, seed=SEED)
        a = A.numpy()[rng.choice(cap, M_EQUAL, replace=False)]
        cen[ti] = a.mean(axis=0)
    return cen, np.array([p for _, p in tasks]), [t for t, _ in tasks]


def lexical(mod):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tasks = mod.design_tasks()
    docs, owner = [], []
    rng = np.random.default_rng(SEED)
    for ti, (t, _) in enumerate(tasks):
        qs = mod.fetch_task(t, cap=mod.EXTRACT_CAP, seed=SEED)
        qs = [qs[i] for i in rng.choice(len(qs), M_EQUAL, replace=False)]
        docs.extend(qs); owner.extend([ti] * len(qs))
    V = TfidfVectorizer(lowercase=True, stop_words="english", min_df=5,
                        sublinear_tf=True)
    X = V.fit_transform(docs); owner = np.array(owner)
    vec = np.stack([np.asarray(X[owner == ti].mean(axis=0)).ravel()
                    for ti in range(len(tasks))])
    return vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)


def perm(vals, groups, sizes, seed, ratio=False):
    """Permutation over groupings of fixed size. ratio=True uses within/between,
    False uses within-minus-between (for residuals, which are signed)."""
    T = len(groups); pairs = list(combinations(range(T), 2))

    def stat(g):
        w = [v for (i, j), v in zip(pairs, vals) if g[i] == g[j]]
        b = [v for (i, j), v in zip(pairs, vals) if g[i] != g[j]]
        return float(np.mean(w) / np.mean(b)) if ratio else \
               float(np.mean(w) - np.mean(b))

    real = stat(groups); rng = np.random.default_rng(seed)
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        p_ = rng.permutation(T); g = np.zeros(T, int); s = 0
        for gi, sz in enumerate(sizes):
            g[p_[s:s + sz]] = gi; s += sz
        null[k] = stat(g)
    z = (real - null.mean()) / (null.std(ddof=1) + 1e-12)
    return real, z, float((null <= real).mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}

    for dname, (mod, tag) in DATASETS.items():
        lex_vec = lexical(mod)
        for mode in MODES:
            cen, par, names = centroids(mod, tag, mode)
            T = len(names)
            pairs = list(combinations(range(T), 2))
            groups = np.unique(par, return_inverse=True)[1]
            sizes = sorted(np.unique(groups, return_counts=True)[1],
                           reverse=True)
            nw = sum(1 for i, j in pairs if groups[i] == groups[j])
            lex = np.array([1 - float(lex_vec[i] @ lex_vec[j])
                            for i, j in pairs])

            ratio = np.full(29, np.nan); z = np.full(29, np.nan)
            pv = np.full(29, np.nan); zr = np.full(29, np.nan)
            pr = np.full(29, np.nan); corr = np.full(29, np.nan)

            for L in LAYERS:
                geo = np.array([float(np.linalg.norm(cen[i, L] - cen[j, L]))
                                for i, j in pairs])
                if not np.isfinite(geo).all() or geo.std() == 0:
                    continue
                geo = geo / geo.mean()
                ratio[L], z[L], pv[L] = perm(geo, groups, sizes, 7, ratio=True)
                corr[L] = float(np.corrcoef(lex, geo)[0, 1])
                A = np.vstack([lex, np.ones_like(lex)]).T
                beta, *_ = np.linalg.lstsq(A, geo, rcond=None)
                _, zr[L], pr[L] = perm(geo - A @ beta, groups, sizes, 11)

            res[(dname, mode)] = dict(ratio=ratio, z=z, p=pv, zr=zr, pr=pr,
                                      corr=corr, nw=nw, names=names)
            print(f"{dname:<6} {mode:<8} branching {sizes}, {nw} within-parent "
                  f"pairs, M={M_EQUAL}")

    for mode in MODES:
        print(f"\n{'='*78}\nmode: {mode}   (z below 0 = more nested than chance)")
        print(f"{'layer':>6}" + "".join(
            f"{d+' ratio':>12}{d+' z':>9}{d+' z|lex':>11}" for d in DATASETS))
        for L in range(0, 29, 2):
            row = f"{L:>6}"
            for d in DATASETS:
                r = res[(d, mode)]
                row += (f"{r['ratio'][L]:>12.3f}{r['z'][L]:>9.2f}"
                        f"{r['zr'][L]:>11.2f}")
            print(row)
        print(f"\n{'':>6}" + "".join(f"{d+' corr(lex,geo)':>32}" for d in DATASETS))
        for L in [2, 14, 28]:
            print(f"{'L'+str(L):>6}" + "".join(
                f"{res[(d,mode)]['corr'][L]:>32.3f}" for d in DATASETS))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    cols = {"SALAD": "#c0392b", "MMLU": "#2471a3"}
    for ax, key, title in zip(
            axes, ["z", "zr"],
            ["nesting vs matched null", "after removing lexical overlap"]):
        for d in DATASETS:
            for mode, ls in zip(MODES, ["-", "--"]):
                ax.plot(res[(d, mode)][key], color=cols[d], ls=ls, lw=1.9,
                        label=f"{d} {mode}")
        ax.axhline(0, color="k", lw=1)
        ax.axhline(-1.645, color="0.5", ls=":", lw=1.2, label="p=0.05 one-sided")
        ax.set_xlabel("layer"); ax.set_ylabel("z vs own null")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.suptitle(f"Two taxonomies, matched at M={M_EQUAL}: harm vs academic",
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "taxonomy_comparison.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\nwrote {p}")
    np.savez(os.path.join(OUT, "taxonomy_comparison.npz"),
             **{f"{k[0]}_{k[1]}_{f}": v[f]
                for k, v in res.items() for f in ("ratio", "z", "p", "zr", "pr", "corr")})


if __name__ == "__main__":
    main()
