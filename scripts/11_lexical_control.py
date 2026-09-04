
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
MODES = ["raw", "chat"]
M_EQUAL = 640
LAYERS = [2, 8, 14, 20, 26, 28]
N_PERM = 4000
SEED = 0


def _no_loader():
    raise RuntimeError("cache miss run scripts/13_extract.py first")


def lexical_vectors():
    """Mean TF-IDF vector per task. -> (T, V), names, parents."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    tasks = salad.design_tasks()
    docs, owner = [], []
    rng = np.random.default_rng(SEED)
    for ti, (t, _) in enumerate(tasks):
        qs = salad.fetch_task(t, cap=salad.EXTRACT_CAP, seed=SEED)
        qs = [qs[i] for i in rng.choice(len(qs), M_EQUAL, replace=False)]
        docs.extend(qs); owner.extend([ti] * len(qs))

    V = TfidfVectorizer(lowercase=True, stop_words="english", min_df=5,
                        sublinear_tf=True)
    X = V.fit_transform(docs)
    owner = np.array(owner)
    vecs = np.stack([np.asarray(X[owner == ti].mean(axis=0)).ravel()
                     for ti in range(len(tasks))])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs, [t for t, _ in tasks], np.array([d for _, d in tasks]), V


def geo_centroids(mode):
    tasks = salad.design_tasks()
    rng = np.random.default_rng(SEED)
    cen = np.zeros((len(tasks), 29, 1536), dtype=np.float32)
    for ti, (t, _) in enumerate(tasks):
        cap = min(len(salad.fetch_task(t, cap=10 ** 9)), salad.EXTRACT_CAP)
        A, _ = cached_acts(t, mode, _no_loader, M=cap, seed=SEED)
        cen[ti] = A.numpy()[rng.choice(cap, M_EQUAL, replace=False)].mean(axis=0)
    return cen


def perm_diff_test(vals, groups, sizes, n_perm, seed):
    """mean(within) - mean(between), permuted over groupings of fixed sizes."""
    T = len(groups)
    pairs = list(combinations(range(T), 2))

    def stat(g):
        w = [v for (i, j), v in zip(pairs, vals) if g[i] == g[j]]
        b = [v for (i, j), v in zip(pairs, vals) if g[i] != g[j]]
        return float(np.mean(w) - np.mean(b))

    real = stat(groups)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(T); g = np.zeros(T, int); s = 0
        for gi, sz in enumerate(sizes):
            g[perm[s:s + sz]] = gi; s += sz
        null[k] = stat(g)
    return real, null, float((null <= real).mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    vecs, names, par, vect = lexical_vectors()
    T = len(names)
    pairs = list(combinations(range(T), 2))
    groups = np.unique(par, return_inverse=True)[1]
    sizes = sorted(np.unique(groups, return_counts=True)[1], reverse=True)
    same = np.array([groups[i] == groups[j] for i, j in pairs])

    lex = np.array([1.0 - float(vecs[i] @ vecs[j]) for i, j in pairs])

    # is Malicious Use lexically tight?
    print("LEXICAL distance (1 - tfidf cosine), within-parent pairs by parent")
    print(f"{'parent':>34} {'pairs':>6} {'mean within':>12} {'vs all pairs':>13}")
    for d in sorted(set(par)):
        idx = [k for k, (i, j) in enumerate(pairs)
               if par[i] == d and par[j] == d]
        if idx:
            print(f"{d[:32]:>34} {len(idx):>6} {lex[idx].mean():>12.4f} "
                  f"{lex[idx].mean() - lex.mean():>+13.4f}")
    print(f"{'ALL PAIRS':>34} {len(pairs):>6} {lex.mean():>12.4f}")

    lr, ln, lp = perm_diff_test(lex, groups, sizes, N_PERM, 5)
    print(f"\nlexical nesting: within - between = {lr:+.4f}, "
          f"null {ln.mean():+.4f} +/- {ln.std(ddof=1):.4f}, p = {lp:.4f}")
    print("  (if this is significant, siblings do share vocabulary and the "
          "geometry result")

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for mi, mode in enumerate(MODES):
        cen = geo_centroids(mode)
        print(f"\n=== {mode}")
        print(f"{'layer':>6} {'corr(lex,geo)':>14} {'raw p':>8} "
              f"{'resid stat':>11} {'resid p':>9}")
        rp, pp = [], []
        for L in LAYERS:
            geo = np.array([float(np.linalg.norm(cen[i, L] - cen[j, L]))
                            for i, j in pairs])
            if not np.isfinite(geo).all() or geo.std() == 0:
                continue
            # Normalise to units of the mean pairwise distance at this specific layer.
            # Residual-stream norms grow ~15,000x from layer 0 to 28, so raw
            # residual magnitudes climb mechanically with depth and cannot be
            # compared across layers. Dividing by the layer mean makes the
            # statistic scale-free; it leaves every p-value unchanged, since a
            # positive rescaling cannot alter a permutation test.
            geo = geo / geo.mean()
            corr = float(np.corrcoef(lex, geo)[0, 1])

            gr, gn, gp = perm_diff_test(geo, groups, sizes, N_PERM, 7)

            # residualise geometric distance on lexical distance
            A = np.vstack([lex, np.ones_like(lex)]).T
            beta, *_ = np.linalg.lstsq(A, geo, rcond=None)
            resid = geo - A @ beta
            rr, rn, rpv = perm_diff_test(resid, groups, sizes, N_PERM, 11)

            print(f"{L:>6} {corr:>14.3f} {gp:>8.4f} {rr:>11.3f} {rpv:>9.4f}")
            rows.append((mode, L, corr, gp, rpv))
            rp.append(corr); pp.append(rpv)

        ax = axes[mi]
        ax.plot(LAYERS[:len(rp)], rp, color="#2471a3", lw=2, marker="o",
                label="corr(lexical, geometric)")
        ax.plot(LAYERS[:len(pp)], pp, color="#c0392b", lw=2, marker="s",
                label="p, taxonomy after removing lexical")
        ax.axhline(0.05, color="#c0392b", ls=":", lw=1.2)
        ax.axhline(0, color="k", lw=.8)
        ax.set_ylim(-0.2, 1.05)
        ax.set_xlabel("layer"); ax.set_title(mode, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=.25)

    fig.suptitle("Is the taxonomy effect conceptual or lexical? "
                 "(red below the dotted line = survives the lexical control)",
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "lexical_control.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\nwrote {p}")

    # what words actually distinguish each parent
    terms = np.array(vect.get_feature_names_out())
    print("\ntop TF-IDF terms per parent (what the lexical signal is made of):")
    for d in sorted(set(par)):
        v = vecs[par == d].mean(axis=0) - vecs.mean(axis=0)
        print(f"  {d[:34]:<36} " + ", ".join(terms[np.argsort(-v)[:8]]))


if __name__ == "__main__":
    main()
