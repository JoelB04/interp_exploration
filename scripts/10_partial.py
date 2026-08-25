"""Session 3g: does the probe carry anything fluency does not?

Session 3f found the probe score correlates with mean token log-prob at
Spearman 0.30-0.80. But both correlate with the label, so that number is
inflated by their shared cause and cannot be read as "the probe reads fluency".

Two analyses, the second much more useful than the first.

1. PARTIAL CORRELATION. Correlate probe score with fluency while holding the
   label fixed -- i.e. within the true statements and within the false ones
   separately, then pool. Whatever survives is shared structure that is NOT
   truth.

2. RESIDUALISED AUROC -- the analysis that actually settles it. Regress the
   probe score on fluency, keep the residual, and ask whether the residual still
   separates true from false. This is the direct question: after removing
   everything a free black-box score already told you, does reading the residual
   stream add anything?

   Note the regression uses no labels -- it is one score against another -- so
   fitting it on the evaluation set is legitimate rather than leakage.

   Applied to the whole transfer matrix, not just the diagonal, because
   out-of-distribution is where the project's claim lives.

Run from the repo root:  python scripts/10_partial.py
Cache only. No forward passes.
"""

import os
import sys

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cache import cached_acts  # noqa: E402
from data import DATASETS, split  # noqa: E402

MAX_N, SEED, LAYER, N_REPEATS = 400, 0, 17, 20
OUT = "results"


def residualise(y, x):
    """Remove the linear dependence of y on x. Uses no labels."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    if denom < 1e-12:
        return y - y.mean()
    return y - y.mean() - xc * ((xc * (y - y.mean())).sum() / denom)


def partial_spearman(a, b, label):
    """Spearman(a, b) holding a binary label fixed: rank within class, pool."""
    ra, rb = np.zeros(len(a)), np.zeros(len(b))
    for c in (0, 1):
        m = label == c
        if m.sum() < 3:
            continue
        ra[m] = rankdata(a[m]) - rankdata(a[m]).mean()
        rb[m] = rankdata(b[m]) - rankdata(b[m]).mean()
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def dom(acts, labels):
    d = acts[labels == 1].mean(axis=0) - acts[labels == 0].mean(axis=0)
    return d / (np.linalg.norm(d) + 1e-12)


def main():
    os.makedirs(OUT, exist_ok=True)
    S = torch.load("cache/raw_baseline.pt", weights_only=False)
    n = len(DATASETS)

    store = {}
    for d in DATASETS:
        a, y, g, _ = cached_acts(d, "raw", MAX_N, SEED)
        store[d] = dict(A=a.numpy()[:, LAYER, :], y=y, groups=g,
                        flu=S[d]["mean_lp"])
        assert np.array_equal(y, S[d]["labels"]), f"{d}: row mismatch"

    # ---- 1. partial correlation, within dataset
    print("1. PROBE vs FLUENCY, raw correlation and holding the label fixed")
    print(f"{'dataset':>24} {'raw rho':>9} {'partial rho':>12}")
    for d in DATASETS:
        A, y, flu = store[d]["A"], store[d]["y"], store[d]["flu"]
        s = A @ dom(A, y)
        print(f"{d:>24} {spearmanr(s, flu).statistic:>+9.3f} "
              f"{partial_spearman(s, flu, y):>+12.3f}")

    # ---- 2. residualised transfer matrix
    raw_M = np.zeros((N_REPEATS, n, n))
    res_M = np.zeros((N_REPEATS, n, n))
    flu_A = np.zeros((N_REPEATS, n))

    for rep in range(N_REPEATS):
        idx = {}
        for d in DATASETS:
            g = store[d]["groups"]
            trval, test_i = split(g, frac_train=0.8, seed=1000 * rep + 7)
            tr_rel, _ = split(g[trval], frac_train=0.75, seed=1000 * rep + 13)
            idx[d] = (trval[tr_rel], test_i)

        for j, te in enumerate(DATASETS):
            _, test_j = idx[te]
            yb, fb = store[te]["y"][test_j], store[te]["flu"][test_j]
            flu_A[rep, j] = roc_auc_score(yb, fb)

        for i, tr in enumerate(DATASETS):
            train_i, _ = idx[tr]
            w = dom(store[tr]["A"][train_i], store[tr]["y"][train_i])
            for j, te in enumerate(DATASETS):
                _, test_j = idx[te]
                yb = store[te]["y"][test_j]
                fb = store[te]["flu"][test_j]
                s = store[te]["A"][test_j] @ w
                raw_M[rep, i, j] = roc_auc_score(yb, s)
                res_M[rep, i, j] = roc_auc_score(yb, residualise(s, fb))

    fold = lambda M: np.maximum(M, 1 - M)
    off = ~np.eye(n, dtype=bool)
    rm, rs = fold(raw_M).mean(0), fold(res_M).mean(0)
    fa = fold(flu_A).mean(0)

    print(f"\n2. TRANSFER AUROC before and after removing fluency (folded, "
          f"L{LAYER}, {N_REPEATS} splits)")
    print(f"{'target':>24} {'fluency':>9} {'probe':>9} {'probe|flu':>11} {'drop':>8}")
    for j, d in enumerate(DATASETS):
        b, a_ = rm[:, j][off[:, j]].mean(), rs[:, j][off[:, j]].mean()
        print(f"{d:>24} {fa[j]:>9.3f} {b:>9.3f} {a_:>11.3f} {b - a_:>+8.3f}")
    print(f"{'mean off-diagonal':>24} {fa.mean():>9.3f} {rm[off].mean():>9.3f} "
          f"{rs[off].mean():>11.3f} {rm[off].mean() - rs[off].mean():>+8.3f}")
    print(f"{'mean diagonal':>24} {'':>9} {np.diag(rm).mean():>9.3f} "
          f"{np.diag(rs).mean():>11.3f} "
          f"{np.diag(rm).mean() - np.diag(rs).mean():>+8.3f}")

    # ---- 3. the residual matrix itself, since the negation cells are the point
    print(f"\n3. NEGATION CELLS after removing fluency (signed, not folded)")
    ix = {d: i for i, d in enumerate(DATASETS)}
    rawS, resS = raw_M.mean(0), res_M.mean(0)
    print(f"{'cell':>34} {'probe':>9} {'probe|flu':>11}")
    for a, b in [("cities", "neg_cities"), ("sp_en_trans", "neg_sp_en_trans"),
                 ("larger_than", "smaller_than")]:
        for i, j, lbl in [(ix[a], ix[b], f"{a} -> {b}"),
                          (ix[b], ix[a], f"{b} -> {a}")]:
            print(f"{lbl:>34} {rawS[i,j]:>9.3f} {resS[i,j]:>11.3f}")

    torch.save(dict(raw=raw_M, resid=res_M, flu=flu_A, datasets=DATASETS),
               os.path.join(OUT, "partial.pt"))
    print(f"\nsaved -> {OUT}/partial.pt")
    print("\nRead the 'drop' column. Near zero means the probe was never leaning")
    print("on fluency and its advantage is its own. Large means the probe's")
    print("apparent performance was substantially a free black-box score.")


if __name__ == "__main__":
    main()
