"""Session 4d: verify the synthetic generator before trusting anything built on it.

Three checks, in order. This file currently implements CHECK 1 only.

  1. Does the recovered spectrum match the spectrum we fed in?     <- here
  2. Does estimated D match ANALYTIC D at large M?
  3. Does the nesting ratio track sigma_ratio?

--------------------------------------------------------------------------
CHECK 1: the recovered spectrum
--------------------------------------------------------------------------

We hand generate() a ground-truth spectrum: n numbers lam_1 >= ... >= lam_n,
the variances along each axis of Sigma_w. We then draw M points, discard all
knowledge of Sigma_w, and ask what covariance those points imply. Eigenvalues
of the sample covariance are lam_hat.

lam_hat CANNOT equal lam, and the way it fails is the finding:

  Rank truncation. The sample covariance of M points has rank at most M-1.
  At M=640 in n=1536 that is 639 non-zero eigenvalues and 897 exact zeros,
  against 1536 true non-zero entries. The tail is invisible at any M < n.

  Trace preservation. The sample's TOTAL variance is close to the truth. So
  the variance that truly occupied 1536 directions is packed into 639, and
  each observed lam_hat_i is inflated to compensate.

  Consequence for D. D = (sum lam)^2 / sum lam^2. Inflating eigenvalues raises
  sum lam^2 while sum lam is roughly fixed, so D_hat < D. The spectrum plot
  does not merely detect that bias, it explains it.

This is why check 1 comes before check 2: it distinguishes "the generator is
wrong" from "the estimator is biased". Generator error persists at every M;
estimator bias shrinks as M grows.

Run from the repo root:  python scripts/14_verify_generator.py
No model, no data. Pure numpy, seconds.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from synthetic import GeometryParams, HierarchySpec, generate, spectrum  # noqa: E402

OUT = "results/synthetic_results"
N = 1536
ALPHA = 1.0
M_VALUES = [160, 640, 2560, 10240]
N_REPEATS = 5


# ---------------------------------------------------------------------------
# YOURS. The scaffolding above and below is bookkeeping; this is the check.
# ---------------------------------------------------------------------------
def recovered_spectrum(X: np.ndarray) -> np.ndarray:
    """Eigenvalues of the sample covariance of one manifold's points.

    X is (M, n) -- the points of a SINGLE child manifold, nothing else.
    Return lam_hat, descending, length min(M, n).

    What to do:
      - centre X on its own mean. Not centring leaves the manifold's offset
        from the origin in the covariance as one huge spurious eigenvalue.
      - take singular values of the centred matrix, not eigenvalues of an
        (n, n) covariance you would have to build first. At M < n the SVD is
        both faster and better conditioned.
      - convert singular values to variances. s^2 / (M - 1), NOT s^2 / M --
        the mean you subtracted costs one degree of freedom, and it is the
        same (M-1) that caps the rank.

    src/geometry.py:category_spectrum already does this for a labelled array.
    Write it here anyway -- it is four lines and you should be able to derive
    the (M-1) from the rank argument in the module docstring.

    Sanity check on your own output before going further:
        lam_hat.sum()  should be close to lam_true.sum()   (trace preserved)
        (lam_hat > 1e-10).sum()  should be min(M - 1, n)   (rank truncated)
    """
    
    M = X.shape[0]
    X_c = X - X.mean(axis = 0)
    lam_hat = []

    s = np.linalg.svd(X_c, compute_uv = False)

    return s**2 / (M-1)



# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------
def one_manifold(M: int, alpha: float, seed: int) -> np.ndarray:
    """Points of a single manifold, no hierarchy involved."""
    spec = HierarchySpec(branching=[1], M=M, n=N)
    params = GeometryParams(alpha=alpha)
    X, _, _ = generate(spec, params, np.random.default_rng(seed))
    return X


def main():
    os.makedirs(OUT, exist_ok=True)
    lam_true = spectrum(N, ALPHA, 1.0)
    D_true = lam_true.sum() ** 2 / (lam_true ** 2).sum()

    print(f"ground truth: n={N}  alpha={ALPHA}  "
          f"total variance={lam_true.sum():.4f}  analytic D={D_true:.1f}")
    print(f"{'M':>7} {'gamma=n/M':>10} {'rank':>6} {'trace_hat':>10} "
          f"{'trace_err':>10} {'D_hat':>8} {'D_hat/D':>8}")

    curves = {}
    for M in M_VALUES:
        reps = [recovered_spectrum(one_manifold(M, ALPHA, s))
                for s in range(N_REPEATS)]
        L = min(len(r) for r in reps)
        lam_hat = np.stack([r[:L] for r in reps]).mean(0)
        curves[M] = lam_hat

        rank = int((lam_hat > 1e-10).sum())
        D_hat = lam_hat.sum() ** 2 / (lam_hat ** 2).sum()
        print(f"{M:>7} {N/M:>10.2f} {rank:>6} {lam_hat.sum():>10.4f} "
              f"{lam_hat.sum()/lam_true.sum()-1:>+10.3f} "
              f"{D_hat:>8.1f} {D_hat/D_true:>8.3f}")

    # ---- plot: truth against recovery, log-log
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    cmap = plt.get_cmap("viridis")

    ax = axes[0]
    ax.loglog(np.arange(1, N + 1), lam_true, color="k", lw=2.4,
              label=f"true (alpha={ALPHA})", zorder=5)
    for i, (M, lam_hat) in enumerate(curves.items()):
        ax.loglog(np.arange(1, len(lam_hat) + 1), np.maximum(lam_hat, 1e-18),
                  color=cmap(i / max(1, len(curves) - 1)), lw=1.6,
                  label=f"M={M}  (n/M={N/M:.1f})")
    ax.set_xlabel("eigenvalue index"); ax.set_ylabel("variance")
    ax.set_ylim(1e-8, None)
    ax.set_title("recovered vs true spectrum", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=.25, which="both")

    ax = axes[1]
    for i, (M, lam_hat) in enumerate(curves.items()):
        L = len(lam_hat)
        ax.semilogx(np.arange(1, L + 1), lam_hat / lam_true[:L],
                    color=cmap(i / max(1, len(curves) - 1)), lw=1.6,
                    label=f"M={M}")
    ax.axhline(1.0, color="k", ls="--", lw=1.2)
    ax.set_xlabel("eigenvalue index"); ax.set_ylabel("lam_hat / lam_true")
    ax.set_title("ratio to truth  (1.0 = perfect recovery)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=.25, which="both")

    fig.suptitle("Check 1: does the sample spectrum recover the planted one?",
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "check1_spectrum_recovery.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    print 
    main()
