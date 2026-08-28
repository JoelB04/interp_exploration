"""Session 4e: CHECK 2 -- does the estimator recover the planted dimension?

Check 1 showed the generator is correct and that finite M distorts the spectrum
in a specific way: middle eigenvalues inflated, tail truncated at rank M-1,
trace preserved. It did that at ONE alpha, where D_true = 38 and D << M -- the
easy regime.

Check 2 asks the question that actually decides whether M=640 is defensible:

    How large can the true dimension be before we can no longer measure it?

This matters because nobody knows what D real harm manifolds have. If they sit
near 40, check 1 already says we are fine. If they sit near 400, we are
estimating a quantity of order M from M samples and the answer will be mostly
an artifact of the sample size.

Sweep D_true from 5 to ~640 at several M. Compare the ESTIMATE against the
ANALYTIC value computed from the spectrum array itself -- no sampling, no
noise, just (sum lam)^2 / sum lam^2 on the numbers we fed in. That is ground
truth, and a sweep without it would only show the estimator is monotonic, which
is far weaker than showing it is correct.

R_M is reported alongside, with a caveat. Under this generative model
spectrum() normalises total variance to within_scale^2 and ||c|| is roughly
sqrt(n) for every manifold, so R_M and D_M are algebraically linked:

    R_M ~ within_scale / (sqrt(n) * sqrt(D_M))

They will trace a clean inverse curve BY CONSTRUCTION. That is a correctness
check, not a discovery. In real data total variance varies across manifolds and
grows with depth, so the two decouple and carry different information -- but
not here, and the write-up should say so.

Predict before running:
  1. At M=640, up to what D_true is recovery within 10%?
  2. Does the estimate saturate, or keep rising past the point it is accurate?
     If it saturates, at what value -- and what is that value related to?
  3. Is the bias multiplicative (constant ratio) or additive (constant offset)?
     This decides whether a correction is even possible.

Run from the repo root:  python scripts/15_verify_dimension.py
Pure numpy. No model, no data.
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
D_TARGETS = [5, 10, 20, 40, 80, 160, 320, 640]
M_VALUES = [160, 640, 2560]
N_REPEATS = 3


# ---------------------------------------------------------------------------
# YOURS.
# ---------------------------------------------------------------------------
def alpha_for_D(n: int, D_target: float, tol: float = 1e-3) -> float:
    """Find the power-law exponent alpha whose spectrum has participation
    ratio D_target.

    Why this exists. alpha is a brutally sharp knob -- D falls from 748 to 38
    between alpha=0.5 and alpha=1.0. Sweeping alpha on a uniform grid gives a
    useless spread of D, and "we used alpha=0.93" is not a sentence anyone can
    interpret. Sweeping D directly is both easier to reason about and easier to
    defend, so invert the relationship once and never think about alpha again.

    How. For a spectrum lam_i proportional to i^-alpha,

        D(alpha) = (sum lam)^2 / sum lam^2

    is continuous and STRICTLY DECREASING in alpha: larger alpha concentrates
    variance into fewer directions. A monotone scalar function on a bounded
    interval is exactly what bisection is for.

    Bracket: alpha = 0 gives D = n exactly (isotropic, every eigenvalue equal).
    Large alpha drives D toward 1. So [0, ~20] brackets any D in [1, n].

    Note spectrum() already normalises total variance, and D is invariant to
    that scaling anyway -- (c*lam) leaves (sum)^2/sum^2 unchanged. So you can
    call spectrum() with any scale and ignore within_scale entirely here.

    Check when you have it:
        D of spectrum(n, alpha_for_D(n, 40)) should come back 40 to within tol
        alpha_for_D(n, n) should return ~0
        the returned alpha should DECREASE as D_target increases
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------
def participation_ratio(lam: np.ndarray) -> float:
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def effective_radius(lam: np.ndarray, c_norm: float) -> float:
    """CLS convention, fixed project-wide: sqrt(sum lam^2 / sum lam) / ||c||."""
    return float(np.sqrt((lam ** 2).sum() / lam.sum()) / c_norm)


def measure(M: int, alpha: float, seed: int):
    """One manifold -> (lam_hat, ||c_hat||)."""
    spec = HierarchySpec(branching=[1], M=M, n=N)
    X, _, _ = generate(spec, GeometryParams(alpha=alpha),
                       np.random.default_rng(seed))
    c = X.mean(axis=0)
    Xc = X - c
    s = np.linalg.svd(Xc, compute_uv=False)
    return s ** 2 / (M - 1), float(np.linalg.norm(c))


def main():
    os.makedirs(OUT, exist_ok=True)

    alphas, D_true, R_true = {}, {}, {}
    print("solving alpha for each target D")
    print(f"{'D_target':>9} {'alpha':>8} {'D_analytic':>11} {'err':>8}")
    for D in D_TARGETS:
        a = alpha_for_D(N, D)
        lam = spectrum(N, a, 1.0)
        alphas[D], D_true[D] = a, participation_ratio(lam)
        R_true[D] = effective_radius(lam, np.sqrt(N))   # ||c|| ~ sqrt(n)
        print(f"{D:>9} {a:>8.4f} {D_true[D]:>11.2f} {D_true[D]/D-1:>+8.4f}")

    res = {M: {"D": [], "Dsd": [], "R": []} for M in M_VALUES}
    print(f"\n{'D_true':>8}" + "".join(f"{'M='+str(M):>20}" for M in M_VALUES))
    for D in D_TARGETS:
        row = f"{D_true[D]:>8.1f}"
        for M in M_VALUES:
            reps = [measure(M, alphas[D], s) for s in range(N_REPEATS)]
            ds = [participation_ratio(l) for l, _ in reps]
            rs = [effective_radius(l, c) for l, c in reps]
            res[M]["D"].append(np.mean(ds))
            res[M]["Dsd"].append(np.std(ds, ddof=1) if len(ds) > 1 else 0.0)
            res[M]["R"].append(np.mean(rs))
            row += f"{np.mean(ds):>13.1f} ({np.mean(ds)/D_true[D]:.2f})"
        print(row)

    Dt = np.array([D_true[D] for D in D_TARGETS])
    Rt = np.array([R_true[D] for D in D_TARGETS])
    cmap = plt.get_cmap("plasma")
    col = {M: cmap(i / max(1, len(M_VALUES) - 1)) for i, M in enumerate(M_VALUES)}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    ax = axes[0]
    lo, hi = Dt.min() * 0.7, Dt.max() * 1.5
    ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=1.3, label="perfect")
    for M in M_VALUES:
        ax.errorbar(Dt, res[M]["D"], yerr=res[M]["Dsd"], color=col[M],
                    lw=1.8, marker="o", ms=4, capsize=2, label=f"M={M}")
        ax.axhline(M - 1, color=col[M], ls=":", lw=1, alpha=.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("analytic D"); ax.set_ylabel("estimated D")
    ax.set_title("recovery of the participation ratio", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=.25, which="both")

    ax = axes[1]
    for M in M_VALUES:
        ax.semilogx(Dt, np.array(res[M]["D"]) / Dt, color=col[M], lw=1.8,
                    marker="o", ms=4, label=f"M={M}")
    ax.axhline(1.0, color="k", ls="--", lw=1.2)
    ax.axhspan(0.9, 1.1, color="0.5", alpha=.15)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("analytic D"); ax.set_ylabel("D_hat / D_true")
    ax.set_title("relative bias   (band = within 10%)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=.25, which="both")

    ax = axes[2]
    ax.loglog(Dt, Rt, color="k", ls="--", lw=1.3, label="analytic")
    for M in M_VALUES:
        ax.loglog(Dt, res[M]["R"], color=col[M], lw=1.8, marker="o", ms=4,
                  label=f"M={M}")
    ax.set_xlabel("analytic D"); ax.set_ylabel("effective radius R_M")
    ax.set_title("R_M -- inverse to D by construction here", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=.25, which="both")

    fig.suptitle("Check 2: how large can D be before we cannot measure it? "
                 f"(n={N}; dotted lines = the M-1 rank cap)", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "check2_dimension_recovery.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\nwrote {p}")

    print("\nusable range -- largest analytic D recovered within 10%:")
    for M in M_VALUES:
        ok = [d for d, e in zip(Dt, res[M]["D"]) if e / d >= 0.9]
        print(f"  M={M:>6}  D <= {max(ok):.0f}" if ok else
              f"  M={M:>6}  none of the tested D values")


if __name__ == "__main__":
    main()
