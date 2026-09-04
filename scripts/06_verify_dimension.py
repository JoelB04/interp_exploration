import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np  
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from synthetic import GeometryParams, HierarchySpec, generate, spectrum  
from geometry import participation_ratio, effective_radius  

OUT = "results/synthetic_results"
N = 1536
D_TARGETS = [5, 10, 15, 30, 40, 50, 60]
M_VALUES = [160, 640, 2560]
N_REPEATS = 3


def alpha_for_D(n: int, D_target: float, tol: float = 1e-3) -> float:
    """Find the power-law exponent alpha whose spectrum has participation
    ratio D_target.

    For a spectrum lam_i proportional to i^-alpha,

        D(alpha) = (sum lam)^2 / sum lam^2

    is continuous and strictly decreasing in alpha so larger alpha concentrates
    variance into fewer directions. 

    Bracket: alpha = 0 gives D = n exactly (isotropic, every eigenvalue equal).
    Large alpha drives D toward 1.

    Spectrum() already normalises total variance, and D is invariant to
    that scaling.
    """

    hi=20.0
    lo=0.0

    for _ in range(100):
        mid = (lo+hi)/2
        D_mid = participation_ratio(spectrum(n,mid,1.0))

        if abs(D_mid - D_target)<tol:
            return mid
        if D_mid > D_target:
            lo = mid
        else:
            hi = mid

    return (lo+hi)/2


# participation_ratio and effective_radius come from src/geometry.py, which is
# where the conventions are fixed. Defining them here too was how the project
# ended up with three copies of the same formula.


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

        ok = [d for d, e in zip(Dt, res[M]["D"]) if abs(e / d - 1) <= 0.1]
        print(f"  M={M:>6}  D <= {max(ok):.0f}" if ok else
              f"  M={M:>6}  none of the tested D values")


if __name__ == "__main__":
    main()
