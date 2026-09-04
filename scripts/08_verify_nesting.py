import os
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synthetic import GeometryParams, HierarchySpec, generate  

OUT = "results/synthetic_results"
N = 1536
M = 640
BRANCHING = [4, 3, 2, 2, 2]          
SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]
N_REPEATS = 12
ALPHA = 1.12                          #  empirical D of 20 at n=1536


def nesting_ratio(X, y_parent, y_child):
    """mean within-parent centroid distance / mean between-parent.

    Returns (ratio, n_within, n_between) so the pair counts travel with the
    number ie 12 against 66 here.
    """
    kids = np.unique(y_child)
    cen = np.stack([X[y_child == c].mean(axis=0) for c in kids])
    par = np.array([y_parent[y_child == c][0] for c in kids])

    within, between = [], []
    for i, j in combinations(range(len(kids)), 2):
        d = float(np.linalg.norm(cen[i] - cen[j]))
        (within if par[i] == par[j] else between).append(d)
    return float(np.mean(within) / np.mean(between)), len(within), len(between)


def analytic(sigma):
    """sigma / sqrt(1 + sigma^2)"""
    return sigma / np.sqrt(1.0 + sigma ** 2)


def main():
    os.makedirs(OUT, exist_ok=True)
    spec = HierarchySpec(branching=BRANCHING, M=M, n=N)
    w, b = spec.pair_counts()
    print(f"design mirrored: {spec.n_parents} parents, {spec.n_children} children, "
          f"M={M}, n={N}")
    print(f"nesting rests on {w} within-parent pairs against {b} between-parent\n")

    print(f"{'sigma':>7} {'rho':>7} {'analytic':>9} {'measured':>10} {'sd':>7} "
          f"{'ratio':>7}")
    mu, sd = [], []
    for s in SIGMAS:
        vals = []
        for r in range(N_REPEATS):
            X, yp, yc = generate(spec, GeometryParams(sigma_ratio=s, alpha=ALPHA),
                                 np.random.default_rng(1000 + r))
            vals.append(nesting_ratio(X, yp, yc)[0])
        m, s_ = float(np.mean(vals)), float(np.std(vals, ddof=1))
        mu.append(m); sd.append(s_)
        a = analytic(s)
        print(f"{s:>7.2f} {1-s**2:>7.2f} {a:>9.4f} {m:>10.4f} {s_:>7.4f} "
              f"{m/a if a > 0 else float('nan'):>7.3f}")

    mu, sd = np.array(mu), np.array(sd)
    ss = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(ss, analytic(ss), color="k", ls="--", lw=1.6,
            label=r"analytic  $\sigma/\sqrt{1+\sigma^2}$")
    ax.plot(ss, ss, color="0.6", ls=":", lw=1.2, label=r"$\sigma$ (naive guess)")
    ax.errorbar(SIGMAS, mu, yerr=sd, color="#2471a3", lw=1.8, marker="o",
                ms=5, capsize=3, label=f"measured, {N_REPEATS} seeds")
    ax.set_xlabel(r"planted $\sigma$  (child scatter / parent scatter)")
    ax.set_ylabel("measured nesting ratio")
    ax.set_title("does the measured nesting track the planted value?",
                 fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(alpha=.25)

    ax = axes[1]
    ax.errorbar(SIGMAS, sd, color="#c0392b", lw=1.8, marker="o", ms=5)
    ax.set_xlabel(r"planted $\sigma$")
    ax.set_ylabel("sd of the nesting ratio over seeds")
    ax.set_title(f"noise floor of a nesting measurement\n"
                 f"({w} within-parent pairs, M={M})", fontweight="bold")
    ax.grid(alpha=.25)

    fig.suptitle("Check 3: between-manifold structure, under the exact "
                 "empirical design", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "check3_nesting.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
