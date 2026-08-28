"""Session 4g: CHECK 3 -- does the measured nesting track the planted one?

Checks 1 and 2 verified the WITHIN-manifold machinery: the generator produces
the spectrum it claims, and M=640 measures D reliably up to about 30. Check 3
verifies the BETWEEN-manifold machinery, which is what the hierarchy claim is
actually made of.

The statistic. For every pair of child manifolds, take the distance between
their centroids. Split those pairs by whether the two children share a parent:

    nesting ratio = mean(within-parent distance) / mean(between-parent distance)

Small means children huddle around their parent -- a real hierarchy. Near 1
means the tree is decorative and children are scattered as widely as parents.

WHAT IT SHOULD EQUAL, and this is worth deriving rather than assuming.
Parent centroids are drawn N(0, I) in n dimensions, so two parents sit about
sqrt(2n) apart. Two children of the SAME parent differ only by their own
scatter: sigma * sqrt(2n). Two children of DIFFERENT parents differ by both,
and the two contributions are independent, so they add in quadrature:
sqrt(2n) * sqrt(1 + sigma^2). Hence

    nesting ratio  =  sigma / sqrt(1 + sigma^2)

NOT sigma itself. At sigma=0.3 that is 0.287, and an earlier one-off measurement
gave 0.288. In Joel's rho parameterisation, sigma = sqrt(1 - rho).

Two things this check is really testing:
  - that generate() places children around parents the way it claims;
  - that the estimator recovers it at M=640 with only 12 within-parent pairs,
    which is the real sample size of every nesting claim in this project.

That second point is the one that matters. 12 pairs is thin, 6 of them come
from Malicious Use alone, and the error bars here are the first honest look at
how noisy a nesting measurement is under the actual design.

Run from the repo root:  python scripts/17_verify_nesting.py
Pure numpy.
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

from synthetic import GeometryParams, HierarchySpec, generate  # noqa: E402

OUT = "results/synthetic_results"
N = 1536
M = 640
BRANCHING = [4, 3, 2, 2, 2]          # the empirical design, mirrored exactly
SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]
N_REPEATS = 12
ALPHA = 1.12                          # ~ empirical D of 20 at n=1536


def nesting_ratio(X, y_parent, y_child):
    """mean within-parent centroid distance / mean between-parent.

    Returns (ratio, n_within, n_between) so the pair counts travel with the
    number -- 12 against 66 here, and nobody should read the ratio without
    them.
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
    """sigma / sqrt(1 + sigma^2) -- see the module docstring."""
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

    print("\nThe right-hand panel is the number to carry forward. It is the "
          "spread of a nesting\nmeasurement under this exact design with "
          "nothing but seed varying -- so any\nempirical difference smaller "
          "than it is not a finding.")


if __name__ == "__main__":
    main()
