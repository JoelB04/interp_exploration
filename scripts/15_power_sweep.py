"""Power sweep at the real regime: what nesting could this design have detected?

The gap this closes. Check 3 verified that measured nesting tracks planted
nesting, but it ran with within-manifold spread at 0.018 of the centroid
separation. Real manifolds sit at 1.5-5.8 (script 09), two orders of magnitude
away, so check 3's noise floor says nothing about the design that was actually
run. Without this, every null in the project is "we saw nothing" rather than a
bound on what there was to see.

Three stages:

  A  calibrate within_scale so the synthetic reproduces the measured
     within-spread / between-centroid ratio
  B  validate the fast path against src/synthetic.generate()
  C  sweep sigma_ratio at the calibrated regime, run the EXACT empirical test
     (13 manifolds, branching [4,3,2,2,2], M=640, matched random partition),
     and report the weakest nesting still caught at 80% power

Why this is cheap. The nesting statistic reads centroids only, and the sample
mean of M points drawn from N(0, Sigma_w) is exactly N(0, Sigma_w / M). So a
manifold is fully represented by its measured centroid and the M x n point cloud
never has to exist. Distances are invariant under the orthogonal U in
Sigma_w = U diag(lam) U^T and every other term in the model is isotropic, so the
sweep runs in the eigenbasis and U is never formed either. Stage B checks that
claim against the real generator instead of asserting it.
"""

import os
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synthetic import GeometryParams, HierarchySpec, generate, spectrum

OUT = "results/synthetic_results"
N = 1536
M = 640
BRANCHING = [4, 3, 2, 2, 2]
ALPHA = 1.3                 # fitted 1.0-1.9 on real data; only the trace matters
N_PERM = 2000
N_SEED = 300
SEED = 0

# Measured on real activations by scripts/09 (b=0, the full sample).
EMPIRICAL = {"raw": dict(calib=2.38, ratio=0.772),
             "request": dict(calib=1.85, ratio=0.756)}

SIGMAS = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40,
                   0.50, 0.65, 0.80, 1.00])

PAIRS = list(combinations(range(sum(BRANCHING)), 2))
TRUE_G = np.concatenate([[p] * k for p, k in enumerate(BRANCHING)])
SIZES = sorted(BRANCHING, reverse=True)
SAME_TRUE = np.array([TRUE_G[i] == TRUE_G[j] for i, j in PAIRS])


def pair_distances(cen):
    return np.array([np.linalg.norm(cen[i] - cen[j]) for i, j in PAIRS])


def ratio_from(d, same):
    return float(d[same].mean() / d[~same].mean())


def draw_centroids(sigma, within_scale, rng, n=N, m=M, alpha=ALPHA):
    """Measured centroids of the 13 child manifolds, in the eigenbasis.

    Exact rather than approximate: M enters only through the variance of the
    centroid estimate, and that variance is analytic.
    """
    lam = spectrum(n, alpha, within_scale)
    parents = rng.normal(size=(len(BRANCHING), n))
    true_c = np.stack([parents[p] + sigma * rng.normal(size=n) for p in TRUE_G])
    est_err = rng.normal(size=(len(TRUE_G), n)) * np.sqrt(lam / m)
    return true_c + est_err, float(np.sqrt(lam.sum()))


def perm_p(d, rng):
    """Matched random partition, identical in form to scripts/10."""
    real = ratio_from(d, SAME_TRUE)
    T = len(TRUE_G)
    hits = 0
    for _ in range(N_PERM):
        perm = rng.permutation(T)
        g = np.zeros(T, int)
        s = 0
        for gi, sz in enumerate(SIZES):
            g[perm[s:s + sz]] = gi
            s += sz
        sm = np.array([g[i] == g[j] for i, j in PAIRS])
        if ratio_from(d, sm) <= real:
            hits += 1
    return real, hits / N_PERM


def measure_calib(sigma, within_scale, rng, reps=40):
    """within-manifold spread / mean between-parent centroid distance."""
    out = []
    for _ in range(reps):
        cen, spread = draw_centroids(sigma, within_scale, rng)
        d = pair_distances(cen)
        out.append(spread / d[~SAME_TRUE].mean())
    return float(np.mean(out))


def solve_within_scale(target, sigma, rng):
    """Bisect within_scale to hit a target calibration ratio."""
    lo, hi = 1.0, 20000.0
    for _ in range(35):
        mid = 0.5 * (lo + hi)
        if measure_calib(sigma, mid, rng) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def validate_fast_path():
    """The fast path must agree with generate(). n is reduced so the full point
    cloud stays affordable; the identity being checked does not depend on n."""
    print("\nStage B: fast path vs src/synthetic.generate()")
    n_s, m_s, reps = 256, 640, 80
    ws = 30.0
    print(f"  n={n_s}, M={m_s}, within_scale={ws}, {reps} reps per method")
    print(f"  {'sigma':>6} {'generate()':>12} {'fast path':>12} {'diff':>9}")
    worst = 0.0
    for sigma in (0.1, 0.3, 0.6):
        spec = HierarchySpec(branching=BRANCHING, M=m_s, n=n_s)
        params = GeometryParams(sigma_ratio=sigma, within_scale=ws, alpha=ALPHA)
        slow = []
        for r in range(reps):
            X, yp, yc = generate(spec, params, np.random.default_rng(9000 + r))
            ids = np.unique(yc)
            cen = np.stack([X[yc == c].mean(axis=0) for c in ids])
            g = np.array([yp[yc == c][0] for c in ids])
            same = np.array([g[i] == g[j] for i, j in PAIRS])
            slow.append(ratio_from(pair_distances(cen), same))
        fast = []
        for r in range(reps):
            rg = np.random.default_rng(11000 + r)
            cen, _ = draw_centroids(sigma, ws, rg, n=n_s, m=m_s)
            fast.append(ratio_from(pair_distances(cen), SAME_TRUE))
        a, b = float(np.mean(slow)), float(np.mean(fast))
        worst = max(worst, abs(b - a))
        print(f"  {sigma:>6.2f} {a:>12.4f} {b:>12.4f} {b - a:>+9.4f}")
    print(f"  worst absolute disagreement: {worst:.4f}")
    return worst


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("=" * 74)
    print("Stage A: recalibrate within_scale to the measured regime")
    print("=" * 74)
    base = measure_calib(0.3, 1.0, rng)
    print(f"  synthetic default within_scale=1.0 gives calib = {base:.4f}")
    print("  that is the 0.018 check 3 ran at. Real data is 1.5-5.8.\n")
    print(f"  {'mode':>9} {'target':>8} {'within_scale':>13} {'achieved':>10}")
    scales = {}
    for mode, emp in EMPIRICAL.items():
        ws = solve_within_scale(emp["calib"], 0.3, rng)
        got = measure_calib(0.3, ws, rng, reps=200)
        scales[mode] = ws
        print(f"  {mode:>9} {emp['calib']:>8.2f} {ws:>13.1f} {got:>10.2f}")

    validate_fast_path()

    print("\n" + "=" * 74)
    print(f"Stage C: power sweep, {N_SEED} seeds x {N_PERM} permutations per point")
    print("=" * 74)

    results = {}
    for mode, ws in scales.items():
        ratios = np.zeros((len(SIGMAS), N_SEED))
        pvals = np.zeros((len(SIGMAS), N_SEED))
        for si, sigma in enumerate(SIGMAS):
            for k in range(N_SEED):
                rg = np.random.default_rng(100000 + 1000 * si + k)
                cen, _ = draw_centroids(sigma, ws, rg)
                ratios[si, k], pvals[si, k] = perm_p(pair_distances(cen), rg)
            print(f"\r  {mode}: sigma {sigma:.2f}    ", end="", flush=True)
        print()
        results[mode] = dict(ratio=ratios, p=pvals, within_scale=ws)

        power = (pvals < 0.05).mean(axis=1)
        print(f"\n  {mode}   within_scale={ws:.0f}, calib={EMPIRICAL[mode]['calib']}")
        print(f"  {'sigma':>6} {'mean ratio':>11} {'sd':>7} {'power':>7}")
        for si, sigma in enumerate(SIGMAS):
            print(f"  {sigma:>6.2f} {ratios[si].mean():>11.4f} "
                  f"{ratios[si].std(ddof=1):>7.4f} {power[si]:>7.2f}")

        obs = EMPIRICAL[mode]["ratio"]
        ok = np.where(power >= 0.80)[0]
        if len(ok):
            si = ok[-1]
            print(f"\n  weakest nesting caught at 80% power: sigma = {SIGMAS[si]:.2f}, "
                  f"which reads as ratio {ratios[si].mean():.3f}")
            print(f"  observed on real data: {obs:.3f}  ->  "
                  f"{'inside' if obs <= ratios[si].mean() else 'OUTSIDE'} "
                  "the detectable range")
        else:
            print("\n  no sigma on the grid reaches 80% power")

    np.savez(os.path.join(OUT, "power_sweep.npz"), sigmas=SIGMAS,
             **{f"{k}_{m}": results[m][k] for m in results for k in ("ratio", "p")})

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    cols = {"raw": "#2471a3", "request": "#c0392b"}

    ax = axes[0]
    for mode, r in results.items():
        mu = r["ratio"].mean(axis=1)
        sd = r["ratio"].std(axis=1, ddof=1)
        ax.plot(SIGMAS, mu, color=cols[mode], lw=2, marker="o",
                label=f"{mode}  (calib {EMPIRICAL[mode]['calib']})")
        ax.fill_between(SIGMAS, mu - sd, mu + sd, color=cols[mode], alpha=.18)
        ax.axhline(EMPIRICAL[mode]["ratio"], color=cols[mode], ls="--", lw=1.2)
    ax.plot(SIGMAS, SIGMAS / np.sqrt(1 + SIGMAS ** 2), color="k", ls=":", lw=1.4,
            label="noiseless  sigma/sqrt(1+sigma^2)")
    ax.set_xlabel("planted sigma  (child scatter / parent scatter)")
    ax.set_ylabel("measured nesting ratio")
    ax.set_title("dashed = observed on real data;\n"
                 "gap to dotted = what the real regime costs",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axes[1]
    for mode, r in results.items():
        ax.plot(SIGMAS, (r["p"] < 0.05).mean(axis=1), color=cols[mode], lw=2,
                marker="s", label=mode)
    ax.axhline(0.80, color="0.3", ls="--", lw=1.2, label="80% power")
    ax.axhline(0.05, color="0.6", ls=":", lw=1)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("planted sigma")
    ax.set_ylabel("P(detected at p < 0.05)")
    ax.set_title("power of the design as run\n"
                 "13 manifolds, 12 within-parent pairs, M=640",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    fig.suptitle("What nesting could this design have detected, at the regime "
                 "the real data occupies?", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "power_sweep.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
