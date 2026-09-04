import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np  

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import salad  
from cache import cached_acts 

OUT = "results/synthetic_results"
MODES = ["raw", "chat"]
M_EQUAL = 640           
FIT_LO = 10             # skip the leading few eigenvalues
FIT_HI = M_EQUAL // 4   # stop short of the rank cliff at M-1
N_MASSIVE = 5           # how many top-variance dims to drop in the ablation
SEED = 0


def _no_loader():
    raise RuntimeError("cache miss - run scripts/13_extract.py first")


def load_task(task: str, mode: str, rng) -> np.ndarray:
    """Cached activations for one task, subsampled to M_EQUAL. -> (M, 29, n)"""
    cap = min(len(salad.fetch_task(task, cap=10 ** 9)), salad.EXTRACT_CAP)
    A, _ = cached_acts(task, mode, _no_loader, M=cap, seed=SEED)
    a = A.numpy()
    idx = rng.choice(len(a), M_EQUAL, replace=False)
    return a[idx]


def spec_of(A_layer: np.ndarray) -> np.ndarray:
    """Covariance eigenvalues of one (M, n) block, descending."""
    M = A_layer.shape[0]
    Xc = A_layer - A_layer.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    return s ** 2 / (M - 1)


def participation_ratio(lam):
    """NaN for a degenerate manifold rather than 0/0.
    """
    lam = lam[lam > 0]
    if lam.size == 0 or lam.sum() <= 0:
        return float("nan")
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def fit_alpha(lam: np.ndarray) -> tuple:
    """Least-squares slope of log(lam) vs log(index) over [FIT_LO, FIT_HI].

    Returns (alpha, r2). alpha is the negative slope, so it matches the
    convention lam_i ~ i^-alpha. r2 near 1 means the power law fits.
    """
    i = np.arange(1, len(lam) + 1)
    m = (i >= FIT_LO) & (i <= FIT_HI) & (lam > 0)
    if m.sum() < 5:                      # degenerate, or fit range too narrow
        return float("nan"), float("nan")
    x, y = np.log(i[m]), np.log(lam[m])
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1 - resid.var() / y.var()
    return float(-slope), float(r2)


def main():
    os.makedirs(OUT, exist_ok=True)
    tasks = [t for t, _ in salad.design_tasks()]
    rng = np.random.default_rng(SEED)
    n_layers = 29

    # full layer sweep, every task, both modes
    print(f"computing spectra: {len(tasks)} tasks x {len(MODES)} modes x "
          f"{n_layers} layers at M={M_EQUAL}")
    D = {m: np.zeros((len(tasks), n_layers)) for m in MODES}
    AL = {m: np.zeros((len(tasks), n_layers)) for m in MODES}
    R2 = {m: np.zeros((len(tasks), n_layers)) for m in MODES}
    keep = {}

    for mi, mode in enumerate(MODES):
        for ti, t in enumerate(tasks):
            A = load_task(t, mode, np.random.default_rng(SEED))
            for l in range(n_layers):
                lam = spec_of(A[:, l, :])
                D[mode][ti, l] = participation_ratio(lam)
                AL[mode][ti, l], R2[mode][ti, l] = fit_alpha(lam)
                if ti == 0:
                    keep[(mode, l)] = lam
            print(f"\r  {mode} {ti+1}/{len(tasks)}", end="", flush=True)
        print()

    print(f"\n{'layer':>6} " + " ".join(
        f"{m+' D':>9} {m+' a':>7} {m+' r2':>6}" for m in MODES))
    for l in range(0, n_layers, 2):
        row = f"{l:>6} "
        for m in MODES:
            row += (f"{np.nanmean(D[m][:,l]):>9.1f} "
                    f"{np.nanmean(AL[m][:,l]):>7.3f} "
                    f"{np.nanmean(R2[m][:,l]):>6.3f}")
        print(row)

    # massive-activation ablation, mid layer
    L = 14
    print(f"\nmassive-activation ablation at layer {L} "
          f"(dropping the {N_MASSIVE} highest-variance dims)")
    print(f"{'task':>44} {'mode':>5} {'D full':>8} {'D ablated':>10} {'top-5 var share':>16}")
    for t in tasks[:4]:
        for mode in MODES:
            A = load_task(t, mode, np.random.default_rng(SEED))[:, L, :]
            v = A.var(axis=0)
            top = np.argsort(-v)[:N_MASSIVE]
            share = v[top].sum() / v.sum()
            keepmask = np.ones(A.shape[1], bool); keepmask[top] = False
            print(f"{t[:42]:>44} {mode:>5} {participation_ratio(spec_of(A)):>8.1f} "
                  f"{participation_ratio(spec_of(A[:, keepmask])):>10.1f} {share:>16.3f}")

    # plots
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    cmap = plt.get_cmap("viridis")
    show = [1, 7, 14, 21, 28]

    ax = axes[0]
    for i, l in enumerate(show):
        lam = keep[("raw", l)]
        ax.loglog(np.arange(1, len(lam) + 1), np.maximum(lam, 1e-14),
                  color=cmap(i / (len(show) - 1)), lw=1.6, label=f"layer {l}")
    ax.axvspan(FIT_LO, FIT_HI, color="0.5", alpha=.15)
    ax.axvline(M_EQUAL - 1, color="r", ls=":", lw=1.2)
    ax.set_xlabel("eigenvalue index"); ax.set_ylabel("variance")
    ax.set_title(f"{tasks[0][:26]} (raw)\nshaded = fit range, red = rank cap",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.25, which="both")

    ax = axes[1]
    for mode, c in zip(MODES, ["#2471a3", "#c0392b"]):
        mu, sd = np.nanmean(D[mode], 0), np.nanstd(D[mode], 0)
        ax.plot(mu, color=c, lw=2, label=mode)
        ax.fill_between(range(n_layers), mu - sd, mu + sd, color=c, alpha=.18)
    ax.axhline(M_EQUAL - 1, color="k", ls=":", lw=1.2, label="rank cap M-1")
    ax.set_xlabel("layer"); ax.set_ylabel("participation ratio D")
    ax.set_title("D by layer, mean +/- sd over 13 tasks", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axes[2]
    for mode, c in zip(MODES, ["#2471a3", "#c0392b"]):
        ax.plot(np.nanmean(AL[mode], 0), color=c, lw=2, label=f"{mode}: alpha")
        ax.plot(np.nanmean(R2[mode], 0), color=c, lw=1.2, ls="--",
                label=f"{mode}: fit r2")
    ax.axhline(1.0, color="0.4", ls=":", lw=1)
    ax.set_xlabel("layer"); ax.set_ylabel("alpha  /  r2")
    ax.set_title("power-law exponent and goodness of fit", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    fig.suptitle("Is the real within-manifold spectrum a power law, and what D "
                 f"does it imply?  (M={M_EQUAL}, n=1536)", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "empirical_spectrum.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\nwrote {p}")
    np.savez(os.path.join(OUT, "empirical_spectrum.npz"),
             tasks=tasks, **{f"D_{m}": D[m] for m in MODES},
             **{f"alpha_{m}": AL[m] for m in MODES},
             **{f"r2_{m}": R2[m] for m in MODES})




if __name__ == "__main__":
    main()
