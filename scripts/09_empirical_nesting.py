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
MODES = ["raw", "request"]
M_EQUAL = 640
N_BOOT = 50
SEED = 0
N_LAYERS = 29
SYNTH_FLOOR = 0.0010    


def _no_loader():
    raise RuntimeError("cache miss run scripts/13_extract.py first")


def load_all(mode, rng):
    """-> centroids (T, B, L, n), spreads (T, B, L), parent id (T,), names."""
    tasks = salad.design_tasks()
    T, B, L = len(tasks), N_BOOT, N_LAYERS
    cen = np.zeros((T, B, L, 1536), dtype=np.float32)
    spr = np.zeros((T, B, L), dtype=np.float64)

    for ti, (t, dom) in enumerate(tasks):
        cap = min(len(salad.fetch_task(t, cap=10 ** 9)), salad.EXTRACT_CAP)
        A, _ = cached_acts(t, mode, _no_loader, M=cap, seed=SEED)
        a = A.numpy()
        base = rng.choice(len(a), M_EQUAL, replace=False)   # equalize M first
        a = a[base]
        for b in range(B):
            # b = 0 is the full sample; b > 0 are bootstrap resamples of it.
            draw = a if b == 0 else a[rng.integers(0, M_EQUAL, M_EQUAL)]
            c = draw.mean(axis=0)
            cen[ti, b] = c
            # within-manifold spread = sqrt(trace(Sigma)) = RMS distance to centroid
            spr[ti, b] = np.sqrt(((draw - c) ** 2).sum(axis=2).mean(axis=0))
        print(f"\r  {mode} {ti+1}/{T}", end="", flush=True)
    print()
    par = np.array([dom for _, dom in tasks])
    return cen, spr, par, [t for t, _ in tasks]


def pair_split(par):
    """-> list of (i, j, is_within)."""
    return [(i, j, par[i] == par[j]) for i, j in combinations(range(len(par)), 2)]


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}

    for mode in MODES:
        print(f"\nloading {mode}")
        cen, spr, par, names = load_all(mode, np.random.default_rng(SEED))
        pairs = pair_split(par)
        nw = sum(1 for *_, w in pairs if w)
        T, B, L = cen.shape[:3]

        ratio = np.zeros((B, L))
        calib = np.zeros((B, L))
        for b in range(B):
            for l in range(L):
                d = {True: [], False: []}
                for i, j, w in pairs:
                    d[w].append(np.linalg.norm(cen[i, b, l] - cen[j, b, l]))
                between = np.mean(d[False])
                ratio[b, l] = np.mean(d[True]) / between if between > 0 else np.nan
                calib[b, l] = spr[:, b, l].mean() / between if between > 0 else np.nan

        res[mode] = dict(ratio=ratio, calib=calib, cen=cen, spr=spr,
                         par=par, names=names, pairs=pairs)

        mu, sd = np.nanmean(ratio, 0), np.nanstd(ratio, 0, ddof=1)
        sem = sd / np.sqrt(B)
        cm, cs = np.nanmean(calib, 0), np.nanstd(calib, 0, ddof=1)
        print(f"\n{mode}: {nw} within-parent pairs, {len(pairs)-nw} between, "
              f"M={M_EQUAL}, {B} bootstraps")
        print(f"{'layer':>6} {'nesting':>9} {'sem':>8} {'sd':>8} "
              f"{'within/between spread':>22}")
        for l in range(0, L, 2):
            print(f"{l:>6} {mu[l]:>9.4f} {sem[l]:>8.4f} {sd[l]:>8.4f} "
                  f"{cm[l]:>15.4f} +/-{cs[l]:.4f}")

    # ---- per-pair detail at the layer with the strongest nesting
    for mode in MODES:
        r = res[mode]
        mu = np.nanmean(r["ratio"], 0)
        L = int(np.nanargmin(mu))
        print(f"\n{mode}: strongest nesting at layer {L} "
              f"(ratio {mu[L]:.4f}). The 12 within-parent pairs individually,")
        print("as a fraction of the mean between-parent distance:")
        cen = r["cen"][:, 0, L]
        between = np.mean([np.linalg.norm(cen[i] - cen[j])
                           for i, j, w in r["pairs"] if not w])
        for i, j, w in r["pairs"]:
            if w:
                d = np.linalg.norm(cen[i] - cen[j]) / between
                print(f"   {d:>7.3f}  {r['names'][i][:34]:<36} | "
                      f"{r['names'][j][:34]}")

    # ---- plots
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    cols = {"raw": "#2471a3", "request": "#c0392b", "chat": "#7d3c98"}

    ax = axes[0]
    for mode in MODES:
        r = res[mode]["ratio"]
        mu, sem = np.nanmean(r, 0), np.nanstd(r, 0, ddof=1) / np.sqrt(N_BOOT)
        ax.plot(mu, color=cols[mode], lw=2, label=mode)
        ax.fill_between(range(N_LAYERS), mu - sem, mu + sem,
                        color=cols[mode], alpha=.3)
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="no nesting")
    ax.axhline(SYNTH_FLOOR, color="0.4", ls=":", lw=1.2,
               label=f"synthetic floor ({SYNTH_FLOOR})")
    ax.set_xlabel("layer"); ax.set_ylabel("nesting ratio")
    ax.set_title("within-parent / between-parent centroid distance\n"
                 "(bands = +/- sem over bootstraps)", fontweight="bold",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axes[1]
    for mode in MODES:
        c = res[mode]["calib"]
        mu, sem = np.nanmean(c, 0), np.nanstd(c, 0, ddof=1) / np.sqrt(N_BOOT)
        ax.plot(mu, color=cols[mode], lw=2, label=mode)
        ax.fill_between(range(N_LAYERS), mu - sem, mu + sem,
                        color=cols[mode], alpha=.3)
    ax.set_xlabel("layer")
    ax.set_ylabel("within-manifold spread / centroid separation")
    ax.set_title("CALIBRATION for the toy model\n"
                 "synthetic used ~1/55 = 0.018", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axes[2]
    for mode in MODES:
        r = res[mode]["ratio"]
        ax.plot(np.nanstd(r, 0, ddof=1), color=cols[mode], lw=1.6,
                label=f"{mode}: sd")
        ax.plot(np.nanstd(r, 0, ddof=1) / np.sqrt(N_BOOT), color=cols[mode],
                lw=1.6, ls="--", label=f"{mode}: sem")
    ax.set_xlabel("layer"); ax.set_ylabel("spread over bootstraps")
    ax.set_title("resampling noise\n(sd = trial spread, sem = precision of mean)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    fig.suptitle("Empirical nesting: do SALAD child categories huddle around "
                 f"their parent? (13 manifolds, M={M_EQUAL})", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "empirical_nesting.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\nwrote {p}")
    np.savez(os.path.join(OUT, "empirical_nesting.npz"),
             **{f"ratio_{m}": res[m]["ratio"] for m in MODES},
             **{f"calib_{m}": res[m]["calib"] for m in MODES})

if __name__ == "__main__":
    main()
