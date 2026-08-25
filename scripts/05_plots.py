"""Session 3c: plots for the transfer matrix.

Pure analysis -- reads results/transfer_n400_s0.pt, writes PNGs to results/.
No model, no activations, runs in a second. Re-run it as often as you like.

Colour convention, used everywhere: diverging, centred on 0.5. Blue is above
chance, red is BELOW chance. Anti-generalisation is a signed phenomenon and the
whole point is that it should be visually distinct from failure-to-transfer, not
lumped in with it as "low".

Run from the repo root:  python scripts/05_plots.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

RESULTS = "results/transfer_n400_s0.pt"
OUT = "results"
PAIRS = [("cities", "neg_cities"),
         ("sp_en_trans", "neg_sp_en_trans"),
         ("larger_than", "smaller_than")]


def selected_matrix(res):
    """AUROC at each row's val-selected layer. -> (n_ds, n_ds)"""
    g, ch, ds = res["grid"], res["chosen"], res["datasets"]
    return np.array([[g[i, j, ch[i]] for j in range(len(ds))]
                     for i in range(len(ds))])


# ---------------------------------------------------------------------------
def plot_heatmaps(R, path):
    ds = R["raw"]["datasets"]
    short = [d.replace("_true_false", "").replace("_", " ") for d in ds]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6))

    for ax, mode in zip(axes, ["raw", "chat"]):
        M = selected_matrix(R[mode])
        im = ax.imshow(M, cmap="RdBu", vmin=0.0, vmax=1.0)

        for i in range(len(ds)):
            for j in range(len(ds)):
                v = M[i, j]
                # White text on the saturated ends, black in the middle band.
                c = "white" if (v < 0.22 or v > 0.78) else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=c, fontsize=8.5,
                        fontweight="bold" if v < 0.5 else "normal")

        ax.set_xticks(range(len(ds)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(ds)))
        ax.set_yticklabels(short, fontsize=8)
        ax.set_xlabel("tested on")
        ax.set_ylabel("probe trained on")
        ax.set_title(f"readout mode: {mode}", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, label="transfer AUROC")

    fig.suptitle("Probe transfer. Red = below chance = the probe is "
                 "systematically inverted, not merely uninformative.",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_negation_layers(R, path):
    """The money plot: negation transfer vs layer, raw against chat."""
    ds = R["raw"]["datasets"]
    ix = {d: i for i, d in enumerate(ds)}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True, sharey=True)

    for col, (a, b) in enumerate(PAIRS):
        for row, (i, j, lbl) in enumerate([(ix[a], ix[b], f"{a} -> {b}"),
                                           (ix[b], ix[a], f"{b} -> {a}")]):
            ax = axes[row, col]
            for mode, colour in [("raw", "#c0392b"), ("chat", "#2471a3")]:
                prof = R[mode]["grid"][i, j, :]
                ax.plot(prof, color=colour, lw=1.9, label=mode)
                ax.scatter([R[mode]["chosen"][i]], [prof[R[mode]["chosen"][i]]],
                           color=colour, s=42, zorder=5, edgecolor="white")

            ax.axhline(0.5, color="0.35", ls="--", lw=1)
            ax.axhspan(0.0, 0.5, color="#c0392b", alpha=0.055)
            ax.set_ylim(0, 1)
            ax.set_title(lbl, fontsize=9.5)
            ax.grid(alpha=0.25)
            if row == 1:
                ax.set_xlabel("layer index (0 = embedding)")
            if col == 0:
                ax.set_ylabel("transfer AUROC")
            if row == 0 and col == 0:
                ax.legend(fontsize=8.5, loc="lower left")

    fig.suptitle("Negation-pair transfer by layer. Shaded band is below chance. "
                 "Dots mark each probe's val-selected layer.", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_diag_vs_off(R, path):
    """Does a good probe transfer? Diagonal against mean off-diagonal."""
    ds = R["raw"]["datasets"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))

    for ax, mode in zip(axes, ["raw", "chat"]):
        M = selected_matrix(R[mode])
        off = ~np.eye(len(ds), dtype=bool)
        diag = np.diag(M)
        mean_off = np.array([M[i][off[i]].mean() for i in range(len(ds))])

        ax.scatter(diag, mean_off, s=70, color="#2471a3", zorder=3)
        for i, d in enumerate(ds):
            ax.annotate(d.replace("_true_false", ""), (diag[i], mean_off[i]),
                        fontsize=7.5, xytext=(4, 4), textcoords="offset points")

        ax.axhline(0.5, color="0.35", ls="--", lw=1)
        ax.set_xlim(0.5, 1.02)
        ax.set_ylim(0, 1)
        ax.set_xlabel("within-dataset AUROC (the diagonal)")
        ax.set_ylabel("mean transfer to the other seven")
        ax.set_title(f"{mode}", fontweight="bold")
        ax.grid(alpha=0.25)

    fig.suptitle("In-distribution accuracy tells you almost nothing about "
                 "transfer. That gap is the project.", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_layer_profiles(R, path):
    """Diagonal AUROC by layer, all datasets, both modes. Where does truth live?"""
    ds = R["raw"]["datasets"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    cmap = plt.get_cmap("tab10")

    for ax, mode in zip(axes, ["raw", "chat"]):
        for i, d in enumerate(ds):
            ax.plot(R[mode]["grid"][i, i, :], lw=1.7, color=cmap(i % 10),
                    label=d.replace("_true_false", ""))
        ax.axhline(0.5, color="0.35", ls="--", lw=1)
        ax.set_ylim(0.35, 1.02)
        ax.set_xlabel("layer index (0 = embedding)")
        ax.set_title(f"{mode}", fontweight="bold")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("within-dataset AUROC")
    axes[1].legend(fontsize=7.5, loc="lower right", ncol=2)

    fig.suptitle("Within-dataset AUROC by layer. Layer 0 should sit at chance; "
                 "if it does not, something leaked.", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    R = torch.load(RESULTS, weights_only=False)

    jobs = [
        ("transfer_heatmaps.png", plot_heatmaps),
        ("negation_by_layer.png", plot_negation_layers),
        ("diagonal_vs_transfer.png", plot_diag_vs_off),
        ("layer_profiles.png", plot_layer_profiles),
    ]
    for fname, fn in jobs:
        p = os.path.join(OUT, fname)
        fn(R, p)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
