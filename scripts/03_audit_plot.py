import os
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  
from matplotlib.patches import Patch  

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from datasets import load_dataset  

OUT = "results/salad_task_counts.png"

DOMAIN_COLOURS = {
    "O5: Malicious Use":              "#2E8B57",   # green
    "O1: Representation & Toxicity":  "#C0392B",   # red
    "O2: Misinformation Harms":       "#2471A3",   # blue
    "O6: Human Autonomy & Integrity": "#8E44AD",   # purple
    "O4: Information & Safety":       "#D68910",   # amber
    "O3: Socioeconomic Harms":        "#17A2A2",   # teal
}


def main():
    os.makedirs("results", exist_ok=True)
    d = load_dataset("OpenSafetyLab/Salad-Data", name="base_set", split="train")

    counts = Counter(d["2-category"])
    parent = {}
    kids = defaultdict(set)
    for l1, l2, l3 in zip(d["1-category"], d["2-category"], d["3-category"]):
        parent[l2] = l1
        kids[l2].add(l3)

    # Group by domain (largest domain first), then by count within domain.
    domain_total = Counter()
    for t, n in counts.items():
        domain_total[parent[t]] += n
    order = sorted(counts,
                   key=lambda t: (-domain_total[parent[t]], parent[t], -counts[t]))
    # barh draws bottom-up, so reverse to read top-down.
    order = order[::-1]

    vals = [counts[t] for t in order]
    cols = [DOMAIN_COLOURS[parent[t]] for t in order]
    labels = [f"{t}  ({len(kids[t])} cat)" for t in order]

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    bars = ax.barh(range(len(order)), vals, color=cols, height=.72,
                   edgecolor="white", linewidth=.6)

    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(v + 45, i, f"{v:,}", va="center", fontsize=9, color="#333")

    for n, style in [(200, (0, (4, 3))), (400, (0, (1, 2)))]:
        ax.axvline(n, color="#555", ls=style, lw=1.1, zorder=0)
        ax.text(n, len(order) - .3, f" n={n}", fontsize=8, color="#555",
                rotation=90, va="top")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("prompts available")
    ax.set_xlim(0, max(vals) * 1.12)
    ax.set_title("SALAD-Bench: prompts per level-2 task, coloured by parent domain",
                 fontweight="bold", fontsize=12, pad=14)
    ax.grid(axis="x", alpha=.25)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    legend = [Patch(facecolor=DOMAIN_COLOURS[k],
                    label=f"{k}  ({sum(1 for t in counts if parent[t]==k)} tasks, "
                          f"{domain_total[k]:,} rows)")
              for k in sorted(DOMAIN_COLOURS, key=lambda k: -domain_total[k])]
    ax.legend(handles=legend, fontsize=8.5, loc="lower right", frameon=True,
              title="level-1 domain", title_fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    print(f"wrote {OUT}\n")
    print(f"{'task':<48} {'n':>6} {'cats':>5}  parent")
    for t in order[::-1]:
        print(f"{t:<48} {counts[t]:>6} {len(kids[t]):>5}  {parent[t]}")
    print(f"\nsmallest task: {min(counts.values())} "
          f"({min(counts, key=counts.get)})")
    print("That number is the ceiling on n if you keep all 16 tasks.")
    print(f"branching: " + ", ".join(
        f"{k.split(':')[0]}={sum(1 for t in counts if parent[t]==k)}"
        for k in sorted(DOMAIN_COLOURS, key=lambda k: -domain_total[k])))


if __name__ == "__main__":
    main()
