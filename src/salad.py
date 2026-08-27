"""SALAD-Bench loading, and the parent/child design.

The design decisions live here as data, not scattered through scripts, so the
one place to change them is the one place to read them.

Return contract, matching src/toy.generate() exactly:

    X         (n_points, n_layers + 1, n)   -- note the layer axis
    y_parent  (n_points,) int
    y_child   (n_points,) int

so every measurement function runs unchanged on synthetic and empirical data.

Design as of 2026-08-27, per the audit in scripts/11 and 12:

  DROPPED   domain O3 Socioeconomic Harms -- holds the smallest task (200),
            the most length-atypical questions (112 chars vs 65-99), and only
            two tasks. Three independent reasons.
  DROPPED   task O11 Defamation -- smallest Malicious Use task (437), and
            trimming it takes the most over-represented parent from 5 children
            to 4.
  NOTE      O6 Risky Financial Practices (651) sits UNDER Socioeconomic Harms,
            so dropping that domain drops O6 with it. Result is 13 tasks, not
            14. To keep O6 as a single-child parent -- it costs nothing in M,
            since 640 is still the floor -- drop only the task O7 instead of
            the whole domain:
                DROP_DOMAINS = set()
                DROP_TASKS = {"O11: Defamation", "O7: Trade and Compliance"}

  M = 640, set by O15: Persuasion and Manipulation. NOT 657, which would drop
  O15 and leave Human Autonomy with one child, costing a whole parent's worth
  of within-parent pairs.

Cache more than you need: `fetch_task` takes a cap above M so the analysis can
subsample to any M <= the cap without re-running forward passes. Equalisation
happens at analysis time via geometry.equalize_class_n.
"""

import numpy as np

DROP_DOMAINS = {"O3: Socioeconomic Harms"}
DROP_TASKS = {"O11: Defamation"}

M_DESIGN = 640          # the equalised M the design is built around
EXTRACT_CAP = 800       # cache this many per task where available, for headroom

_DATA = None            # lazily loaded HF dataset


def _rows():
    global _DATA
    if _DATA is None:
        from datasets import load_dataset
        d = load_dataset("OpenSafetyLab/Salad-Data", name="base_set", split="train")
        _DATA = [dict(q=q, l1=a, l2=b, l3=c) for q, a, b, c in
                 zip(d["question"], d["1-category"], d["2-category"],
                     d["3-category"])]
    return _DATA


def design_tasks():
    """The kept level-2 tasks, and the parent of each. -> [(task, domain), ...]

    Sorted by domain then task so the ordering is deterministic across runs --
    an unstable ordering would silently permute y_parent between the synthetic
    mirror and the empirical data.
    """
    seen = {}
    for r in _rows():
        if r["l1"] in DROP_DOMAINS or r["l2"] in DROP_TASKS:
            continue
        seen[r["l2"]] = r["l1"]
    return sorted(seen.items(), key=lambda kv: (kv[1], kv[0]))


def fetch_task(task: str, cap: int = EXTRACT_CAP, seed: int = 0):
    """Prompts for one level-2 task, shuffled and capped. -> list[str]

    Shuffled before capping so the cap does not correlate with whatever order
    the dataset happens to be in -- SALAD is grouped by source, and taking the
    first k would sample one source preferentially.
    """
    qs = [r["q"] for r in _rows() if r["l2"] == task]
    rng = np.random.default_rng(abs(hash(task)) % (2 ** 32) + seed)
    rng.shuffle(qs)
    return qs[:cap]


def branching():
    """Children per parent, descending -- the argument to HierarchySpec."""
    tasks = design_tasks()
    counts = {}
    for _, dom in tasks:
        counts[dom] = counts.get(dom, 0) + 1
    return sorted(counts.values(), reverse=True)


def summary():
    tasks = design_tasks()
    doms = sorted({d for _, d in tasks})
    lines = [f"{len(tasks)} tasks across {len(doms)} domains, "
             f"branching {branching()}, M_design={M_DESIGN}"]
    for d in doms:
        kids = [t for t, dd in tasks if dd == d]
        lines.append(f"  {d}")
        for t in kids:
            lines.append(f"     {t:<48} {len(fetch_task(t, cap=10**9)):>5} available")
    return "\n".join(lines)
