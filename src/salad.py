import numpy as np

DROP_DOMAINS = {"O3: Socioeconomic Harms"}
DROP_TASKS = {"O11: Defamation"}

M_DESIGN = 640         
EXTRACT_CAP = 800       

_DATA = None           


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

    Sorted by domain then task so the ordering is deterministic across runs so
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

    qs = [r["q"] for r in _rows() if r["l2"] == task]
    rng = np.random.default_rng(abs(hash(task)) % (2 ** 32) + seed)
    rng.shuffle(qs)
    return qs[:cap]


def branching():
    """Children per parent, descending, the argument to HierarchySpec."""
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
