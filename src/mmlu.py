import numpy as np


DESIGN = {
    # STEM
    "conceptual_physics":         "STEM",             # 235 rows,  76 chars
    "elementary_mathematics":     "STEM",             # 378,      113
    "high_school_chemistry":      "STEM",             # 203,      129
    # humanities
    "philosophy":                 "humanities",       # 311,       81
    "prehistory":                 "humanities",       # 324,       90
    "moral_disputes":             "humanities",       # 346,       98
    # social sciences
    "sociology":                  "social sciences",  # 201,       76
    "security_studies":           "social sciences",  # 245,       87
    "high_school_macroeconomics": "social sciences",  # 390,       96
    "high_school_microeconomics": "social sciences",  # 238,      101
    # other
    "human_aging":                "other",            # 223,       78
    "miscellaneous":              "other",            # 783,       87
    "nutrition":                  "other",            # 306,       91
}

M_DESIGN = 200         
EXTRACT_CAP = 300      

_DATA = None


def _rows():
    global _DATA
    if _DATA is None:
        from datasets import load_dataset
        d = load_dataset("cais/mmlu", "all", split="test")
        keep = set(DESIGN)
        _DATA = [dict(q=q, subject=s) for q, s in
                 zip(d["question"], d["subject"]) if s in keep]
    return _DATA


def design_tasks():
    """[(subject, category), ...], deterministically ordered.

    Sorted by category then subject so y_parent cannot permute between
    runs or between this and the synthetic mirror.
    """
    return sorted(DESIGN.items(), key=lambda kv: (kv[1], kv[0]))


def fetch_task(task: str, cap: int = EXTRACT_CAP, seed: int = 0):
    """Question stems for one subject, shuffled then capped."""
    qs = [r["q"] for r in _rows() if r["subject"] == task]
    rng = np.random.default_rng(abs(hash(task)) % (2 ** 32) + seed)
    rng.shuffle(qs)
    return qs[:cap]


def branching():
    counts = {}
    for _, cat in design_tasks():
        counts[cat] = counts.get(cat, 0) + 1
    return sorted(counts.values(), reverse=True)


def summary():
    tasks = design_tasks()
    cats = sorted({c for _, c in tasks})
    lines = [f"{len(tasks)} subjects across {len(cats)} categories, "
             f"branching {branching()}, M_design={M_DESIGN}"]
    for c in cats:
        lines.append(f"  {c}")
        for t, cc in tasks:
            if cc == c:
                qs = fetch_task(t, cap=10 ** 9)
                ln = np.mean([len(q) for q in qs])
                lines.append(f"     {t:<32} {len(qs):>5} available, "
                             f"{ln:>6.1f} chars")
    return "\n".join(lines)
