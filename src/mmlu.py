"""MMLU as a second taxonomy, mirroring src/salad.py exactly.

Not a "control" so much as a second test of the same question: does a model's
representational geometry reflect an externally-given category hierarchy? SALAD
is a harm taxonomy; MMLU is an academic one. They share almost nothing in
content or form, which makes agreement between them informative and makes
disagreement hard to attribute.

Interface is identical to salad.py -- design_tasks, fetch_task, branching,
summary -- so every analysis script switches dataset by swapping the import.

--------------------------------------------------------------------------
DESIGN DECISIONS
--------------------------------------------------------------------------
GROUPING is MMLU's own published 4-way split (STEM / humanities / social
sciences / other), not one invented here. That matters: the whole value of a
second taxonomy is that somebody else drew the boundaries. Hand-building a
grouping to mirror SALAD's [4,3,2,2,2] would destroy exactly the property being
relied on. The 13 subjects below are mapped by hand rather than by encoding all
57, so the mapping can be checked against the MMLU repo at a glance.

M = 200, forced by the data. Only 3 of 57 subjects have >=640 questions, 26
have >=200. SALAD must therefore be SUBSAMPLED to 200 for the matched
comparison -- free, since 800 per task are already cached. Report SALAD at
M=640 as the primary result and SALAD@200 vs MMLU@200 as the comparison.

BRANCHING is [4,3,3,3] against SALAD's [4,3,2,2,2]: 13 manifolds either way,
but 15 within-parent pairs instead of 12. Four official categories cannot be
made into five without inventing structure. Compare z-scores against each
dataset's own matched permutation null rather than raw nesting ratios, and the
difference stops mattering.

QUESTION STEM ONLY, no answer choices. Appending four options would make these
much longer and structurally unlike a request.

SUBJECT SELECTION prefers subjects whose mean question length falls near
SALAD's 77-102 characters, from among those with >=200 questions. This is a
deliberate reduction of the register mismatch, not cherry-picking on the
outcome -- length is fixed before any activation is computed. Excluded on
length: professional_law (830 chars), high_school_world_history (1342),
professional_medicine (654), moral_scenarios (322), high_school_statistics
(267), professional_accounting (239).
"""

import numpy as np

# subject -> official MMLU category. Verify against the categories.py in
# hendrycks/test if you want to check the mapping.
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

M_DESIGN = 200          # forced by sociology at 201 rows
EXTRACT_CAP = 300       # cache above M for headroom, as with SALAD

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

    Sorted by category then subject so y_parent cannot silently permute between
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
