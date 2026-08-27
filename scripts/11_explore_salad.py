"""Session 4a: look at SALAD-Bench before designing anything.

The point of this script is to let you DECIDE the hierarchy, not to analyse it.
It prints the taxonomy tree with counts, checks the tree is actually a tree,
samples questions at whatever node you ask for, and flags the surface confounds
that could masquerade as geometry later.

Read the output, then answer three questions:

  1. Which two levels are your parent/child pair? Level 1 (6 domains) is almost
     certainly too few manifolds to say anything. Level 3 (66 categories) has
     plenty but some are near-synonymous, which would show up as a spurious
     "these manifolds overlap" result. Level 2 (16 tasks) may be the sweet spot,
     or 2-and-3 as a genuine parent/child pair.

  2. How many points per manifold? Bounded by the smallest class you keep. Every
     spectral quantity is biased by n, so this number must be IDENTICAL across
     manifolds -- see equalize_class_n in src/predictors.py.

  3. Which categories would you merge or drop? A taxonomy designed for
     benchmarking coverage is not automatically a good set of manifolds.

Run from the repo root:
    python scripts/11_explore_salad.py              # tree + confound report
    python scripts/11_explore_salad.py "O14"        # sample from matching nodes
"""

import os
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datasets import load_dataset  # noqa: E402

OUT = "results/salad_taxonomy.txt"
SAMPLES_PER_NODE = 3


def load():
    d = load_dataset("OpenSafetyLab/Salad-Data", name="base_set", split="train")
    rows = [dict(q=r["question"], l1=r["1-category"],
                 l2=r["2-category"], l3=r["3-category"]) for r in d]
    return rows


def check_tree(rows):
    """A clean tree means each child has exactly one parent. If not, the
    'hierarchy' is a DAG and nesting claims get much harder to state."""
    p2, p3 = defaultdict(set), defaultdict(set)
    for r in rows:
        p2[r["l2"]].add(r["l1"])
        p3[r["l3"]].add(r["l2"])

    bad2 = {k: v for k, v in p2.items() if len(v) > 1}
    bad3 = {k: v for k, v in p3.items() if len(v) > 1}
    return bad2, bad3


def dup_report(rows):
    """Exact and near-duplicate questions. Duplicates inflate apparent manifold
    tightness -- a repeated point has zero variance and shrinks the radius."""
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    c = Counter(norm(r["q"]) for r in rows)
    exact = sum(v - 1 for v in c.values() if v > 1)
    return exact, len(c)


def main():
    rows = load()
    os.makedirs("results", exist_ok=True)

    query = sys.argv[1] if len(sys.argv) > 1 else None
    if query:
        hits = [r for r in rows if query.lower() in
                (r["l1"] + r["l2"] + r["l3"]).lower()]
        print(f"{len(hits)} rows matching {query!r}\n")
        seen = set()
        for r in hits:
            key = (r["l2"], r["l3"])
            if key in seen:
                continue
            seen.add(key)
            ex = [x["q"] for x in hits if (x["l2"], x["l3"]) == key][:6]
            print(f"[{r['l1']}]\n  {r['l2']}  >  {r['l3']}")
            for e in ex:
                print(f"     - {e[:150]}")
            print()
        return

    lines = []
    w = lines.append

    # ---- structural integrity
    bad2, bad3 = check_tree(rows)
    exact_dups, uniq = dup_report(rows)
    w("=" * 78)
    w("STRUCTURE")
    w("=" * 78)
    w(f"rows: {len(rows)}   unique questions (normalised): {uniq}   "
      f"exact duplicates: {exact_dups}")
    w(f"level-2 nodes with >1 parent: {len(bad2)}   "
      f"level-3 nodes with >1 parent: {len(bad3)}")
    if bad2 or bad3:
        w("  NOT a clean tree. Offending nodes:")
        for k, v in list(bad2.items())[:5]:
            w(f"    {k} <- {sorted(v)}")
        for k, v in list(bad3.items())[:5]:
            w(f"    {k} <- {sorted(v)}")
    else:
        w("  Clean tree: every child has exactly one parent.")

    # ---- the taxonomy itself
    c1, c2, c3 = (Counter(r["l1"] for r in rows), Counter(r["l2"] for r in rows),
                  Counter(r["l3"] for r in rows))
    tree = defaultdict(lambda: defaultdict(int))
    parent = {}
    for r in rows:
        tree[r["l1"]][r["l2"]] += 1
        parent[r["l2"]] = r["l1"]
    kids = defaultdict(list)
    for r in rows:
        if r["l3"] not in [k for k in kids[r["l2"]]]:
            kids[r["l2"]].append(r["l3"])

    w("")
    w("=" * 78)
    w(f"TAXONOMY   {len(c1)} domains / {len(c2)} tasks / {len(c3)} categories")
    w("=" * 78)
    for d, n1 in c1.most_common():
        w(f"\n{d}   [{n1} rows]")
        for t, n2 in sorted(tree[d].items(), key=lambda x: -x[1]):
            w(f"  |- {t}   [{n2}]   ({len(set(kids[t]))} categories)")
            for cat in sorted(set(kids[t]), key=lambda c: -c3[c]):
                flag = "  << thin" if c3[cat] < 150 else ""
                w(f"  |    - {cat:<52} {c3[cat]:>5}{flag}")

    # ---- how many manifolds survive an equalised-n choice
    w("")
    w("=" * 78)
    w("MANIFOLD BUDGET   (n must be identical across manifolds)")
    w("=" * 78)
    w(f"{'n per manifold':>16} {'level-2 tasks kept':>20} {'level-3 cats kept':>20}")
    for n in [100, 150, 200, 250, 300, 400, 500]:
        k2 = sum(1 for v in c2.values() if v >= n)
        k3 = sum(1 for v in c3.values() if v >= n)
        w(f"{n:>16} {k2:>20} {k3:>20}")
    w("")
    w("Read this as your core design tradeoff: more points per manifold means")
    w("less estimator bias but fewer manifolds. D_M is capped at n-1 and biased")
    w("well below it, so n also caps the dimensionality you can even observe.")

    # ---- surface confounds
    w("")
    w("=" * 78)
    w("SURFACE CONFOUNDS   (could masquerade as geometry)")
    w("=" * 78)
    for lvl, cnt, name in [("l1", c1, "domain"), ("l2", c2, "task")]:
        L = defaultdict(list)
        for r in rows:
            L[r[lvl]].append(len(r["q"]))
        w(f"\nmean question length by {name}:")
        for k in sorted(L, key=lambda k: -sum(L[k]) / len(L[k])):
            v = L[k]
            w(f"  {sum(v)/len(v):7.1f} chars  (n={len(v):5d})  {k}")
        spread = max(sum(v)/len(v) for v in L.values()) / \
                 min(sum(v)/len(v) for v in L.values())
        w(f"  -> max/min ratio {spread:.2f}"
          f"{'   CONTROL FOR THIS' if spread > 1.25 else '   probably fine'}")

    # ---- first words, a proxy for phrasing template
    w("")
    w("opening word distribution by domain (template leakage check):")
    for d in c1:
        fw = Counter(r["q"].split()[0].lower() for r in rows
                     if r["l1"] == d and r["q"].split())
        top = ", ".join(f"{k} {v*100//c1[d]}%" for k, v in fw.most_common(4))
        w(f"  {d[:34]:<36} {top}")

    w("")
    w("If one domain is 60% 'how' and another 60% 'what', a probe or a manifold")
    w("statistic can separate them on syntax alone. Check before believing.")

    text = "\n".join(lines)
    print(text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n\nwritten to {OUT}")
    print("Drill into any node with:  python scripts/11_explore_salad.py \"O14\"")


if __name__ == "__main__":
    main()
