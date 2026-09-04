import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mmlu  
from cache import cached_acts, cache_status  

MODES = ["raw", "request"]
SEED = 0
TAG = "mmlu:"         
                       


def main():
    tasks = mmlu.design_tasks()
    print(mmlu.summary())

    caps = {t: min(len(mmlu.fetch_task(t, cap=10 ** 9)), mmlu.EXTRACT_CAP)
            for t, _ in tasks}
    total = sum(caps.values()) * len(MODES)

    todo = [(t, m) for t, _ in tasks for m in MODES
            if not cache_status([TAG + t], [m], caps[t], SEED)[0][2]]
    print(f"\n{total:,} forward passes across {len(MODES)} modes")
    print(f"{len(tasks)*len(MODES) - len(todo)} pairs cached, "
          f"{len(todo)} to compute\n")

    t0 = time.time()
    for i, (task, mode) in enumerate(todo, 1):
        cap = caps[task]
        loader = lambda t=task, c=cap: (
            mmlu.fetch_task(t, cap=c, seed=SEED),
            {"subject": t, "category": dict(tasks)[t], "dataset": "mmlu"},
        )
        cached_acts(TAG + task, mode, loader, M=cap, seed=SEED)
        el = time.time() - t0
        print(f"  [{i}/{len(todo)}] {task[:34]:<36} {mode:<8} "
              f"elapsed {el/60:5.1f}m  eta {el/i*(len(todo)-i)/60:5.1f}m")

    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    for t, _ in tasks:
        row = "  ".join(
            f"{m}:{'ok' if cache_status([TAG+t],[m],caps[t],SEED)[0][2] else 'MISSING'}"
            for m in MODES)
        print(f"  {t:<34} M={caps[t]:>4}  {row}")


if __name__ == "__main__":
    main()
