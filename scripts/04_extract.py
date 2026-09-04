import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import salad  
from cache import cached_acts, cache_status  

MODES = ["raw", "chat", "request"]
SEED = 0


def main():
    tasks = salad.design_tasks()
    print(salad.summary())

    caps = {t: min(len(salad.fetch_task(t, cap=10 ** 9)), salad.EXTRACT_CAP)
            for t, _ in tasks}
    total = sum(caps.values()) * len(MODES)

    todo = [(t, m) for t, _ in tasks for m in MODES
            if not cache_status([t], [m], caps[t], SEED)[0][2]]
    done = len(tasks) * len(MODES) - len(todo)
    print(f"\n{total:,} forward passes total across {len(MODES)} modes")
    print(f"{done} task/mode pairs already cached, {len(todo)} to compute\n")

    t0 = time.time()
    for i, (task, mode) in enumerate(todo, 1):
        cap = caps[task]
        loader = lambda t=task, c=cap: (
            salad.fetch_task(t, cap=c, seed=SEED),
            {"task": t, "domain": dict(tasks)[t]},
        )
        cached_acts(task, mode, loader, M=cap, seed=SEED)

        el = time.time() - t0
        print(f"  [{i}/{len(todo)}] {task[:38]:<40} {mode:<5} "
              f"elapsed {el/60:5.1f}m  eta {el/i*(len(todo)-i)/60:5.1f}m")

    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    print("cache status:")
    for t, _ in tasks:
        row = "  ".join(
            f"{m}:{'ok' if cache_status([t], [m], caps[t], SEED)[0][2] else 'MISSING'}"
            for m in MODES)
        print(f"  {t[:44]:<46} M={caps[t]:>4}  {row}")


if __name__ == "__main__":
    main()
