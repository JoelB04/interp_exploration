"""Session 3b: is the `chat` probe just reading the model's answer token?

The suspicion. Under `chat` the prompt is "Is the following true or false?" and
the readout sits on the generation-prompt token -- exactly where the model has
finished deciding what to emit next. If the probe direction is really the
"about to say true vs about to say false" direction, then the chat results are
not evidence that truth is linearly represented. They are a roundabout way of
asking the model, and a black-box baseline would match them.

Two tests, both cheap:

  1. Unembed the probe direction and read off the top tokens. If +d promotes
     "true"/"Yes" and -d promotes "false"/"No", that is the answer direction.

  2. Cosine between the probe direction and the explicit answer-token direction
     W_U[true] - W_U[false]. A number, not a vibe.

Predict before running: which mode has the higher cosine, and roughly how big
is the gap? Write it down.

Run from the repo root:  python scripts/04_unembed_check.py
Needs the cache from 03_transfer.py plus one model load.
"""

import os
import sys

import numpy as np
import torch

# Qwen's vocab is full of CJK tokens and the Windows console is cp1252, which
# cannot encode them. Without this the script dies on the first print.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from acts import load  # noqa: E402
from cache import cached_acts  # noqa: E402
from data import DATASETS, split  # noqa: E402

MAX_N, SEED = 400, 0
RESULTS = "results/transfer_n400_s0.pt"
TOPK = 8

# Candidate answer pairs. Leading space matters -- that is how these tokenise
# mid-sentence, which is where the model would actually emit them.
ANSWER_PAIRS = [(" true", " false"), (" True", " False"), (" Yes", " No")]


def probe_direction(dataset, mode, layer):
    """Recompute the exact direction 03_transfer.py used at this layer.

    Same seeds and same split calls, so the directions match the grid.
    """
    acts, labels, groups, _ = cached_acts(dataset, mode, MAX_N, SEED)
    trval, _ = split(groups, frac_train=0.8, seed=SEED)
    tr_rel, _ = split(groups[trval], frac_train=0.75, seed=SEED + 1)
    train_i = trval[tr_rel]

    a, y = acts[train_i].numpy(), labels[train_i]
    d = a[y == 1].mean(axis=0) - a[y == 0].mean(axis=0)
    return d[layer] / (np.linalg.norm(d[layer]) + 1e-12)


def main():
    R = torch.load(RESULTS, weights_only=False)
    model, tok, device = load()

    W_U = model.lm_head.weight.detach().float()          # (vocab, d)
    gain = model.model.norm.weight.detach().float()      # RMSNorm learned gain

    # Answer directions in residual space, with the same gain applied.
    answers = {}
    for pos, neg in ANSWER_PAIRS:
        ip, ineg = tok.encode(pos), tok.encode(neg)
        if len(ip) != 1 or len(ineg) != 1:
            print(f"  skipping {pos!r}/{neg!r}: not single tokens")
            continue
        v = (W_U[ip[0]] - W_U[ineg[0]]) * gain
        answers[f"{pos.strip()}-{neg.strip()}"] = (v / v.norm()).numpy()

    for mode in ["raw", "chat"]:
        chosen = R[mode]["chosen"]
        print(f"\n{'=' * 74}\nmode: {mode}\n{'=' * 74}")

        cos_table = {k: [] for k in answers}
        for i, ds in enumerate(DATASETS):
            layer = int(chosen[i])
            d = probe_direction(ds, mode, layer)

            # Logit-lens the direction. Scale by the final-norm gain first;
            # RMSNorm's magnitude normalisation is irrelevant because we only
            # look at the RANKING of tokens, not the logit values.
            dt = torch.tensor(d, dtype=torch.float32) * gain
            logits = W_U @ dt

            top = torch.topk(logits, TOPK).indices.tolist()
            bot = torch.topk(-logits, TOPK).indices.tolist()
            fmt = lambda ids: " ".join(repr(tok.decode([t])) for t in ids)

            print(f"\n{ds}  (layer {layer})")
            print(f"  +d -> {fmt(top)}")
            print(f"  -d -> {fmt(bot)}")

            cosines = {k: float(d @ v) for k, v in answers.items()}
            for k, c in cosines.items():
                cos_table[k].append(c)
            print("  cos with answer dir: " +
                  "  ".join(f"{k} {c:+.3f}" for k, c in cosines.items()))

        print(f"\n--- {mode}: mean |cos| with answer direction ---")
        for k, vals in cos_table.items():
            v = np.abs(vals)
            print(f"  {k:14s} mean {v.mean():.3f}   max {v.max():.3f}   "
                  f"per-dataset {np.round(vals, 2)}")

    print("\nHow to read this. A direction drawn at random in 1536 dimensions has")
    print("|cos| ~ 1/sqrt(1536) ~ 0.026 with anything. Treat that as your floor.")
    print("If chat sits far above it and raw sits near it, the chat probe is")
    print("largely the answer-token direction and the chat results are a")
    print("black-box measurement in disguise.")


if __name__ == "__main__":
    main()
