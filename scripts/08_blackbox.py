"""Session 3e: the black-box baseline. Standard 1, finally paid.

The claim any probe result has to beat: "just ask the model."

Under `chat` the prompt already IS the question -- "Is the following true or
false?" -- and the readout sits on the generation-prompt token. Session 1
established that the final norm is already applied to hidden_states[-1], so
lm_head(hs[-1]) reproduces the logits exactly. The cached chat activations
therefore already contain the model's answer distribution.

So this costs ZERO forward passes. Score each statement by

    logit(" true") - logit(" false")

at the generation position, and take AUROC against the label. No internals, no
training, no probe. Pure black box.

Then compare against the probe. If a layer-20 chat probe scores 0.96 and simply
asking scores 0.96, the probe has bought you nothing and every chat result in
this project is a black-box measurement wearing a lab coat.

The comparison is only fair in `chat`. Under `raw` the model was never asked a
question, so there is no answer distribution to read and the black box does not
apply -- which is itself the argument for why raw results are the interesting
ones if they hold up.

Run from the repo root:  python scripts/08_blackbox.py
Loads the model once for the unembedding rows. No generation.
"""

import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cache import cached_acts  # noqa: E402
from data import DATASETS, split  # noqa: E402

MAX_N, SEED = 400, 0
N_REPEATS = 20
PROBE_LAYER = 20          # the stable/best fixed layer from session 3d (chat)
UNEMBED_CACHE = "cache/answer_unembed.pt"
# The model emits these WITHOUT a leading space at the readout token -- the top
# predictions are 'False', 'True', 'No'. Including both forms and letting the
# best one win per dataset; leading-space variants are what you would need
# mid-sentence, which is not where this readout sits.
ANSWER_PAIRS = [("True", "False"), ("true", "false"), ("Yes", "No"),
                (" True", " False"), (" true", " false")]


def answer_rows():
    """Raw W_U rows for the answer tokens. NOT gain-scaled: hs[-1] is already
    post-norm, so logits = W_U @ hs[-1] directly."""
    if os.path.exists(UNEMBED_CACHE):
        return torch.load(UNEMBED_CACHE, weights_only=False)

    from acts import load
    model, tok, _ = load()
    W_U = model.lm_head.weight.detach().float()
    out = {}
    for pos, neg in ANSWER_PAIRS:
        ip, ineg = tok.encode(pos), tok.encode(neg)
        if len(ip) == 1 and len(ineg) == 1:
            out[f"{pos.strip()}/{neg.strip()}"] = (W_U[ip[0]].numpy(),
                                                   W_U[ineg[0]].numpy())
    # A few top predictions, to confirm the model is actually answering.
    acts, _, _, stmts = cached_acts("cities", "chat", MAX_N, SEED)
    logits = acts[:3, -1, :].numpy() @ W_U.numpy().T
    out["_examples"] = [
        (stmts[i][-90:], [tok.decode([t]) for t in np.argsort(-logits[i])[:5]])
        for i in range(3)
    ]
    os.makedirs("cache", exist_ok=True)
    torch.save(out, UNEMBED_CACHE)
    del model
    return out


def dom(acts, labels):
    d = acts[labels == 1].mean(axis=0) - acts[labels == 0].mean(axis=0)
    return d / (np.linalg.norm(d) + 1e-12)


def main():
    rows = answer_rows()

    print("what the model actually predicts at the readout token:")
    for stmt, top in rows["_examples"]:
        print(f"  ...{stmt!r}\n     -> {top}")

    print(f"\n{'=' * 78}")
    print(f"{'dataset':>24}  {'black box':>21}  {'probe L' + str(PROBE_LAYER):>14}  "
          f"{'delta':>7}")
    print(f"{'':>24}  {'(ask the model)':>21}  {'(chat, DoM)':>14}")
    print("=" * 78)

    bb_all, pr_all = [], []
    for name in DATASETS:
        acts, y, groups, _ = cached_acts(name, "chat", MAX_N, SEED)
        A = acts.numpy()

        # --- black box: no training, so score every example and average over
        # the same test splits the probe is scored on, for a like-for-like n.
        h_final = A[:, -1, :]
        # Filter BEFORE unpacking: rows also holds the "_examples" entry.
        bb_scores = {k: h_final @ (v[0] - v[1])
                     for k, v in rows.items() if not k.startswith("_")}

        bb_reps, pr_reps = {k: [] for k in bb_scores}, []
        for rep in range(N_REPEATS):
            trval, test_i = split(groups, frac_train=0.8, seed=1000 * rep + 7)
            tr_rel, _ = split(groups[trval], frac_train=0.75, seed=1000 * rep + 13)
            train_i = trval[tr_rel]

            for k, s in bb_scores.items():
                bb_reps[k].append(roc_auc_score(y[test_i], s[test_i]))

            d = dom(A[train_i, PROBE_LAYER, :], y[train_i])
            pr_reps.append(roc_auc_score(y[test_i], A[test_i, PROBE_LAYER, :] @ d))

        best_k = max(bb_reps, key=lambda k: np.mean(bb_reps[k]))
        bb_mu, bb_sd = np.mean(bb_reps[best_k]), np.std(bb_reps[best_k], ddof=1)
        pr_mu, pr_sd = np.mean(pr_reps), np.std(pr_reps, ddof=1)
        bb_all.append(bb_mu); pr_all.append(pr_mu)

        print(f"{name:>24}  {bb_mu:.3f}+-{bb_sd:.2f} {best_k:>8}  "
              f"{pr_mu:.3f}+-{pr_sd:.2f}  {pr_mu - bb_mu:+7.3f}")

    print("=" * 78)
    print(f"{'mean':>24}  {np.mean(bb_all):>21.3f}  {np.mean(pr_all):>14.3f}  "
          f"{np.mean(pr_all) - np.mean(bb_all):+7.3f}")

    print("\nHow to read this. Positive delta means the probe beats simply asking.")
    print("A delta near zero means the chat probe is reading the model's answer")
    print("and the internals bought you nothing -- in which case the honest")
    print("headline is that RAW is where the interesting representation lives,")
    print("because raw has no answer to read.")


if __name__ == "__main__":
    main()
