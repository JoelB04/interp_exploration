"""Session 3f: the black-box baseline for RAW mode.

Session 3e killed the chat results: the layer-20 chat probe scored 0.963 and
simply asking the model scored 0.957. Raw is the remaining hope, because under
raw the model is never asked a question and there is no answer token to read.

But raw owes its own baseline, and it is not "ask the model" -- there is no
question. It is FLUENCY. True statements tend to be more probable than false
ones under the model's own distribution ("Paris is in France" is a more likely
string than "Paris is in Japan"), so a raw probe could be reading how surprising
the sentence is rather than whether it is true.

Three black-box scores, none of which look inside the model:

  mean_lp   mean token log-probability of the statement, teacher-forced.
            Length-normalised, so it is "how unsurprising per token".
  sum_lp    total log-probability. Confounded with length by construction and
            included mainly so you can see that confound rather than trip on it.
  n_tokens  statement length alone. A pure sanity check -- if this separates
            true from false, the dataset is broken and nothing else matters.

Costs real forward passes (~3200 short sequences, roughly 10 min on CPU) because
the activation cache stores last-token states only, and this needs logits at
every position. Results are cached to cache/raw_baseline.pt, so it runs once.

Run from the repo root:  python scripts/09_raw_baseline.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cache import cached_acts  # noqa: E402
from data import DATASETS, prepare, split  # noqa: E402

MAX_N, SEED = 400, 0
N_REPEATS = 20
PROBE_LAYER = 17          # best fixed transfer layer for raw, from session 3d
BATCH = 16
SCORE_CACHE = "cache/raw_baseline.pt"


@torch.no_grad()
def sequence_scores(statements, model, tok, device):
    """Teacher-forced log-probs of each bare statement. -> mean_lp, sum_lp, n_tok."""
    mean_lp, sum_lp, n_tok = [], [], []

    for i in range(0, len(statements), BATCH):
        chunk = statements[i:i + BATCH]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        ids, am = enc.input_ids, enc.attention_mask

        logits = model(**enc).logits.float()
        lp = F.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)     # (B, T-1)

        # A position counts only if the target AND its predecessor are real.
        # With left padding the first real token has a pad predecessor, so it is
        # correctly excluded rather than scored against garbage context.
        m = (am[:, 1:] * am[:, :-1]).float()

        s = (tok_lp * m).sum(1)
        n = m.sum(1).clamp(min=1)
        sum_lp.append(s.cpu().numpy())
        mean_lp.append((s / n).cpu().numpy())
        n_tok.append(am.sum(1).cpu().numpy())

        print(f"\r    {min(i + BATCH, len(statements))}/{len(statements)}",
              end="", flush=True)
    print()
    return (np.concatenate(mean_lp), np.concatenate(sum_lp),
            np.concatenate(n_tok).astype(float))


def get_scores():
    if os.path.exists(SCORE_CACHE):
        return torch.load(SCORE_CACHE, weights_only=False)

    from acts import load
    model, tok, device = load()

    out = {}
    for name in DATASETS:
        # prepare() with the same seed reproduces exactly the rows that were
        # embedded into the activation cache, so scores line up row-for-row.
        statements, labels, groups = prepare(name, max_n=MAX_N, seed=SEED)
        print(f"  {name}  n={len(statements)}")
        mean_lp, sum_lp, n_tok = sequence_scores(statements, model, tok, device)
        out[name] = dict(mean_lp=mean_lp, sum_lp=sum_lp, n_tokens=n_tok,
                         labels=labels, groups=groups)

    os.makedirs("cache", exist_ok=True)
    torch.save(out, SCORE_CACHE)
    del model
    return out


def dom(acts, labels):
    d = acts[labels == 1].mean(axis=0) - acts[labels == 0].mean(axis=0)
    return d / (np.linalg.norm(d) + 1e-12)


def main():
    S = get_scores()

    print(f"\n{'=' * 86}")
    print(f"{'dataset':>24} {'mean_lp':>11} {'sum_lp':>11} {'n_tokens':>11} "
          f"{'probe L' + str(PROBE_LAYER):>12} {'delta':>8}")
    print(f"{'':>24} {'(fluency)':>11} {'(+length)':>11} {'(length)':>11} "
          f"{'(raw, DoM)':>12} {'probe-best':>8}")
    print("=" * 86)

    agg = {k: [] for k in ["mean_lp", "sum_lp", "n_tokens", "probe", "delta"]}

    for name in DATASETS:
        d = S[name]
        y, groups = d["labels"], d["groups"]
        acts, y2, _, _ = cached_acts(name, "raw", MAX_N, SEED)
        A = acts.numpy()
        assert np.array_equal(y, y2), f"{name}: score/activation rows disagree"

        res = {k: [] for k in ["mean_lp", "sum_lp", "n_tokens", "probe"]}
        for rep in range(N_REPEATS):
            trval, test_i = split(groups, frac_train=0.8, seed=1000 * rep + 7)
            tr_rel, _ = split(groups[trval], frac_train=0.75, seed=1000 * rep + 13)
            train_i = trval[tr_rel]

            for k in ["mean_lp", "sum_lp", "n_tokens"]:
                res[k].append(roc_auc_score(y[test_i], d[k][test_i]))

            w = dom(A[train_i, PROBE_LAYER, :], y[train_i])
            res["probe"].append(roc_auc_score(y[test_i], A[test_i, PROBE_LAYER, :] @ w))

        mu = {k: float(np.mean(v)) for k, v in res.items()}
        sd = {k: float(np.std(v, ddof=1)) for k, v in res.items()}

        # A black-box score below 0.5 is still informative -- it just means the
        # sign is flipped. Fold to |AUROC - 0.5| so the baseline gets full credit.
        best_bb = max(abs(mu[k] - 0.5) for k in ["mean_lp", "sum_lp", "n_tokens"]) + 0.5
        delta = mu["probe"] - best_bb
        for k in ["mean_lp", "sum_lp", "n_tokens", "probe"]:
            agg[k].append(mu[k])
        agg["delta"].append(delta)

        print(f"{name:>24} {mu['mean_lp']:>6.3f}+-{sd['mean_lp']:.2f} "
              f"{mu['sum_lp']:>6.3f}+-{sd['sum_lp']:.2f} "
              f"{mu['n_tokens']:>6.3f}+-{sd['n_tokens']:.2f} "
              f"{mu['probe']:>6.3f}+-{sd['probe']:.2f} {delta:>+8.3f}")

    print("=" * 86)
    print(f"{'mean':>24} {np.mean(agg['mean_lp']):>11.3f} "
          f"{np.mean(agg['sum_lp']):>11.3f} {np.mean(agg['n_tokens']):>11.3f} "
          f"{np.mean(agg['probe']):>12.3f} {np.mean(agg['delta']):>+8.3f}")

    print("\nHow to read this.")
    print("  n_tokens far from 0.5  -> the dataset leaks the label through length.")
    print("  mean_lp far from 0.5   -> fluency alone separates true from false,")
    print("                            and the raw probe must beat it to mean")
    print("                            anything.")
    print("  delta near zero        -> raw is fluency, and the project has no")
    print("                            surviving positive result.")
    print("  delta clearly positive -> raw reads something fluency does not, and")
    print("                            THAT is the finding worth writing up.")


if __name__ == "__main__":
    main()
