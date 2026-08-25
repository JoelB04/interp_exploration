"""Session 3c: error bars, and cosines at full precision.

Two jobs.

1. Recompute the answer-direction cosines from the actual vectors instead of
   reading them off a printed table at two decimal places.

2. Put error bars on everything by resampling the SPLIT many times.

What is and is not resampled, because this determines what the error bars mean:

  resampled      which examples land in train / val / test, at the group level,
                 and therefore also which layer gets selected on val
  NOT resampled  which 400 rows were drawn from each CSV, the model, the weights

The second one is fixed because the cache key includes the seed, so varying it
would re-embed everything -- 30-60 min per trial on CPU. The split is the
dominant noise source and is free to resample, so that is what we do.

CONSEQUENCE, and say this in the write-up: the test sets across repeats overlap
heavily, being drawn from the same fixed 400 examples. The spread below is
SPLIT SENSITIVITY, not a confidence interval on a population value, and it
understates true uncertainty. It answers "does this result depend on where I
happened to cut the data", which is a real question, but not "would this
replicate on fresh data", which needs new activations.

Run from the repo root:  python scripts/06_stability.py
Reads the existing cache. Loads the model only once, and only to grab the
unembedding matrix -- cached to cache/answer_dirs.pt after the first run.
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
MODES = ["raw", "chat"]
N_REPEATS = 20
ANSWER_CACHE = "cache/answer_dirs.pt"
OUT = "results"

# The five families. larger_than/smaller_than additionally share group keys by
# construction, so they are not independent even as a family.
FAMILIES = {
    "cities": "cities", "neg_cities": "cities",
    "sp_en_trans": "sp_en", "neg_sp_en_trans": "sp_en",
    "larger_than": "ordinal", "smaller_than": "ordinal",
    "companies_true_false": "companies",
    "common_claim_true_false": "common_claim",
}
ANSWER_PAIRS = [(" true", " false"), (" True", " False"), (" Yes", " No")]


def answer_directions():
    """Unit answer-token directions in residual space. Model loaded once, ever."""
    if os.path.exists(ANSWER_CACHE):
        return torch.load(ANSWER_CACHE, weights_only=False)

    from acts import load
    model, tok, _ = load()
    W_U = model.lm_head.weight.detach().float()
    gain = model.model.norm.weight.detach().float()

    out = {}
    for pos, neg in ANSWER_PAIRS:
        ip, ineg = tok.encode(pos), tok.encode(neg)
        if len(ip) != 1 or len(ineg) != 1:
            continue
        v = (W_U[ip[0]] - W_U[ineg[0]]) * gain
        out[f"{pos.strip()}-{neg.strip()}"] = (v / v.norm()).numpy()

    os.makedirs("cache", exist_ok=True)
    torch.save(out, ANSWER_CACHE)
    del model
    return out


def dom_all_layers(acts, labels):
    d = acts[labels == 1].mean(axis=0) - acts[labels == 0].mean(axis=0)
    return d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12)


def auroc_by_layer(scores, labels):
    return np.array([roc_auc_score(labels, scores[:, l])
                     for l in range(scores.shape[1])])


def one_repeat(store, rep):
    """One split draw -> (matrix, selected layers, directions at those layers)."""
    n_ds = len(DATASETS)
    idx = {}
    for name in DATASETS:
        g = store[name]["groups"]
        trval, test_i = split(g, frac_train=0.8, seed=1000 * rep + 7)
        tr_rel, val_rel = split(g[trval], frac_train=0.75, seed=1000 * rep + 13)
        idx[name] = (trval[tr_rel], trval[val_rel], test_i)

    M = np.zeros((n_ds, n_ds))
    chosen = np.zeros(n_ds, dtype=int)
    dirs = []

    for i, tr in enumerate(DATASETS):
        A, y = store[tr]["acts"], store[tr]["labels"]
        train_i, val_i, _ = idx[tr]
        D = dom_all_layers(A[train_i], y[train_i])

        val_auroc = auroc_by_layer(np.einsum("nld,ld->nl", A[val_i], D), y[val_i])
        chosen[i] = int(np.nanargmax(val_auroc))
        dirs.append(D[chosen[i]])

        for j, te in enumerate(DATASETS):
            B, yb = store[te]["acts"], store[te]["labels"]
            _, _, test_j = idx[te]
            # Take the TRAIN probe's selected layer on the target's activations.
            # B[test_j] still carries the layer axis; index it before projecting
            # or roc_auc_score receives a 2-D score array.
            s = B[test_j, chosen[i], :] @ D[chosen[i]]
            M[i, j] = roc_auc_score(yb[test_j], s)

    return M, chosen, np.stack(dirs)


def summarize(name, vals):
    v = np.asarray(vals)
    return (f"  {name:34s} {v.mean():+.3f} +/- {v.std(ddof=1):.3f}   "
            f"[{np.percentile(v, 5):+.3f}, {np.percentile(v, 95):+.3f}]")


def main():
    from scipy.stats import spearmanr
    os.makedirs(OUT, exist_ok=True)
    answers = answer_directions()
    n_ds = len(DATASETS)
    fam = np.array([FAMILIES[d] for d in DATASETS])
    report = {}

    for mode in MODES:
        print(f"\n{'=' * 72}\nmode: {mode}   ({N_REPEATS} split repeats)\n{'=' * 72}")

        store = {}
        for name in DATASETS:
            a, y, g, _ = cached_acts(name, mode, MAX_N, SEED)
            store[name] = dict(acts=a.numpy(), labels=y, groups=g)

        mats, chosens, coss = [], [], {k: [] for k in answers}
        rho_ans, rho_diag = {k: [] for k in answers}, []

        for rep in range(N_REPEATS):
            M, ch, D = one_repeat(store, rep)
            mats.append(M); chosens.append(ch)

            off = ~np.eye(n_ds, dtype=bool)
            rowmean = np.array([M[i][off[i]].mean() for i in range(n_ds)])
            rho_diag.append(spearmanr(np.diag(M), rowmean).statistic)

            for k, v in answers.items():
                c = D @ v                       # full precision, per dataset
                coss[k].append(c)
                rho_ans[k].append(spearmanr(c, rowmean).statistic)

        mats = np.stack(mats)
        M_mu, M_sd = mats.mean(0), mats.std(0, ddof=1)

        print("\ntransfer AUROC, mean +/- sd over splits")
        short = [d[:9] for d in DATASETS]
        print(f"{'':>24s}" + " ".join(f"{s:>13s}" for s in short))
        for i, d in enumerate(DATASETS):
            print(f"{d:>24s}" + " ".join(
                f"{M_mu[i,j]:>7.3f}+-{M_sd[i,j]:.2f}" for j in range(n_ds)))

        ch = np.stack(chosens)
        print("\nselected layer, mode of the repeats (and how often)")
        for i, d in enumerate(DATASETS):
            vals, cnt = np.unique(ch[:, i], return_counts=True)
            print(f"  {d:>24s}  layer {vals[cnt.argmax()]:2d}  "
                  f"({cnt.max()}/{N_REPEATS})   range {ch[:,i].min()}-{ch[:,i].max()}")

        print("\nanswer-direction cosine, per dataset (mean +/- sd over splits)")
        key = "True-False"
        C = np.stack(coss[key])
        for i, d in enumerate(DATASETS):
            print(f"  {d:>24s}  {C[:,i].mean():+.4f} +/- {C[:,i].std(ddof=1):.4f}")
        print(f"  {'mean |cos|':>24s}  {np.abs(C).mean():.4f}"
              f"   (random floor 1/sqrt(1536) = {1/np.sqrt(1536):.4f})")

        print("\nSpearman rho vs mean transfer, over splits")
        for k in answers:
            print(summarize(f"answer-cos ({k})", rho_ans[k]))
        print(summarize("in-distribution AUROC (diagonal)", rho_diag))

        # Leave-one-family-out: is the correlation carried by a single family?
        print("\nleave-one-family-out rho, answer-cos (True-False)")
        base = np.stack(coss[key]).mean(0)
        rm = np.stack([np.array([m[i][~np.eye(n_ds, dtype=bool)[i]].mean()
                                 for i in range(n_ds)]) for m in mats]).mean(0)
        print(f"  {'all 8 datasets':>24s}  rho={spearmanr(base, rm).statistic:+.3f}")
        for f in sorted(set(fam)):
            keep = fam != f
            if keep.sum() < 4:
                continue
            r = spearmanr(base[keep], rm[keep]).statistic
            print(f"  {'drop ' + f:>24s}  rho={r:+.3f}   (n={keep.sum()})")

        report[mode] = dict(M_mu=M_mu, M_sd=M_sd, chosen=ch, cos=coss,
                            rho_answer=rho_ans, rho_diag=rho_diag)

    torch.save(report, os.path.join(OUT, "stability.pt"))
    print(f"\nsaved -> {OUT}/stability.pt")
    print("\nReminder: these bars are SPLIT SENSITIVITY on a fixed 400-example")
    print("draw. Test sets overlap across repeats. They do not tell you whether")
    print("the result replicates on fresh data.")


if __name__ == "__main__":
    main()
