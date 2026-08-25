# Research log

Timestamped, append-only. **Write the prediction before running anything.** A
prediction recorded after seeing the output is worth nothing, and the habit is
the point — calibration is the skill being built here.

Dead ends stay in. They get deleted from the code, never from this log.

Entry template:

```
## YYYY-MM-DD — session N: title

**Question.** What is being asked, in one sentence.

**Prediction (written before running).** What I expect, and roughly how strongly.
Name the outcome that would surprise me.

**Setup.** Dataset, n, model, layers, readout mode, probe type. Anything a
reader would need to reproduce the number.

**Result.** What actually happened. Report n alongside every AUROC.

**Baseline / null.** Permutation null, black-box comparison, or why neither
applies yet.

**Read.** What I now believe, and what I don't. Flag anything untested.

**Next.** The single next thing.
```

---

## 2026-08-23 — session 1: smoke test / plumbing

**Question.** Can I get activations out of Qwen2.5-1.5B-Instruct and know exactly
what tensor I'm holding?

**Prediction.** *(not recorded — this log was created retroactively on 2026-08-25.
Session 1 and 2 predictions exist only in my head or not at all. From session 3
onward the prediction goes in before the run.)*

**Setup.** `scripts/01_smoke_test.py`. Qwen2.5-1.5B-Instruct, CPU, fp32.

**Result.** All asserts pass. Established:

- `hidden_states` is a tuple of length `n_layers + 1`; index 0 is the embedding,
  index `i` is the output of block `i-1`.
- The final norm is already applied to `hidden_states[-1]` for this model —
  `lm_head(hs[-1])` reproduces the logits.
- Forward hook on `model.model.layers[i]` matches `hidden_states[i+1]`.
- Left padding works; batched vs unbatched last-token activations differ by
  ~3e-4 (fp32 accumulation noise, not a bug).
- Under the chat template the last token is the generation-prompt token and is
  identical across all examples — all example-specific information must be
  attention-transported there. Readout position is therefore a live experimental
  variable, hence the `mode` parameter in `src/acts.py`.

**Baseline / null.** N/A — plumbing only, no claim made.

**Read.** Setup is trustworthy. The four-statement similarity pilot at the end is
too small to conclude anything and was not intended to.

**Next.** First real probe.

---

## 2026-08-25 — session 2: first probe *(in progress)*

**Question.** Is truth linearly decodable from the residual stream, at which
layer, and does readout position (`raw` vs `chat`) matter?

**Prediction (written before running).** *(fill this in before the first run)*

**Setup.** `scripts/02_first_probe.py`. `cities` from geometry_of_truth,
MAX_EXAMPLES=400, 70/30 random row split, diff-of-means and logistic regression
(C=0.1), AUROC at all 29 layer indices, both readout modes.

**Result.** *(pending)*

**Baseline / null.** *(pending — permutation null not yet implemented in the
script; standard 2 says it must be before any conclusion is drawn.)*

**Read.** *(pending)*

**Next.** Superseded by session 3 — the transfer matrix subsumes this, and
`03_transfer.py` fixes the two bugs that would have made session 2's numbers
wrong anyway (404ing dataset URL, pair-straddling random split).

---

## 2026-08-25 — session 3: the transfer matrix

**Question.** Train a truth probe on each of 8 datasets, test on all 8. Which
probes transfer, which collapse to chance, and which anti-generalise?

**Prediction (written before running).** *(FILL THIS IN BEFORE THE FIRST RUN.
Specifically: (a) which cells go below 0.5, (b) whether cities→neg_cities is
symmetric with neg_cities→cities, (c) how far the diagonal drops now that the
split is group-aware, (d) whether `raw` or `chat` transfers better, and whether
that is the same mode with the higher diagonal.)*

**Setup.** `scripts/03_transfer.py`. 8 geometry-of-truth datasets, MAX_N=400,
group-aware 60/20/20 split, diff-of-means at all 29 layer indices, both readout
modes. Layer chosen per train-dataset on its own val split; reported on test.
Permutation null, 20 replicates, two-sided.

**Result.** *(pending)*

**Baseline / null.** Permutation null is implemented and two-sided (anti-
generalisation is a real outcome, so a one-sided test would miss it). Black-box
baseline — just asking the model — is NOT yet implemented and is still owed.

**Read.** *(pending)*

**Next.** *(pending)*

---

### Pipeline validation, 2026-08-25 (not a result)

Before any real run, `03_transfer.py` was exercised on synthetic activations with
a known planted structure: two orthogonal "truth" directions defining two dataset
families, and a deliberately sign-flipped direction for `neg_cities` and
`neg_sp_en_trans`. The script recovered all of it — within-family transfer ~0.97,
cross-family ~0.5, and the planted anti-generalisation at 0.007 and 0.029, each
correctly flagged by the two-sided null.

This tests the plumbing (splitting, layer selection, null, grid assembly), not
the science. It says the script does what it claims. It says nothing about Qwen.

One caveat it exposed: 64 cells at a two-sided 95% threshold means roughly 3
false stars expected by chance, and the synthetic run showed a few. Do not read
individual starred cells as findings without a multiple-comparisons correction.
