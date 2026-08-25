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

**Result.** Ran 2026-08-25. n_test = 71-80 per column, SE on a 0.5 AUROC
~0.056. Layer 0 sits at exactly 0.500 in both modes, diagonal and off-diagonal
(standard-5 check passes).

Headline: **readout position determines whether negation breaks the probe.**

    negation pair              raw (A->B / B->A)    chat (A->B / B->A)
    cities <-> neg_cities        0.332 / 0.002        0.951 / 0.864
    sp_en  <-> neg_sp_en         0.001 / 0.019        0.010 / 0.041
    larger <-> smaller           0.013 / 0.649        0.941 / 0.323

In raw, all three pairs anti-generalise (5 of 6 cells below chance). In chat the
anti-generalisation vanishes for cities and mostly for larger/smaller, but
survives at full strength for sp_en_trans.

Diagonal mean 0.951 (raw) / 0.967 (chat). Off-diagonal mean 0.621 / 0.793.
Rank-1 explains 69.3% (raw) / 51.5% (chat) of centred logit-AUROC variance, so
between a third and a half of the matrix is genuine relational structure rather
than "some datasets are easy". Row spread > column spread in both modes: which
probe you trained matters more than which target you test on.

**Baseline / null.** Permutation null implemented, two-sided, 20 replicates.
Black-box baseline STILL OWED and now load-bearing -- see session 3b.

**Read.** The chat probes were suspected of reading the model's about-to-be-
emitted answer token. Session 3b tests this directly and the answer is "partly".
Not yet safe to claim chat-mode results show truth is linearly represented.

**Next.** Session 3b, then geometric predictors against this grid.

---

## 2026-08-25 — session 3b: is the chat probe reading the answer token?

**Question.** Under `chat` the readout sits where the model is about to answer
"true" or "false". Is the probe direction just the answer-token direction?

**Prediction (written before running).** *(not recorded -- ran immediately after
3. From here on, in before the run.)*

**Setup.** `scripts/04_unembed_check.py`. Logit-lens each probe direction
(scaled by the final RMSNorm gain), read top/bottom tokens; and cosine against
W_U[" true"] - W_U[" false"] and two variants. Random-direction floor in d=1536
is |cos| ~ 0.026.

**Result.** Not the answer direction, but meaningfully tilted toward it.

    mode   mean |cos| with True-False dir   max
    raw    0.034                            0.062   (at the random floor)
    chat   0.121                            0.242   (3-5x the floor)

cos = 0.24 means ~6% of the direction's variance is the answer direction. So the
chat probe is NOT simply the answer readout.

The token readout is more striking than the cosines. Raw directions unembed to
semantic garbage. Chat directions unembed to clean correctness vocabulary:
cities -d gives ' Incorrect', ' incorrect', ' wrong', '_invalid'; larger_than -d
gives ' inconsistent', ' destructive', ' unethical', ' ineffective';
companies/common_claim +d give 'yes', ' yes', ' checkmark', assertTrue.

**The unplanned finding.** Cosine with the answer direction PREDICTS transfer:

    mode   answer-cos vs mean transfer    diagonal vs mean transfer
    raw    rho = +0.647  p = 0.083        rho = -0.619  p = 0.102
    chat   rho = +0.838  p = 0.009        rho = -0.452  p = 0.260

Two things at once. The obvious predictor (in-distribution AUROC) is NEGATIVELY
correlated with transfer -- the best probes in-distribution are the worst
travellers. And answer-direction cosine is a fully LEGAL predictor: it needs the
train probe and the unembedding matrix, no target data of any kind.

Caveats, load-bearing: n = 8 datasets clustered into ~5 families, so p = 0.009
is optimistic by a wide margin. One seed, one model. Cosines here were read at
2dp from the printed table and should be recomputed at full precision before any
claim. And there is a whiff of tautology to guard against -- "probes that found
the general correctness feature generalise" is close to true by definition; the
content is that it is MEASURABLE in advance, from weights alone.

**Read.** Chat results are not a black-box measurement in disguise, but they are
answer-adjacent in a way raw results are not. The mechanistic story that fits:
raw probes read an upstream content feature that never integrates the negation;
chat probes read a downstream correctness feature that does. sp_en_trans is the
exception on both counts (lowest answer-cos, still anti-generalises in chat),
which is consistent rather than coincidental.

**Next.** Recompute cosines at full precision. Then the black-box baseline --
it is now the single most important missing number.

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
