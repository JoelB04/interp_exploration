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


---

## 2026-08-25 — session 3c: error bars

**Question.** How much of session 3/3b depends on which split I happened to draw?

**Prediction (written before running).** *(not recorded)*

**Setup.** `scripts/06_stability.py`. 20 resampled group-aware 60/20/20 splits on
the SAME cached 400-example draw. Cosines recomputed from the direction vectors
at full precision rather than read off a printed table.

Scope of the bars, and this matters: split sensitivity only. The 400 rows per
dataset, the model and the weights are all fixed, and test sets overlap heavily
across repeats. These bars say "does the result depend on where I cut the data".
They do NOT say "would this replicate on fresh data".

**Result. Two session-3 conclusions do not survive.**

1. "Chat mode fixes negation for cities" was a single-split artifact.
   chat `cities → neg_cities` = 0.587 +/- 0.44 over 20 splits; the 0.951 logged
   in session 3 was one draw. `neg_cities → cities` = 0.511 +/- 0.46. Both are
   effectively bimodal, flipping between ~0.95 and ~0.05.

2. The raw-mode answer-cosine correlation is largely a negation confound.
   rho = +0.738 on all 8, but +0.371 on the six non-negated datasets. Raw
   cosines also sit at the random-direction floor (mean |cos| 0.038 vs 0.026),
   so there is little real signal for them to carry.

**What does survive.**

- sp_en_trans anti-generalisation is rock solid: 0.013 +/- 0.03 and
  0.026 +/- 0.03. Diagonals are all 0.98-1.00 +/- <=0.01.
- The CHAT answer-cosine result is robust. rho = +0.760 +/- 0.220 over splits,
  90% interval [+0.251, +0.929], never crossing zero. Leave-one-family-out
  gives 0.929-0.952. Dropping both negated datasets leaves +0.943, so it is NOT
  the negation confound that sinks the raw version. Chat cosines are ~5x the
  random floor (0.123).
- In-distribution AUROC anti-predicts transfer in both modes:
  -0.452 +/- 0.261 (raw), -0.398 +/- 0.244 (chat).

**Mechanism for the instability.** Val-based layer selection is unstable --
`cities` selects anywhere in 16-28 across repeats. Different layers give
qualitatively different transfer, hence the bimodality.

**Read.** The headline flips. The robust result is the chat answer-cosine
predictor, not the readout-position story about negation. The negation story now
rests on sp_en_trans alone, where it is very solid, plus unstable cities cells
that should not be reported as a finding.

**Next.** Fix the layer rather than selecting it per split, and see how much
variance that removes. Then the black-box baseline.


---

## 2026-08-25 — session 3d: fix the layer

**Question.** How much of the session-3c instability was val-based layer selection?

**Prediction (written before running).** *(not recorded)*

**Setup.** `scripts/07_fixed_layer.py`. Full transfer matrix at every layer,
layer held FIXED across all eight probes, 20 resampled splits at each. Cache
only, no forward passes.

**Result. Essentially all of it.**

    chat, mean sd over off-diagonal cells
      selected-layer (3c)   up to 0.46 on individual cells
      fixed layer 20        0.015 mean, 0.031 max

And with the layer fixed, a sharp TRANSITION appears in chat:

    chat negation cells        L16          L17          L20          L28
      cities->neg_cities    0.071+-0.02  0.106+-0.03  0.956+-0.02  0.939+-0.03
      neg_cities->cities    0.015+-0.01  0.013+-0.01  0.975+-0.02  0.980+-0.01
      sp_en->neg_sp_en      0.105+-0.04  0.358+-0.06  0.964+-0.02  0.959+-0.02
      neg_sp_en->sp_en      0.056+-0.03  0.196+-0.06  1.000+-0.00  1.000+-0.00
      larger->smaller       0.458+-0.05  0.906+-0.03  0.973+-0.02  0.971+-0.02
      smaller->larger       0.415+-0.07  0.917+-0.03  0.991+-0.01  0.990+-0.01

Before ~L17 every negation pair anti-generalises. After ~L20 every one transfers
near-perfectly. Mean off-diagonal transfer jumps 0.677 (L16) -> 0.946 (L18).

This also retires the "sp_en_trans is the exception" claim from 3b. It was not an
exception; its selected layer was 14, on the wrong side of the transition.

**Read.** All of the negation anti-generalisation is a pre-transition-layer
phenomenon. Session 3 was reading different sides of a sharp transition for
different datasets and calling the mixture a readout-mode effect.

**Next.** Black-box baseline before interpreting the transition.

---

## 2026-08-25 — session 3e: the black-box baseline

**Question.** Does the layer-20 chat probe beat just asking the model?

**Prediction (written before running).** Expected the probe to win by a little.

**Setup.** `scripts/08_blackbox.py`. hs[-1] is post-norm, so W_U @ hs[-1] is the
logit vector -- the cached chat activations already contain the answer
distribution and this costs zero forward passes. Score = logit(True) -
logit(False) at the readout token, best of five token pairs per dataset. Same 20
test splits as the probe, for like-for-like n.

Verified the model is genuinely answering: top predictions at the readout token
are 'False' for a false city statement and 'True' for true ones.

**Result. It does not. Mean delta +0.006.**

    dataset                black box    probe L20    delta
    cities                 0.982        0.984        +0.002
    neg_cities             0.959        0.973        +0.014
    sp_en_trans            1.000        1.000        +0.000
    neg_sp_en_trans        0.956        0.988        +0.032
    larger_than            0.991        0.993        +0.002
    smaller_than           0.978        0.983        +0.005
    companies_true_false   0.941        0.943        +0.002
    common_claim_true_false 0.849       0.840        -0.009
    mean                   0.957        0.963        +0.006

Every delta is inside the split-to-split sd (0.01-0.03).

**Read. The chat results are a black-box measurement in disguise.** A layer-20
chat probe is reading the model's answer token, and reading internals buys
nothing over prompting. By extension the L17->L20 transition is most likely the
model COMPUTING ITS ANSWER, not "negation being integrated into a truth
representation" -- the deflationary reading is the one the baseline supports.

The chat half of this project is therefore closed as an interpretability result.
It stands as a cautionary result: probe AUROC of 0.96, stable to +-0.015, on a
readout position that made it worthless.

**Next.** Move to raw, where the model was never asked a question and there is
no answer to read. But raw needs its OWN black-box baseline before it gets any
credit -- the natural one is mean token log-probability of the statement, since
true facts tend to be more probable than false ones. That is the next number.


---

## 2026-08-25 — session 3f: the fluency baseline for raw

**Question.** Does the raw probe beat the model's own sense of how likely the
sentence is?

**Prediction (written before running).** *(not recorded)*

**Setup.** `scripts/09_raw_baseline.py`. Teacher-forced mean token log-prob of
each bare statement, plus sum log-prob and token count as controls. ~3200 forward
passes. Compared against the raw diff-of-means probe at fixed layer 17, same 20
splits.

**Result 1: within-dataset, the probe wins.** Mean probe 0.954 vs best black-box
0.826, delta +0.128.

**Result 2: on TRANSFER -- the quantity the project is actually about -- it does
not.** Probe transfer 0.827 vs fluency-on-target 0.811. Delta **+0.017**, well
inside the spread (per-cell sd 0.06-0.19).

    target                probe transfer   fluency|folded    delta
    cities                    0.935            0.874        +0.062
    neg_cities                0.875            0.802        +0.072
    sp_en_trans               0.936            0.999        -0.063
    neg_sp_en_trans           0.704            0.998        -0.294
    larger_than               0.788            0.769        +0.018
    smaller_than              0.959            0.633        +0.326
    companies                 0.697            0.667        +0.030
    common_claim              0.724            0.743        -0.019
    mean                      0.827            0.811        +0.017

**Result 3, and the interesting one: the verdict hinges on ONE BIT.** The table
above lets fluency know its sign on the target (folding |AUROC-0.5|), which
requires target labels. Denied that bit -- the honest deployment condition, where
you have no labels on the distribution you are monitoring -- fluency averages
0.577 and the transferred probe averages 0.787, a gap of +0.21.

So: the probe beats fluency if and only if fluency is applied blind. A single
labelled example on the target would supply the missing bit and erase most of
the advantage.

**Result 4: two datasets are pure fluency and are void as evidence.**
sp_en_trans mean_lp AUROC = 0.999, neg_sp_en_trans = 0.002. Fluency alone solves
both perfectly. Nothing measured on that pair says anything about truth
representations -- which retracts the "sp_en_trans anti-generalisation is rock
solid" claim from 3c as evidence, though the number itself stands.

**Result 5: the probe is substantially a fluency reader.** Spearman between probe
score and mean_lp, per dataset: cities +0.553, neg_cities -0.521, sp_en +0.793,
neg_sp_en -0.802, larger +0.472, smaller -0.298, companies +0.321,
common_claim +0.580. The sign flips on the negated sets exactly as the
fluency-truth relationship does.

But it is a PARTIAL explanation, not a complete one: larger_than and
smaller_than have opposite fluency signs (0.769 vs 0.367) yet the probe transfers
between them at 0.946, not below chance. Fluency sign-flip predicts inversion
there and does not get it.

**Also: a dataset defect.** n_tokens alone gives AUROC 0.654 on larger_than and
0.346 on smaller_than. Statement length leaks the label in that pair. The
+0.326 delta on smaller_than is therefore measured against a length-confounded
baseline and should not be leaned on.

**Read.** No surviving claim that reading internals beats a trivial baseline
out of distribution. The honest headline is now a negative one, with a genuinely
interesting hinge: the advantage is worth about one bit of target label
information.

**Next.** Partial correlation of probe vs fluency controlling for the label, to
separate "the probe reads fluency" from "both read truth". Then write up.


---

## 2026-08-25 — session 3g: partial correlation and residualised transfer

**Question.** Session 3f said the probe correlates with fluency at rho 0.30-0.80.
But both correlate with the label. Does anything survive holding the label fixed,
and does the probe carry truth information fluency does not?

**Setup.** `scripts/10_partial.py`. (a) Spearman(probe, fluency) within each
label class, pooled. (b) Regress probe score on fluency, keep the residual, take
AUROC of the residual against the label. The regression uses no labels -- it is
one score against another -- so fitting on the evaluation set is legitimate.

**Result 1: I overstated "the probe is a fluency reader". It mostly is not.**

    dataset            raw rho    partial rho
    cities              +0.553      +0.001
    neg_cities          -0.521      -0.188
    sp_en_trans         +0.793      +0.175
    neg_sp_en_trans     -0.802      -0.216
    larger_than         +0.472      +0.180
    smaller_than        -0.298      -0.170
    companies           +0.321      +0.122
    common_claim        +0.580      +0.436

cities collapses to +0.001. The large raw correlations were almost entirely
mediated by the shared cause. Only common_claim retains substantial non-truth
shared structure.

**Result 2: the probe survives fluency removal, but loses about a tenth.**
Mean off-diagonal transfer 0.832 -> 0.724 after residualising on fluency
(diagonal 0.954 -> 0.835). 0.724 is well above chance, so there IS truth
information beyond the free baseline. Per target the drop is very uneven:
smaller_than 0.959 -> 0.946 (drop 0.013), sp_en_trans 0.941 -> 0.608 (0.333).

**Result 3, and this is the important one: the negation anti-generalisation
splits into two kinds.** Residualised at the layers where the effect is strong:

    cell                       L12            L14            L16
    cities -> neg_cities   0.044|0.148    0.013|0.089    0.344|0.414
    neg_cities -> cities   0.006|0.118    0.002|0.086    0.066|0.212
    sp_en -> neg_sp_en     0.003|0.437    0.000|0.381    0.069|0.479
    neg_sp_en -> sp_en     0.003|0.432    0.000|0.393    0.013|0.450
    larger -> smaller      0.076|0.095    0.030|0.056    0.851|0.853
    smaller -> larger      0.097|0.196    0.060|0.159    0.678|0.680

- sp_en pair: 0.000 -> 0.381-0.437. Removing fluency returns it to near chance.
  The anti-generalisation there was a FLUENCY ARTIFACT, entirely.
- cities pair: 0.013 -> 0.089 at L14. Still catastrophically inverted. NOT
  explained by fluency.
- larger/smaller: 0.030 -> 0.056 at L14. Barely moves. NOT explained by fluency.

**Read.** This partially rescues the negation finding, and sharpens it. Two of
the three pairs anti-generalise for reasons a free black-box baseline cannot
account for; one does not, and that one (sp_en) was the pair I had been citing
as the cleanest evidence. The surviving claim is narrower and better supported:
early-to-mid-layer raw probes invert across negation for cities and ordinal
comparisons, and fluency does not explain it.

**Next.** Understand WHY cities and ordinal survive but translation does not.
Candidate: sp_en truth is fully determined by lexical association, so there is no
truth feature separate from fluency to find in the first place.
