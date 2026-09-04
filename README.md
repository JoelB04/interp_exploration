# Do misaligned concepts have a hierarchical order to their geometry, and how much of concept geometry is just vocabulary?

Measuring whether a language model's representational geometry reflects an
externally-given category taxonomy and how much of any such structure survives
controlling for the words the categories happen to use.

**Model:** Qwen2.5-1.5B-Instruct (28 layers, d_model 1536), CPU only.
**Data:** SALAD-Bench harm taxonomy, with MMLU as a second, unrelated taxonomy.

---

> **Motivation** Mechanistic interpretability has been of deep interest to me
> lately. It combines my background in machine learning, computational
> neuroscience, and philosophy in a way that is hard to find. I believe there is
> much to learn from the discipline of computational neuroscience for interp
> research, primarily in how it sets out to describe neural circuits as
> representations and dynamic systems. This is why I have found representation
> geometry so interesting: it gives us a microscope to view the actual
> organization of information inside a model. The main motivation of this project
> is to identify whether there are distinguishing features of misaligned concept
> manifolds, which would hopefully help future researchers identify malignant
> concepts when their misalignment-status is unclear.
>
> This project is inspired by Haim Sompolinsky and his work on concept manifolds
> in neural networks, as well as my work within his lab. The thesis for the
> project was this: do misaligned concepts have different geometric structure in
> the representation space of a neural network? If so, what features are
> different and why? As it turns out, the hierarchical arrangement of concept
> manifolds may give insights into how misaligned concepts are represented.

---

## The question

Safety taxonomies carve harmful requests into categories and sub-categories.
SALAD-Bench, for instance, groups 13 tasks under 5 domains: *Illegal Activities*,
*Fraud*, *Security Threats* and *Influence Operations* all sit under
*Malicious Use*.

Does the model's internal geometry reflect that carving? Do sibling categories
sit closer together in activation space than the taxonomy-blind baseline would
predict? If so, is that because the model has abstracted something about
harm, or simply because siblings use similar words?

## Headline findings

| | |
|---|---|
| **The taxonomy beats a matched random partition** | p = 0.0005–0.015 at every layer 2–28, in both readout modes |
| **One branch carries it** | Removing *Malicious Use* is the only exclusion that kills the effect (p 0.002 to 0.13) |
| **~80% of it is vocabulary** | corr(lexical distance, geometric distance) = 0.73–0.91 |
| **A residual survives the lexical control, in `raw`** | flat with depth; p falls because the null tightens, not because the effect grows. Patchier in `request` |
| **MMLU shows no comparable residual** | its z goes *positive* after the lexical control |
| **Refusal shows no structure** | but refusal is near-ceiling (0.72–0.90), so this dataset cannot test it |

Plus one methodological result that mattered more than any of the above: **the
synthetic null model was calibrated 100× wrong**, and measuring caught it before any power analysis was built on it.

---

## Design

13 SALAD level-2 tasks under 5 level-1 domains, branching `[4,3,2,2,2]`.

![SALAD prompts per task](figures/salad_task_counts.png)

The whole design falls out of this one plot. *Persuasion and Manipulation* at 640
prompts is the binding constraint; *Defamation* (437) and the two Socioeconomic
Harms tasks (651 and 200) are what had to be dropped to keep M high enough.

**Every manifold uses exactly M = 640 prompts.** The participation
ratio $D$ is capped at $M-1$ and biased below it, so comparing manifolds of
different sizes measures sample size and reports it as geometry. Parent manifolds
contain more points than children by construction, so this confound points in
exactly the direction of the hypothesis.

M = 640 is set by the smallest kept task. Not 657, that would have dropped
*Persuasion and Manipulation* and left a parent with a single child, costing a
whole parent's worth of within-parent pairs.

**Main number**

```
branching [4,3,2,2,2]  ->  12 within-parent pairs, 66 between-parent
                           6 of the 12 come from Malicious Use alone
```

Every nesting claim rests on those 12 pairs. They are reported individually
rather than only as a mean.

**Readout position is a live variable**, never hardcoded. Every result below is
reported in two independent readouts: `raw` (bare prompt, last token) and
`request` (plain user turn under the chat template, generation-prompt token). A
third mode, `chat`, was retired mid-project once it turned out to carry a stale
instruction; see *Bugs found*. No figure in this README uses it.

The two modes are not redundant. Under `raw` the last token is content; under
`request` it is a fixed template token, so every example-specific fact has to be
attention-transported there. An effect appearing in both is not an artifact of
readout position.

## The instrument, before measurement

The synthetic part exists as a verification baseline, before looking at data. Hierarchical manifolds with known
ground truth, then three checks.

**1, Does the generator produce the geometry it claims?**

```
      M    rank   trace_err   D_hat/D
    160     159      -0.001   0.837
    640     639      +0.004   0.938
  10240    1536      -0.001   0.997
```

Rank is exactly `min(M-1, n)`; trace preserved to 0.4%. The mechanism is
visible in the spectrum: middle eigenvalues inflate (1.40× at M=640) because
trace conservation forces variance from 1536 directions into 639. That inflation
is why $\hat D < D$.

![spectrum recovery](figures/check1_spectrum_recovery.png)

**2, How large can D be before it cannot be measured?** Within 10% up to
D ≈ 30 at M=640. Across the empirical range (D = 10–25) the bias moves only
0.95 → 0.92 — approximately a shared constant, so it **cancels in cross-manifold
comparisons**.

![dimension recovery](figures/check2_dimension_recovery.png)

The middle panel is the one that licenses M=640: the M=640 curve stays inside the
±10% band across the entire range the real data occupies. Note also that at M=160
the curve crosses $\hat D/D = 1$ near D=15 — two opposite biases cancelling, not
accuracy. A ratio near 1 is not evidence of a good estimate.

**3, Does measured nesting track planted nesting?** Recovers the analytic curve
$\sigma/\sqrt{1+\sigma^2}$ to within 0.8%.

![nesting recovery](figures/check3_nesting.png)

The dotted line is the naive expectation that measured nesting equals the planted
σ. It doesn't — the estimator follows $\sigma/\sqrt{1+\sigma^2}$, and the two
diverge visibly above σ ≈ 0.3. Reading the planted value straight off the
measurement would have overstated nesting.

The right panel is also where this project's main gap lives: it establishes a
noise floor only out to σ = 1, while the real data sits at 1.5–5.5. See
*What this does not show*.

**Run out of order, on purpose:** before trusting the synthetic null at all, is
the real within-manifold spectrum actually a power law? It is: r^2 = 0.95–0.999
at every layer. And D = 10–25, far below the rank cap, so M=640 is good.

## Results

### The taxonomy beats a shuffle

Permuting which tasks are siblings while holding group sizes identical:

```
null mean 1.00 ± 0.08     real 0.725–0.821
p = 0.0005–0.015 at every layer 2–28, both readout modes
```

![random partition null](figures/random_partition.png)

27 contiguous layers in two independent readouts. The red line sits below the
grey band everywhere except layer 0.

Two sanity checks pass. Pooling all prompts and cutting them into fake tasks of
the same sizes gives 0.95–1.04, as it must. And **raw layer 0 gives p = 0.33**:
no taxonomy signal in the last token's embedding, strong signal by layer 2. The
structure appears the moment the model integrates context, then saturates.

`request` layer 0 is not plotted because it is *exactly* degenerate — under the
chat template every prompt shares the same final token, so all 13 centroids
coincide and the nesting ratio is undefined. That is a free exact null: any
structure reported there would be a bug.

### One branch carries it

| excluded | tasks | within-pairs | real | p |
|---|---|---|---|---|
| none | 13 | 12 | 0.785 | 0.0020 |
| Repr & Toxicity | 10 | 9 | 0.765 | 0.0088 |
| Misinformation | 11 | 11 | 0.743 | 0.0013 |
| Information & Safety | 11 | 11 | 0.826 | 0.0208 |
| **Malicious Use** | **9** | **6** | **0.906** | **0.1298** |
| Human Autonomy | 11 | 11 | 0.745 | 0.0005 |

Every parent can be dropped except one. **Caveat this design cannot resolve:**
dropping *Malicious Use* also halves the within-parent pairs 12 to 6, so some of
that p = 0.13 is lost power rather than lost effect.

### Most of it is vocabulary

*Malicious Use*'s four children all concern crime and share heavy vocabulary, a
purely lexical grouping would beat the null too. So: TF-IDF per task, residualise
geometric distance on lexical distance, rerun the permutation.

```
Malicious Use lexical distance   0.598 vs 0.713 over all pairs   (p = 0.030)
corr(lexical, geometric)         0.73–0.91, rising with depth
residual p          raw      0.037 at L2  ->  0.004 at L28
                    request  0.112 at L2  ->  0.007 at L28
residual effect size         -0.06 to -0.12, flat with depth
```

![lexical control](figures/lexical_control.png)

Roughly 80% of inter-category geometry is word overlap, and the siblings-share-
vocabulary worry is real rather than hypothetical: *Malicious Use*'s children are
significantly closer lexically than chance (p = 0.030). A residual nonetheless
survives in `raw` at all six sampled layers. **In `request` it does not**: p =
0.11 at layer 2 and 0.063 at layer 20, so the lexical-independent effect is
weaker and patchier under that readout than under `raw`.

The green line is the point of this figure. Distances are normalised by the mean
pairwise distance at each layer, which is necessary because residual-stream norms
grow ~15,000× across depth and raw magnitudes would climb mechanically. Once
normalised, **the effect size is flat while p falls** — the null tightens with
depth, the effect does not grow. Plotting p alone, as an earlier version of this
figure did, reads as the opposite; see *Bugs found*.

### A second taxonomy

MMLU, using its own published 4-way grouping (not one constructed here, the
whole value of a second taxonomy is that someone else drew the boundaries).
Subjects chosen for question length near SALAD's range, fixed before any
activation was computed.

```
raw mode, both datasets subsampled to M=200 for the matched comparison

layer   SALAD z   SALAD z|lex    MMLU z   MMLU z|lex
    4     -3.59        -3.13      0.36        +1.36
   14     -2.98        -1.58     -0.37        +0.97
   28     -3.26        -2.43     -1.04        -0.77
```

![taxonomy comparison](figures/taxonomy_comparison.png)

In `raw`, SALAD clears its null at every layer and survives the lexical control;
MMLU does neither, and after removing lexical overlap its z goes positive — the
wrong direction. z is measured against each dataset's *own* matched null, because
the branching differs (12 within-parent pairs against 15).

`request` complicates this and is plotted alongside. There MMLU *does* nest
(z = −3.6 at layers 22–26, comparable to SALAD), but neither dataset survives the
lexical control cleanly. So "survives the lexical control" is a property of the
`raw` readout, not a general property of the harm taxonomy.

### Behaviour: a null

The cached activations already contain the model's next-token distribution, so
reading behaviour cost zero forward passes. Refusal is near-ceiling (0.72–0.90)
and shows no clustering by parent (p = 0.60), with the point estimate in the
wrong direction.

![refusal](figures/refusal.png)

The right panel is what an honest null looks like: the true taxonomy sits in the
middle of its own null distribution, very slightly on the wrong side. The left
panel shows why the ordering cuts across the taxonomy — *Adult Content* (0.90)
and *Unfair Representation* (0.77) share a parent and sit near opposite ends.

This is deliberately not reported as "representation and behaviour are
separable." Refusal is at ceiling so there is little variance for any structure
to predict, and 12 pairs gives low power.

---

> **Interpretation** The coolest part of this project came from the disagreement
> between MMLU and SALAD. The fact that SALAD survived lexical control more than
> MMLU implies to me, at least, that there is a kind of correlated-manifold
> structure to be found in concepts that are misaligned. What this is, however,
> is incredibly open and I can not claim confidence on any interpretation yet.
> The main result I hope that is taken away from this is that categorical
> structure of representations *can* introduce geometric quirks that, maybe one
> day, in the future can become a kind of dictionary or lens for what kind of
> concept a network is thinking of.

---

## What this does not show

- **One model, one scale.** Everything is Qwen2.5-1.5B. No claim about scaling.
- **13 manifolds, 12 within-parent pairs.** Thin, and half of them from one
  parent.
- **The MMLU comparison may measure taxonomy coherence, not domain.** MMLU's
  4-way grouping is coarser and partly administrative — `other` is an explicit
  residual holding `human_aging`, `miscellaneous`, `nutrition`. SALAD's
  *Malicious Use* is four crime-related tasks.
- **Register differs** between the two datasets (imperative requests vs exam
  stems) and this design cannot exclude it.
- **The lexical residual is readout-dependent.** It survives at all six sampled
  layers under `raw`, but under `request` it fails at layer 2 (p = 0.11) and is
  marginal at layer 20 (p = 0.063). The taxonomy effect itself is robust to
  readout; the *lexical-independent part* of it is not.
- **No power analysis at the real regime.** The synthetic null was calibrated at
  a within-spread/separation ratio of 0.018; reality is 1.5–5.5. Recalibrating
  and rerunning the power sweep is outstanding, so there is currently **no bound
  on what this design could have detected.**

---

> **Next steps** Obviously this experiment would benefit from larger scaling.
> More manifolds, more data, more compute. These claims rest on two datasets.
> Additionally, it would be helpful to run these tests across many different
> regimes/datasets so that we could see whether consistent trends in the manifold
> geometry of certain concepts emerges. This could end up being a useful tool for
> understanding what the model is thinking just by analyzing the geometry of an
> associated batch of responses.

---

## Bugs found, and how

**A 100× calibration error.** The synthetic model assumed manifolds that are
essentially points, within-spread/separation of 0.018. Measured on real data it
is 1.5–5.5: **real manifolds are larger than the gaps between their centroids.**
Caught by measuring the calibration constant rather than assuming it, before the
power analysis depended on it.

![empirical nesting and calibration](figures/empirical_nesting.png)

The middle panel is the error. The synthetic model was built at 0.018; the real
data sits between 1.5 and 5.5, two orders of magnitude away, so the noise floor
the synthetic null implied was meaningless. The left panel is the nesting result
itself and the right panel is why 12 pairs still gives a usable measurement:
resampling sd is 0.005–0.014, because each distance is a sum over 1536
coordinates and concentrates. High ambient dimension works *for* this
measurement, which is not the intuition.

**A retracted interpretation.** An earlier reading — "the non-lexical effect
emerges in late layers" — was wrong. After normalising for norm growth the
effect size is flat and it is the null that tightens. The figure in *Most of it
is vocabulary* originally plotted only the p-value, which made the retracted
reading look correct; it now plots the effect size beside it.

**A framing bug, found by a check written for something else.** The refusal
script's descriptive stage printed the model's top predicted tokens as a sanity
check. They came back `True` and `False` — because `chat` mode was still
wrapping every prompt in *"Is the following true or false?"*, a leftover from an
earlier project. The model was answering a quiz, not deciding whether to refuse.
Fixed by adding a new `request` mode rather than redefining `chat`, since the
activation cache keys on the mode string and silently changing its meaning would
have left stale tensors under a valid label.

## Repo

```
src/acts.py          activation extraction; readout modes
src/cache.py         disk cache, dataset-agnostic, model loads lazily
src/geometry.py      R_M, D_M, centre statistics, equalize_class_n
src/synthetic.py     hierarchical generative model
src/salad.py         SALAD design and loading
src/mmlu.py          MMLU design and loading

scripts/01_smoke_test.py           plumbing verification
scripts/02, 03                     taxonomy exploration and audit
scripts/04_extract.py              SALAD activation extraction
scripts/05, 06, 08                 synthetic checks 1, 2, 3
scripts/07_empirical_spectrum.py   empirical spectra: is it a power law?
scripts/09_empirical_nesting.py    empirical nesting + calibration
scripts/10_random_partition.py     random-partition null
scripts/11_lexical_control.py      lexical control
scripts/12_refusal.py              refusal readout
scripts/13, 14                     MMLU extraction and comparison

logs/research_log.md    logs
logs/weird.md           surprises from the process
figures/                the plots reproduced in this README
```

## Running it

CPU only. Extraction is the only slow part, everything downstream reads cached
tensors and runs in seconds.

```bash
python scripts/04_extract.py          # ~10 h, 3 readout modes, 13 tasks
python scripts/13_extract_mmlu.py     # ~3 h
python scripts/07_empirical_spectrum.py
python scripts/09_empirical_nesting.py
python scripts/10_random_partition.py
python scripts/11_lexical_control.py
python scripts/14_compare_taxonomies.py
```

Synthetic checks need no data:

```bash
python scripts/05_verify_generator.py
python scripts/06_verify_dimension.py
python scripts/08_verify_nesting.py
```

## Method notes

Standards held throughout:

1. **Equal M before any spectral comparison.** $D$ is capped at $M-1$.
2. **Report M and n alongside every geometric quantity**, plus $\gamma = n/M$.
3. **Power before interpretation.** Know what the design could detect before
   looking. Only partially met here, see *What this does not show*.
4. **Suspicion on success.** A clean result triggers an artifact hunt: length,
   opening-word templates, and layer-0 behaviour.
5. **Fixed conventions.** Radius is
   $R_M = \sqrt{\sum\lambda^2 / \sum\lambda}\,/\,\lVert c\rVert$, unchanged
   throughout.
6. **Error bars from resampling, stating what they cover.** Here: a bootstrap
   over prompts within each manifold. It does not cover which categories the
   taxonomy defines, or the model.
