# Do misaligned concepts have a hierarchical order to their geometry and
# how much of concept geometry is just vocabulary?

Measuring whether a language model's representational geometry reflects an
externally-given category taxonomy and how much of any such structure survives
controlling for the words the categories happen to use.

**Model:** Qwen2.5-1.5B-Instruct (28 layers, d_model 1536), CPU only.
**Data:** SALAD-Bench harm taxonomy, with MMLU as a second, unrelated taxonomy.

---

> **Motivation** Mechanistic interpretability has been of deep interest to me lately.
It combines my background in machine learning, computational neuroscience, and philosophy 
in a way that is hard to find. I believe there is much to learn from the discipline of computational
neuroscience for interp research, primarily in how it sets out to describe neural circuits as representations
and dynamic systems. This is why I have found representation geometry so interesting: it gives us a microscope to view the 
actual organizatiom of information inside a model. The main motivation of this project is to identify whether there are distinguishing features of misaligned concept manifolds, which would hopefully help future researchers identify malignant 
concepts when their misalignment-status is unclear.
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
| **The taxonomy beats a matched random partition** | p = 0.001–0.011 at every layer 2–28, in both readout modes |
| **One branch carries it** | Removing *Malicious Use* is the only exclusion that kills the effect (p 0.002 to 0.13) |
| **~80% of it is vocabulary** | corr(lexical distance, geometric distance) = 0.73–0.91 |
| **A residual survives the lexical control** | flat with depth; p falls 0.049 to 0.008 because the null tightens |
| **MMLU shows no comparable residual** | its z goes *positive* after the lexical control |
| **Refusal shows no structure** | but refusal is near-ceiling (0.72–0.90), so this dataset cannot test it |

Plus one methodological result that mattered more than any of the above: **the
synthetic null model was calibrated 100× wrong**, and measuring caught it before any power analysis was built on it.

---

## Design

13 SALAD level-2 tasks under 5 level-1 domains, branching `[4,3,2,2,2]`.

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

**Readout position is a live variable**, never hardcoded, `raw` (bare prompt),
`request` (plain user turn), and a deprecated `chat` (see *Bugs found*).

## The instrument, before measurement

The synthetic part exists as a verification baseline, before looking at data. Hierarchical manifolds with known
ground truth, then three checks.

**1, Does the generator produce the geometry it claims?**

```
      M    rank   trace_err   $\hat{D}$/D
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
D \approx 30 at M=640. Across the empirical range (D = 10–25) the bias moves only
0.95 → 0.92 — approximately a shared constant, so it **cancels in cross-manifold
comparisons**.

**3, Does measured nesting track planted nesting?** Recovers the analytic curve
$\sigma/\sqrt{1+\sigma^2}$ to within 0.8%.

**Run out of order, on purpose:** before trusting the synthetic null at all, is
the real within-manifold spectrum actually a power law? It is: r^2 = 0.95–0.999
at every layer. And D = 10–25, far below the rank cap, so M=640 is good.

## Results

### The taxonomy beats a shuffle

Permuting which tasks are siblings while holding group sizes identical:

```
null mean 1 ± 0.08        real 0.75–0.80
p = 0.001–0.011 at layers 2–28, both readout modes
```

![random partition null](figures/random_partition.png)

27 contiguous layers in two independent readouts.

Two sanity checks pass. Pooling all prompts and cutting them into fake tasks
gives 0.92–1.02, as it must. And **raw layer 0 gives p = 0.33**: no taxonomy
signal in the last token's embedding, strong signal by layer 2. The structure
appears the moment the model integrates context, then saturates.

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
Malicious Use lexical distance   0.567 vs 0.710 over all pairs
corr(lexical, geometric)         0.73–0.91, rising with depth
residual p                       0.049 at L2  ->  0.008 at L28
```

![lexical control](figures/lexical_control.png)

Roughly 80% of inter-category geometry is word overlap. A residual survives,
and after normalising by the mean pairwise distance per layer which is necessary, since
residual-stream norms grow ~15,000× across depth. The effect size is flat
while p falls. The null tightens with depth, the effect does not grow.

### A second taxonomy

MMLU, using its own published 4-way grouping (not one constructed here, the
whole value of a second taxonomy is that someone else drew the boundaries).
Subjects chosen for question length near SALAD's range, fixed before any
activation was computed.

```
layer   SALAD z   SALAD z|lex    MMLU z   MMLU z|lex
    4     -3.59        -3.13      0.36        +1.36
   14     -2.98        -1.58     -0.37        +0.97
   28     -3.26        -2.43     -1.04        -0.77
```

![taxonomy comparison](figures/taxonomy_comparison.png)

SALAD clears its null at every layer and survives the lexical control. MMLU does
neither, after removing lexical overlap its z goes positive.

### Behaviour: a null

The cached activations already contain the model's next-token distribution, so
reading behaviour cost zero forward passes. Refusal is near-ceiling (0.72–0.90)
and shows no clustering by parent (p = 0.60), with the point estimate in the
wrong direction.

This is deliberately not reported as "representation and behaviour are
separable." Refusal is at ceiling so there is little variance for any structure
to predict, and 12 pairs gives low power.

---

> **Interpretation** The coolest part of this project came from the disagreement between MMLU and
SALAD. The fact that SALAD survived lexical control more than MMLU implies to me, at least, that there is a kind 
of correlated-manifold structure to be found in concepts that are misaligned. What this is, however, is incredibly open
and I can not claim confidence on any interpretation yet. The main result I hope that is taken away from this is that categorical structure of representations *can* introduce geometric quirks that, maybe one day, in the future can become a kind of dictionary or lens for what kind of concept a network is thinking of.

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
- **No power analysis at the real regime.** The synthetic null was calibrated at
  a within-spread/separation ratio of 0.018; reality is 1.6–5.5. Recalibrating
  and rerunning the power sweep is outstanding, so there is currently **no bound
  on what this design could have detected.**

---

> **Next steps** Obviously this experiment would benefit from larger scaling. More manifolds, more data,
more compute. These claims rest on two datasets. Additionally, it would be helpful to run these tests across many 
different regimes/datasets so that we could see whether consistent trends in the manifold geometry of certain concepts emerges.
This could end up being a useful tool for understanding what the model is thinking just by analyzing the geometry of an associated batch of responses.

---

## Bugs found, and how

**A 100× calibration error.** The synthetic model assumed manifolds that are
essentially points, within-spread/separation of 0.018. Measured on real data it
is 1.6–5.5: **real manifolds are larger than the gaps between their centroids.**
Caught by measuring the calibration constant rather than assuming it, before the
power analysis depended on it.

**A retracted interpretation.** An earlier reading — "the non-lexical effect
emerges in late layers," was wrong. After normalising for norm growth the
effect size is flat and it is the null that tightens. Retraction is in logs.

## Repo

```
src/acts.py          activation extraction; readout modes
src/cache.py         disk cache, dataset-agnostic, model loads lazily
src/geometry.py      R_M, D_M, centre statistics, equalize_class_n
src/synthetic.py     hierarchical generative model
src/salad.py         SALAD design and loading
src/mmlu.py          MMLU design and loading

scripts/1,2        taxonomy exploration and audit
scripts/3           SALAD activation extraction
scripts/4,5,6     synthetic checks 1, 2, 3
scripts/7          empirical spectra: is it a power law?
scripts/9           empirical nesting + calibration
scripts/10          random-partition null
scripts/11           lexical control
scripts/12           refusal readout
scripts/13,14        MMLU extraction and comparison

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
3. **Power before interpretation**
5. **Fixed conventions.** Radius is
   $R_M = \sqrt{\sum\lambda^2 / \sum\lambda}\,/\,\lVert c\rVert$, unchanged
   throughout.
6. **Error bars from resampling, stating what they cover.** Here: a bootstrap
   over prompts within each manifold. It does not cover which categories the
   taxonomy defines, or the model.
