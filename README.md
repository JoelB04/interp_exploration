# How much of concept geometry is just vocabulary?

Measuring whether a language model's representational geometry reflects an
externally-given category taxonomy — and how much of any such structure survives
controlling for the words the categories happen to use.

**Model:** Qwen2.5-1.5B-Instruct (28 layers, d_model 1536), CPU only.
**Data:** SALAD-Bench harm taxonomy, with MMLU as a second, unrelated taxonomy.

---

> **[YOUR VOICE — motivation]** *Your own draft, recovered from the working tree.
> It stops mid-sentence — finish it and delete this note.*
>
> This project is inspired by Haim Sompolinsky and his work on concept manifolds
> in neural networks, as well as my work within his lab. The thesis for the
> project was this: do misaligned concepts have different geometric structure in
> the representation space of a neural network? If so, what features are
> different and why? As it turns out, the hierarchical arrangement of concept
> manifolds may give insights into how misaligned concepts are represented.
>
> To research this I use Qwen2.5-1.5B on the SALAD-Bench dataset, which organizes
> **⟵ [unfinished]**

---

## The question

Safety taxonomies carve harmful requests into categories and sub-categories.
SALAD-Bench, for instance, groups 13 tasks under 5 domains: *Illegal Activities*,
*Fraud*, *Security Threats* and *Influence Operations* all sit under
*Malicious Use*.

Does the model's internal geometry reflect that carving? Do sibling categories
sit closer together in activation space than the taxonomy-blind baseline would
predict — and if so, is that because the model has abstracted something about
harm, or simply because siblings use similar words?

## Headline findings

| | |
|---|---|
| **The taxonomy beats a matched random partition** | p = 0.001–0.011 at every layer 2–28, in both readout modes |
| **One branch carries it** | Removing *Malicious Use* is the only exclusion that kills the effect (p 0.002 → 0.13) |
| **~80% of it is vocabulary** | corr(lexical distance, geometric distance) = 0.73–0.91 |
| **A residual survives the lexical control** | flat with depth; p falls 0.049 → 0.008 because the null tightens, not because the effect grows |
| **MMLU shows no comparable residual** | its z goes *positive* after the lexical control |
| **Refusal shows no structure** | but refusal is near-ceiling (0.72–0.90), so this dataset cannot test it |

Plus one methodological result that mattered more than any of the above: **the
synthetic null model was calibrated 100× wrong**, and measuring rather than
assuming caught it before any power analysis was built on it.

---

## Design

13 SALAD level-2 tasks under 5 level-1 domains, branching `[4,3,2,2,2]`.

**Every manifold uses exactly M = 640 prompts.** Not optional: the participation
ratio $D$ is capped at $M-1$ and biased below it, so comparing manifolds of
different sizes measures sample size and reports it as geometry. Parent manifolds
contain more points than children by construction, so this confound points in
exactly the direction of the hypothesis.

M = 640 is set by the smallest kept task. Not 657 — that would have dropped
*Persuasion and Manipulation* and left a parent with a single child, costing a
whole parent's worth of within-parent pairs.

**The number that governs everything:**

```
branching [4,3,2,2,2]  ->  12 within-parent pairs, 66 between-parent
                           6 of the 12 come from Malicious Use alone
```

Every nesting claim rests on those 12 pairs. They are reported individually
rather than only as a mean.

**Readout position is a live variable**, never hardcoded — `raw` (bare prompt),
`request` (plain user turn), and a deprecated `chat` (see *Bugs found*).

## The instrument, before the measurement

The synthetic arm exists to answer *could this design detect the thing I am
looking for?* before looking at any data. Hierarchical manifolds with known
ground truth, then three checks.

**1 — Does the generator produce the geometry it claims?**

```
      M    rank   trace_err   D̂/D
    160     159      -0.001   0.837
    640     639      +0.004   0.938
  10240    1536      -0.001   0.997
```

Rank is exactly `min(M-1, n)`; trace preserved to 0.4%. The mechanism is
visible in the spectrum: middle eigenvalues *inflate* (1.40× at M=640) because
trace conservation forces variance from 1536 directions into 639. That inflation
is precisely why $\hat D < D$.

![spectrum recovery](figures/check1_spectrum_recovery.png)

**2 — How large can D be before it cannot be measured?** Within 10% up to
D ≈ 30 at M=640. Across the empirical range (D = 10–25) the bias moves only
0.95 → 0.92 — approximately a shared constant, so it **cancels in cross-manifold
comparisons**, which is what every claim here is made of.

**3 — Does measured nesting track planted nesting?** Recovers the analytic curve
$\sigma/\sqrt{1+\sigma^2}$ — not $\sigma$ — to within 0.8%.

**Run out of order, on purpose:** before trusting the synthetic null at all, is
the real within-manifold spectrum actually a power law? It is: r² = 0.95–0.999
at every layer. And D = 10–25, far below the rank cap, so M=640 is comfortably
adequate.

## Results

### The taxonomy beats a shuffle

Permuting which tasks are siblings while holding group sizes identical:

```
null mean 1.000 ± 0.08        real 0.75–0.80
p = 0.001–0.011 at layers 2–28, both readout modes
```

![random partition null](figures/random_partition.png)

27 contiguous layers in two independent readouts — not a scattered p < 0.05.

Two sanity checks pass. Pooling all prompts and cutting them into fake tasks
gives 0.92–1.02, as it must. And **raw layer 0 gives p = 0.33**: no taxonomy
signal in the last token's embedding, strong signal by layer 2. The structure
appears the moment the model integrates context, then holds flat.

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
dropping *Malicious Use* also halves the within-parent pairs 12 → 6, so some of
that p = 0.13 is lost power rather than lost effect.

### Most of it is vocabulary

*Malicious Use*'s four children all concern crime and share heavy vocabulary — a
purely lexical grouping would beat the null too. So: TF-IDF per task, residualise
geometric distance on lexical distance, rerun the permutation.

```
Malicious Use lexical distance   0.567    vs 0.710 over all pairs
corr(lexical, geometric)         0.73–0.91, rising with depth
residual p                       0.049 at L2  ->  0.008 at L28
```

![lexical control](figures/lexical_control.png)

Roughly **80% of inter-category geometry is word overlap.** A residual survives,
and after normalising by the mean pairwise distance per layer — necessary, since
residual-stream norms grow ~15,000× across depth — the **effect size is flat**
while p falls. The null tightens with depth; the effect does not grow.

### A second taxonomy

MMLU, using its own published 4-way grouping (not one constructed here — the
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
neither — after removing lexical overlap its z goes *positive*.

### Behaviour: a null

The cached activations already contain the model's next-token distribution, so
reading behaviour cost zero forward passes. Refusal is near-ceiling (0.72–0.90)
and shows **no clustering by parent (p = 0.60)**, with the point estimate in the
wrong direction.

This is deliberately **not** reported as "representation and behaviour are
separable." Refusal is at ceiling so there is little variance for any structure
to predict, and 12 pairs gives low power. The honest statement is that this
dataset cannot test the question.

---

> **[YOUR VOICE — interpretation]** *What do you actually think is going on? The
> SALAD-survives / MMLU-doesn't dissociation is the most interesting thing here
> and there's no settled explanation. Worth saying plainly what you believe and
> how confident you are.*

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

> **[YOUR VOICE — next steps]** *What you would do with more time or a GPU.
> Candidates: a second model via Kempner; more taxonomies to turn n=2 into
> something with power; a dataset with actual variance in refusal so the
> behavioural question becomes testable.*

---

## Bugs found, and how

Kept visible because the checks that caught them are part of the method.

**A 100× calibration error.** The synthetic model assumed manifolds that are
essentially points — within-spread/separation of 0.018. Measured on real data it
is 1.6–5.5: **real manifolds are larger than the gaps between their centroids.**
Caught by measuring the calibration constant rather than assuming it, before the
power analysis depended on it.

**A framing bug, found by a sanity check written for another purpose.** The
refusal script's descriptive stage printed the model's top predicted tokens
purely to check the refusal token set was sensible. They came back `'True'` and
`'False'` — because `chat` mode still wrapped every prompt in *"Is the following
true or false?"*, a leftover from an earlier project. The model was answering a
true/false question, not deciding whether to refuse. Fixed by adding a **new**
mode rather than redefining `chat` in place, since the activation cache keys on
the mode string and silently changing its meaning would have let stale tensors be
reused under a new label.

**A retracted interpretation.** An earlier reading — "the non-lexical effect
emerges in late layers" — was wrong. After normalising for norm growth the
effect size is flat and it is the null that tightens. Retraction is in the log
rather than quietly corrected.

## Repo

```
src/acts.py          activation extraction; readout modes
src/cache.py         disk cache, dataset-agnostic, model loads lazily
src/geometry.py      R_M, D_M, centre statistics, equalize_class_n
src/synthetic.py     hierarchical generative model
src/salad.py         SALAD design and loading
src/mmlu.py          MMLU design and loading

scripts/11,12        taxonomy exploration and audit
scripts/13           SALAD activation extraction
scripts/14,15,17     synthetic checks 1, 2, 3
scripts/16           empirical spectra: is it a power law?
scripts/18           empirical nesting + calibration
scripts/19           random-partition null
scripts/20           lexical control
scripts/21           refusal readout
scripts/22,23        MMLU extraction and comparison

logs/research_log.md    timestamped, prediction before each run
logs/weird.md           one-line surprises, not chased
logs/archive_*          an earlier, unrelated project on truth-probe transfer
figures/                the plots reproduced in this README
```

`logs/research_log.md` is the honest record — predictions written before runs,
results, and the corrections. Probably more informative about how this was done
than the README is.

## Running it

CPU only. Extraction is the only slow part; everything downstream reads cached
tensors and runs in seconds.

```bash
python scripts/13_extract.py          # ~10 h, 3 readout modes, 13 tasks
python scripts/22_extract_mmlu.py     # ~3 h
python scripts/16_empirical_spectrum.py
python scripts/18_empirical_nesting.py
python scripts/19_random_partition.py
python scripts/20_lexical_control.py
python scripts/23_compare_taxonomies.py
```

Synthetic checks need no data:

```bash
python scripts/14_verify_generator.py
python scripts/15_verify_dimension.py
python scripts/17_verify_nesting.py
```

## Method notes

Standards held throughout, each because something was missed once:

1. **Equal M before any spectral comparison.** $D$ is capped at $M-1$.
2. **Report M and n alongside every geometric quantity**, plus $\gamma = n/M$.
3. **Power before interpretation** — know what the design could detect before
   looking. *(Partially unmet; see limitations.)*
4. **Suspicion on success.** A clean result triggers an artifact hunt.
5. **Fixed conventions.** Radius is
   $R_M = \sqrt{\sum\lambda^2 / \sum\lambda}\,/\,\lVert c\rVert$, unchanged
   throughout.
6. **Error bars from resampling, stating what they cover.** Here: a bootstrap
   over prompts within each manifold. It does *not* cover which categories the
   taxonomy defines, or the model.
