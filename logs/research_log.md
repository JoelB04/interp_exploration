# Research log -- hierarchical geometry of misalignment concepts

**Question.** Does Qwen2.5-1.5B organise misalignment-related concepts according
to an externally-given taxonomy, where in depth does that structure appear, and
how much of it is vocabulary rather than concept?

Timestamped, append-only. **The prediction goes in before the run.** A prediction
recorded after seeing the output is worth nothing, and the habit is the point.
Dead ends stay in; they get deleted from the code, never from here.

Two arms, deliberately separated. Synthetic hierarchical manifolds with known
ground truth, to calibrate the estimators and bound what this design could
detect. And real prompts through the model, geometry per layer.

An earlier and unrelated project on truth-probe transfer is in
`logs/archive_truth_probes.md`.

Entry template:

```
## YYYY-MM-DD -- session N: title

**Question.** One sentence.
**Prediction (written before running).** What I expect, how strongly, and which
outcome would surprise me.
**Setup.** Dataset, M, n, model, layers, readout mode. Enough to reproduce.
**Result.** What happened. Report M and n alongside every geometric quantity.
**Baseline / null.** Which of the three tiers was paid, or why not yet.
**Read.** What I now believe and what I don't.
**Next.** The single next thing.
```

--------------------------------------------------------------------------------

## 2026-08-27/28 -- session 4: verifying the synthetic generator

**Question.** Before measuring anything real: does the generator produce the
geometry it claims, and can M=640 measure it?

**Prediction (written before running, verbatim):**

Synthetic spectra test:
  I expect the recovered spectra of lam_hat to pull away from the baseline truth at M=640 since this is where our eigenvalues stop since our data only goes up to 640 dimensions, whereas the true spectra will have values in all 1536 dimensions.

  The ratio lam_hat/lam_true should increase up to M=640 since all the variance from the spectra is preserved, it just condensed into fewer dimensions. This early-mid M we will see lam_hat dominate, but as we approach M=640 and go past it this ratio drops to zero, since all eigenvalues of lam_hat are zero after that.

  I predict the trace error will shrink with M and better approximate the true spectra as M increases.

  The participation ratio for D'/D is going to remain small since for finite M D' is always less than D by construction since in the equation (\sum\lam)^2 remains the same whereas \sum\lam^2 will be large for fewer M.

**Result -- check 1, spectrum recovery.** Generator correct; the finite-M
distortion is the predicted shape.

      M   gamma=n/M   rank   trace_err   D_hat   D_hat/D
    160        9.60    159      -0.001    31.9     0.837
    640        2.40    639      +0.004    35.8     0.938
   2560        0.60   1536      +0.001    37.5     0.984
  10240        0.15   1536      -0.001    38.0     0.997

Rank is exactly min(M-1, n) at every M; trace preserved to within 0.4%. The
ratio panel shows the mechanism: leading eigenvalues recover at ~1.0, the MIDDLE
of the spectrum inflates (1.97x at M=160, 1.40x at M=640), everything past rank
M-1 is exactly zero. Trace conservation forces it, and that inflation is
precisely why D_hat < D.

The prediction was right about mechanism and wrong on one detail: the ratio does
not rise monotonically to M then drop. It rises, peaks mid-spectrum, then falls
to zero at the rank cliff. The trace-conservation reasoning was correct.

Vindicated standard 7 as well: a single draw had reported 0.87 at M=640 and
showed M=200 apparently BEATING M=640. With 5 replicates the ordering is
monotonic. The anomaly was noise.

**Result -- check 2, dimension recovery.** Within 10% up to D<=30 at M=640,
D<=60 at M=2560. Across the empirical range D=10-25 the bias moves only
0.95 -> 0.92 -- approximately a shared multiplicative constant, so it CANCELS in
cross-manifold comparisons, which is the comparison every hierarchy claim here
is made of. Saturation (prediction Q2) untested: the grid was narrowed to D<=60
for relevance, so the rank caps are never approached.

Two competing biases at low D: sampling noise inflates small eigenvalues and
pushes D_hat up, rank truncation pushes it down. At M=160 they cancel exactly at
D=15, giving a ratio of 1.00 that is two errors offsetting rather than accuracy.

**Result -- check 3, nesting.** Measured nesting recovers the analytic curve
sigma/sqrt(1+sigma^2) -- NOT sigma -- to within 0.3-0.8% throughout. The
residual is a consistent small overestimate shrinking as sigma grows: centroid
estimation error, which inflates small within-parent distances proportionally
more than large between-parent ones. At sigma=0 the measured ratio is 0.0010
rather than 0, the pure noise floor.

Surprise: sd is 0.001-0.006 despite only 12 within-parent pairs. Concentration
of measure -- each distance is a sum over 1536 coordinates, so individual
distances are themselves low-variance. High ambient dimension works FOR this
measurement.

**Read.** Instrument calibrated. Power-law family confirmed on real data
(r2 0.95-0.999), D is 10-25 so M=640 is comfortably adequate, and the bias
cancels in comparisons.

--------------------------------------------------------------------------------

## 2026-08-28 -- session 5: the empirical measurement

*(Predictions not recorded for this session. Entries assembled from the commit
record; every number is reproducible from the scripts named.)*

**5a, first nesting measurement** (18_empirical_nesting.py). Nesting ratio
0.75-0.93 (raw), 0.76-0.81 (chat), sem 0.001-0.006 over 50 prompt bootstraps.

The calibration result matters more than the headline. Within-manifold spread /
between-parent centroid distance is 2.0-5.5 (raw) and 1.6-2.6 (chat), against
0.018 in the synthetic. Real manifolds are LARGER than the gaps between their
centroids -- roughly 100x off the regime check 3 ran in. Check 3's noise floor
does not apply and within_scale must be recalibrated before any power analysis
means anything. STILL OUTSTANDING.

Per-pair detail exposed the mean as misleading: Malicious Use children cluster
at 0.54-0.75 (6 of the 12 pairs), Misinformation Harms is ANTI-nested at 1.13,
Information & Safety shows none at 1.02.

**5b, random-partition null** (19_random_partition.py). Standard 1, first tier.
Permutes which tasks are siblings, holding group sizes at [4,3,2,2,2].

  null mean 1.000 +/- 0.08;  real 0.75-0.80;  p = 0.001-0.011 at layers 2-28,
  BOTH modes -- 27 contiguous layers in two independent readouts.

Sanity checks pass: the task-shuffle control gives 0.92-1.02 as it must, and raw
layer 0 gives p = 0.33, i.e. no taxonomy signal in the last token's embedding
with strong signal by layer 2. There IS a depth transition; it happens
immediately then holds flat, which is what I mistook for absence of structure.

LEAVE-ONE-PARENT-OUT at layer 14 qualifies it sharply. Dropping Malicious Use is
the only exclusion that kills the effect (0.906, p = 0.13); every other parent
can be removed and it survives. Caveat: that exclusion also halves within-parent
pairs from 12 to 6, so some of p=0.13 is lost power rather than lost effect, and
this design cannot separate the two.

**5c, lexical control** (20_lexical_control.py). Malicious Use IS lexically
tight -- within-parent lexical distance 0.567 against 0.710 over all pairs, top
terms fake/steal/create/access/computer. Lexical nesting alone is significant at
p = 0.015.

corr(lexical, geometric) is 0.73-0.91 and rises with depth. Roughly 80% of the
variance in inter-manifold distance is word overlap.

Residual structure survives: p = 0.049 at L2 falling monotonically to 0.008 at
L28 (raw), 0.003 (chat). After normalising distances by the mean pairwise
distance per layer -- necessary, since residual-stream norms grow ~15,000x --
the EFFECT SIZE is flat with depth (-0.06 to -0.10) while p falls. So the
correct reading is not "the effect emerges late" but "the effect is roughly
constant and the null tightens with depth". An earlier reading of this as
late-emerging was wrong and is retracted.

**5d, refusal** (21_refusal.py). The descriptive stage caught a bug: chat mode
wraps every prompt in "Is the following true or false?", a leftover from the
truth-probe project, so the model was answering a true/false question rather
than deciding whether to refuse. Added a request mode and re-extracted.

On corrected data: refusal is near-ceiling, 0.72-0.90, mean 0.83. No clustering
by parent -- stat +0.070, null -0.003 +/- 0.201, p = 0.60, point estimate in the
wrong direction. No better after lexical control.

NOT evidence of separability. Refusal is at ceiling so there is little variance
for any structure to predict, and 12 within-parent pairs gives low power. The
honest statement is that this dataset cannot test whether geometry predicts
behaviour. Descriptively the refusal ordering cuts across the taxonomy: Adult
Content (0.904) and Unfair Representation (0.771) share a parent and sit nearly
the full range apart.

**Read.** Standing claim: the SALAD parent structure is reflected in Qwen's
geometry above a matched random partition, carried mainly by Malicious Use;
~80% of it is vocabulary; a small residual survives the lexical control at
roughly constant magnitude across depth. Refusal shows no corresponding
structure, on a dataset that cannot test it properly.

--------------------------------------------------------------------------------

--------------------------------------------------------------------------------

## 2026-09-01 -- session 6: MMLU as a second taxonomy

**Question.** Does representational geometry reflect an externally-given category
hierarchy in general, or is the SALAD result specific to harm?

**Prediction (written before the analysis).** I think the MMLU dataset will
experience less semantic clustering and a random baseline compared to misaligned
concepts. *(Joel's, written before the run; recovered 2026-09-03 from a
working-tree copy that predated the log split.)*

**Setup.** `src/mmlu.py`, `22_extract_mmlu.py`, `23_compare_taxonomies.py`.
13 MMLU subjects under MMLU's own published 4-way grouping, branching [4,3,3,3],
M=200 -- forced, since only 3 of 57 subjects have >=640 rows. Subjects chosen for
question length near SALAD's 77-102 chars; achieved 76-129, mean 92. Question
stems only, no answer choices. Modes raw and request.

SALAD subsampled to M=200 for the matched comparison; SALAD@640 remains the
primary result. Compared by z against each dataset's OWN matched permutation
null, since branching differs (12 within-parent pairs against 15).

**Result. A dissociation in raw mode.**

    layer   SALAD z   SALAD z|lex    MMLU z   MMLU z|lex
        4     -3.59        -3.13      0.36        +1.36
       14     -2.98        -1.58     -0.37        +0.97
       26     -3.14        -2.11     -2.01        +0.01
       28     -3.26        -2.43     -1.04        -0.77

SALAD clears its null at every layer 2-28 and survives the lexical control at
most of them. MMLU does neither: it barely nests in raw mode, and after removing
lexical overlap its z is POSITIVE -- the wrong direction.

Request mode complicates this. There MMLU does nest (z = -3.6 at layers 22-26,
comparable to SALAD) but neither dataset survives the lexical control cleanly
(SALAD -0.6 to -2.0, MMLU -1.3 to +0.4). So "survives lexical" is specific to
raw readout, not a general property.

corr(lexical, geometric): SALAD 0.76-0.89, MMLU 0.20-0.47 (raw). SALAD's
inter-category geometry is overwhelmingly vocabulary; MMLU's much less so.

**Baseline / null.** Tier 1 (random partition) and tier 3 (matched second
taxonomy) both paid. The synthetic null band under the real regime is still
outstanding -- see below.

**Read.** The prediction was borne out: MMLU clusters less than the harm
taxonomy, and after the lexical control it is indistinguishable from its own
random baseline. Worth flagging that this is a confirmed pre-registered
prediction rather than a pattern found by looking, since it is the only one in
this project that was recorded before the run and then held.

Two caveats keep it from being "harm is special": MMLU's 4-way grouping is
coarser and partly administrative -- STEM/humanities/social sciences is a
university-department carving and "other" is an explicit residual holding
human_aging, miscellaneous and nutrition -- so this may measure taxonomy
coherence rather than domain. And the register difference (imperative requests
vs exam stems) is not excluded by this design. Neither is fixable with more
compute.

**Next.** Write-up.

--------------------------------------------------------------------------------

## STANDING CLAIMS

1. SALAD's parent structure is reflected in Qwen2.5-1.5B's geometry above a
   matched random partition: p = 0.001-0.011 at every layer 2-28, both readout
   modes, 27 contiguous layers.
2. It is carried by Malicious Use. Dropping that parent is the only exclusion
   that kills it (0.906, p = 0.13); every other parent can be removed and it
   survives. That exclusion also halves within-parent pairs 12 -> 6, so power
   and effect are not separable here.
3. Roughly 80% of inter-category geometry is word overlap
   (corr(lexical, geometric) 0.73-0.91, rising with depth).
4. A residual survives the lexical control at roughly constant magnitude across
   depth (-0.06 to -0.10 normalised); p falls with depth because the null
   tightens, not because the effect grows.
5. MMLU shows no comparable lexical-independent structure in raw mode, though
   taxonomy coherence and register are unexcluded explanations.
6. Refusal is near-ceiling (0.72-0.90) and shows no clustering by parent
   (p = 0.60). This dataset cannot test whether geometry predicts behaviour.

## OUTSTANDING

- within_scale recalibration and the synthetic null band / power sweep at the
  real regime (spread/separation ~2-5, not the 0.018 check 3 used). Without it
  there is no bound on what this design could have detected.
- README and write-up.
