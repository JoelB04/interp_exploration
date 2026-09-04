# Research log -- hierarchical geometry of misalignment concepts

**Question.** Does Qwen2.5-1.5B organise misalignment-related concepts according
to an externally-given taxonomy, where in depth does that structure appear, and
how much of it is vocabulary rather than concept?

There are synthetic hierarchical manifolds with known
ground truth, to calibrate the estimators and bound what this design could
detect. And real prompts through the model, geometry per layer.


## 2026-08-27/28: verifying the synthetic generator

**Question.** Before measuring, does the generator produce the
geometry it claims, and can M=640 measure it?

**Prediction**
Synthetic spectra test:
  I expect the recovered spectra of lam_hat to pull away from the baseline truth at M=640 since this is where our eigenvalues stop since our data only goes up to 640 dimensions, whereas the true spectra will have values in all 1536 dimensions.

  The ratio lam_hat/lam_true should increase up to M=640 since all the variance from the spectra is preserved, it just condensed into fewer dimensions. This early-mid M we will see lam_hat dominate, but as we approach M=640 and go past it this ratio drops to zero, since all eigenvalues of lam_hat are zero after that.

  I predict the trace error will shrink with M and better approximate the true spectra as M increases.

  The participation ratio for D'/D is going to remain small since for finite M D' is always less than D by construction since in the equation (\sum\lam)^2 remains the same whereas \sum\lam^2 will be large for fewer M.

**Result, check 1, spectrum recovery.** Generator correct and the finite-M
distortion is the predicted shape.

      M   gamma=n/M   rank   trace_err   D_hat   D_hat/D
    160        9.60    159      -0.001    31.9     0.837
    640        2.40    639      +0.004    35.8     0.938
   2560        0.60   1536      +0.001    37.5     0.984
  10240        0.15   1536      -0.001    38.0     0.997

Rank is exactly min(M-1, n) at every M and trace preserved to within 0.4%. The
ratio panel shows the mechanism. Leading eigenvalues recover at about 1, the middle
of the spectrum inflates (1.97x at M=160, 1.40x at M=640), everything past rank
M-1 is exactly zero. Trace conservation forces it, and that inflation is
why D_hat < D.

The prediction was right about mechanism and wrong on one detail, the ratio does
not rise monotonically to M then drop. It rises, peaks mid-spectrum, then falls
to zero at the rank cliff. The trace-conservation reasoning was correct.

Vindicated standard 7 as well, a single draw had reported 0.87 at M=640 and
showed M=200 apparently beating M=640. With 5 replicates the ordering is
monotonic. The anomaly was noise.

**Result, check 2, dimension recovery.** Within 10% up to D<=30 at M=640,
D<=60 at M=2560. Across the empirical range D=10-25 the bias moves only
0.95 -> 0.92 -- approximately a shared multiplicative constant, so it cancels in
cross-manifold comparisons, which is the comparison the hierarchy claim here
is made of.

Two competing biases at low D. First, sampling noise inflates small eigenvalues and
pushes D_hat up, rank truncation pushes it down. Also, t M=160 they cancel exactly at
D=15, giving a ratio of 1 that is two errors offsetting.

**Result check 3, nesting.** Measured nesting recovers the analytic curve
sigma/sqrt(1+sigma^2) to within 0.3-0.8% throughout. The
residual is a consistent small overestimate shrinking as sigma grows, centroid
estimation error, which inflates small within-parent distances proportionally
more than large between parent ones. At sigma=0 the measured ratio is 0.001
rather than 0, the pure noise floor.

Surprise: sd is 0.001-0.006 despite only 12 within-parent pairs. Concentration
of measure since each distance is a sum over 1536 coordinates, so individual
distances are themselves low-variance. High ambient dimension works for this
measurement/regime.

**Read.** Instrument calibrated. Power-law family confirmed on real data
(r2 0.95-0.999), D is 10-25 so M=640 is adequate, and the bias
cancels in comparisons.

--------------------------------------------------------------------------------

## 2026-08-28, session 5: empirical measurement

**5a, first nesting measurement** (18_empirical_nesting.py). Nesting ratio
0.75-0.93 (raw), 0.76-0.81 (chat), sem 0.001-0.006 over 50 prompt bootstraps.

Within-manifold spread /between-parent centroid distance is 2-5.5 (raw) and 1.6-2.6 (chat), against
0.018 in the synthetic. Real manifolds are larger than the gaps between their
centroids, roughly 100x off the regime check 3 ran in. Check 3's noise floor
does not apply and within_scale must be recalibrated before any power analysis
means anything. 

Per-pair detail exposed the mean as misleading: Malicious Use children cluster
at 0.54-0.75 (6 of the 12 pairs), Misinformation Harms is ANTI-nested at 1.13,
Information & Safety shows none at 1.02.

**5b, random-partition null** (19_random_partition.py). Standard 1, first tier.
Permutes which tasks are siblings, holding group sizes at [4,3,2,2,2].

  null mean 1 +/- 0.08;  real 0.75-0.80;  p = 0.001-0.011 at layers 2-28,
 both modes - 27 contiguous layers in two independent readouts.

Sanity checks pass: the task-shuffle control gives 0.92-1.02 as it must, and raw
layer 0 gives p = 0.33, ie no taxonomy signal in the last token's embedding
with strong signal by layer 2. There is a depth transition, it happens
immediately then saturates, which is what I mistook for absence of structure.

LEAVE-ONE-PARENT-OUT at layer 14 qualifies it sharply. Dropping Malicious Use is
the only exclusion that kills the effect (0.906, p = 0.13); every other parent
can be removed and it survives. Caveat: that exclusion also halves within-parent
pairs from 12 to 6, so some of p=0.13 is lost power rather than lost effect, and
this design cannot separate the two.

**5c, lexical control** (20_lexical_control.py). Malicious Use is lexically
tight, within-parent lexical distance 0.567 against 0.710 over all pairs, top
terms fake/steal/create/access/computer. Lexical nesting alone is significant at
p = 0.015.

corr(lexical, geometric) is 0.73-0.91 and rises with depth. Roughly 80% of the
variance in inter-manifold distance is word overlap.

Residual structure survives: p = 0.049 at L2 falling monotonically to 0.008 at
L28 (raw), 0.003 (chat). After normalising distances by the mean pairwise
distance per layer, necessary, since residual-stream norms grow ~15,000x,
the effect size is flat with depth (-0.06 to -0.10) while p falls. So the effect is roughly
constant and the null tightens with depth. An earlier reading of this as
late-emerging was wrong.

**5d, refusal** (21_refusal.py)

On corrected data: refusal is near-ceiling, 0.72-0.9, mean 0.83. No clustering
by parent, stat +0.070, null -0.003 +/- 0.201, p = 0.6, point estimate in the
wrong direction. No better after lexical control.

Not evidence of separability. Refusal is at ceiling so there is little variance
for any structure to predict, and 12 within-parent pairs gives low power. Descriptively the refusal ordering cuts across the taxonomy: Adult
Content (0.904) and Unfair Representation (0.771) share a parent and sit nearly
the full range apart.

**Read.** Standing claim: the SALAD parent structure is reflected in Qwen's
geometry above a matched random partition, carried mainly by Malicious Use;
~80% of it is vocabulary; a small residual survives the lexical control at
roughly constant magnitude across depth. Refusal shows no corresponding
structure, on a dataset that cannot test it properly.

--------------------------------------------------------------------------------

--------------------------------------------------------------------------------

## 2026-09-01, session 6: MMLU as a second taxonomy

**Question.** Does representational geometry reflect an externally-given category
hierarchy in general, or is the SALAD result specific to harm?

**Prediction (written before the analysis).** I think the MMLU dataset will
experience less semantic clustering and a random baseline compared to misaligned
concepts.*

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
lexical overlap its z is positive, the wrong direction.

Request mode complicates this. There MMLU does nest (z = -3.6 at layers 22-26,
comparable to SALAD) but neither dataset survives the lexical control cleanly
(SALAD -0.6 to -2.0, MMLU -1.3 to +0.4). So "survives lexical" is specific to
raw readout, not a general property.

corr(lexical, geometric): SALAD 0.76-0.89, MMLU 0.20-0.47 (raw). SALAD's
inter-category geometry is overwhelmingly vocabulary; MMLU's much less so.

**Baseline / null.** Tier 1 (random partition) and tier 3 (matched second
taxonomy) both paid. The synthetic null band under the real regime is still
outstanding, see below.

**Read.** The prediction was borne out: MMLU clusters less than the harm
taxonomy, and after the lexical control it is indistinguishable from its own
random baseline. Worth flagging that this is a confirmed pre-registered
prediction rather than a pattern found by looking, since it is the only one in
this project that was recorded before the run and then held.

Two caveats keep it from being "harm is special": MMLU's 4-way grouping is
coarser and partly administrative, STEM/humanities/social sciences is a
university-department carving and "other" is an explicit residual holding
human_aging, miscellaneous and nutrition, so this may measure taxonomy
coherence rather than domain. And the register difference (imperative requests
vs exam stems) is not excluded by this design. Neither is fixable with more
compute.

--------------------------------------------------------------------------------

## 2026-09-04 -- session 7: re-running the SALAD results in `request`

**Question.** The README was written against figures plotting `raw` and `chat`,
but `chat` was retired mid-project for carrying the stale "Is the following true
or false?" framing. Do the results hold in `request`, the mode that replaced it?

**Prediction (written before re-running).** Held, since the framing text is a
constant offset shared by every prompt and the nesting statistic is built from
centroid *differences*. Expected the p-values to move a little and the direction
to be unchanged. Would have been surprised by a sign flip.

**Setup.** No new forward passes, `request` activations for all 13 tasks were
already cached from 2026-08-28. Changed MODES in scripts 07, 09, 10, 11 from
["raw","chat"] to ["raw","request"] and reran. M=640, 2000 permutations
(partition) / 4000 (lexical).

**Result.** Direction unchanged, magnitudes slightly different.

    random partition   raw      p 0.0005-0.0050   real 0.747-0.800
                       request  p 0.0005-0.0145   real 0.725-0.821
    (previously quoted as 0.001-0.011 for raw+chat)

    task-shuffle sanity   raw 0.947-0.986, request 1.027-1.037  (want ~1)
    request layer 0       exactly degenerate, as chat layer 0 was

    lexical corr    raw 0.763-0.914, request 0.727-0.871
    residual p      raw 0.037 -> 0.004 (L2 -> L28), survives at all 6 layers
                    request 0.112 at L2, 0.063 at L20, else < 0.021

**One claim weakened.** The residual does not survive at every layer under
`request`. Claim 4 said it survives, full stop; that was true of raw+chat and is
not true of raw+request. Amended below.

**Also.** The lexical control figure plotted only the p-value, which visually
argues the interpretation retracted in session 5 (effect emerging late). It now
plots the normalised effect size beside it: flat, while p falls.

**Read.** The headline result is robust to readout position. The
lexical-independent part of it is not, and that is now stated as a limitation
rather than left implicit in a figure.

**Spectrum, re-measured in request.** Layer-mean fit r2 0.958-0.999 (raw),
0.997-0.999 (request); alpha 1.02-1.85. Layer-mean D 8-22 against the rank cap
639, per-manifold 5.6-37.6 -- so a few manifolds sit above the D ~ 30 that check
2 verified, which the README now says rather than quoting "10-25" as before.

The massive-activation ablation still holds and is sharper in request: dropping
the 5 highest-variance dims at layer 14 raises D by 8-11 in raw (top-5 variance
share 7-10%) but by only 0.1-0.4 in request (share 2-3%). Those dimensions move
the centroid.

--------------------------------------------------------------------------------

## 2026-09-04 -- housekeeping: geometry.py was dead code

Not a result, recorded because the repo convention is that removed code gets a
git pointer here rather than vanishing.

`src/geometry.py` held eight functions and NOTHING imported any of them. Mean-
while scripts 05, 06 and 07 each carried their own copy of the participation
ratio. So the file that the README cites as fixing the project's conventions was
code that never ran, and the conventions were actually being re-derived three
times.

Half of it was truth-probe residue: `diff_of_means`, `equalize_class_n`,
`centre_separation` and `centre_cosine_degeneracy_check` all assume labels in
{0,1} and cannot express 13 manifolds. That is why the manifold scripts quietly
reimplemented what they needed instead of importing.

Removed those four; kept and generalised the rest; added `participation_ratio`
and `effective_radius` as primitives on the eigenvalues, which is the level the
scripts actually work at. Scripts 05, 06, 07 now import them.

    git show e1d086b:src/geometry.py     # the four removed functions

Verified inert: check 1 and check 2 reproduce their tables exactly, and
check1/check2/empirical_spectrum are byte-identical to the pre-refactor PNGs.

**Read.** Worth noticing how this happened. The conventions were written down in
CLAUDE.md and in the README, and were followed -- the arithmetic is right
everywhere. What drifted was the claim about WHERE they lived. Documentation
saying a module is authoritative does not make it imported.


## STANDING CLAIMS

*(amended 2026-09-04 after the re-run in `request`; see session 7)*

1. SALAD's parent structure is reflected in Qwen2.5-1.5B's geometry above a
   matched random partition: p = 0.0005-0.015 at every layer 2-28, both readout
   modes (raw and request), 27 contiguous layers.
2. It is carried by Malicious Use. Dropping that parent is the only exclusion
   that kills it (0.906, p = 0.13); every other parent can be removed and it
   survives. That exclusion also halves within-parent pairs 12 -> 6, so power
   and effect are not separable here.
3. Roughly 80% of inter-category geometry is word overlap
   (corr(lexical, geometric) 0.73-0.91, rising with depth).
4. A residual survives the lexical control at roughly constant magnitude across
   depth (-0.06 to -0.12 normalised), p falls with depth because the null
   tightens, not because the effect grows. **Readout-dependent**: survives at all
   six sampled layers in raw, but in request it fails at L2 (p = 0.11) and is
   marginal at L20 (p = 0.063).
5. MMLU shows no comparable lexical-independent structure in raw mode, though
   taxonomy coherence and register are unexcluded explanations.
6. Refusal is near-ceiling (0.72-0.90) and shows no clustering by parent
   (p = 0.60). This dataset cannot test whether geometry predicts behaviour.

