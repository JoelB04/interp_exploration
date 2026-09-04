# CLAUDE.md

Context for Claude Code working in this repo. Read before making changes.

## Who and why

Joel. Background: concept manifolds and Gardner-style classification capacity of
hierarchical manifolds (Sompolinsky lab), plus philosophy of mind/language. New to
mechanistic interpretability, competent at ML and statistical mechanics of learning.

Originally written for a MATS 12.0 application, which closed early. The work
now stands as a portfolio project, so the bar is that a reader can follow the
reasoning and check it, not that it hits a deadline.

Priority is developing as a researcher over getting a perfect result. Being wrong
in an interesting way is a good outcome here.

## Current direction

*(as of 2026-08-27)*

**Does the model organise misalignment-related concepts hierarchically, and where
in depth does that structure appear?**

Two arms, deliberately separated:

- **Synthetic.** Generative hierarchical manifolds with known ground truth
  (`src/synthetic.py`). Three jobs: calibrate the estimators at the M and n we
  actually have; produce a null band under the exact empirical design; and
  measure how strong a hierarchy must be before this design can detect it. The
  third is the important one — it converts a null empirical result from "we
  found nothing" into a bound.
- **Empirical.** SALAD-Bench prompts through Qwen2.5-1.5B, manifold geometry per
  layer.

Framing rule: the question is about the model, not about the method. "Does the
model organise harm concepts hierarchically" has a result; "what is the manifold
geometry of harm categories" has a table. Avoid hammer-and-nail — the geometry
has to answer something that could not be got otherwise.

### Previous direction (closed 2026-08-27)

Session 3 asked whether probe transfer across truth datasets was predictable
from geometry. Seven rounds, five retractions, no surviving claim that reading
internals beats a free baseline out of distribution. The full record is in
`logs/research_log.md`; the code is in git at `73f06af`. It was closed because
the correlational framing needed many independent datasets and only ~5 families
existed — a structural cap, not a fixable one. The pivot fixes this by making
the unit of analysis manifolds rather than datasets.

## Environment

- Windows, `C:\Python313\python.exe`, CPU only — no CUDA. Runtime is the binding
  constraint. Keep batch sizes small and sample counts modest; prefer fast
  iteration over statistical power during exploration.
- Model: `Qwen/Qwen2.5-1.5B-Instruct` (28 layers, n = d_model = 1536, vocab 151936).
- HF transformers with `output_hidden_states` and forward hooks. Not
  TransformerLens. nnsight may come later for larger models.
- Dependencies, verified 2026-08-27: torch 2.13.0+cpu, transformers 5.15.1,
  scikit-learn 1.8.0, pandas 3.0.3, numpy 2.4.4, scipy 1.17.1, datasets 4.8.5,
  matplotlib 3.10.9. No virtualenv — packages are installed against the system
  interpreter above. transformers is on 5.x: `torch_dtype=` still works but is
  backwards-compat only; `dtype=` is current.

Run scripts from the repo root; `sys.path` inserts and output paths assume it:

```bash
C:\Python313\python.exe scripts\02_explore_salad.py
```

## Notation

Fixed project-wide, because the earlier project used `n` for both and it caused
confusion:

- **M** — points (prompts) per manifold. Equal across manifolds by construction.
- **n** — ambient dimension / neurons. 1536 here.
- **P** — number of parents; **C_p** — children of parent p.

## Repo layout

```
src/acts.py               load(), format_prompts(), get_acts()
src/cache.py              disk cache for activations; model loads lazily
src/geometry.py           participation_ratio, effective_radius, spectra.
                          Conventions are fixed HERE and imported, never
                          re-derived inside a script.
src/synthetic.py          hierarchical generative model (Joel's)
src/salad.py              SALAD design and loading; design is data, not code
src/mmlu.py               MMLU design and loading
scripts/01..03            plumbing, taxonomy exploration, audit plot
scripts/04, 13            activation extraction (SALAD, MMLU) — the slow parts
scripts/05, 06, 08        synthetic checks 1, 2, 3
scripts/07, 09            empirical spectrum, empirical nesting + calibration
scripts/10, 11, 12        random-partition null, lexical control, refusal
scripts/14                taxonomy comparison
data/                     gitignored; fetched on demand
cache/                    gitignored; cached activation tensors
results/                  gitignored; figures and grids
logs/research_log.md      timestamped, prediction recorded BEFORE each run
logs/weird.md             one-line surprises, not chased during exploration
```

Compute activations once, then iterate on analysis. `cache.py` keys on
(model, dataset, mode, max_n, seed) and only constructs the model on a miss.

## Established facts

Verified on this setup — do not re-derive.

**Model plumbing**

- `hidden_states` is a tuple of length `n_layers + 1`; index 0 is the embedding,
  index `i` is the output of block `i-1`.
- The final norm is already applied to `hidden_states[-1]` for this model —
  `lm_head(hs[-1])` reproduces the logits. This means the cached last-layer
  activations already contain the model's output distribution, and any
  black-box baseline costs zero forward passes.
- Forward hook on `model.model.layers[i]` matches `hidden_states[i+1]`.
- Left padding works; batched vs unbatched last-token activations differ by
  ~3e-4, which is fp32 accumulation noise.
- Under the chat template, the last token is the generation-prompt token and is
  identical across all examples. All example-specific information must be
  attention-transported there. This is why `src/acts.py` exposes a `mode`
  parameter — readout position is a live experimental variable, never a
  hardcoded default. Three modes exist: `raw`, `request`, and `chat`, which is
  RETIRED. `chat` still carries "Is the following true or false?" from the
  closed truth-probe project; it is kept only so old cache entries stay
  reproducible. Do not report results in it.
- Naive cosine similarity between activations is useless here: a few massive-
  activation dimensions push every pair to ~0.99. Subtract means.
- **RESOLVED 2026-08-28**: those dimensions do NOT dominate the covariance
  spectrum. At layer 14 the top-5 highest-variance dims carry 7–10% of total
  variance under `raw` and only ~3% under `chat`. Dropping them *raises* D
  (23.0 → 31.7 on Toxic Content raw), so they mildly concentrate the spectrum
  but are nowhere near dominating it. A large constant offset moves the
  centroid without contributing to covariance, and that is what was happening.
- Qwen's vocab contains CJK tokens and the Windows console is cp1252. Any script
  that prints decoded tokens needs
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`.

**SALAD-Bench** (`OpenSafetyLab/Salad-Data`, config `base_set`, 21,318 rows)

- Columns: `question`, `1-category`, `2-category`, `3-category`.
  6 domains / 16 tasks / 66 categories.
- Clean tree: every level-2 and level-3 node has exactly one parent. Zero exact
  duplicate questions after normalisation.
- Manifold budget: at M=200, 55 of 66 leaf categories survive; at M=150, 63.
  All 16 level-2 tasks have >= 200.
- Length confound: mean question length varies 77–102 chars across domains
  (ratio 1.33). Must be controlled. Template leakage is mild — no domain exceeds
  33% on a single opening word.
- Current design under consideration: drop domain O3 (Socioeconomic Harms) and
  task O11 (Defamation), giving 13 tasks across 5 domains at **M = 640** — set
  by O15: Persuasion and Manipulation. Not 657; that would drop O15 and leave a
  parent with one child.
- That design yields only **12 within-parent pairs** against 66 between-parent,
  and 6 of the 12 come from Malicious Use alone. Nesting claims rest on those 12.

**Real manifold geometry** (measured 2026-08-28, M=640, all 13 tasks, all layers)

- The within-manifold spectrum IS a power law. Fit r² is 0.95–0.999 at every
  layer in both modes, so the toy model's `lambda_i ~ i^-alpha` family is the
  right one and the synthetic null is well-founded.
- Fitted alpha sits at 1.15–1.7 (raw) and 1.2–1.65 (chat).
- **D is small: 10–25, not hundreds.** Far below the M-1 = 639 rank cap, so the
  measurement is nowhere near censored and M=640 is comfortably adequate —
  check 1 showed only a 6% undershoot at D=38, and the bias is smaller still
  at D≈20.
- `raw` manifolds are consistently higher-dimensional than `chat` (peak D ≈ 22
  vs ≈ 13–18). Consistent with chat forcing everything through one
  attention-transported token.
- `raw` D rises from 3 at the embedding to ~22 by layer 14, holds, then falls
  to 13 at layer 28. `chat` is flatter, ~10, rising to 18 at the last layer.
- `chat` layer 0 is EXACTLY degenerate — one distinct point, all eigenvalues
  zero, D undefined. Free exact null: any structure reported there is a bug.
- `raw` layer 0 fits badly (r² 0.76, alpha 48) because the last token of a bare
  prompt takes only ~93 distinct values across 800 examples. Exclude it.

## Methodological standards for this repo

Non-negotiable, because the eventual write-up is judged on them. Session 3
produced five retractions; each of these exists because something was missed.

1. **Baselines before conclusions.** Three tiers, all required before any
   hierarchy claim: a random partition of the same prompts into fake categories
   of identical size; a matched ordinary (non-misalignment) hierarchy; and the
   synthetic null under the exact empirical design.
2. **Equal M before any spectral comparison.** `D_M` is capped at M-1 and biased
   well below it. Comparing manifolds of different sizes measures sample size
   and reports it as geometry. Subsample to a common M before comparing.
3. **Report M and n alongside every geometric quantity**, plus the ratio
   `gamma = n/M`. Estimator bias is governed by that ratio.
4. **Power before interpretation.** Know what effect size the design could
   detect before looking at the data. A null result without a power analysis is
   uninterpretable — this is the single biggest lesson from session 3.
5. **Suspicion on success.** A clean result should trigger an artifact hunt.
   Check length, opening-word template, and layer-0 behaviour. Layer 0 is
   embeddings only and should show no abstract structure.
6. **Fix conventions and never change them mid-project.** Radius convention is
   `R_M = sqrt(sum lam^2 / sum lam) / ||c||`. Write down which convention, keep
   it fixed, and state it in the write-up.
7. **Error bars from resampling, and say what they cover.** Session 3 had a
   headline result of 0.951 that was 0.587 +/- 0.44 once splits were resampled.
   Always state what is and is not being resampled.

## How to work with Joel

- He records a written prediction before running anything. Don't spoil outputs;
  when handing him a script, tell him what to predict, not what will happen.
- **He writes the load-bearing code himself.** Claude does scaffolding, sweeps,
  plots, caching, and formatting. Joel writes generative models, geometry
  functions, and anything he will have to defend under questioning. Flag which
  is which when proposing work.
- Dead ends get logged, not deleted. Code may be removed from the working tree
  when a direction closes, but only with a git pointer added to the research log.
- Say when something is untested or uncertain rather than asserting fluently.
- He asked for blunt evaluation of whether work is interesting. Give it.

## Planned session sequence

1. ~~Smoke test / plumbing~~ — done
2. ~~Truth probes and the transfer matrix~~ — closed, see logs
3. ~~SALAD-Bench audit and hierarchy design~~ — done
4. ~~Synthetic checks 1, 2 and 3: estimator calibration and nesting recovery~~ — done
5. ~~Empirical: activations for the 13 tasks, geometry per layer~~ — done
6. ~~Controls: random partition, lexical control, MMLU as a second taxonomy~~ — done
7. ~~Refusal readout~~ — done, a null, and reported as one
8. **Outstanding.** Recalibrate `within_scale` and run the power sweep at the
   real regime (within-spread/separation 1.5–5.5, not the 0.018 check 3 used).
   Until that exists there is no bound on what this design could have detected,
   which is the one gap the README admits to.
9. If ever revisited: does the hierarchy survive the `attack_enhanced_set`
   jailbreak rewrites of the same base questions?
