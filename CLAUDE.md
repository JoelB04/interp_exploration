# CLAUDE.md

Context for Claude Code working in this repo. Read before making changes.

## Who and why

Joel. Background: concept manifolds and Gardner-style classification capacity of
hierarchical manifolds (Sompolinsky lab), plus philosophy of mind/language. New to
mechanistic interpretability, competent at ML and statistical mechanics of learning.

Goal: applying to Neel Nanda's MATS 12.0 stream, deadline Fri Sept 4 2026,
11:59pm PT. The application requires a ~20-hour research project. This repo is the
exploration phase before committing to that project — building fluency and
generating a real question, not producing the deliverable yet.

Priority is developing as a researcher over getting a perfect result. Being wrong
in an interesting way is a good outcome here.

## Direction under consideration

*(as of 2026-08-25 — under active discussion, expect this section to move)*

Topic area: concept representations (Neel's suggested-problems list).

Leading candidate for the 20-hour project: does the J-space fraction of a probe's
direction predict how well that probe transfers out of distribution? Anthropic's
Jacobian-lens work found that concept probes hold only ~6–7% of their variance in
J-space yet most of their causal power lives there. Hypothesis: J-space is a
corpus-averaged canonical basis, and the non-J-space residual is where
dataset-specific artifact lives — so J-space fraction should predict transfer.

Fallback if J-lens tooling proves painful: same transfer-matrix experiment with
purely geometric predictors (manifold radius/dimension/centre correlation, cosine
similarity of diff-of-means directions as the baseline).

Neither is committed. The exploration may generate something better.

## Environment

- Windows, `C:\Python313\python.exe`, CPU only — no CUDA. Runtime is the binding
  constraint. Keep batch sizes small and sample counts modest; prefer fast
  iteration over statistical power during exploration.
- Model: `Qwen/Qwen2.5-1.5B-Instruct` (28 layers, d_model 1536, vocab 151936).
- HF transformers with `output_hidden_states` and forward hooks. Not
  TransformerLens. nnsight may come later for larger models.
- Dependencies, verified installed 2026-08-25: torch 2.13.0+cpu, transformers
  5.15.1, scikit-learn 1.8.0, pandas 3.0.3, numpy 2.4.4. No virtualenv — packages
  are installed against the system interpreter above. Note transformers is on 5.x:
  `torch_dtype=` still works but is backwards-compat only; `dtype=` is current.

Run scripts from the repo root; `DATA_DIR` and the `sys.path` insert both assume it:

```bash
C:\Python313\python.exe scripts\01_smoke_test.py
```

```bash
C:\Python313\python.exe scripts\02_first_probe.py
```

## Repo layout

```
src/acts.py               load(), format_prompts(), get_acts()
src/data.py               fetch/prepare + GROUP-AWARE split (pairs never straddle)
src/cache.py              disk cache for activations; model loads lazily
src/predictors.py         transfer predictors — STUBS, Joel writes these
scripts/01_smoke_test.py  plumbing verification — passing
scripts/02_first_probe.py first probe, AUROC by layer — superseded by 03
scripts/03_transfer.py    the transfer matrix — the project substrate
data/                     gitignored; geometry-of-truth CSVs, fetched on demand
cache/                    gitignored; cached activation tensors
results/                  gitignored; transfer grids
logs/research_log.md      timestamped, prediction recorded before each run
logs/weird.md             one-line surprises, not chased during exploration
```

Compute activations once, then iterate on analysis. `cache.py` keys on
(model, dataset, mode, max_n, seed) and only constructs the model on a miss.

## Established facts

Verified on this setup — do not re-derive.

- `hidden_states` is a tuple of length `n_layers + 1`; index 0 is the embedding,
  index `i` is the output of block `i-1`.
- The final norm is already applied to `hidden_states[-1]` for this model —
  `lm_head(hs[-1])` reproduces the logits.
- Left padding works; batched vs unbatched last-token activations differ by
  ~3e-4, which is fp32 accumulation noise.
- Under the chat template, the last token is the generation-prompt token and is
  identical across all examples. All example-specific information must be
  attention-transported there. This is why `src/acts.py` exposes a `mode`
  parameter (`raw` vs `chat`) — readout position is a live experimental variable,
  never a hardcoded default.
- Naive cosine similarity between activations is useless here: a few massive-
  activation dimensions push every pair to ~0.99. Subtract means.
- The geometry-of-truth repo is `saprmarks/geometry-of-truth` — **hyphens**. The
  underscore spelling 404s. This bug sat in `02_first_probe.py`, so that script
  never got past its fetch step.
- Dataset pair structure, measured: `cities`/`neg_cities` and
  `larger_than`/`smaller_than` are fully paired (every entity carries both
  labels). `sp_en_trans` is not — 344 distinct Spanish words across 354 rows, so
  group-aware splitting is nearly a no-op there. `companies_true_false` and
  `common_claim_true_false` have no pair structure at all.
- AUROC is invariant to centering. Subtracting any constant vector shifts every
  score by the same amount and cannot change a ranking — so train-mean vs
  test-mean centering is moot for probe AUROC. It is *not* moot for the
  geometric predictors, where distances are not rank-only.

## Methodological standards for this repo

Non-negotiable, because the eventual write-up is judged on them:

1. **Baselines before conclusions.** Any claim that reading internals helped needs
   the black-box comparison — ask the model directly, or token entropy. For J-lens
   claims specifically, logit lens is the mandatory baseline; tuned lens is a
   cheap third.
2. **Permutation null on every probe result.** Shuffle labels, confirm AUROC ≈ 0.5
   at all layers. At n < d, separability is guaranteed by Cover's theorem and
   train accuracy carries no information. *(Not yet implemented in
   `02_first_probe.py` — must be in before any session-2 conclusion is logged.)*
3. **Report n alongside every AUROC.** At ~120 test examples the standard error is
   roughly 0.04. Differences under ~0.05 are noise.
4. **AUROC is rank-only.** Invariant to intercept and base rate. Fine for "is the
   information present"; insufficient for any monitoring/usefulness claim, which
   needs TPR at fixed low FPR against a control distribution.
5. **Suspicion on success.** High AUROC should trigger an artifact hunt, not
   satisfaction. Check whether the direction is dominated by massive-activation
   dimensions; check that layer 0 under `chat` sits at chance.
6. **Diff-of-means is the default estimator.** It cancels confounds that are
   mean-independent of the label, and only those. If logistic regression beats it
   substantially, suspect exploitation of n < d separability and sweep `C` before
   believing it.

## How to work with Joel

- He records a written prediction before running anything. Don't spoil outputs;
  when handing him a script, tell him what to predict, not what will happen.
- Flag which parts of any code he should understand deeply and implement himself
  versus copy and move on. He's building durable skill, not just results.
- Dead ends get logged, not deleted. Neel explicitly wants them in the write-up.
- Say when something is untested or uncertain rather than asserting fluently.

## Planned session sequence

1. ~~Smoke test / plumbing~~ — done
2. First probe end to end, diff-of-means, AUROC by layer, `raw` vs `chat` — current
3. Break it. Train on `cities`, test on `sp_en_trans` and `larger_than`. Watch
   transfer collapse. Try `neg_cities`, where probes sometimes anti-generalize
   (AUROC < 0.5). This is the most educational hour in the sequence.
4. Causal validation. Steer along the direction during generation; ablate it.
   Establishes the decodable-vs-used distinction firsthand. Expect steering to be
   finicky — likely needs coefficients large enough to degrade text quality before
   behaviour shifts cleanly.
5. Sample two other concepts from Neel's list. Uncertainty is cheapest (trivia
   labelled by correctness, free strong baseline in token entropy). Then eval
   awareness. Deception after that. Misalignment direction last — needs finetuned
   model organisms.
6. J-lens: load a pre-fitted lens from Neuronpedia, decompose one already-trusted
   probe direction into J-space and residual. Only after a probe he understands
   exists, or tooling bugs will be indistinguishable from findings.

Keep a running `logs/weird.md` of anything surprising, one line each. Don't chase
them during exploration. That list is the best source of a project question that's
genuinely his rather than lifted from a suggested-topics doc.

## Reading list

**Read properly:** the Anthropic global-workspace/J-lens paper and Neel's
LessWrong review of it; Arditi et al. on the refusal direction (for argument
structure, not result); Goldowsky-Dill et al. on deception probes; Parrack et al.
on black-to-white performance boosts.

**Skim:** Marks & Tegmark Geometry of Truth; Mamou et al. on separable manifolds
in LM representations (the "someone already did manifold analysis" objection he
needs an answer to); Belrose et al. tuned lens; Liars' Bench as a dataset source.
