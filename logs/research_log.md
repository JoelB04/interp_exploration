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

**Next.** *(pending)*
