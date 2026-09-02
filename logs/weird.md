# Weird

One line per surprise from the manifold-geometry project. Don't chase them during
exploration — the point is to have a list to mine later. This is the best source
of a project question that is genuinely mine rather than lifted from a
suggested-topics doc.

Surprises from the earlier truth-probe project are in
`logs/archive_truth_probes_weird.md`.

Format: `YYYY-MM-DD — observation. (where it came up)`

---

- 2026-08-28 — Massive-activation dimensions do NOT dominate the covariance
  spectrum, only the centroid. Top-5 variance share is 7–10% (raw), ~3% (chat),
  and dropping them RAISES D (23.0 → 31.7). A large constant offset moves the
  mean without contributing to covariance. (16_empirical_spectrum)
- 2026-08-28 — Real manifolds are LARGER than the gaps between their centroids.
  Within-spread / between-centroid distance is 1.6–5.5, against 0.018 in the
  synthetic model. Two orders of magnitude off, and it invalidated check 3's
  noise floor. (18_empirical_nesting)
- 2026-08-28 — Nesting measured to sd 0.001–0.006 despite only 12 within-parent
  pairs. Concentration of measure: each distance is a sum over 1536 coordinates,
  so individual distances are themselves low-variance. High ambient dimension
  works FOR this measurement, which is not the intuition. (17_verify_nesting)
- 2026-08-28 — At M=160 the two estimator biases cancel exactly at D=15, giving
  D_hat/D = 1.00 that is two errors offsetting rather than accuracy. A ratio near
  1 is not evidence of a good estimate. (15_verify_dimension)
- 2026-08-28 — `chat` layer 0 is EXACTLY degenerate: one distinct row across 800
  prompts, total variance 0.0. Free exact null — any structure reported there is
  a bug. (13_extract verification)
- 2026-08-28 — The refusal script's descriptive sanity check found a framing bug
  it was not looking for: `chat` mode still carried "Is the following true or
  false?" from the closed project, so the model was answering true/false rather
  than deciding whether to refuse. Top tokens were 'True' and 'False'. (21_refusal)
- 2026-08-29 — Refusal ordering cuts across the taxonomy. Adult Content (0.904)
  and Unfair Representation (0.771) share a parent and sit nearly the full range
  apart. (21_refusal)
- 2026-09-01 — MMLU has much LOWER lexical–geometric correlation than SALAD
  (0.20–0.47 vs 0.76–0.89), yet MMLU's nesting is the one that vanishes entirely
  under the lexical control while SALAD's survives. Those two facts together are
  odd and I have no story for them. (23_compare_taxonomies)
- 2026-09-01 — MMLU nests in `request` mode (z = −3.6 at layers 22–26) but barely
  at all in `raw`. SALAD nests in both. Readout position changes which taxonomy
  shows structure. (23_compare_taxonomies)
