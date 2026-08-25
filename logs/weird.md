# Weird

One line per surprise. Don't chase them during exploration — the point is to have
a list to mine later. This is the best source of a project question that is
genuinely mine rather than lifted from a suggested-topics doc.

Format: `YYYY-MM-DD — observation. (where it came up)`

---

- 2026-08-23 — Naive cosine similarity between activations is ~0.99 for every
  pair, true/false alike. A handful of massive-activation dimensions dominate the
  norm and swamp everything else. Subtracting the mean fixes it. (session 1)
- 2026-08-25 — Transfer is wildly asymmetric. `neg_cities → cities` = 0.002 but
  `cities → neg_cities` = 0.332 (raw). A pure sign flip would be symmetric. (s3)
- 2026-08-25 — `larger_than → smaller_than` = 0.013 but `smaller_than →
  larger_than` = 0.649 (raw). Two logically mirror-image datasets, opposite
  behaviour. In `chat` the asymmetry REVERSES: 0.941 vs 0.323. (s3)
- 2026-08-25 — In raw, `smaller_than` transfers well to nearly everything (row
  mean 0.84) while `larger_than` transfers terribly (0.37). Same task, mirrored. (s3)
- 2026-08-25 — Within-dataset AUROC ANTI-predicts transfer: Spearman -0.62 (raw),
  -0.45 (chat). The best in-distribution probes are the worst travellers. (s3)
- 2026-08-25 — Unembedding the raw probe directions gives semantic garbage
  ('imates', 'otty', '=sub'). The chat directions give clean correctness
  vocabulary (' incorrect', ' invalid', ' Neither', 'yes', ' ✓'). Same probe
  method, same model, different readout token. (s3b)
- 2026-08-25 — Val-based layer selection is unstable: `cities` picks anywhere in
  layers 16-28 across 20 split repeats (mode 16, only 7/20). `companies` picks
  layer 27 in 4/20 and ranges 16-28. Layer choice drives most of the transfer
  variance. (s3c)
- 2026-08-25 — Some transfer cells are effectively BIMODAL across splits:
  chat `cities → neg_cities` = 0.587 +/- 0.44, i.e. sometimes ~0.95 and
  sometimes ~0.05. A single split reads as a clean finding either way. (s3c)
