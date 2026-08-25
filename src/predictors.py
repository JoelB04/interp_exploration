"""Candidate predictors of transfer.

The mechanics here are now written. The *choices* are still yours, and they are
what you will have to defend: which radius convention, how many samples per
class, whether the massive-activation dimensions are included, and which of
these you are willing to call a predictor at all.

The contract. A legal predictor sees the TRAIN side in full -- activations,
labels, fitted direction -- and may see the target side's ACTIVATIONS ONLY. The
moment it touches target labels it stops being a prediction and becomes a
post-hoc summary. Two functions below deliberately break this rule and are
labelled as references, not predictors. Keep them out of the predictor table.

Grouped by legality:

  legal, train-side only   split_half_stability, manifold_radius,
                           manifold_dimension, centre_separation
  legal, uses target acts  target_variance_along_direction
  reference only           direction_cosine, centre_correlation_matrix
  not yet reachable        j_space_fraction

Before comparing any spectral quantity across datasets, run everything through
`equalize_class_n`. See the warning on `manifold_dimension` for why.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------
def diff_of_means(acts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Unit-norm diff-of-means direction at one layer. acts (n, d) -> (d,)."""
    d = acts[labels == 1].mean(axis=0) - acts[labels == 0].mean(axis=0)
    return d / (np.linalg.norm(d) + 1e-12)


def equalize_class_n(acts: np.ndarray, labels: np.ndarray, n_per_class: int,
                     seed: int = 0):
    """Subsample to exactly n_per_class examples of each label.

    Not optional if you intend to compare spectra across datasets. The
    covariance of n points in d dimensions has rank at most n-1, so every
    spectral quantity you compute is partly a measurement of your sample size.
    Datasets here range from 354 to 4450 rows; without this, a "lower-dimensional
    manifold" finding may just be a smaller dataset.
    """
    rng = np.random.default_rng(seed)
    idx = []
    for lab in (0, 1):
        pool = np.where(labels == lab)[0]
        if len(pool) < n_per_class:
            raise ValueError(
                f"label {lab} has {len(pool)} examples, need {n_per_class}"
            )
        idx.append(rng.choice(pool, n_per_class, replace=False))
    idx = np.concatenate(idx)
    return acts[idx], labels[idx]


def category_spectrum(acts: np.ndarray, labels: np.ndarray, label: int):
    """Covariance eigenvalues and centroid norm for one class manifold.

    Returns (lam, ||c||) where lam is descending and length min(n_c, d).

    SVD of the centred data rather than eigh of the covariance: with n_c ~ 200
    and d = 1536 you would otherwise build a 1536x1536 matrix of rank <= 199,
    which is slower and numerically worse for no gain.
    """
    X = acts[labels == label]
    c = X.mean(axis=0)                          # (d,) centroid
    Xc = X - c                                  # (n_c, d) centred data

    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2 / (len(X) - 1)                 # NOTE the parens: n-1, not n, minus 1
    return lam, float(np.linalg.norm(c))


# ---------------------------------------------------------------------------
# Legal predictors -- train side only
# ---------------------------------------------------------------------------
def split_half_stability(acts: np.ndarray, labels: np.ndarray,
                         groups: np.ndarray = None, n_repeats: int = 20,
                         seed: int = 0) -> float:
    """Mean cosine between diff-of-means directions fitted on disjoint halves.

    The cheapest sanity predictor: if a direction does not agree with itself
    across a random split of its own training data, it will certainly not agree
    with a different dataset.

    Understand its limit. This measures ESTIMATION NOISE, not BIAS. A confound
    like "correct city-country pairing" is perfectly stable across halves --
    both halves find it enthusiastically -- so high stability is necessary for
    transfer and nowhere near sufficient. Expect it to be a floor, not a ranker.

    Pass `groups` to halve at the group level, so a true/false pair never lands
    with one half seeing the statement and the other seeing its negation.
    """
    rng = np.random.default_rng(seed)
    out = []

    for _ in range(n_repeats):
        if groups is None:
            perm = rng.permutation(len(labels))
            h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
        else:
            uniq = rng.permutation(np.unique(groups))
            g1 = set(uniq[: len(uniq) // 2])
            in1 = np.array([g in g1 for g in groups])
            h1, h2 = np.where(in1)[0], np.where(~in1)[0]

        if min(len(np.unique(labels[h1])), len(np.unique(labels[h2]))) < 2:
            continue
        out.append(float(diff_of_means(acts[h1], labels[h1])
                         @ diff_of_means(acts[h2], labels[h2])))

    return float(np.mean(out)) if out else np.nan


def manifold_radius(acts: np.ndarray, labels: np.ndarray, label: int) -> float:
    """Effective radius of one class manifold at one layer.

    Convention, fixed here and not to be changed mid-project
    (Chung/Lee/Sompolinsky):

        R_M = sqrt( sum_i lam_i^2 / sum_i lam_i ) / ||c||

    This is NOT the naive RMS radius sqrt(sum_i lam_i). It is weighted toward
    the large axes -- an effective radius that pairs with D_M. The naive version
    is equally defensible; the only sin is switching between them.

    The ||c|| normalisation is load-bearing. Residual stream norm grows a lot
    across layers, and without it your layer-wise plots mostly show that growth
    rather than anything about shape.

    Open question, not an established fact: whether the massive-activation
    dimensions dominate the SPECTRUM. They certainly dominate the centroid --
    that is the session-1 cosine finding, and it inflates ||c|| here. But a
    dimension with a huge constant offset contributes nothing to the covariance;
    it only matters if it also VARIES across examples. Compute the spectrum with
    and without those dims and find out. Either answer is a line in weird.md.
    """
    lam, c_norm = category_spectrum(acts, labels, label)
    R_M = np.sqrt((lam ** 2).sum() / lam.sum())
    return float(R_M / c_norm)


def manifold_dimension(acts: np.ndarray, labels: np.ndarray, label: int) -> float:
    """Participation-ratio dimension of one class manifold at one layer.

        D_M = (sum_i lam_i)^2 / sum_i lam_i^2

    Interpretation: roughly the number of directions the manifold meaningfully
    occupies. A probe fitted to a high-D manifold has more room to latch onto
    dataset-specific structure, which is the mechanism you are proposing when
    you say the residual carries the artifact.

    HARD CEILING: D_M cannot exceed n_c - 1, and is biased downward well below
    that. With 200 points per class it can never report more than 199 however
    the geometry actually looks. Always call equalize_class_n first, or you will
    be comparing sample sizes and calling it geometry.
    """
    lam, _ = category_spectrum(acts, labels, label)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def centre_separation(acts: np.ndarray, labels: np.ndarray) -> float:
    """Centroid separation relative to spread along the separation axis.

    This replaces the centre-correlation quantity, which degenerates for two
    manifolds -- see centre_cosine_degeneracy_check below for why.

    Be honest in the write-up: this is Cohen's d along the diff-of-means axis.
    The manifold vocabulary adds nothing in the two-class case; what actually
    governs linear separability is a signal-to-noise ratio. The genuinely
    manifold-flavoured content lives in R_M and D_M, which use the whole
    spectrum rather than a single axis. A reviewer will recognise a
    t-statistic, so name it first.
    """
    X1, X0 = acts[labels == 1], acts[labels == 0]
    dc = X1.mean(axis=0) - X0.mean(axis=0)
    u = dc / (np.linalg.norm(dc) + 1e-12)

    s1, s0 = X1 @ u, X0 @ u
    pooled = np.sqrt(0.5 * (s1.var(ddof=1) + s0.var(ddof=1)))
    return float(np.linalg.norm(dc) / (pooled + 1e-12))


# ---------------------------------------------------------------------------
# Legal predictor -- uses target ACTIVATIONS, never target labels
# ---------------------------------------------------------------------------
def target_variance_along_direction(target_acts: np.ndarray,
                                    direction: np.ndarray) -> float:
    """Variance of the target dataset along the train probe direction,
    as a multiple of the average per-dimension variance.

    The most mechanically obvious predictor on the list, and possibly the
    strongest. If the target data barely varies along your direction, the probe
    physically cannot separate anything there -- every example scores about the
    same and AUROC collapses toward 0.5 regardless of how good the direction is
    in principle.

    Scale-free by construction: 1.0 means the direction is as variable as a
    typical dimension, >> 1 means it is a high-variance direction in the target.

    Needs only the target's activations. That makes it legal, and it is the one
    predictor here you could actually deploy -- you always have unlabelled data
    from the distribution you are about to monitor.
    """
    u = direction / (np.linalg.norm(direction) + 1e-12)
    proj_var = float(np.asarray(target_acts @ u).var(ddof=1))
    mean_dim_var = float(target_acts.var(axis=0, ddof=1).mean())
    return proj_var / (mean_dim_var + 1e-12)


# ---------------------------------------------------------------------------
# References -- these touch target labels. NOT predictors.
# ---------------------------------------------------------------------------
def direction_cosine(d_train: np.ndarray, d_test: np.ndarray) -> float:
    """Cosine between the two datasets' diff-of-means directions at one layer.

    Uses target labels to form d_test, so it is illegal as a predictor. It is
    here as a ceiling-ish reference: the most direct measure of "do these two
    datasets want the same direction". If the legal predictors find nothing
    while this correlates strongly, the transfer signal is real and your
    train-side proxies simply are not capturing it -- which is a result.
    """
    a = d_train / (np.linalg.norm(d_train) + 1e-12)
    b = d_test / (np.linalg.norm(d_test) + 1e-12)
    return float(a @ b)


def centre_correlation_matrix(centroids: np.ndarray) -> np.ndarray:
    """Pairwise cosine between manifold centroids, after removing the grand mean.

    centroids (M, d), one row per manifold -> (M, M).

    This is where centre correlation is actually meaningful: across the full
    ensemble of M = 16 manifolds (8 datasets x 2 classes), not within a pair.
    Removing the mean over all M kills the shared massive-activation offset
    without creating the antipodal degeneracy that ruins the two-class case.

    Building the rows requires per-class centroids for every dataset, hence
    target labels, hence: reference only. Its value is diagnostic -- if
    cities-true and sp_en_trans-true point in aligned directions, transfer has
    somewhere to come from; if they are orthogonal, no probe can bridge them.
    """
    C = centroids - centroids.mean(axis=0, keepdims=True)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return C @ C.T


def centre_cosine_degeneracy_check(acts: np.ndarray, labels: np.ndarray) -> dict:
    """Demonstrate why two-class centre correlation carries no information.

    Run this once, look at the numbers, put a line in logs/weird.md, and never
    use centroid cosine as a two-class predictor again.

    Returns both variants:

      raw       cosine between the two class centroids. Near 1.0 for every
                dataset and every layer, because a handful of massive-activation
                dimensions dominate both centroids. Your session-1 finding.

      centred   cosine after subtracting the grand mean. Exactly -1, always,
                for ANY class balance -- this is algebra, not measurement.
                The grand mean is a convex combination g = w1*c1 + w0*c0, so
                c1 - g = w0*(c1 - c0) and c0 - g = -w1*(c1 - c0). The two
                residual centroids are antiparallel by construction; the weights
                change their lengths but not the sign. Verified numerically at
                both balanced and 40/120 splits.

    One constant near +1 and one constant at exactly -1. Neither varies, so
    neither can predict anything.
    """
    c1 = acts[labels == 1].mean(axis=0)
    c0 = acts[labels == 0].mean(axis=0)

    def cos(a, b):
        return float(a @ b / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))

    grand = acts.mean(axis=0)
    return dict(raw=cos(c1, c0), centred=cos(c1 - grand, c0 - grand))


# ---------------------------------------------------------------------------
# Not yet reachable
# ---------------------------------------------------------------------------
def j_space_fraction(direction: np.ndarray, lens_basis: np.ndarray) -> float:
    """Fraction of a probe direction's squared norm lying in J-space.

    Needs a pre-fitted Jacobian lens. Timeboxed day-6 upgrade, not a dependency
    -- the project stands on the geometric predictors alone.

    Sanity check before believing anything: the Anthropic result is that concept
    probes hold only ~6-7% of their variance in J-space. If you compute 90% or
    0.01%, you have a basis convention wrong; that is a bug, not a finding.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------
def rank_correlation(predictor_values: np.ndarray,
                     transfer_auroc: np.ndarray) -> dict:
    """Spearman rho between a predictor and measured transfer, off-diagonal only.

    Both arrays are (n_ds, n_ds). The diagonal is excluded: it is not transfer,
    and leaving it in would manufacture a correlation out of the fact that
    within-dataset AUROC is high and self-similarity is 1.

    READ THE n BEFORE THE p. The 56 off-diagonal cells are not 56 independent
    observations. The datasets cluster into roughly five families
    ({cities, neg_cities}, {sp_en_trans, neg_sp_en_trans},
    {larger_than, smaller_than}, {companies}, {common_claim}), and
    larger_than/smaller_than share group keys by construction. Effective n is
    closer to 20, and the cells are not independent draws in any case, so the
    reported p is optimistic by a wide margin.

    Consequence for the write-up: do not stage this as a horse race between
    four correlated predictors. You do not have the power to win it. Report the
    designed contrast -- the negation pairs, where competing accounts predict
    opposite signs -- as the primary result, and treat this sweep as
    exploratory.
    """
    from scipy.stats import spearmanr

    n = predictor_values.shape[0]
    off = ~np.eye(n, dtype=bool)
    x, y = predictor_values[off], transfer_auroc[off]
    ok = np.isfinite(x) & np.isfinite(y)

    if ok.sum() < 4:
        return dict(rho=np.nan, p=np.nan, n=int(ok.sum()))

    rho, p = spearmanr(x[ok], y[ok])
    return dict(rho=float(rho), p=float(p), n=int(ok.sum()))
