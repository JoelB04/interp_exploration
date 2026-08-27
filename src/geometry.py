"""Manifold geometry.

Chung/Lee/Sompolinsky-style statistics for concept manifolds in the residual
stream. Written during the session-3 transfer project, retained because the
geometry is the core of the hierarchy project -- the transfer-specific
predictors that used to live alongside these were removed on 2026-08-27 and are
in git history at 73f06af (src/predictors.py).

Two rules that apply to everything here:

  1. Run `equalize_class_n` before comparing ANY spectral quantity across
     manifolds. D_M is capped at M-1 and biased well below it, so unequal
     sample sizes are measured as differences in geometry. See the warning on
     manifold_dimension.

  2. Radius convention is fixed and must not change mid-project:
     R_M = sqrt(sum lam^2 / sum lam) / ||c||. Written out on manifold_radius.

Notation, project-wide: M = points per manifold, n = ambient dimension.
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
# References -- these touch target labels. NOT predictors.
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
# Evaluation harness
