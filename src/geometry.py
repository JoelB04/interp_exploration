"""Manifold geometry: the two shape statistics this project reports.

Conventions are fixed here and nowhere else, so that a number quoted in the
README can be traced to one definition. Both statistics are functions of the
covariance eigenvalues alone, so the primitives take `lam` directly and the
`acts`/`label` wrappers are conveniences on top.

Notation, project-wide:
    M   points per manifold (samples), equal across manifolds by construction
    n   ambient dimension (neurons), 1536 for Qwen2.5-1.5B
"""

import numpy as np


def participation_ratio(lam: np.ndarray) -> float:
    """D_M = (sum_i lam_i)^2 / sum_i lam_i^2.

    NaN for a degenerate manifold rather than 0/0. That case is real: under the
    chat template every prompt shares a final token, so layer 0 has one distinct
    point and zero variance.
    """
    lam = np.asarray(lam, dtype=float)
    lam = lam[lam > 0]
    if lam.size == 0:
        return float("nan")
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def effective_radius(lam: np.ndarray, c_norm: float) -> float:
    """R_M = sqrt(sum_i lam_i^2 / sum_i lam_i) / ||c||   (Chung/Lee/Sompolinsky).

    Not the naive RMS radius sqrt(sum_i lam_i). This one is weighted toward the
    large axes and pairs with the participation ratio above.

    The ||c|| normalisation is load-bearing. Residual-stream norm grows by four
    orders of magnitude across depth, and without it a layer-wise plot mostly
    shows that growth rather than anything about shape.
    """
    lam = np.asarray(lam, dtype=float)
    if lam.sum() <= 0 or c_norm <= 0:
        return float("nan")
    return float(np.sqrt((lam ** 2).sum() / lam.sum()) / c_norm)


def category_spectrum(acts: np.ndarray, labels: np.ndarray, label):
    """Covariance eigenvalues and centroid norm for one manifold.

    Returns (lam, ||c||), lam descending, length min(M - 1, n).

    SVD of the centred data rather than eigh of the covariance: with M ~ 640 and
    n = 1536 the covariance is a 1536x1536 matrix of rank <= 639, so forming it
    is both slower and numerically worse.
    """
    X = acts[labels == label]
    c = X.mean(axis=0)
    s = np.linalg.svd(X - c, compute_uv=False)
    return s ** 2 / (len(X) - 1), float(np.linalg.norm(c))


def manifold_dimension(acts: np.ndarray, labels: np.ndarray, label) -> float:
    """Participation ratio of one manifold, from raw activations."""
    lam, _ = category_spectrum(acts, labels, label)
    return participation_ratio(lam)


def manifold_radius(acts: np.ndarray, labels: np.ndarray, label) -> float:
    """Effective radius of one manifold, from raw activations."""
    lam, c_norm = category_spectrum(acts, labels, label)
    return effective_radius(lam, c_norm)


def centre_correlation_matrix(centroids: np.ndarray) -> np.ndarray:
    """Pairwise cosine between manifold centroids, after removing the grand mean.

    centroids (K, n), one row per manifold -> (K, K). Removing the grand mean is
    not optional: a handful of massive-activation dimensions push every raw
    pairwise cosine to ~0.99 regardless of content.
    """
    C = centroids - centroids.mean(axis=0, keepdims=True)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return C @ C.T
