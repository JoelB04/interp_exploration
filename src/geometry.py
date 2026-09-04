import numpy as np



def diff_of_means(acts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Unit-norm diff-of-means direction at one layer. acts (n, d) -> (d,)."""
    d = acts[labels == 1].mean(axis=0) - acts[labels == 0].mean(axis=0)
    return d / (np.linalg.norm(d) + 1e-12)


def equalize_class_n(acts: np.ndarray, labels: np.ndarray, n_per_class: int,
                     seed: int = 0):
    """Subsample to exactly n_per_class examples of each label.
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
    and d = 1536 one would otherwise build a 1536x1536 matrix of rank <= 199,
    which is slower and numerically worse.
    """
    X = acts[labels == label]
    c = X.mean(axis=0)                          # (d,) centroid
    Xc = X - c                                  # (n_c, d) centred data

    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2 / (len(X) - 1)                 
    return lam, float(np.linalg.norm(c))



def manifold_radius(acts: np.ndarray, labels: np.ndarray, label: int) -> float:
    """Effective radius of one class manifold at one layer.

    Convention
    (Chung/Lee/Sompolinsky):

        R_M = sqrt( sum_i lam_i^2 / sum_i lam_i ) / ||c||

    This is not the naive RMS radius sqrt(sum_i lam_i). It is weighted toward
    the large axes with an effective radius that pairs with D_M. 

    The ||c|| normalisation is load-bearing. Residual stream norm grows a lot
    across layers, and without it layer-wise plots mostly show that growth
    rather than anything about shape.
    """
    lam, c_norm = category_spectrum(acts, labels, label)
    R_M = np.sqrt((lam ** 2).sum() / lam.sum())
    return float(R_M / c_norm)


def manifold_dimension(acts: np.ndarray, labels: np.ndarray, label: int) -> float:
    """Participation-ratio dimension of one class manifold at one layer.

        D_M = (sum_i lam_i)^2 / sum_i lam_i^2
    """
    lam, _ = category_spectrum(acts, labels, label)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def centre_separation(acts: np.ndarray, labels: np.ndarray) -> float:

    X1, X0 = acts[labels == 1], acts[labels == 0]
    dc = X1.mean(axis=0) - X0.mean(axis=0)
    u = dc / (np.linalg.norm(dc) + 1e-12)

    s1, s0 = X1 @ u, X0 @ u
    pooled = np.sqrt(0.5 * (s1.var(ddof=1) + s0.var(ddof=1)))
    return float(np.linalg.norm(dc) / (pooled + 1e-12))



def centre_correlation_matrix(centroids: np.ndarray) -> np.ndarray:
    """Pairwise cosine between manifold centroids, after removing the grand mean.

    centroids (M, d), one row per manifold -> (M, M).

    """
    C = centroids - centroids.mean(axis=0, keepdims=True)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return C @ C.T


def centre_cosine_degeneracy_check(acts: np.ndarray, labels: np.ndarray) -> dict:
    """Demonstrate why two-class centre correlation carries no information."""

    c1 = acts[labels == 1].mean(axis=0)
    c0 = acts[labels == 0].mean(axis=0)

    def cos(a, b):
        return float(a @ b / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))

    grand = acts.mean(axis=0)
    return dict(raw=cos(c1, c0), centred=cos(c1 - grand, c0 - grand))

