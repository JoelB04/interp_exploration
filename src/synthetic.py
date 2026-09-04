"""Synthetic hierarchical manifolds.

Notation:
    M   points per child manifold  (samples)
    n   ambient dimension          (neurons), 1536 for Qwen2.5-1.5B
    P   number of parents
    C_p number of children of parent p
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class HierarchySpec:

    branching: list       # children per parent
    M: int                # points per child manifold, equal by construction
    n: int = 1536         # ambient dimension

    @property
    def n_parents(self) -> int:
        return len(self.branching)

    @property
    def n_children(self) -> int:
        return sum(self.branching)

    @property
    def total_points(self) -> int:
        return self.n_children * self.M

    def pair_counts(self) -> tuple:
        """(within-parent pairs, between-parent pairs) among child manifolds. 
        """
        c = self.branching
        within = sum(k * (k - 1) // 2 for k in c)
        total = self.n_children * (self.n_children - 1) // 2
        return within, total - within

    @classmethod
    def from_labels(cls, y_parent, y_child, M, n=1536):
        y_parent, y_child = np.asarray(y_parent), np.asarray(y_child)
        branching = []
        for p in np.unique(y_parent):
            branching.append(len(np.unique(y_child[y_parent == p])))
        return cls(branching=sorted(branching, reverse=True), M=M, n=n)


@dataclass
class GeometryParams:
    """sigma_ratio is the correlation parameter, and the thing the power sweep
    varies:

        0  children sit exactly on their parent, so nesting is perfect
        1  children scatter as widely as the parents do, so the hierarchy is
           effectively flat

    Only the ratio is identifiable. The overall scale cancels out of every
    dimensionless statistic, so sigma_p is held at 1.
    """

    sigma_ratio: float = 0.3      # sigma_c / sigma_p
    within_scale: float = 1.0     # within-manifold spread, relative to sigma_p
    alpha: float = 1.5            # within-manifold spectrum: lambda_i ~ i^-alpha

    parent_subspace: int = 0      # 0 = parents isotropic in R^n
                                  # k > 0 = parent centres confined to a
                                  # k-dimensional subspace
                                  
    shared_within: bool = True    # same within-covariance for every child


def spectrum(n: int, alpha: float, scale: float = 1.0) -> np.ndarray:
    """Power-law eigenvalue spectrum, normalised to total variance scale^2.
    lambda_i ~ i^-alpha for i = 1..n, then rescaled so lambda.sum() == scale^2.
    """
    lam = np.arange(1, n + 1, dtype=float) ** (-alpha)
    return lam * (scale ** 2) / lam.sum()


def generate(spec: HierarchySpec, params: GeometryParams, rng) -> tuple:
    """Sample a hierarchical point cloud.
    Returns
        X         (spec.total_points, spec.n) array
        y_parent  (spec.total_points,) int, which parent each point belongs to
        y_child   (spec.total_points,) int, which child manifold
    Sigma_w carries the within-manifold shape. A power-law spectrum
    lambda_i proportional to i^-alpha gives an adjustable participation ratio;
    alpha near 0 is isotropic (D_M goes to n), large alpha is low dimensional.
    Scale it so its total variance is within_scale^2.

    By default parents are drawn isotropically in the whole space. That is
    unlikely to hold empirically, since concepts of the same type probably share
    a common subspace; parent_subspace controls this. shared_within does the
    same job for child shape, deciding whether every child gets the same
    within-manifold covariance.

    A sweep is reproducible from the seed alone.
    """

    X, y_parent, y_child = [], [], []

    if params.parent_subspace == 0:
        parent_centroids = rng.normal(size = (spec.n_parents, spec.n))
    else:
        k_sub = params.parent_subspace
        B,_ = np.linalg.qr(rng.normal(size = (spec.n, k_sub))) #(n,k) shared subspace
        parent_centroids = rng.normal(size = (spec.n_parents, k_sub)) @ B.T

    child_id = 0
    U_shared, _ = np.linalg.qr(rng.normal(size=(spec.n, spec.n))) # Universal eigenbasis, assumes parents are drawn isotropically and share no common subspace
    lam = spectrum(spec.n, params.alpha, params.within_scale)

    for p in range(spec.n_parents):
        for k in range(spec.branching[p]):

            if params.shared_within:
                U = U_shared
            else:
                U, _= np.linalg.qr(rng.normal(size = (spec.n, spec.n)))

            centroid = rng.normal(loc = parent_centroids[p], scale = params.sigma_ratio) #(n,)
            points = rng.normal(size = (spec.M, spec.n))*np.sqrt(lam) @ U.T # (M,n)

            X.append(points + centroid)
            y_parent.append(np.full(spec.M, p))
            y_child.append(np.full(spec.M, child_id))
            child_id+=1

    X = np.concatenate(X)
    y_parent = np.concatenate(y_parent)
    y_child = np.concatenate(y_child)

    return X, y_parent, y_child


