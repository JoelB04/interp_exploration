"""Synthetic hierarchical manifolds.

Design note, since this is the part that matters architecturally:

`generate()` returns EXACTLY the format the empirical loader returns --
(X, y_parent, y_child). Every measurement function then works unchanged on
synthetic and real data, so a null comparison cannot be corrupted by the two
paths handling things differently. If you change the return signature here,
change it in the data loader too.

The two dataclasses below hold values and nothing else. No logic lives in them.
They are separate because the two sweeps vary them independently:

    calibration  vary the SPEC (M), pin the geometry
    power        pin the SPEC at the real design, vary the GEOMETRY

Notation, fixed for the whole project:
    M   points per child manifold  (samples)
    n   ambient dimension          (neurons)  -- 1536 for Qwen2.5-1.5B
    P   number of parents
    C_p number of children of parent p
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HierarchySpec:
    """What tree, and how big. No geometry here."""

    branching: list       # children per parent, e.g. [4, 3, 2, 2, 2]
    M: int                # points per child manifold -- EQUAL by construction
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

        This is the actual n of your nesting test, and it is much smaller than
        the number of manifolds. Print it before trusting any nesting claim.
        """
        c = self.branching
        within = sum(k * (k - 1) // 2 for k in c)
        total = self.n_children * (self.n_children - 1) // 2
        return within, total - within

    @classmethod
    def from_labels(cls, y_parent, y_child, M, n=1536):
        """Read the branching structure off real data, so mirroring the
        empirical design cannot silently drift from it."""
        y_parent, y_child = np.asarray(y_parent), np.asarray(y_child)
        branching = []
        for p in np.unique(y_parent):
            branching.append(len(np.unique(y_child[y_parent == p])))
        return cls(branching=sorted(branching, reverse=True), M=M, n=n)


@dataclass
class GeometryParams:
    """What shape. No tree structure here.

    sigma_ratio is THE signal knob and the thing your power sweep varies:

        0.0  children sit exactly on their parent  -> perfect nesting
        1.0  children scattered as widely as parents are -> effectively flat

    Only the ratio is identifiable -- the overall scale cancels out of every
    dimensionless statistic -- so sigma_p is pinned at 1 and not exposed.
    """

    sigma_ratio: float = 0.3      # sigma_c / sigma_p
    within_scale: float = 1.0     # within-manifold spread, relative to sigma_p
    alpha: float = 1.5            # within-manifold spectrum: lambda_i ~ i^-alpha
    parent_subspace: int = 0      # 0 = parents isotropic in R^n;
                                  # k > 0 = parent centres confined to a
                                  # k-dimensional subspace
    shared_within: bool = True    # same within-covariance for every child


# ---------------------------------------------------------------------------
# YOURS. The dataclasses above are containers; all the work happens here.
# ---------------------------------------------------------------------------
def generate(spec: HierarchySpec, params: GeometryParams, rng) -> tuple:
    """Sample a hierarchical point cloud.

    Returns
        X         (spec.total_points, spec.n) float array
        y_parent  (spec.total_points,) int, which parent each point belongs to
        y_child   (spec.total_points,) int, which child manifold

    The generative story to implement:

        parent centres   c_p   ~  N(0, sigma_p^2 * I)        sigma_p := 1
        child centres    c_pk  ~  N(c_p, sigma_ratio^2 * I)
        points           x     ~  N(c_pk, Sigma_w)

    Sigma_w carries the within-manifold shape. A power-law spectrum
    lambda_i proportional to i^-alpha gives a tunable participation ratio;
    alpha near 0 is isotropic (D_M -> n), large alpha is low-dimensional.
    Scale it so its total variance is within_scale^2.

    Decisions the stubs leave open, which are yours to make and defend:

      - parent_subspace. Isotropic parent centres in 1536 dimensions are
        mutually near-orthogonal, which is a strong and possibly wrong claim
        about what a hierarchy looks like. Confining them to a k-dimensional
        subspace says the model has a low-rank "kind of harm" axis. These give
        very different nulls.

      - shared_within. Shared is the clean null. Per-child covariance is more
        realistic and adds a parameter you would then have to fit.

      - Whether to draw Sigma_w's eigenbasis fresh per child or share it. If
        every child shares an eigenbasis, the manifolds are parallel slabs;
        if not, they are randomly oriented. This is a real modelling claim.

    Keep it a pure function of (spec, params, rng) so a sweep is reproducible
    from the seed alone.
    """
    raise NotImplementedError


def spectrum(n: int, alpha: float, scale: float = 1.0) -> np.ndarray:
    """Power-law eigenvalue spectrum, normalised to total variance scale^2.

    Provided because it is fiddly rather than interesting. lambda_i ~ i^-alpha
    for i = 1..n, then rescaled so lambda.sum() == scale^2.

    Sanity check worth running once: the participation ratio of this spectrum,
    (sum lam)^2 / sum lam^2, should fall as alpha rises, and should equal n
    exactly at alpha = 0.
    """
    lam = np.arange(1, n + 1, dtype=float) ** (-alpha)
    return lam * (scale ** 2) / lam.sum()
