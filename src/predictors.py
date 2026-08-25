"""Candidate predictors of transfer.

THIS FILE IS THE PART YOU WRITE. Everything else in the repo is plumbing you
should read once and then stop thinking about. This is where your geometry
background is doing the actual work, and every choice here is one you will have
to defend in the write-up.

The contract: each predictor takes the TRAIN side only -- activations, labels,
and the fitted direction -- plus whatever the target side exposes *without*
looking at target labels. It returns one scalar per (train, test) pair. You then
ask which predictor correlates with the measured transfer AUROC from
scripts/03_transfer.py.

The discipline that makes this a prediction rather than a description: if a
predictor needs target labels, it is not a predictor. It is a post-hoc summary.

`direction_cosine` is implemented as your baseline. Beat it or report that you
could not -- either is a result.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Baseline -- given to you, deliberately dumb
# ---------------------------------------------------------------------------
def direction_cosine(d_train: np.ndarray, d_test: np.ndarray) -> float:
    """Cosine between the two datasets' diff-of-means directions at one layer.

    This uses the target's LABELS (to form d_test), so it is not a legal
    predictor under the contract above. It is here as a ceiling-ish reference:
    it is the most direct measure of "do these two datasets want the same
    direction", and if your legal predictors cannot beat chance while this one
    correlates strongly, that tells you the transfer signal is real but your
    train-side proxies are not capturing it.

    Report it as what it is. Do not quietly let it into the predictor table.
    """
    a = d_train / (np.linalg.norm(d_train) + 1e-12)
    b = d_test / (np.linalg.norm(d_test) + 1e-12)
    return float(a @ b)


# ---------------------------------------------------------------------------
# Yours. Signatures are suggestions; change them if the geometry wants otherwise.
# ---------------------------------------------------------------------------
def manifold_radius(acts: np.ndarray, labels: np.ndarray, label: int) -> float:
    """Effective radius of one class manifold at one layer.

    acts (n, d), labels (n,), label picks the class.

    Sketch: centre the class on its own centroid, take the eigenvalues of the
    covariance, and form a scale-free radius. The convention matters and there
    is more than one in the literature -- Chung/Lee/Sompolinsky normalise by the
    centroid norm, which makes R_M dimensionless and comparable across layers
    where the residual stream norm grows a lot. Pick one, write down which, and
    keep it fixed.

    Watch out: the massive-activation dimensions will dominate the covariance
    unless you handle them. That is not a nuisance -- whether the geometry is
    dominated by those dims is itself a finding worth a line in logs/weird.md.
    """
    raise NotImplementedError


def manifold_dimension(acts: np.ndarray, labels: np.ndarray, label: int) -> float:
    """Participation-ratio dimension of one class manifold at one layer.

    D = (sum_i lambda_i)^2 / sum_i lambda_i^2, on the covariance eigenvalues.

    Interpretation to hold onto: D is the number of directions the manifold
    meaningfully occupies. A probe fitted to a high-D manifold has more room to
    latch onto dataset-specific structure -- which is the mechanism you are
    proposing when you say the residual carries the artifact.
    """
    raise NotImplementedError


def centre_correlation(acts: np.ndarray, labels: np.ndarray) -> float:
    """Alignment between the two class centroids at one layer.

    In the capacity theory this is the term that controls how much of the
    separation is trivially available from the centroids alone versus requiring
    the manifold structure. Low centre correlation with high transfer would be
    interesting; the reverse would be the boring outcome.
    """
    raise NotImplementedError


def j_space_fraction(direction: np.ndarray, lens_basis: np.ndarray) -> float:
    """Fraction of a probe direction's squared norm lying in J-space.

    Only reachable once you have a pre-fitted Jacobian lens. This is the
    timeboxed day-6 upgrade, not a dependency -- if the tooling fights you, the
    project stands on the geometric predictors alone.

    Sanity check before believing anything: the Anthropic result is that concept
    probes hold only ~6-7% of their variance in J-space. If you compute 90% or
    0.01%, you have a basis convention wrong, not a finding.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Evaluation harness -- given, so you only have to write the predictors
# ---------------------------------------------------------------------------
def rank_correlation(predictor_values: np.ndarray, transfer_auroc: np.ndarray) -> dict:
    """Spearman rho between a predictor and measured transfer, off-diagonal only.

    Both arrays are (n_ds, n_ds). The diagonal is excluded: it is not transfer,
    and leaving it in would manufacture a correlation out of the fact that
    within-dataset AUROC is high and self-similarity is 1.
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
