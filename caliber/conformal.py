"""One-sided conformal p-values and thresholds (higher score = more anomalous)."""

import math

import numpy as np


def pvalues(cal_scores, test_scores):
    """p(x) = (1 + #{cal >= x}) / (n + 1). Valid (super-uniform) for test points
    exchangeable with the calibration points."""
    cal = np.sort(np.asarray(cal_scores, dtype=np.float64))
    x = np.asarray(test_scores, dtype=np.float64)
    n = cal.size
    n_ge = n - np.searchsorted(cal, x, side="left")
    return (1.0 + n_ge) / (n + 1.0)


def insample_pvalues(cal_scores):
    """Rank-based p-values of the calibration points against themselves
    (for normalising, not for guarantees)."""
    cal = np.asarray(cal_scores, dtype=np.float64)
    return pvalues(cal, cal)


def threshold(cal_scores, alpha):
    """Score threshold t with p(x) <= alpha  <=>  x >= t... approximately;
    returns the ceil((n+1)(1-alpha))-th order statistic (inf if alpha < 1/(n+1))."""
    cal = np.sort(np.asarray(cal_scores, dtype=np.float64))
    n = cal.size
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return np.inf
    return cal[k - 1]


def empirical_far(p_benign, alphas):
    p = np.asarray(p_benign)
    return {float(a): float(np.mean(p <= a)) for a in alphas}
