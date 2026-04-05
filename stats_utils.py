"""
stats_utils.py — Statistical analysis utilities.

Effect sizes (Hedges' g, Cliff's delta), bootstrap CIs,
and Brier score decomposition (Murphy 1973).
"""

import numpy as np


def hedges_g(x, y):
    """Hedges' g effect size (positive = x > y)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 +
                      (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2))
    g = (np.mean(x) - np.mean(y)) / pooled if pooled > 0 else 0.0
    return g * (1 - 3 / (4 * (nx + ny) - 9))  # small-sample correction


def cliffs_delta(x, y):
    """Cliff's delta (non-parametric effect size for ordinal/discrete)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    diffs = x[:, None] - y[None, :]
    return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / diffs.size)


def boot_ci(func, x, y, n=5000, alpha=0.05, seed=42):
    """Bootstrap 95% CI for a two-sample statistic."""
    rng = np.random.default_rng(seed)
    stats = [func(rng.choice(x, len(x)), rng.choice(y, len(y))) for _ in range(n)]
    return np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def brier_decomposition(y_true, y_prob, n_bins=8):
    """Brier decomposition: REL - RES + UNC = Brier (Murphy 1973).

    Returns: (reliability, resolution, uncertainty)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    base_rate = y_true.mean()
    unc = base_rate * (1 - base_rate)

    bins = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bins[0], bins[-1] = 0.0, 1.0 + 1e-9

    rel, res, N = 0.0, 0.0, len(y_true)
    for j in range(n_bins):
        mask = (y_prob >= bins[j]) & (y_prob < bins[j + 1])
        nk = mask.sum()
        if nk == 0:
            continue
        yk = y_true[mask].mean()
        pk = y_prob[mask].mean()
        rel += nk * (pk - yk) ** 2
        res += nk * (yk - base_rate) ** 2

    return rel / N, res / N, unc
