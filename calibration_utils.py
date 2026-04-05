"""
calibration_utils.py — Shared calibration toolkit for MIE1517 project.
Person C writes this file. Person A and B import it to calibrate their models.

v2 changelog (review fixes applied):
  [Critical] Cross-fitted OOF evaluation to eliminate calibration-stage leakage
  [Medium]   Temperature scaling: bounded log-space optimization (minimize_scalar)
  [Medium]   Robust quantile binning that handles tied probabilities
  [Medium]   Input validation for shape/label sanity
  [Medium]   run_calibration_suite returns fitted_models for test-time use
  [Low]      bootstrap_ci() for uncertainty intervals
  [Low]      Fixed cube-root comment (108^(1/3) ≈ 4.76)
  [Low]      All comments translated to English

Usage:
    from calibration_utils import run_calibration_suite, reliability_diagram
    results = run_calibration_suite(oof_logits, oof_labels)
"""

import os

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit  # sigmoid
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


# ============================================================
# 0. INTERNAL HELPERS — validation and binning
# ============================================================

def _validate_binary_inputs(logits, labels):
    """
    Validate that logits and labels are compatible binary classification arrays.
    Raises ValueError with clear message if anything is wrong.

    Returns:
        logits (np.ndarray): shape (n,), dtype float
        labels (np.ndarray): shape (n,), dtype int
    """
    logits = np.asarray(logits, dtype=float).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Length mismatch: logits has {logits.shape[0]} elements, "
            f"labels has {labels.shape[0]} elements"
        )
    unique = np.unique(labels)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(
            f"Labels must be binary {{0, 1}}, got unique values: {unique}"
        )
    if unique.size < 2:
        raise ValueError(
            "Both classes (0 and 1) are required to fit calibration. "
            f"Only found class {unique[0]}."
        )
    return logits, labels


def _quantile_bins(y_prob, n_bins):
    """
    Compute quantile-based bin edges, robust to tied probabilities.
    When many probabilities are identical (e.g., saturated near 0 or 1),
    np.quantile can produce duplicate edges. This deduplicates them
    so the actual number of bins may be fewer than requested.

    Returns:
        bins (np.ndarray): monotonically increasing bin edges,
                           first edge = 0.0, last edge = 1.0 + eps
    """
    y_prob = np.clip(np.asarray(y_prob, dtype=float).ravel(), 0.0, 1.0)
    bins = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)  # remove duplicates from tied values
    if bins.size < 2:
        # Degenerate case: all probabilities are identical
        return np.array([0.0, 1.0 + 1e-8], dtype=float)
    bins[0] = 0.0
    bins[-1] = 1.0 + 1e-8
    return bins


def _uniform_bins(n_bins):
    """Compute uniform-width bin edges on [0, 1]."""
    return np.linspace(0.0, 1.0 + 1e-8, int(n_bins) + 1)


def _resolve_bins(y_prob, n_bins=4, strategy='quantile', bins=None):
    """
    Resolve bin edges from either provided bins or a strategy.

    Parameters:
        y_prob: probabilities (used when bins is None)
        n_bins: requested number of bins
        strategy: "quantile" or "uniform"
        bins: explicit bin edges (overrides strategy if provided)
    """
    if bins is not None:
        resolved = np.asarray(bins, dtype=float).ravel()
        if resolved.size < 2:
            raise ValueError("Provided bins must contain at least two edges.")
        resolved = np.unique(resolved)
        if resolved.size < 2:
            raise ValueError("Provided bins collapse to fewer than two unique edges.")
        resolved[0] = 0.0
        resolved[-1] = 1.0 + 1e-8
        return resolved

    if strategy == 'quantile':
        return _quantile_bins(y_prob, n_bins)
    if strategy == 'uniform':
        return _uniform_bins(n_bins)
    raise ValueError(f"Unknown binning strategy: {strategy}")


def _wilson_interval(successes, n, z=1.959963984540054):
    """
    Wilson score confidence interval for a binomial proportion.

    Returns:
        (low, high) bounded to [0, 1]
    """
    if n <= 0:
        return np.nan, np.nan
    p_hat = successes / n
    denom = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2.0 * n)) / denom
    margin = (
        z
        * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z ** 2) / (4.0 * (n ** 2)))
        / denom
    )
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return float(low), float(high)


def _compute_reliability_bin_stats(y_true, y_prob, n_bins=4, strategy='quantile',
                                   bins=None, include_ci=True):
    """
    Compute reliability-diagram bin statistics.

    Returns:
        dict with keys:
            edges, count, conf_mean, acc_mean, gap, ci_low, ci_high
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.clip(np.asarray(y_prob, dtype=float).ravel(), 0.0, 1.0)
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true has {y_true.shape[0]}, y_prob has {y_prob.shape[0]}"
        )

    edges = _resolve_bins(y_prob, n_bins=n_bins, strategy=strategy, bins=bins)

    counts = []
    conf_mean = []
    acc_mean = []
    gaps = []
    ci_low = []
    ci_high = []

    for i in range(len(edges) - 1):
        mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        n_i = int(mask.sum())
        if n_i == 0:
            continue
        y_bin = y_true[mask]
        p_bin = y_prob[mask]

        n_pos = int(np.round(y_bin.sum()))
        acc = float(y_bin.mean())
        conf = float(p_bin.mean())

        if include_ci:
            low, high = _wilson_interval(n_pos, n_i)
        else:
            low, high = np.nan, np.nan

        counts.append(n_i)
        conf_mean.append(conf)
        acc_mean.append(acc)
        gaps.append(acc - conf)
        ci_low.append(low)
        ci_high.append(high)

    return {
        'edges': edges,
        'count': np.asarray(counts, dtype=int),
        'conf_mean': np.asarray(conf_mean, dtype=float),
        'acc_mean': np.asarray(acc_mean, dtype=float),
        'gap': np.asarray(gaps, dtype=float),
        'ci_low': np.asarray(ci_low, dtype=float),
        'ci_high': np.asarray(ci_high, dtype=float),
    }


# ============================================================
# 1. METRICS — three measures of probability quality
# ============================================================

def brier_score(y_true, y_prob):
    """
    Brier Score: mean squared error between predicted probabilities and true labels.
    Lower is better. Range [0, 1], perfect calibration = 0.

    Example:
        y_true = [1, 0, 1, 0]
        y_prob = [0.9, 0.1, 0.8, 0.3]
        brier_score(y_true, y_prob)  # ~ 0.0375 (good)
    """
    return brier_score_loss(y_true, y_prob)


def negative_log_likelihood(y_true, y_prob):
    """
    NLL: negative log-likelihood. Lower is better.
    Heavily penalizes confident wrong predictions
    (e.g., predicting 0.99 when truth is 0 -> very large NLL).

    Probabilities are clipped to [1e-15, 1-1e-15] for numerical stability.
    """
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    nll = -np.mean(
        y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
    )
    return nll


def expected_calibration_error(y_true, y_prob, n_bins=4, strategy='quantile',
                               bins=None):
    """
    ECE: partition predictions into n_bins bins, then compute the
    weighted average of |mean_predicted_prob - actual_accuracy| per bin.

    Uses 'quantile' (equal-count) binning by default instead of 'uniform'
    (equal-width) because with small samples, equal-width creates empty bins.

    n_bins=4: we have 108 OOF samples. Futami et al. (2024) recommend
    optimal bins ~ n^(1/3) ~ 4.76, so 4 is conservative.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bins = _resolve_bins(y_prob, n_bins=n_bins, strategy=strategy, bins=bins)

    ece = 0.0
    for i in range(len(bins) - 1):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()       # actual positive rate in this bin
        bin_conf = y_prob[mask].mean()      # mean predicted probability in this bin
        bin_weight = mask.sum() / len(y_true)
        ece += bin_weight * abs(bin_acc - bin_conf)

    return ece


def compute_all_metrics(y_true, y_prob, n_bins=4, strategy='quantile',
                        bins=None):
    """
    Compute all three metrics in one call.

    Returns:
        {'brier': 0.18, 'nll': 0.52, 'ece': 0.07}
    """
    return {
        'brier': brier_score(y_true, y_prob),
        'nll': negative_log_likelihood(y_true, y_prob),
        'ece': expected_calibration_error(
            y_true, y_prob, n_bins=n_bins, strategy=strategy, bins=bins
        ),
    }


# ============================================================
# 2. CALIBRATION METHODS — three tools to correct probabilities
# ============================================================

def logits_to_probs(logits):
    """Logits -> probabilities (sigmoid for binary classification)."""
    return expit(logits)


# --- Method A: Temperature Scaling ---

def fit_temperature_scaling(logits, y_true, t_min=1e-2, t_max=1e2):
    """
    Temperature Scaling: learn a single parameter T so that
    sigmoid(logits / T) produces better-calibrated probabilities.

    T > 1: makes model less confident (common case)
    T < 1: makes model more confident
    T = 1: no change

    Optimization is done in log-space with bounded scalar search,
    which avoids the numerical issues of unconstrained Nelder-Mead.

    Parameters:
        logits: raw logits from the model
        y_true: true binary labels (0/1)
        t_min: lower bound for T (default 0.01)
        t_max: upper bound for T (default 100)

    Returns:
        T (float): learned temperature parameter
    """
    logits, y_true = _validate_binary_inputs(logits, y_true)

    def objective(log_t):
        T = np.exp(log_t)
        scaled_probs = expit(logits / T)
        return negative_log_likelihood(y_true, scaled_probs)

    result = minimize_scalar(
        objective,
        bounds=(np.log(t_min), np.log(t_max)),
        method='bounded',
    )
    if not result.success:
        raise RuntimeError(f"Temperature scaling optimization failed: {result.message}")
    return float(np.exp(result.x))


def apply_temperature_scaling(logits, T):
    """Apply learned T to convert logits -> calibrated probabilities."""
    return expit(np.asarray(logits, dtype=float) / T)


# --- Method B: Platt Scaling ---

def fit_platt_scaling(logits, y_true, C=1.0):
    """
    Platt Scaling: fit a logistic regression on raw logits.
    Learns two parameters a and b: calibrated_prob = sigmoid(a * logit + b)

    More flexible than Temperature Scaling (2 params vs 1),
    but also more prone to overfitting on small samples.

    Parameters:
        logits: raw logits
        y_true: true binary labels
        C: inverse regularization strength (default 1.0)

    Returns:
        LogisticRegression model (fitted)
    """
    logits, y_true = _validate_binary_inputs(logits, y_true)
    lr = LogisticRegression(C=C, solver='lbfgs', max_iter=1000)
    lr.fit(logits.reshape(-1, 1), y_true)
    return lr


def apply_platt_scaling(logits, platt_model):
    """Apply fitted Platt model to convert logits -> calibrated probabilities."""
    logits = np.asarray(logits, dtype=float).ravel()
    return platt_model.predict_proba(logits.reshape(-1, 1))[:, 1]


# --- Method C: Isotonic Regression ---

def fit_isotonic_regression(probs, y_true):
    """
    Isotonic Regression: non-parametric method that fits a monotonically
    increasing function mapping probabilities -> calibrated probabilities.
    NOTE: input is probabilities (post-sigmoid), NOT logits.

    Pros: makes no distributional assumptions.
    Cons: easily overfits when n < 1000.
    For our 108 samples, it will likely overfit — this is itself a finding.

    Returns:
        IsotonicRegression model (fitted)
    """
    probs = np.asarray(probs, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=int).ravel()
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    ir.fit(probs, y_true)
    return ir


def apply_isotonic_regression(probs, iso_model):
    """Apply fitted isotonic model to convert probabilities -> calibrated probabilities."""
    return iso_model.predict(np.asarray(probs, dtype=float).ravel())


# ============================================================
# 3. CROSS-FITTED CALIBRATION — unbiased OOF evaluation
# ============================================================

def cross_fitted_calibrated_probs(logits, y, method, n_splits=5,
                                  random_state=42, platt_C=1.0):
    """
    Cross-fitted calibration to avoid leakage during OOF evaluation.

    Problem: if we fit a calibrator on all 108 OOF samples and then
    evaluate metrics on those same 108 samples, the calibrator has
    "seen" the evaluation data. This is especially bad for isotonic
    regression, which can perfectly memorize the training set.

    Solution: split OOF into K inner folds. For each inner fold,
    fit calibrator on the other K-1 folds, predict on the held-out fold.
    The result is truly out-of-sample calibrated probabilities.

    Parameters:
        logits: OOF logits, shape (n,)
        y: true labels, shape (n,)
        method: one of "temperature", "platt", "isotonic"
        n_splits: number of inner CV folds (default 5)
        random_state: random seed for reproducibility
        platt_C: regularization for Platt scaling (default 1.0)

    Returns:
        out: cross-fitted calibrated probabilities, shape (n,)
    """
    logits = np.asarray(logits, dtype=float).ravel()
    y = np.asarray(y, dtype=int).ravel()
    raw_probs = expit(logits)
    out = np.zeros_like(raw_probs, dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    for tr, va in skf.split(logits, y):
        if method == "temperature":
            T = fit_temperature_scaling(logits[tr], y[tr])
            out[va] = apply_temperature_scaling(logits[va], T)
        elif method == "platt":
            m = fit_platt_scaling(logits[tr], y[tr], C=platt_C)
            out[va] = apply_platt_scaling(logits[va], m)
        elif method == "isotonic":
            m = fit_isotonic_regression(raw_probs[tr], y[tr])
            out[va] = apply_isotonic_regression(raw_probs[va], m)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    return out


# ============================================================
# 4. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=2000, alpha=0.05,
                 random_state=42):
    """
    Compute bootstrap confidence interval for any metric.

    Parameters:
        y_true: true labels
        y_prob: predicted probabilities
        metric_fn: callable(y_true, y_prob) -> float
        n_boot: number of bootstrap samples (default 2000)
        alpha: significance level (default 0.05 -> 95% CI)
        random_state: random seed

    Returns:
        (lower, upper): tuple of floats, the alpha/2 and 1-alpha/2 quantiles

    Example:
        lo, hi = bootstrap_ci(labels, probs, brier_score)
        print(f"Brier Score 95% CI: [{lo:.4f}, {hi:.4f}]")
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = metric_fn(y_true[idx], y_prob[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ============================================================
# 5. RELIABILITY DIAGRAM — visualize calibration quality
# ============================================================

def reliability_diagram(y_true, y_prob, n_bins=4, ax=None, label=None,
                        color='steelblue', show_ece=True, bins=None,
                        show_bin_ci=False, show_bin_counts=False,
                        show_gap=False):
    """
    Plot a reliability diagram.

    Backward-compatible wrapper for teammates who already import this function.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    stats = _compute_reliability_bin_stats(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy='quantile',
        bins=bins,
        include_ci=show_bin_ci,
    )
    conf = stats['conf_mean']
    acc = stats['acc_mean']

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label='Perfect calibration')

    if conf.size > 0:
        yerr = None
        if show_bin_ci:
            yerr = np.vstack([
                np.maximum(0.0, acc - stats['ci_low']),
                np.maximum(0.0, stats['ci_high'] - acc),
            ])

        ax.errorbar(
            conf,
            acc,
            yerr=yerr,
            fmt='o-',
            color=color,
            linewidth=1.8,
            markersize=5,
            capsize=3 if show_bin_ci else 0,
            label=label,
        )

        if show_gap:
            for x_i, y_i in zip(conf, acc):
                ax.vlines(
                    x_i,
                    ymin=min(x_i, y_i),
                    ymax=max(x_i, y_i),
                    colors=color,
                    linestyles=':',
                    linewidth=1.0,
                    alpha=0.7,
                )

        if show_bin_counts:
            for x_i, y_i, n_i in zip(conf, acc, stats['count']):
                ax.text(
                    x_i,
                    min(y_i + 0.04, 0.98),
                    f"n={n_i}",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color='#444444',
                )

    if show_ece:
        ece = expected_calibration_error(y_true, y_prob, bins=stats['edges'])
        ax.text(
            0.05,
            0.92,
            f'ECE = {ece:.4f}',
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor='white',
                edgecolor='#BBBBBB',
                alpha=0.95,
            ),
        )

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, linewidth=0.6)
    return ax


def _format_metrics_annotation(metrics, temperature_T=None):
    """Format per-panel metric text."""
    lines = [
        f"Brier = {metrics['brier']:.4f}",
        f"NLL = {metrics['nll']:.4f}",
        f"ECE = {metrics['ece']:.4f}",
    ]
    if temperature_T is not None:
        lines.append(f"T = {temperature_T:.3f}")
    return "\n".join(lines)


def plot_calibration_comparison(
    y_true,
    prob_dict,
    metrics_dict,
    model_name='Model',
    n_bins=4,
    strategy='quantile',
    bins=None,
    plot_style='publication',
    show_bin_counts=True,
    show_bin_ci=True,
    save_dir='.',
    save_formats=('png', 'pdf'),
    plot_dpi=300,
):
    """
    Plot a standardized Section 7 figure (2x4):
    top row reliability, bottom row prediction histograms.

    Returns:
        fig, figure_paths, shared_bins
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    method_order = ['before', 'temperature', 'platt', 'isotonic']
    method_order = [m for m in method_order if m in prob_dict]
    if len(method_order) == 0:
        raise ValueError("prob_dict must contain at least one known method key.")

    base_prob = np.asarray(prob_dict[method_order[0]], dtype=float).ravel()
    shared_bins = _resolve_bins(base_prob, n_bins=n_bins, strategy=strategy, bins=bins)

    method_titles = {
        'before': 'Before Calibration',
        'temperature': 'Temperature Scaling',
        'platt': 'Platt Scaling',
        'isotonic': 'Isotonic Regression',
    }
    method_colors = {
        'before': '#4C78A8',
        'temperature': '#59A14F',
        'platt': '#F28E2B',
        'isotonic': '#9C7CC5',
    }

    rc_publication = {
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.edgecolor': '#222222',
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'font.size': 10,
    }
    rc_ctx = rc_publication if plot_style == 'publication' else {}

    n_methods = len(method_order)
    with plt.rc_context(rc_ctx):
        fig, axes = plt.subplots(
            2,
            n_methods,
            figsize=(4.3 * n_methods, 7.2),
            gridspec_kw={'height_ratios': [2.8, 1.2]},
            sharex='col',
        )
        if n_methods == 1:
            axes = np.asarray(axes).reshape(2, 1)

        for col, method in enumerate(method_order):
            probs = np.clip(np.asarray(prob_dict[method], dtype=float).ravel(), 0.0, 1.0)
            color = method_colors.get(method, '#4C78A8')

            ax_main = axes[0, col]
            stats = _compute_reliability_bin_stats(
                y_true=y_true,
                y_prob=probs,
                bins=shared_bins,
                include_ci=show_bin_ci,
            )
            conf = stats['conf_mean']
            acc = stats['acc_mean']
            yerr = None
            if show_bin_ci and conf.size > 0:
                yerr = np.vstack([
                    np.maximum(0.0, acc - stats['ci_low']),
                    np.maximum(0.0, stats['ci_high'] - acc),
                ])

            ax_main.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.6)
            if conf.size > 0:
                ax_main.errorbar(
                    conf,
                    acc,
                    yerr=yerr,
                    fmt='o-',
                    color=color,
                    linewidth=2.0,
                    markersize=6,
                    capsize=3 if show_bin_ci else 0,
                )
                # Shaded fill between curve and diagonal (replaces dotted vlines)
                ax_main.fill_between(
                    conf, acc, conf,
                    alpha=0.15,
                    color=color,
                    linewidth=0,
                )
                if show_bin_counts:
                    for x_i, y_i, n_i in zip(conf, acc, stats['count']):
                        # Place label above or below curve depending on
                        # whether curve is above or below diagonal
                        offset = 0.04 if y_i >= x_i else -0.06
                        label_y = np.clip(y_i + offset, 0.03, 0.97)
                        ax_main.text(
                            x_i,
                            label_y,
                            f"n={n_i}",
                            ha='center',
                            va='bottom' if offset > 0 else 'top',
                            fontsize=7,
                            color='#666666',
                        )

            panel_metrics = metrics_dict.get(method, {})
            if {'brier', 'nll', 'ece'}.issubset(panel_metrics.keys()):
                metric_text = _format_metrics_annotation(
                    panel_metrics,
                    temperature_T=panel_metrics.get('T') if method == 'temperature' else None,
                )
                ax_main.text(
                    0.03,
                    0.97,
                    metric_text,
                    transform=ax_main.transAxes,
                    ha='left',
                    va='top',
                    fontsize=9,
                    bbox=dict(
                        boxstyle='round,pad=0.25',
                        facecolor='white',
                        edgecolor='#BBBBBB',
                        alpha=0.95,
                    ),
                )

            ax_main.set_title(method_titles.get(method, method.title()))
            ax_main.set_xlim(0, 1)
            ax_main.set_ylim(0, 1)
            ax_main.grid(alpha=0.25, linewidth=0.6)
            if col == 0:
                ax_main.set_ylabel('Fraction of Positives')

            ax_hist = axes[1, col]
            ax_hist.hist(
                probs,
                bins=shared_bins,
                color=color,
                alpha=0.75,
                edgecolor='white',
                linewidth=0.8,
            )
            ax_hist.set_xlim(0, 1)
            ax_hist.grid(alpha=0.20, linewidth=0.5)
            ax_hist.set_xlabel('Predicted Probability')
            if col == 0:
                ax_hist.set_ylabel('Count')

        fig.suptitle(
            (
                f"{model_name} - Reliability Diagrams (cross-fitted OOF)\n"
                f"N={len(y_true)}, bins={len(shared_bins) - 1}, strategy={strategy}"
            ),
            fontsize=14,
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        os.makedirs(save_dir, exist_ok=True)
        safe_model_name = model_name.lower().replace(" ", "_")
        base_name = f"{safe_model_name}_calibration_{plot_style}"
        formats = (save_formats,) if isinstance(save_formats, str) else tuple(save_formats)

        figure_paths = []
        for fmt in formats:
            fmt_clean = str(fmt).lower().lstrip(".")
            out_path = os.path.join(save_dir, f"{base_name}.{fmt_clean}")
            save_kwargs = {'bbox_inches': 'tight'}
            if fmt_clean in {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'webp'}:
                save_kwargs['dpi'] = int(plot_dpi)
            fig.savefig(out_path, **save_kwargs)
            figure_paths.append(out_path)

        plt.show()

    return fig, figure_paths, shared_bins


# ============================================================
# 6. RUN ALL — the main function each person calls
# ============================================================

def run_calibration_suite(oof_logits, oof_labels, model_name='Model',
                          platt_C=1.0, n_inner_folds=5, n_bins=4,
                          plot_n_bins=None, plot_style='publication',
                          save_dir='.', save_formats=('png', 'pdf'),
                          plot_dpi=300, show_bin_counts=True,
                          show_bin_ci=True):
    """
    Run the complete calibration analysis in one call.

    IMPORTANT:
    1) OOF metrics use cross-fitted calibration to avoid leakage.
    2) Fitted models in results are trained on ALL OOF data for test-time use.

    Parameters:
        oof_logits: numpy array, shape (n,)
            Out-of-fold logits from CV (values before sigmoid).
        oof_labels: numpy array, shape (n,)
            True labels (0 = HC, 1 = AD)
        model_name: str
            Model name for printed summary and plot title
        platt_C: float
            Platt scaling regularization (inverse strength)
        n_inner_folds: int
            Inner folds for cross-fitted calibration
        n_bins: int
            Shared bins for ECE in the result dict
        plot_n_bins: int or None
            Bin count for figure; if None, uses n_bins
        plot_style: str
            Figure style preset ('publication' or 'default')
        save_dir: str
            Figure output directory
        save_formats: tuple/list/str
            Figure formats, e.g. ('png', 'pdf')
        plot_dpi: int
            DPI for raster formats
        show_bin_counts: bool
            Whether to annotate n per bin in reliability panels
        show_bin_ci: bool
            Whether to show Wilson 95% CI in reliability panels

    Returns:
        dict with keys:
            before, temperature, platt, isotonic, fitted_models, figure_paths
    """
    oof_logits, oof_labels = _validate_binary_inputs(oof_logits, oof_labels)

    results = {}
    raw_probs = logits_to_probs(oof_logits)

    # Shared ECE bins for fair method comparison.
    metric_bins = _resolve_bins(raw_probs, n_bins=n_bins, strategy='quantile')

    # --- Before calibration ---
    results['before'] = compute_all_metrics(
        oof_labels, raw_probs, n_bins=n_bins, strategy='quantile', bins=metric_bins
    )
    results['before']['brier_ci'] = bootstrap_ci(oof_labels, raw_probs, brier_score)
    results['before']['nll_ci'] = bootstrap_ci(
        oof_labels, raw_probs, negative_log_likelihood
    )

    print(f"\n{'='*60}")
    print(f"  {model_name} - Calibration Analysis (cross-fitted OOF)")
    print(f"{'='*60}")
    _print_metrics('Before calibration', results['before'])

    # --- Cross-fitted calibrated probabilities for unbiased evaluation ---
    cf_temp_probs = cross_fitted_calibrated_probs(
        oof_logits, oof_labels, 'temperature',
        n_splits=n_inner_folds, platt_C=platt_C
    )
    cf_platt_probs = cross_fitted_calibrated_probs(
        oof_logits, oof_labels, 'platt',
        n_splits=n_inner_folds, platt_C=platt_C
    )
    cf_iso_probs = cross_fitted_calibrated_probs(
        oof_logits, oof_labels, 'isotonic',
        n_splits=n_inner_folds
    )

    # --- Fit calibrators on all OOF data for test-time use ---
    T_full = fit_temperature_scaling(oof_logits, oof_labels)
    platt_full = fit_platt_scaling(oof_logits, oof_labels, C=platt_C)
    iso_full = fit_isotonic_regression(raw_probs, oof_labels)

    # --- Temperature metrics ---
    results['temperature'] = compute_all_metrics(
        oof_labels, cf_temp_probs, n_bins=n_bins, strategy='quantile', bins=metric_bins
    )
    results['temperature']['T'] = T_full
    results['temperature']['brier_ci'] = bootstrap_ci(
        oof_labels, cf_temp_probs, brier_score
    )
    _print_metrics(f'Temperature Scaling (T = {T_full:.3f})', results['temperature'])

    # --- Platt metrics ---
    results['platt'] = compute_all_metrics(
        oof_labels, cf_platt_probs, n_bins=n_bins, strategy='quantile', bins=metric_bins
    )
    results['platt']['brier_ci'] = bootstrap_ci(
        oof_labels, cf_platt_probs, brier_score
    )
    _print_metrics('Platt Scaling', results['platt'])

    # --- Isotonic metrics ---
    results['isotonic'] = compute_all_metrics(
        oof_labels, cf_iso_probs, n_bins=n_bins, strategy='quantile', bins=metric_bins
    )
    results['isotonic']['brier_ci'] = bootstrap_ci(
        oof_labels, cf_iso_probs, brier_score
    )
    _print_metrics('Isotonic Regression', results['isotonic'])

    # --- Store fitted models for test-time application ---
    results['fitted_models'] = {
        'temperature_T': T_full,
        'platt_model': platt_full,
        'isotonic_model': iso_full,
    }

    # --- Plot standardized Section 7 figure ---
    if plot_n_bins is None:
        plot_bins = metric_bins
    else:
        plot_bins = _resolve_bins(
            raw_probs, n_bins=int(plot_n_bins), strategy='quantile'
        )

    plot_metrics = {
        'before': dict(results['before']),
        'temperature': dict(results['temperature']),
        'platt': dict(results['platt']),
        'isotonic': dict(results['isotonic']),
    }

    # Keep panel ECE annotation consistent when plot bins differ.
    if len(plot_bins) != len(metric_bins) or not np.allclose(plot_bins, metric_bins):
        plot_metrics['before']['ece'] = expected_calibration_error(
            oof_labels, raw_probs, bins=plot_bins
        )
        plot_metrics['temperature']['ece'] = expected_calibration_error(
            oof_labels, cf_temp_probs, bins=plot_bins
        )
        plot_metrics['platt']['ece'] = expected_calibration_error(
            oof_labels, cf_platt_probs, bins=plot_bins
        )
        plot_metrics['isotonic']['ece'] = expected_calibration_error(
            oof_labels, cf_iso_probs, bins=plot_bins
        )

    _, figure_paths, _ = plot_calibration_comparison(
        y_true=oof_labels,
        prob_dict={
            'before': raw_probs,
            'temperature': cf_temp_probs,
            'platt': cf_platt_probs,
            'isotonic': cf_iso_probs,
        },
        metrics_dict=plot_metrics,
        model_name=model_name,
        n_bins=len(plot_bins) - 1,
        strategy='quantile',
        bins=plot_bins,
        plot_style=plot_style,
        show_bin_counts=show_bin_counts,
        show_bin_ci=show_bin_ci,
        save_dir=save_dir,
        save_formats=save_formats,
        plot_dpi=plot_dpi,
    )
    results['figure_paths'] = figure_paths

    # --- Summary ---
    print(f"\n  Best method (lowest cross-fitted Brier): ", end='')
    methods = ['before', 'temperature', 'platt', 'isotonic']
    best = min(methods, key=lambda m: results[m]['brier'])
    print(f"{best} (Brier = {results[best]['brier']:.4f})")

    if best == 'before':
        print('  NOTE: no calibration method improved over raw probabilities.')
        print('  This is a valid finding - report it as-is.')

    print()
    return results

def _print_metrics(title, metrics):
    """Pretty-print a metrics dict with optional CI."""
    print(f"\n  {title}:")
    brier = metrics['brier']
    nll = metrics['nll']
    ece = metrics['ece']

    ci_str = ""
    if 'brier_ci' in metrics:
        lo, hi = metrics['brier_ci']
        ci_str = f"  [95% CI: {lo:.4f} - {hi:.4f}]"
    print(f"    Brier = {brier:.4f}{ci_str}")

    ci_str = ""
    if 'nll_ci' in metrics:
        lo, hi = metrics['nll_ci']
        ci_str = f"  [95% CI: {lo:.4f} - {hi:.4f}]"
    print(f"    NLL   = {nll:.4f}{ci_str}")

    print(f"    ECE   = {ece:.4f}")


# ── Test Set Calibration ──────────────────────────────────────

def evaluate_test_calibration(test_logits, y_test, cal_results, model_name,
                              n_bins=4, save_dir='.'):
    """Apply OOF-fitted calibrators to test logits, compute metrics + plot."""
    fitted = cal_results['fitted_models']
    T = fitted['temperature_T']
    platt_model = fitted['platt_model']
    iso_model = fitted['isotonic_model']

    raw_probs = logits_to_probs(test_logits)
    temp_probs = apply_temperature_scaling(test_logits, T)
    platt_probs = apply_platt_scaling(test_logits, platt_model)
    iso_probs = apply_isotonic_regression(raw_probs, iso_model)

    metric_bins = _resolve_bins(raw_probs, n_bins=n_bins, strategy='quantile')

    results = {}
    for name, probs in [('before', raw_probs), ('temperature', temp_probs),
                         ('platt', platt_probs), ('isotonic', iso_probs)]:
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        results[name] = compute_all_metrics(y_test, probs, bins=metric_bins)

    results['temperature']['T'] = T
    results['fitted_models'] = fitted

    print(f"\n{'=' * 60}")
    print(f"  {model_name} - TEST SET Calibration")
    print(f"{'=' * 60}")
    for method in ['before', 'temperature', 'platt', 'isotonic']:
        m = results[method]
        t_str = f"  (T={T:.3f})" if method == 'temperature' else ''
        print(f"  {method:15s}: Brier={m['brier']:.4f}  NLL={m['nll']:.4f}  ECE={m['ece']:.4f}{t_str}")

    best = min(['before', 'temperature', 'platt', 'isotonic'],
               key=lambda m: results[m]['brier'])
    print(f"  Best (Brier): {best} ({results[best]['brier']:.4f})")

    # Plot
    prob_dict = {'before': raw_probs, 'temperature': temp_probs,
                 'platt': platt_probs, 'isotonic': iso_probs}
    plot_metrics = {k: dict(results[k]) for k in ['before', 'temperature', 'platt', 'isotonic']}
    fig, paths, _ = plot_calibration_comparison(
        y_true=y_test, prob_dict=prob_dict, metrics_dict=plot_metrics,
        model_name=f'{model_name} (Test Set)', bins=metric_bins,
        show_bin_counts=True, show_bin_ci=True,
        save_dir=save_dir, save_formats=('png',), plot_dpi=200)
    results['figure_paths'] = paths
    return results
