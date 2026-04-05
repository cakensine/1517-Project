"""
visualization.py — Publication-quality figures for AD detection project.

Contains:
  - plot_eda_panel(): 1x3 EDA figure (temporal, WER, semantic)
  - plot_feature_effect_sizes(): Forest plot of Hedges' g
  - plot_model_comparison(): Bar chart comparing model metrics
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stats_utils import hedges_g
from feature_extraction import count_content_units

PALETTE = {'HC': '#4C72B0', 'AD': '#DD8452'}


def plot_eda_panel(df_train, save_path='fig_eda.png', show=True):
    """Publication-quality 1×3 EDA figure.

    Panels:
      (a) Speech rate vs pause-to-speech ratio (scatter)
      (b) WER distribution by group (box)
      (c) Content units vs information density (scatter)

    Parameters:
        df_train: DataFrame with columns [speech_rate, pause_to_speech_ratio,
                  wer, text, label, group]. 'group' = {'HC', 'AD'}.
        save_path: output path (set None to skip saving)
        show: call plt.show()

    Returns:
        fig, axes
    """
    df = df_train.copy()
    if 'group' not in df.columns:
        df['group'] = df.label.map({0: 'HC', 1: 'AD'})

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # ---- (a) Temporal scatter ----
    ax = axes[0]
    for grp, color in PALETTE.items():
        mask = df.group == grp
        ax.scatter(df.loc[mask, 'speech_rate'],
                   df.loc[mask, 'pause_to_speech_ratio'],
                   c=color, label=grp, alpha=0.7, edgecolors='white', s=50)
    ax.set_xlabel('Speech Rate (words/sec)')
    ax.set_ylabel('Pause-to-Speech Ratio')
    ax.set_title('(a) Temporal Features')
    ax.legend()

    # ---- (b) WER box plot ----
    ax = axes[1]
    data_hc = df.loc[df.group == 'HC', 'wer']
    data_ad = df.loc[df.group == 'AD', 'wer']
    bp = ax.boxplot([data_hc, data_ad], labels=['HC', 'AD'], patch_artist=True,
                    widths=0.5, medianprops=dict(color='black', linewidth=1.5))
    bp['boxes'][0].set_facecolor(PALETTE['HC'])
    bp['boxes'][1].set_facecolor(PALETTE['AD'])
    for b in bp['boxes']:
        b.set_alpha(0.7)
    g = hedges_g(data_ad, data_hc)
    ax.set_title(f'(b) WER Distribution (g={g:.2f})')
    ax.set_ylabel('Word Error Rate')

    # ---- (c) Semantic scatter ----
    ax = axes[2]
    df['_wc'] = df['text'].apply(lambda t: len(re.findall(r'\b\w+\b', t)))
    df['_cu'] = df['text'].apply(count_content_units)
    df['_id'] = df['_cu'] / df['_wc']
    for grp, color in PALETTE.items():
        mask = df.group == grp
        ax.scatter(df.loc[mask, '_cu'], df.loc[mask, '_id'],
                   c=color, label=grp, alpha=0.7, edgecolors='white', s=50)
    ax.set_xlabel('Content Units Mentioned')
    ax.set_ylabel('Information Density')
    ax.set_title('(c) Semantic Features')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


def plot_feature_effect_sizes(stats_df, save_path=None, show=True):
    """Forest plot of Hedges' g effect sizes per feature.

    Parameters:
        stats_df: DataFrame with columns [Feature, Hedges_g, CI_95, Sig]
                  (as produced by the stats analysis cell)
        save_path: output path (set None to skip)
        show: call plt.show()
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    feats = stats_df['Feature'].values
    gs = stats_df['Hedges_g'].astype(float).values
    y = np.arange(len(feats))

    colors = ['#DD8452' if g > 0 else '#4C72B0' for g in gs]
    ax.barh(y, gs, color=colors, alpha=0.7, edgecolor='white')
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Hedges' g (AD vs HC)")
    ax.set_title('Feature Effect Sizes')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax


def plot_model_comparison(save_path=None, show=True):
    """Bar chart comparing Brier scores across models.

    Hardcoded values — update after all calibration results are in.
    """
    models = ['MLP\n(13 feat)', 'Frozen\nDistilBERT', 'Fine-tuned\nDistilBERT']
    brier  = [0.1227, 0.1251, 0.1592]
    acc    = [0.806, 0.833, 0.787]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    bars = ax.bar(models, acc, color=['#4C72B0', '#55A868', '#DD8452'], alpha=0.8)
    ax.set_ylabel('OOF Accuracy')
    ax.set_title('(a) Accuracy')
    ax.set_ylim(0.5, 0.9)
    for bar, v in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
                ha='center', va='bottom', fontsize=10)

    ax = axes[1]
    bars = ax.bar(models, brier, color=['#4C72B0', '#55A868', '#DD8452'], alpha=0.8)
    ax.set_ylabel('Brier Score')
    ax.set_title('(b) Calibration (Brier)')
    ax.set_ylim(0, 0.25)
    for bar, v in zip(bars, brier):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.4f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes
