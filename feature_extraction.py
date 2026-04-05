"""
feature_extraction.py — 13 handcrafted features across 5 dimensions.

Dim 1 — Semantic (gold text):    n_content_units, info_density
Dim 2 — Lexical (gold text):     pronoun_noun_ratio, mattr, word_count
Dim 3 — Disfluency (.cha files): repetition_rate, filler_rate
Dim 4 — Temporal (from ASR):     speech_rate, pause_to_speech_ratio,
                                  n_long_pauses, mean_pause_dur
Dim 5 — ASR error:               wer, deletion_rate
"""

import os
import re
import numpy as np
import spacy
from jiwer import process_words
from asr_pipeline import normalize_for_wer

nlp = spacy.load('en_core_web_sm')

# --- Feature column names (ordered) ---
FEATURE_COLS = [
    'n_content_units', 'info_density',
    'pronoun_noun_ratio', 'mattr', 'word_count',
    'repetition_rate', 'filler_rate',
    'speech_rate', 'pause_to_speech_ratio', 'n_long_pauses', 'mean_pause_dur',
    'wer', 'deletion_rate',
]

# Cookie Theft standard information units (16 groups)
# Croisile et al. (1996)
CONTENT_UNITS = [
    ['boy', 'son', 'kid'],
    ['girl', 'daughter', 'sister'],
    ['woman', 'mother', 'mom', 'lady'],
    ['cookie', 'cookies'],
    ['jar'],
    ['stool', 'step stool', 'stepstool'],
    ['sink'],
    ['dish', 'dishes', 'plate', 'plates'],
    ['cupboard', 'cabinet', 'counter', 'countertop'],
    ['window'],
    ['curtain', 'curtains'],
    ['water'],
    ['steal', 'stealing', 'taking', 'reaching', 'climb', 'climbing'],
    ['fall', 'falling', 'tipping', 'tilting', 'tilt', 'topple'],
    ['wash', 'washing', 'drying', 'wiping', 'dry'],
    ['overflow', 'overflowing', 'spilling', 'running over', 'overspilling'],
]


# ============================================================
# 6.1  Semantic features (gold text)
# ============================================================

def _kw_match(text_lower, kw):
    """Match keyword with word boundaries; handles multi-word phrases."""
    if ' ' in kw:
        pat = r'(?<!\w)' + re.escape(kw).replace(r'\ ', r'[\s-]+') + r'(?!\w)'
    else:
        pat = rf'\b{re.escape(kw)}\b'
    return re.search(pat, text_lower) is not None


def count_content_units(text):
    """Count Cookie Theft information units mentioned in text."""
    text_lower = text.lower()
    return sum(
        any(_kw_match(text_lower, kw) for kw in kws)
        for kws in CONTENT_UNITS
    )


# ============================================================
# 6.2  Lexical features (gold text)
# ============================================================

def pronoun_noun_ratio(text):
    """PRON / (PRON + NOUN) via spaCy POS tagging."""
    doc = nlp(text)
    pron = sum(1 for t in doc if t.pos_ == 'PRON')
    noun = sum(1 for t in doc if t.pos_ == 'NOUN')
    total = pron + noun
    return pron / total if total > 0 else 0.0


def compute_mattr(text, window=10):
    """Moving Average Type-Token Ratio (MATTR).
    Length-corrected lexical diversity."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    n = len(tokens)
    if n < window:
        return len(set(tokens)) / n if n > 0 else 0.0
    ttrs = []
    for i in range(n - window + 1):
        w = tokens[i:i + window]
        ttrs.append(len(set(w)) / window)
    return np.mean(ttrs)


# ============================================================
# 6.3  Disfluency features (.cha annotation)
# ============================================================

def extract_disfluency_from_cha(cha_path):
    """Extract repetition and filler rates from raw .cha PAR utterances.

    Returns: (repetition_rate, filler_rate)
    """
    with open(cha_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    par_utterances = []
    current_speaker, current_text = None, ''

    for line in lines:
        line = line.rstrip('\n')
        if line.startswith('*'):
            if current_speaker == '*PAR' and current_text:
                par_utterances.append(current_text)
            current_speaker = line.split(':')[0]
            current_text = line[len(current_speaker) + 1:].strip()
        elif line.startswith('\t') and current_speaker:
            current_text += ' ' + line.strip()
        elif line.startswith(('%', '@')):
            if current_speaker == '*PAR' and current_text:
                par_utterances.append(current_text)
            current_speaker, current_text = None, ''
    if current_speaker == '*PAR' and current_text:
        par_utterances.append(current_text)

    n_utt = len(par_utterances)
    if n_utt == 0:
        return 0.0, 0.0

    n_reps = sum(utt.count('[/]') for utt in par_utterances)
    n_fillers = sum(
        len(re.findall(r'&(uh|um|ah|er|hm)\b', utt, re.IGNORECASE))
        for utt in par_utterances
    )

    total_words = 0
    for utt in par_utterances:
        cleaned = re.sub(r'\x15\d+_\d+\x15', '', utt)
        cleaned = re.sub(r'\d+_\d+', '', cleaned)
        cleaned = re.sub(r'\[.*?\]', '', cleaned)
        cleaned = re.sub(r'<[^>]*>', '', cleaned)
        cleaned = re.sub(r'[+<>]', '', cleaned)
        cleaned = re.sub(r'&\S+', '', cleaned)
        lex_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", cleaned)
        total_words += len(lex_words)

    repetition_rate = n_reps / n_utt
    filler_rate = n_fillers / total_words if total_words > 0 else 0.0
    return repetition_rate, filler_rate


# ============================================================
# 6.4  ASR error features
# ============================================================

def compute_deletion_rate(gold_text, asr_text):
    """Deletion rate from WER alignment (deletions / reference words)."""
    gold_norm = normalize_for_wer(gold_text)
    asr_norm = normalize_for_wer(asr_text)
    if not gold_norm:
        return np.nan
    n_ref = len(gold_norm.split())
    if n_ref == 0:
        return np.nan
    if not asr_norm:
        return 1.0
    result = process_words(gold_norm, asr_norm)
    n_del = sum(
        c.ref_end_idx - c.ref_start_idx
        for sent in result.alignments
        for c in sent if c.type == 'delete'
    )
    return n_del / n_ref


# ============================================================
# 6.5  Apply all features to DataFrame
# ============================================================

def get_cha_path(row, train_dir, test_dir):
    """Construct .cha file path from sample metadata."""
    if row.split == 'train':
        group = 'cc' if row.label == 0 else 'cd'
        return os.path.join(train_dir, 'transcription', group, f'{row.id}.cha')
    else:
        return os.path.join(test_dir, 'transcription', f'{row.id}.cha')


def extract_all_features(df, train_dir, test_dir, verbose=True):
    """Apply all 13 features to DataFrame. Modifies df in-place.

    Requires columns from ASR pipeline: speech_rate, pause_to_speech_ratio,
    n_long_pauses, mean_pause_dur, wer, asr_text.
    """
    required = ['speech_rate', 'pause_to_speech_ratio', 'n_long_pauses',
                'mean_pause_dur', 'wer', 'asr_text']
    missing = [c for c in required if c not in df.columns]
    assert not missing, f'ASR pipeline must run first. Missing columns: {missing}'

    if verbose:
        print('Extracting 13 features across 5 dimensions ...')

    # Semantic
    if verbose:
        print('  [1/5] Semantic features ...', end=' ', flush=True)
    df['n_content_units'] = df['text'].apply(count_content_units)
    df['word_count'] = df['text'].apply(lambda t: len(re.findall(r'\b\w+\b', t)))
    df['info_density'] = df['n_content_units'] / df['word_count']
    if verbose:
        print('done')

    # Lexical
    if verbose:
        print('  [2/5] Lexical features ...', end=' ', flush=True)
    df['pronoun_noun_ratio'] = df['text'].apply(pronoun_noun_ratio)
    df['mattr'] = df['text'].apply(compute_mattr)
    if verbose:
        print('done')

    # Disfluency
    if verbose:
        print('  [3/5] Disfluency features ...', end=' ', flush=True)
    rep_rates, filler_rates = [], []
    for _, row in df.iterrows():
        cha_path = get_cha_path(row, train_dir, test_dir)
        rr, fr = extract_disfluency_from_cha(cha_path)
        rep_rates.append(rr)
        filler_rates.append(fr)
    df['repetition_rate'] = rep_rates
    df['filler_rate'] = filler_rates
    if verbose:
        print('done')

    # Temporal (already in df from ASR pipeline)
    if verbose:
        print('  [4/5] Temporal features ... already in df')

    # ASR error
    if verbose:
        print('  [5/5] ASR error features ...', end=' ', flush=True)
    df['deletion_rate'] = df.apply(
        lambda row: compute_deletion_rate(row['text'], row['asr_text']), axis=1
    )
    if verbose:
        print('done')

    if verbose:
        print(f'\n  Feature matrix shape: {df[FEATURE_COLS].shape}')

    return df
