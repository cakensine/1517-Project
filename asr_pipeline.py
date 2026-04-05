"""
asr_pipeline.py — WhisperX ASR pipeline with PAR-aware processing.

Pipeline:
  0. PAR-aware audio normalization (zero INV regions + RMS gain)
  1. WhisperX transcribe (large-v3-turbo, batched)
  2. Forced alignment (wav2vec2 -> word-level timestamps)
  3. Filter words by overlap with PAR time ranges from .cha files
  4. Compute pause/timing statistics from word timestamps
"""

import re
import numpy as np


def normalize_audio_par_aware(audio, par_ranges, sample_rate,
                              target_rms_db=-25.0, max_gain_db=40.0,
                              buffer_s=0.2):
    """Zero out INV regions and apply RMS gain to PAR speech.

    Parameters:
        audio: numpy float32 array (raw waveform)
        par_ranges: list of (start_sec, end_sec) tuples
        sample_rate: audio sample rate (e.g. 16000)
        target_rms_db: target RMS level in dB
        max_gain_db: maximum gain to apply
        buffer_s: seconds of buffer around PAR regions

    Returns:
        (audio, gain_db): normalized array and applied gain in dB.
    """
    n_samples = len(audio)
    mask = np.zeros(n_samples, dtype=bool)
    for ps, pe in par_ranges:
        i_start = max(0, int((ps - buffer_s) * sample_rate))
        i_end = min(n_samples, int((pe + buffer_s) * sample_rate))
        mask[i_start:i_end] = True

    audio[~mask] = 0.0

    par_samples = audio[mask]
    gain_db = 0.0
    if len(par_samples) > 0:
        par_rms = np.sqrt(np.mean(par_samples ** 2))
        if par_rms > 0:
            par_rms_db = 20.0 * np.log10(par_rms)
            gain_db = float(np.clip(target_rms_db - par_rms_db, 0.0, max_gain_db))
            gain_linear = 10.0 ** (gain_db / 20.0)
            audio *= gain_linear
            audio = np.clip(audio, -1.0, 1.0)

    return audio, gain_db


def extract_par_timestamps(cha_path):
    """Extract participant (*PAR:) time ranges from .cha file."""
    with open(cha_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    par_ranges, current_speaker, current_text = [], None, ''
    for line in lines:
        line = line.rstrip('\n')
        if line.startswith('*'):
            if current_speaker == '*PAR':
                ts = re.findall(r'(\d+)_(\d+)', current_text)
                if ts:
                    par_ranges.append((int(ts[-1][0]) / 1000, int(ts[-1][1]) / 1000))
            current_speaker = line.split(':')[0]
            current_text = line
        elif line.startswith('\t') and current_speaker:
            current_text += ' ' + line.strip()
        elif line.startswith('%'):
            if current_speaker == '*PAR':
                ts = re.findall(r'(\d+)_(\d+)', current_text)
                if ts:
                    par_ranges.append((int(ts[-1][0]) / 1000, int(ts[-1][1]) / 1000))
            current_speaker, current_text = None, ''
    if current_speaker == '*PAR':
        ts = re.findall(r'(\d+)_(\d+)', current_text)
        if ts:
            par_ranges.append((int(ts[-1][0]) / 1000, int(ts[-1][1]) / 1000))
    return par_ranges


def filter_par_words(segments, par_ranges, threshold=0.5):
    """Filter ASR words to PAR-only using timestamp overlap.

    Returns:
        par_text: all PAR words as text (for WER / models)
        par_words: list of {word, start, end} dicts with timestamps (for timing)
    """
    par_words = []
    all_par_tokens = []

    for seg in segments:
        seg_start = seg.get('start', 0.0)
        seg_end = seg.get('end', 0.0)
        seg_dur = seg_end - seg_start
        if seg_dur > 0:
            seg_overlap = sum(
                max(0.0, min(seg_end, pe) - max(seg_start, ps))
                for ps, pe in par_ranges
            )
            seg_is_par = (seg_overlap / seg_dur) >= threshold
        else:
            seg_is_par = False

        for word in seg.get('words', []):
            ws = word.get('start')
            we = word.get('end')

            if ws is not None and we is not None:
                word_dur = we - ws
                if word_dur <= 0:
                    mid = ws
                    if any(ps <= mid <= pe for ps, pe in par_ranges):
                        par_words.append({'word': word['word'], 'start': ws, 'end': we})
                        all_par_tokens.append(word['word'])
                    continue
                total_overlap = sum(
                    max(0.0, min(we, pe) - max(ws, ps))
                    for ps, pe in par_ranges
                )
                if (total_overlap / word_dur) >= threshold:
                    par_words.append({'word': word['word'], 'start': ws, 'end': we})
                    all_par_tokens.append(word['word'])
            else:
                if seg_is_par:
                    all_par_tokens.append(word['word'])

    par_text = ' '.join(all_par_tokens).strip()
    return par_text, par_words


def compute_pause_stats(par_words, short_threshold=0.5, long_threshold=1.0):
    """Compute pause statistics from word-level timestamps."""
    if len(par_words) < 2:
        return dict(n_pauses=0, n_long_pauses=0, total_pause_dur=0.0,
                    mean_pause_dur=0.0, total_speech_dur=0.0,
                    speech_rate=0.0, pause_to_speech_ratio=0.0)

    gaps = []
    for i in range(1, len(par_words)):
        gap = par_words[i]['start'] - par_words[i - 1]['end']
        if gap >= short_threshold:
            gaps.append(gap)

    MAX_WORD_DUR = 2.0
    total_speech = sum(min(w['end'] - w['start'], MAX_WORD_DUR) for w in par_words)
    total_pause = sum(gaps)

    return dict(
        n_pauses=len(gaps),
        n_long_pauses=sum(1 for g in gaps if g >= long_threshold),
        total_pause_dur=round(total_pause, 3),
        mean_pause_dur=round(total_pause / len(gaps), 3) if gaps else 0.0,
        total_speech_dur=round(total_speech, 3),
        speech_rate=round(len(par_words) / total_speech, 2) if total_speech > 0 else 0.0,
        pause_to_speech_ratio=round(total_pause / total_speech, 3) if total_speech > 0 else 0.0,
    )


def normalize_for_wer(text):
    """Normalize for fair WER: lowercase, remove punctuation, collapse spaces."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()
