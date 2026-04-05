"""
data_loading.py — ADReSS CHAT file parser and data loader.

Parses DementiaBank .cha transcription files and builds a DataFrame
with columns: [id, text, label, age, gender, split, audio_path].
"""

import os
import re
import glob
import pandas as pd


# --- Default paths (override via function args if needed) ---
DEFAULT_TRAIN_DIR = 'Data/ADReSS-IS2020-train/ADReSS-IS2020-data/train'
DEFAULT_TEST_DIR = 'Data/ADReSS-IS2020-test/ADReSS-IS2020-data/test'


def clean_chat_utterance(text):
    """Clean CHAT annotations from a single *PAR: utterance.
    Keeps filled pauses (uh, um) as they are AD diagnostic signals."""
    text = re.sub(r'\x15', '', text)
    text = re.sub(r'\s*\d+_\d+\s*', ' ', text)
    text = re.sub(r'&=[a-z:_]+', '', text)
    text = re.sub(r'&(uh|um|ah|hm|er|oh)', r'\1', text)
    text = re.sub(r'&\w+', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\S*@\w+', '', text)
    text = re.sub(r'\bxxx\b', '', text)
    text = re.sub(r'\bww\b', '', text)
    text = re.sub(r'\(\.\.*\)', '', text)
    text = re.sub(r'\((\w+)\)', r'\1', text)
    text = re.sub(r'(\w)\+(\w)', r'\1\2', text)
    text = re.sub(r'\+\S*', '', text)
    text = re.sub(r'\S*Last_name\S*', '', text)
    text = re.sub(r'\S*First_name\S*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[.\s]+', '', text)
    return '' if text in ('', '.') else text


def parse_cha_file(filepath):
    """Parse .cha -> (text, age, gender, diagnosis)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    age, gender, diagnosis = None, None, None
    for line in lines:
        if line.startswith('@ID') and 'PAR' in line:
            parts = line.strip().split('|')
            if len(parts) >= 6:
                age_str = parts[3].replace(';', '').strip()
                age = int(age_str) if age_str.isdigit() else None
                gender = parts[4].strip() or None
                dx = parts[5].strip()
                diagnosis = 0 if dx == 'Control' else (1 if dx == 'ProbableAD' else None)
            break

    utterances, current = [], None
    for line in lines:
        line = line.rstrip('\n')
        if line.startswith('*PAR:'):
            if current is not None:
                utterances.append(current)
            current = line[5:].strip()
        elif line.startswith('\t') and current is not None:
            current += ' ' + line.strip()
        elif line.startswith(('*', '%', '@')):
            if current is not None:
                utterances.append(current)
                current = None
    if current:
        utterances.append(current)

    cleaned = [clean_chat_utterance(u) for u in utterances]
    return ' '.join(c for c in cleaned if c), age, gender, diagnosis


def load_adress_data(train_dir=None, test_dir=None):
    """Load full ADReSS dataset -> DataFrame [id, text, label, age, gender, split, audio_path]."""
    train_dir = train_dir or DEFAULT_TRAIN_DIR
    test_dir = test_dir or DEFAULT_TEST_DIR
    records = []

    # Train (labels from folder: cc=0, cd=1)
    for group, label in [('cc', 0), ('cd', 1)]:
        for f in sorted(glob.glob(os.path.join(train_dir, 'transcription', group, '*.cha'))):
            sid = os.path.splitext(os.path.basename(f))[0]
            text, age, gender, _ = parse_cha_file(f)
            audio = os.path.join(train_dir, 'Full_wave_enhanced_audio', group, f'{sid}.wav')
            records.append(dict(id=sid, text=text, label=label, age=age,
                                gender=gender, split='train', audio_path=audio))

    # Test (no cc/cd split, labels hidden)
    meta = {}
    mp = os.path.join(test_dir, 'meta_data.txt')
    if os.path.exists(mp):
        for line in open(mp):
            p = [x.strip() for x in line.strip().split(';')]
            if len(p) >= 3:
                meta[p[0]] = {'age': int(p[1]), 'gender': p[2]}

    for f in sorted(glob.glob(os.path.join(test_dir, 'transcription', '*.cha'))):
        sid = os.path.splitext(os.path.basename(f))[0]
        text, age, gender, dx = parse_cha_file(f)
        m = meta.get(sid, {})
        audio = os.path.join(test_dir, 'Full_wave_enhanced_audio', f'{sid}.wav')
        records.append(dict(id=sid, text=text, label=dx,
                            age=m.get('age', age), gender=m.get('gender', gender),
                            split='test', audio_path=audio))

    return pd.DataFrame(records)
