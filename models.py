"""
models.py — All model definitions and training for MIE1517 project.

Models:
  1. MLP (13 -> 32 -> 1) with multi-seed averaging and 5-fold CV
  2. BiLSTM with GloVe embeddings (from teammate's Main (2).ipynb)
  3a. Frozen DistilBERT + Logistic Regression
  3b. Fine-tuned DistilBERT (last 2 layers unfrozen)
"""

import os
import re
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from collections import Counter


# ══════════════════════════════════════════════════════════════
# MODEL 1: MLP
# ══════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Single hidden layer MLP. 481 parameters for 13 input features (hidden=32)."""

    def __init__(self, input_dim=13, hidden=32, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


DEFAULT_CONFIG = dict(
    seed=42,
    n_seeds=5,
    n_epochs=200,
    patience=15,
    lr=1e-3,
    batch_size=16,
    wd_grid=[0.01, 0.1, 0.5],
    dropout_grid=[0.3, 0.4, 0.5],
)


def train_mlp_cv(wd, dropout, X_all, y_all, X_test=None, verbose=False,
                 save_weights=False, weight_dir=None,
                 seed=42, n_seeds=5, n_epochs=200, patience=15, lr=1e-3, batch_size=16):
    """
    Run full 5-fold x N_SEEDS training with mini-batch SGD.

    If X_test is provided, each fold×seed model also predicts test.
    If save_weights=True, saves each model to weight_dir.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_logits = np.zeros(len(X_all), dtype=np.float32)
    oof_preds = np.zeros(len(X_all), dtype=np.int64)
    if X_test is not None:
        test_logits_all = np.zeros((5, n_seeds, X_test.shape[0]), dtype=np.float32)
    fold_accs, fold_val_losses = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[tr_idx])
        X_va = scaler.transform(X_all[va_idx])
        if X_test is not None:
            X_te = scaler.transform(X_test)
            X_te_t = torch.tensor(X_te, dtype=torch.float32)
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]
        X_va_t = torch.tensor(X_va, dtype=torch.float32)
        y_va_t = torch.tensor(y_va, dtype=torch.float32)
        train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                 torch.tensor(y_tr, dtype=torch.float32))
        seed_logits = np.zeros((n_seeds, len(va_idx)), dtype=np.float32)
        seed_vl = []

        for s in range(n_seeds):
            s_seed = seed + fold * 100 + s
            torch.manual_seed(s_seed)
            g = torch.Generator().manual_seed(s_seed)
            loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
            model = MLP(input_dim=X_all.shape[1], dropout=dropout)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            crit = nn.BCEWithLogitsLoss()
            best_vl, patience_ctr, best_state = float('inf'), 0, None

            for epoch in range(n_epochs):
                model.train()
                for xb, yb in loader:
                    opt.zero_grad()
                    loss = crit(model(xb), yb)
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    vl = crit(model(X_va_t), y_va_t).item()
                if vl < best_vl:
                    best_vl, patience_ctr = vl, 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        break

            seed_vl.append(best_vl)
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                seed_logits[s] = model(X_va_t).numpy()
                if X_test is not None:
                    test_logits_all[fold, s] = model(X_te_t).numpy()
            if save_weights and weight_dir:
                torch.save(best_state, os.path.join(weight_dir, f'mlp_fold{fold}_seed{s}.pt'))

        oof_logits[va_idx] = seed_logits.mean(axis=0)
        oof_preds[va_idx] = (oof_logits[va_idx] > 0).astype(int)
        fold_accs.append((oof_preds[va_idx] == y_va).mean())
        fold_val_losses.append(np.mean(seed_vl))
        if verbose:
            print(f'  Fold {fold + 1}: acc={fold_accs[-1]:.3f}, val_loss={fold_val_losses[-1]:.4f}')

    oof_probs = 1 / (1 + np.exp(-oof_logits))
    result = {
        'oof_logits': oof_logits,
        'oof_preds': oof_preds,
        'oof_acc': accuracy_score(y_all, oof_preds),
        'oof_f1': f1_score(y_all, oof_preds),
        'oof_brier': brier_score_loss(y_all, oof_probs),
        'fold_accs': fold_accs,
    }
    if X_test is not None:
        result['test_logits'] = test_logits_all.mean(axis=(0, 1))
    return result


def grid_search_mlp(X_all, y_all, wd_grid=None, dropout_grid=None, **kwargs):
    """Run grid search over weight_decay x dropout. Returns sorted results."""
    wd_grid = wd_grid or DEFAULT_CONFIG['wd_grid']
    dropout_grid = dropout_grid or DEFAULT_CONFIG['dropout_grid']

    print(f'MLP Grid Search: {len(wd_grid)}x{len(dropout_grid)} combos')
    print(f'Architecture: {X_all.shape[1]}->32->1 '
          f'({sum(p.numel() for p in MLP(input_dim=X_all.shape[1]).parameters())} params)')

    grid_results = []
    for wd in wd_grid:
        for dp in dropout_grid:
            res = train_mlp_cv(wd, dp, X_all, y_all, **kwargs)
            grid_results.append({'wd': wd, 'dropout': dp, **res})
            print(f'  wd={wd:.2f}, dropout={dp:.1f} -> '
                  f'acc={res["oof_acc"]:.3f}, brier={res["oof_brier"]:.4f}')

    grid_results.sort(key=lambda r: r['oof_brier'])
    best = grid_results[0]
    print(f'\nBest: wd={best["wd"]:.2f}, dropout={best["dropout"]:.1f} '
          f'(Brier={best["oof_brier"]:.4f})')
    return grid_results


# ══════════════════════════════════════════════════════════════
# MODEL 2: BiLSTM with GloVe (from teammate's Main (2).ipynb)
# ══════════════════════════════════════════════════════════════

class BiLSTMClassifier(nn.Module):
    """BiLSTM with frozen GloVe embeddings for binary classification."""

    def __init__(self, embedding_matrix, hidden_dim=128, fc_dim=64, dropout=0.5):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_matrix,
            freeze=True
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, fc_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        forward_hidden = hidden[0]
        backward_hidden = hidden[1]
        combined = torch.cat((forward_hidden, backward_hidden), dim=1)
        x = self.dropout(combined)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x).squeeze(1)


class TextDataset(Dataset):
    """Simple dataset for tokenized text sequences."""
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def tokenize(text):
    """Lowercase + keep contractions (matches teammate's tokenizer)."""
    text = text.lower().strip()
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


def build_vocab(texts, min_freq=2):
    """Build vocabulary from texts with min frequency filter."""
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab_words = [word for word, freq in counter.items() if freq >= min_freq]
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for word in sorted(vocab_words):
        word2idx[word] = len(word2idx)
    return word2idx


def load_glove_embeddings(glove_path, word2idx, embed_dim=300):
    """Load GloVe vectors and build embedding matrix."""
    glove = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove[word] = vector

    vocab_size = len(word2idx)
    embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    hits, misses = 0, 0
    for word, idx in word2idx.items():
        if word in glove:
            embedding_matrix[idx] = glove[word]
            hits += 1
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embed_dim,))
            misses += 1
    print(f'GloVe: {hits} found, {misses} missed out of {vocab_size}')
    return torch.tensor(embedding_matrix, dtype=torch.float32)


def encode_and_pad(texts, word2idx, max_len):
    """Tokenize, encode, and pad texts to fixed length."""
    seqs = []
    for text in texts:
        tokens = tokenize(text)
        seq = [word2idx.get(tok, word2idx['<UNK>']) for tok in tokens]
        if len(seq) >= max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [0] * (max_len - len(seq))
        seqs.append(seq)
    return np.array(seqs, dtype=np.int64)


def run_bilstm_epoch(model, loader, criterion, optimizer=None, device='cpu'):
    """Run one epoch (train or eval). Returns loss, acc, logits, labels."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []

    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch_y.detach().cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss = total_loss / len(loader.dataset)
    preds = (all_logits > 0).astype(int)
    acc = accuracy_score(all_labels.astype(int), preds)
    return avg_loss, acc, all_logits, all_labels


# ══════════════════════════════════════════════════════════════
# MODEL 3a: Frozen DistilBERT + Logistic Regression
# ══════════════════════════════════════════════════════════════

class ADRDataset(Dataset):
    """Tokenized dataset for DistilBERT."""
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(texts, truncation=True, padding=True,
                                   max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


def extract_cls_embeddings(texts, bert_model, tokenizer, max_len=256,
                           batch_size=16, device='cpu'):
    """Extract [CLS] embeddings from frozen DistilBERT."""
    dataset = ADRDataset(texts, [0] * len(texts), tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size)
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            cls = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls.cpu().numpy())
    return np.vstack(embeddings)


# ══════════════════════════════════════════════════════════════
# MODEL 3b: Fine-tuned DistilBERT
# ══════════════════════════════════════════════════════════════

class LabeledTextDataset(Dataset):
    """Tokenized dataset for fine-tuned DistilBERT."""
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(texts, truncation=True, padding=True,
                                   max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx],
        }


def train_one_epoch(model, loader, optimizer, device='cpu'):
    """Train one epoch for fine-tuned DistilBERT."""
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['labels'].to(device),
        )
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()
    return total_loss / len(loader)


def evaluate_distilbert(model, loader, device='cpu'):
    """Evaluate fine-tuned DistilBERT. Returns (labels, preds, probs, logits)."""
    model.eval()
    all_preds, all_probs, all_labels, all_logits = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
            )
            logits = outputs.logits  # shape: [batch, 2]
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_logits.append(logits.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].numpy())
    return (np.array(all_labels), np.array(all_preds),
            np.array(all_probs), np.vstack(all_logits))
