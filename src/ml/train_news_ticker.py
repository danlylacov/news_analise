import os
import argparse
import math
import json
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.ml.nn_model import NewsTickerModel
from src.ml.nn_data import NewsDataset, build_vocab, save_vocab, load_vocab, collate_batch


def compute_pos_weight(train_df: pd.DataFrame, ticker_to_idx: Dict[str, int]) -> torch.Tensor:
    counts = np.zeros(len(ticker_to_idx), dtype=np.float64)
    for row in train_df['tickers'].fillna(""):
        seen = set()
        for t in str(row).split(','):
            t = t.strip()
            if t and t in ticker_to_idx and t not in seen:
                counts[ticker_to_idx[t]] += 1
                seen.add(t)
    total = len(train_df) + 1e-6
    pos_weight = (total - counts) / (counts + 1e-6)
    pos_weight = np.clip(pos_weight, 1.0, 10.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def train(args):
    os.makedirs(args.artifacts, exist_ok=True)
    df = pd.read_csv(args.news)
    # тикеры словарь
    all_tickers = sorted({t.strip() for s in df['tickers'].fillna("") for t in str(s).replace(';', ',').split(',') if t.strip()})
    ticker_to_idx = {t: i for i, t in enumerate(all_tickers)}
    with open(os.path.join(args.artifacts, 'tickers.json'), 'w', encoding='utf-8') as f:
        json.dump({"ticker_to_idx": ticker_to_idx}, f, ensure_ascii=False)

    # словарь
    texts = (df['title'].fillna('') + ' ' + df['publication'].fillna('')).tolist()
    vocab = build_vocab(texts, min_freq=3, max_size=args.vocab_size)
    save_vocab(os.path.join(args.artifacts, 'vocab.json'), vocab)

    # сплит
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df)
    val_n = int(0.1 * n)
    train_df, val_df = df.iloc[val_n:], df.iloc[:val_n]

    train_ds = NewsDataset(train_df, vocab, ticker_to_idx, max_len=args.max_len, mode='train')
    val_ds = NewsDataset(val_df, vocab, ticker_to_idx, max_len=args.max_len, mode='train')
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, pad_id=0, mode='train'))
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_batch(b, pad_id=0, mode='train'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NewsTickerModel(vocab_size=len(vocab), num_labels=len(ticker_to_idx), embed_dim=args.embed_dim, rnn_hidden=args.hidden, dropout=args.dropout).to(device)
    opt = AdamW(model.parameters(), lr=args.lr)
    pos_weight = compute_pos_weight(train_df, ticker_to_idx).to(device)

    best_val = float('inf')
    patience = args.patience

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            loss = model.bce_with_logits_loss(logits, labels, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * input_ids.size(0)
        train_loss = total_loss / len(train_ds)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids, attention_mask)
                loss = model.bce_with_logits_loss(logits, labels, pos_weight=pos_weight)
                total_val += loss.item() * input_ids.size(0)
        val_loss = total_val / len(val_ds)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            torch.save({'state_dict': model.state_dict(), 'vocab': vocab, 'ticker_to_idx': ticker_to_idx, 'config': vars(args)}, os.path.join(args.artifacts, 'model.pt'))
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                print('Early stopping')
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--news', required=True)
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=2)
    args = parser.parse_args()
    train(args)
