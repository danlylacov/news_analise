import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch

from nn_model import NewsTickerModel
from nn_data import load_vocab, encode_text


def load_artifacts(artifacts: str):
    with open(os.path.join(artifacts, 'tickers.json'), 'r', encoding='utf-8') as f:
        ticker_to_idx = json.load(f)['ticker_to_idx']
    vocab = load_vocab(os.path.join(artifacts, 'vocab.json'))
    ckpt = torch.load(os.path.join(artifacts, 'model.pt'), map_location='cpu')
    return ticker_to_idx, vocab, ckpt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def score_news(df_news: pd.DataFrame, vocab, model_state, num_labels: int, max_len: int = 256, batch_size: int = 256) -> np.ndarray:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NewsTickerModel(vocab_size=len(vocab), num_labels=num_labels)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    scores = []
    with torch.no_grad():
        for i in range(0, len(df_news), batch_size):
            batch = df_news.iloc[i:i+batch_size]
            texts = (batch['title'].fillna('') + ' [SEP] ' + batch['publication'].fillna('')).tolist()
            ids = [encode_text(t, vocab, max_len) for t in texts]
            maxL = max(len(x) for x in ids) if ids else 1
            input_ids = torch.zeros((len(ids), maxL), dtype=torch.long)
            attention_mask = torch.zeros((len(ids), maxL), dtype=torch.long)
            for j, seq in enumerate(ids):
                L = len(seq)
                input_ids[j, :L] = torch.tensor(seq, dtype=torch.long)
                attention_mask[j, :L] = 1
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask).cpu().numpy()
            scores.append(sigmoid(logits))
    return np.vstack(scores) if scores else np.zeros((0, num_labels))


def aggregate_to_candles(df_candles: pd.DataFrame, df_news: pd.DataFrame, scores: np.ndarray, ticker_to_idx: dict, half_life_days: float = 2.0) -> pd.DataFrame:
    idx_to_ticker = {v: k for k, v in ticker_to_idx.items()}
    df_news = df_news.copy()
    df_news['publish_date'] = pd.to_datetime(df_news['publish_date'], errors='coerce')
    df_news['date'] = df_news['publish_date'].dt.date

    df_candles = df_candles.copy()
    df_candles['begin'] = pd.to_datetime(df_candles['begin'], errors='coerce')
    df_candles['date'] = df_candles['begin'].dt.date

    decay_lambda = math.log(2) / max(half_life_days, 1e-3)

    features = []
    news_dates = pd.to_datetime(df_news['date']).values

    for ticker, g in df_candles.groupby('ticker'):
        if ticker not in ticker_to_idx:
            continue
        label_idx = ticker_to_idx[ticker]
        probs = scores[:, label_idx]
        for dt, sub in g.groupby('date'):
            mask = news_dates <= np.datetime64(dt)
            if not mask.any():
                features.append({'ticker': ticker, 'date': dt, 'nn_news_sum': 0.0, 'nn_news_mean': 0.0, 'nn_news_count': 0})
            else:
                deltas = (pd.to_datetime(dt) - pd.to_datetime(news_dates[mask])).days.astype('int64')
                weights = np.exp(-decay_lambda * deltas)
                wprobs = probs[mask] * weights
                cnt = int(mask.sum())
                ssum = float(wprobs.sum())
                smean = float(wprobs.mean()) if cnt > 0 else 0.0
                features.append({'ticker': ticker, 'date': dt, 'nn_news_sum': ssum, 'nn_news_mean': smean, 'nn_news_count': cnt})
    return pd.DataFrame(features)


def main():
    parser = argparse.ArgumentParser(description='Инференс: связь новости→тикеры и агрегация по свечам')
    parser.add_argument('--news', required=True, help='task_1_news.csv')
    parser.add_argument('--candles', required=True, help='task_1_candles.csv')
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    ticker_to_idx, vocab, ckpt = load_artifacts(args.artifacts)

    df_news = pd.read_csv(args.news)
    df_candles = pd.read_csv(args.candles)

    scores = score_news(df_news, vocab, ckpt['state_dict'], num_labels=len(ticker_to_idx), max_len=ckpt['config'].get('max_len', 256))
    feats = aggregate_to_candles(df_candles, df_news, scores, ticker_to_idx)

    if args.out.endswith('.parquet'):
        feats.to_parquet(args.out, index=False)
    else:
        feats.to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
