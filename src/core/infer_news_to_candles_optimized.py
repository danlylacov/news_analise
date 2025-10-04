import argparse
import json
import math
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from src.ml.nn_model import NewsTickerModel
from src.ml.nn_data import load_vocab, encode_text


# Глобальные переменные для кэширования
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model_cache = {}


@lru_cache(maxsize=1)
def load_artifacts_cached(artifacts: str):
    """Кэшированная загрузка артефактов модели"""
    with open(os.path.join(artifacts, 'tickers.json'), 'r', encoding='utf-8') as f:
        ticker_to_idx = json.load(f)['ticker_to_idx']
    vocab = load_vocab(os.path.join(artifacts, 'vocab.json'))
    ckpt = torch.load(os.path.join(artifacts, 'model.pt'), map_location='cpu')
    
    # Создаем и загружаем модель один раз
    model = NewsTickerModel(vocab_size=len(vocab), num_labels=len(ticker_to_idx))
    model.load_state_dict(ckpt['state_dict'])
    model.to(_device)
    model.eval()
    
    return {
        'ticker_to_idx': ticker_to_idx,
        'vocab': vocab,
        'model': model,
        'config': ckpt.get('config', {})
    }


def score_news_optimized(df_news: pd.DataFrame, artifacts_data: dict, batch_size: int = 512) -> np.ndarray:
    """Оптимизированная функция скоринга новостей с кэшированной моделью"""
    model = artifacts_data['model']
    vocab = artifacts_data['vocab']
    max_len = artifacts_data['config'].get('max_len', 256)
    
    scores = []
    with torch.no_grad():
        for i in range(0, len(df_news), batch_size):
            batch = df_news.iloc[i:i+batch_size]
            texts = (batch['title'].fillna('') + ' [SEP] ' + batch['publication'].fillna('')).tolist()
            
            # Векторизованная токенизация
            ids = [encode_text(t, vocab, max_len) for t in texts]
            maxL = max(len(x) for x in ids) if ids else 1
            
            # Создаем тензоры более эффективно
            batch_size_actual = len(ids)
            input_ids = torch.zeros((batch_size_actual, maxL), dtype=torch.long, device=_device)
            attention_mask = torch.zeros((batch_size_actual, maxL), dtype=torch.long, device=_device)
            
            for j, seq in enumerate(ids):
                L = len(seq)
                input_ids[j, :L] = torch.tensor(seq, dtype=torch.long)
                attention_mask[j, :L] = 1
            
            logits = model(input_ids, attention_mask)
            scores.append(torch.sigmoid(logits).cpu().numpy())
    
    return np.vstack(scores) if scores else np.zeros((0, len(artifacts_data['ticker_to_idx'])))


def aggregate_to_candles_optimized(
    df_candles: pd.DataFrame,
    df_news: pd.DataFrame,
    scores: np.ndarray,
    ticker_to_idx: dict,
    half_life_days: float = 2.0,
    p_threshold: float = 0.5,
    max_days: float = 20.0,
) -> pd.DataFrame:
    """Оптимизированная агрегация новостей к свечам"""
    scores = np.asarray(scores)
    df_news = df_news.copy()
    df_news['publish_date'] = pd.to_datetime(df_news['publish_date'], errors='coerce')
    df_news['date'] = df_news['publish_date'].dt.date

    df_candles = df_candles.copy()
    df_candles['begin'] = pd.to_datetime(df_candles['begin'], errors='coerce')
    df_candles['date'] = df_candles['begin'].dt.date

    news_dates = pd.to_datetime(df_news['date']).values.astype('datetime64[D]')
    decay_lambda = math.log(2) / max(half_life_days, 1e-6)
    max_days = float(max_days) if max_days is not None else np.inf

    features = []
    
    # Предварительно фильтруем тикеры
    valid_tickers = set(ticker_to_idx.keys())
    
    for ticker, g in df_candles.groupby('ticker'):
        if ticker not in valid_tickers:
            continue
            
        label_idx = ticker_to_idx[ticker]
        probs = np.asarray(scores[:, label_idx], dtype=np.float64)
        
        for dt, sub in g.groupby('date'):
            dt64 = np.datetime64(dt)
            
            # Оптимизированное окно времени
            if np.isfinite(max_days):
                min_dt64 = dt64 - np.timedelta64(int(max_days), 'D')
                base_mask = (news_dates <= dt64) & (news_dates >= min_dt64)
            else:
                base_mask = (news_dates <= dt64)
                
            if not base_mask.any():
                features.append({
                    'ticker': ticker, 'date': dt, 
                    'nn_news_sum': 0.0, 'nn_news_mean': 0.0, 
                    'nn_news_max': 0.0, 'nn_news_count': 0
                })
                continue
                
            # Релевантность по порогу
            mask_thr = base_mask & (probs >= p_threshold)
            if not mask_thr.any():
                features.append({
                    'ticker': ticker, 'date': dt, 
                    'nn_news_sum': 0.0, 'nn_news_mean': 0.0, 
                    'nn_news_max': 0.0, 'nn_news_count': 0
                })
                continue
                
            # Векторизованное вычисление затухания
            deltas = (dt64 - news_dates[mask_thr]).astype('timedelta64[D]').astype(int)
            weights = np.exp(-decay_lambda * deltas.astype(np.float64))
            vals = probs[mask_thr]
            wvals = vals * weights
            
            cnt = int(mask_thr.sum())
            ssum = float(np.sum(wvals))
            smean = float(np.mean(wvals)) if cnt > 0 else 0.0
            smax = float(np.max(wvals)) if cnt > 0 else 0.0
            
            features.append({
                'ticker': ticker, 'date': dt, 
                'nn_news_sum': ssum, 'nn_news_mean': smean, 
                'nn_news_max': smax, 'nn_news_count': cnt
            })
    
    return pd.DataFrame(features)


def infer_news_to_candles_df_optimized(news_df: pd.DataFrame, candles_df: pd.DataFrame, artifacts_dir: str, 
                                      p_threshold: float = 0.5, half_life_days: float = 0.5, max_days: int = 5) -> tuple:
    """Оптимизированная основная функция для инференса новостей с DataFrame входом"""
    artifacts_data = load_artifacts_cached(artifacts_dir)
    
    scores = score_news_optimized(news_df, artifacts_data)
    
    features_df = aggregate_to_candles_optimized(
        candles_df, news_df, scores, artifacts_data['ticker_to_idx'],
        half_life_days=half_life_days,
        p_threshold=p_threshold,
        max_days=max_days,
    )
    
    # Объединяем свечи с фичами
    candles_df_copy = candles_df.copy()
    candles_df_copy['date'] = pd.to_datetime(candles_df_copy['begin'], errors='coerce').dt.date
    features_df_copy = features_df.copy()
    
    joined_df = candles_df_copy.merge(features_df_copy, on=['ticker', 'date'], how='left')
    
    # Заполняем пропуски нулями
    feature_cols = ['nn_news_sum', 'nn_news_mean', 'nn_news_max', 'nn_news_count']
    for col in feature_cols:
        if col in joined_df.columns:
            joined_df[col] = joined_df[col].fillna(0.0)
    
    return features_df, joined_df


# Обратная совместимость
def load_artifacts(artifacts: str):
    """Оригинальная функция для обратной совместимости"""
    with open(os.path.join(artifacts, 'tickers.json'), 'r', encoding='utf-8') as f:
        ticker_to_idx = json.load(f)['ticker_to_idx']
    vocab = load_vocab(os.path.join(artifacts, 'vocab.json'))
    ckpt = torch.load(os.path.join(artifacts, 'model.pt'), map_location='cpu')
    return ticker_to_idx, vocab, ckpt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def score_news(df_news: pd.DataFrame, vocab, model_state, num_labels: int, max_len: int = 256, batch_size: int = 256) -> np.ndarray:
    """Оригинальная функция скоринга для обратной совместимости"""
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


def aggregate_to_candles(
    df_candles: pd.DataFrame,
    df_news: pd.DataFrame,
    scores: np.ndarray,
    ticker_to_idx: dict,
    half_life_days: float = 2.0,
    p_threshold: float = 0.5,
    max_days: float = 20.0,
) -> pd.DataFrame:
    """Оригинальная функция агрегации для обратной совместимости"""
    scores = np.asarray(scores)
    df_news = df_news.copy()
    df_news['publish_date'] = pd.to_datetime(df_news['publish_date'], errors='coerce')
    df_news['date'] = df_news['publish_date'].dt.date

    df_candles = df_candles.copy()
    df_candles['begin'] = pd.to_datetime(df_candles['begin'], errors='coerce')
    df_candles['date'] = df_candles['begin'].dt.date

    news_dates = pd.to_datetime(df_news['date']).values.astype('datetime64[D]')

    decay_lambda = math.log(2) / max(half_life_days, 1e-6)
    max_days = float(max_days) if max_days is not None else np.inf

    features = []
    for ticker, g in df_candles.groupby('ticker'):
        if ticker not in ticker_to_idx:
            continue
        label_idx = ticker_to_idx[ticker]
        probs = np.asarray(scores[:, label_idx], dtype=np.float64)
        for dt, sub in g.groupby('date'):
            dt64 = np.datetime64(dt)
            # окно: новости за последние max_days и не позже даты свечи
            if np.isfinite(max_days):
                min_dt64 = dt64 - np.timedelta64(int(max_days), 'D')
                base_mask = (news_dates <= dt64) & (news_dates >= min_dt64)
            else:
                base_mask = (news_dates <= dt64)
            if not base_mask.any():
                features.append({'ticker': ticker, 'date': dt, 'nn_news_sum': 0.0, 'nn_news_mean': 0.0, 'nn_news_max': 0.0, 'nn_news_count': 0})
                continue
            # релевантность по порогу
            mask_thr = base_mask & (probs >= p_threshold)
            if not mask_thr.any():
                features.append({'ticker': ticker, 'date': dt, 'nn_news_sum': 0.0, 'nn_news_mean': 0.0, 'nn_news_max': 0.0, 'nn_news_count': 0})
                continue
            # затухание
            deltas = (dt64 - news_dates[mask_thr]).astype('timedelta64[D]').astype(int)
            weights = np.exp(-decay_lambda * deltas.astype(np.float64))
            vals = probs[mask_thr]
            wvals = vals * weights
            cnt = int(mask_thr.sum())
            ssum = float(np.sum(wvals))
            smean = float(np.mean(wvals)) if cnt > 0 else 0.0
            smax = float(np.max(wvals)) if cnt > 0 else 0.0
            features.append({'ticker': ticker, 'date': dt, 'nn_news_sum': ssum, 'nn_news_mean': smean, 'nn_news_max': smax, 'nn_news_count': cnt})
    return pd.DataFrame(features)


def infer_news_to_candles_df(news_df: pd.DataFrame, candles_df: pd.DataFrame, artifacts_dir: str, 
                            p_threshold: float = 0.5, half_life_days: float = 0.5, max_days: int = 5) -> tuple:
    """Оригинальная основная функция для обратной совместимости"""
    ticker_to_idx, vocab, ckpt = load_artifacts(artifacts_dir)
    
    scores = score_news(news_df, vocab, ckpt['state_dict'], num_labels=len(ticker_to_idx), 
                       max_len=ckpt['config'].get('max_len', 256))
    
    features_df = aggregate_to_candles(
        candles_df, news_df, scores, ticker_to_idx,
        half_life_days=half_life_days,
        p_threshold=p_threshold,
        max_days=max_days,
    )
    
    # Объединяем свечи с фичами
    candles_df_copy = candles_df.copy()
    candles_df_copy['date'] = pd.to_datetime(candles_df_copy['begin'], errors='coerce').dt.date
    features_df_copy = features_df.copy()
    
    joined_df = candles_df_copy.merge(features_df_copy, on=['ticker', 'date'], how='left')
    
    # Заполняем пропуски нулями
    feature_cols = ['nn_news_sum', 'nn_news_mean', 'nn_news_max', 'nn_news_count']
    for col in feature_cols:
        if col in joined_df.columns:
            joined_df[col] = joined_df[col].fillna(0.0)
    
    return features_df, joined_df


def main():
    parser = argparse.ArgumentParser(description='Инференс: связь новости→тикеры и агрегация по свечам')
    parser.add_argument('--news', required=True, help='task_1_news.csv')
    parser.add_argument('--candles', required=True, help='task_1_candles.csv')
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--out', required=True)
    parser.add_argument('--half_life_days', type=float, default=2.0)
    parser.add_argument('--p_threshold', type=float, default=0.5)
    parser.add_argument('--max_days', type=float, default=20.0)
    parser.add_argument('--optimized', action='store_true', help='Использовать оптимизированную версию')
    args = parser.parse_args()

    if args.optimized:
        # Используем оптимизированную версию
        df_news = pd.read_csv(args.news)
        df_candles = pd.read_csv(args.candles)
        
        features_df, joined_df = infer_news_to_candles_df_optimized(
            df_news, df_candles, args.artifacts,
            half_life_days=args.half_life_days,
            p_threshold=args.p_threshold,
            max_days=args.max_days
        )
    else:
        # Используем оригинальную версию
        ticker_to_idx, vocab, ckpt = load_artifacts(args.artifacts)
        df_news = pd.read_csv(args.news)
        df_candles = pd.read_csv(args.candles)

        scores = score_news(df_news, vocab, ckpt['state_dict'], num_labels=len(ticker_to_idx), max_len=ckpt['config'].get('max_len', 256))
        features_df = aggregate_to_candles(
            df_candles, df_news, scores, ticker_to_idx,
            half_life_days=args.half_life_days,
            p_threshold=args.p_threshold,
            max_days=args.max_days,
        )

    if args.out.endswith('.parquet'):
        features_df.to_parquet(args.out, index=False)
    else:
        features_df.to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
