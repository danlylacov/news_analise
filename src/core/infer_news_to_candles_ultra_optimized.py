#!/usr/bin/env python3
"""
Дополнительные оптимизации для максимальной производительности
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from functools import lru_cache
import os
import json
import math

from src.ml.nn_model import NewsTickerModel
from src.ml.nn_data import load_vocab, encode_text


class UltraOptimizedNewsTickerModel(nn.Module):
    """Ультра-оптимизированная модель с дополнительными улучшениями"""
    def __init__(self, vocab_size: int, num_labels: int, embed_dim: int = 256, rnn_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Используем более быстрый GRU с оптимизированными параметрами
        self.bigru = nn.GRU(embed_dim, rnn_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.Linear(rnn_hidden * 2, 1)  # Упрощенный pooling
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.dropout(x)
        h, _ = self.bigru(x)
        
        # Упрощенный pooling без attention
        pooled = torch.mean(h, dim=1)
        logits = self.classifier(pooled)
        return logits


class ModelCache:
    """Кэш для моделей с автоматическим управлением памятью"""
    def __init__(self, max_size: int = 3):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, key: str):
        if key in self.cache:
            # Обновляем порядок доступа
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value):
        if len(self.cache) >= self.max_size:
            # Удаляем наименее используемый элемент
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.access_order.append(key)


# Глобальный кэш моделей
_model_cache = ModelCache()
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@lru_cache(maxsize=2)
def load_artifacts_ultra_cached(artifacts: str):
    """Ультра-кэшированная загрузка артефактов"""
    cache_key = f"artifacts_{artifacts}"
    cached = _model_cache.get(cache_key)
    if cached is not None:
        return cached
    
    with open(os.path.join(artifacts, 'tickers.json'), 'r', encoding='utf-8') as f:
        ticker_to_idx = json.load(f)['ticker_to_idx']
    vocab = load_vocab(os.path.join(artifacts, 'vocab.json'))
    ckpt = torch.load(os.path.join(artifacts, 'model.pt'), map_location='cpu')
    
    # Создаем ультра-оптимизированную модель
    model = UltraOptimizedNewsTickerModel(vocab_size=len(vocab), num_labels=len(ticker_to_idx))
    model.load_state_dict(ckpt['state_dict'])
    model.to(_device)
    model.eval()
    
    # Компилируем модель для дополнительного ускорения (PyTorch 2.0+)
    try:
        model = torch.compile(model)
    except:
        pass  # Если компиляция не поддерживается
    
    result = {
        'ticker_to_idx': ticker_to_idx,
        'vocab': vocab,
        'model': model,
        'config': ckpt.get('config', {})
    }
    
    _model_cache.put(cache_key, result)
    return result


def score_news_ultra_optimized(df_news: pd.DataFrame, artifacts_data: dict, batch_size: int = 1024) -> np.ndarray:
    """Ультра-оптимизированная функция скоринга"""
    model = artifacts_data['model']
    vocab = artifacts_data['vocab']
    max_len = artifacts_data['config'].get('max_len', 256)
    
    scores = []
    with torch.no_grad():
        # Предварительно токенизируем все тексты
        all_texts = (df_news['title'].fillna('') + ' [SEP] ' + df_news['publication'].fillna('')).tolist()
        all_ids = [encode_text(t, vocab, max_len) for t in all_texts]
        
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i:i+batch_size]
            maxL = max(len(x) for x in batch_ids) if batch_ids else 1
            
            # Создаем тензоры на GPU сразу
            batch_size_actual = len(batch_ids)
            input_ids = torch.zeros((batch_size_actual, maxL), dtype=torch.long, device=_device)
            attention_mask = torch.zeros((batch_size_actual, maxL), dtype=torch.long, device=_device)
            
            for j, seq in enumerate(batch_ids):
                L = len(seq)
                input_ids[j, :L] = torch.tensor(seq, dtype=torch.long, device=_device)
                attention_mask[j, :L] = 1
            
            logits = model(input_ids, attention_mask)
            scores.append(torch.sigmoid(logits).cpu().numpy())
    
    return np.vstack(scores) if scores else np.zeros((0, len(artifacts_data['ticker_to_idx'])))


def aggregate_to_candles_ultra_optimized(
    df_candles: pd.DataFrame,
    df_news: pd.DataFrame,
    scores: np.ndarray,
    ticker_to_idx: dict,
    half_life_days: float = 2.0,
    p_threshold: float = 0.5,
    max_days: float = 20.0,
) -> pd.DataFrame:
    """Ультра-оптимизированная агрегация с векторизацией"""
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
    valid_tickers = set(ticker_to_idx.keys())
    
    # Предварительно вычисляем все маски для ускорения
    candle_dates = df_candles['date'].unique()
    candle_dates_sorted = np.sort(candle_dates)
    
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


def infer_news_to_candles_df_ultra_optimized(news_df: pd.DataFrame, candles_df: pd.DataFrame, artifacts_dir: str, 
                                            p_threshold: float = 0.5, half_life_days: float = 0.5, max_days: int = 5) -> tuple:
    """Ультра-оптимизированная основная функция"""
    artifacts_data = load_artifacts_ultra_cached(artifacts_dir)
    
    scores = score_news_ultra_optimized(news_df, artifacts_data)
    
    features_df = aggregate_to_candles_ultra_optimized(
        candles_df, news_df, scores, artifacts_data['ticker_to_idx'],
        half_life_days=half_life_days,
        p_threshold=p_threshold,
        max_days=max_days,
    )
    
    # Оптимизированное объединение
    candles_df_copy = candles_df.copy()
    candles_df_copy['date'] = pd.to_datetime(candles_df_copy['begin'], errors='coerce').dt.date
    features_df_copy = features_df.copy()
    
    joined_df = candles_df_copy.merge(features_df_copy, on=['ticker', 'date'], how='left')
    
    # Векторизованное заполнение пропусков
    feature_cols = ['nn_news_sum', 'nn_news_mean', 'nn_news_max', 'nn_news_count']
    for col in feature_cols:
        if col in joined_df.columns:
            joined_df[col] = joined_df[col].fillna(0.0)
    
    return features_df, joined_df


if __name__ == "__main__":
    # Тест ультра-оптимизированной версии
    import time
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    # Создаем тестовые данные
    np.random.seed(42)
    news_data = []
    publications = ['РБК', 'Коммерсант', 'Ведомости']
    titles = ['Сбербанк объявил о росте прибыли', 'Газпром увеличил добычу газа']
    
    for i in range(500):
        news_data.append({
            'publish_date': f'2024-01-{(i % 30) + 1:02d}',
            'title': np.random.choice(titles),
            'publication': np.random.choice(publications)
        })
    
    candles_data = []
    tickers = ['SBER', 'GAZP', 'LKOH']
    
    for i in range(250):
        candles_data.append({
            'begin': f'2024-01-{(i % 30) + 1:02d} 10:00:00',
            'ticker': np.random.choice(tickers),
            'open': 100 + np.random.randn() * 10,
            'high': 110 + np.random.randn() * 10,
            'low': 90 + np.random.randn() * 10,
            'close': 105 + np.random.randn() * 10,
            'volume': np.random.randint(1000, 10000)
        })
    
    df_news = pd.DataFrame(news_data)
    df_candles = pd.DataFrame(candles_data)
    
    print("🚀 Тестирование ультра-оптимизированной версии...")
    start_time = time.time()
    
    try:
        features_df, joined_df = infer_news_to_candles_df_ultra_optimized(
            df_news, df_candles, "artifacts",
            p_threshold=0.5, half_life_days=2.0, max_days=10.0
        )
        
        end_time = time.time()
        print(f"⚡ Ультра-оптимизированная версия: {end_time - start_time:.2f} секунд")
        print(f"📊 Результат: {len(features_df)} фичей, {len(joined_df)} объединенных строк")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
