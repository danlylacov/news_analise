import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch

from src.ml.nn_model import NewsTickerModel
from src.ml.nn_data import load_vocab, encode_text
from src.core.sentiment_analysis import add_sentiment_to_news


def load_artifacts(artifacts: str):
    with open(os.path.join(artifacts, 'tickers.json'), 'r', encoding='utf-8') as f:
        ticker_to_idx = json.load(f)['ticker_to_idx']
    vocab = load_vocab(os.path.join(artifacts, 'vocab.json'))
    ckpt = torch.load(os.path.join(artifacts, 'model.pt'), map_location='cpu')
    return ticker_to_idx, vocab, ckpt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def score_news(df_news: pd.DataFrame, vocab, model_state, num_labels: int, max_len: int = 256, batch_size: int = 256, add_sentiment: bool = True) -> tuple:
    """
    Оценка новостей с помощью нейронной сети и сентимент-анализа
    
    Args:
        df_news: Датафрейм с новостями
        vocab: Словарь для токенизации
        model_state: Состояние модели
        num_labels: Количество меток
        max_len: Максимальная длина последовательности
        batch_size: Размер батча
        add_sentiment: Добавлять ли сентимент-анализ
        
    Returns:
        Кортеж (scores, sentiment_features) где:
        - scores: массив оценок релевантности новостей к тикерам
        - sentiment_features: датафрейм с сентимент-фичами (если add_sentiment=True)
    """
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
    
    scores_array = np.vstack(scores) if scores else np.zeros((0, num_labels))
    
    # Добавляем сентимент-анализ если требуется
    sentiment_features = None
    if add_sentiment:
        try:
            sentiment_features = add_sentiment_to_news(df_news)
        except Exception as e:
            print(f"Предупреждение: не удалось выполнить сентимент-анализ: {e}")
            sentiment_features = None
    
    return scores_array, sentiment_features


def aggregate_to_candles(
    df_candles: pd.DataFrame,
    df_news: pd.DataFrame,
    scores: np.ndarray,
    ticker_to_idx: dict,
    sentiment_features: pd.DataFrame = None,
    half_life_days: float = 2.0,
    p_threshold: float = 0.5,
    max_days: float = 20.0,
) -> pd.DataFrame:
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
                feature_row = {'ticker': ticker, 'date': dt, 'nn_news_sum': 0.0, 'nn_news_mean': 0.0, 'nn_news_max': 0.0, 'nn_news_count': 0}
                if sentiment_features is not None:
                    feature_row.update({
                        'sentiment_mean': 1.0,
                        'sentiment_sum': 0.0,
                        'sentiment_count': 0,
                        'sentiment_positive_count': 0,
                        'sentiment_negative_count': 0,
                        'sentiment_neutral_count': 0
                    })
                features.append(feature_row)
                continue
            # релевантность по порогу
            mask_thr = base_mask & (probs >= p_threshold)
            if not mask_thr.any():
                feature_row = {'ticker': ticker, 'date': dt, 'nn_news_sum': 0.0, 'nn_news_mean': 0.0, 'nn_news_max': 0.0, 'nn_news_count': 0}
                if sentiment_features is not None:
                    feature_row.update({
                        'sentiment_mean': 1.0,
                        'sentiment_sum': 0.0,
                        'sentiment_count': 0,
                        'sentiment_positive_count': 0,
                        'sentiment_negative_count': 0,
                        'sentiment_neutral_count': 0
                    })
                features.append(feature_row)
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
            
            feature_row = {'ticker': ticker, 'date': dt, 'nn_news_sum': ssum, 'nn_news_mean': smean, 'nn_news_max': smax, 'nn_news_count': cnt}
            
            # Добавляем сентимент-фичи если доступны
            if sentiment_features is not None:
                # Получаем сентимент для релевантных новостей
                relevant_news_indices = np.where(mask_thr)[0]
                if len(relevant_news_indices) > 0:
                    sentiment_subset = sentiment_features.iloc[relevant_news_indices]
                    weighted_sentiment = sentiment_subset['sentiment_score'] * weights
                    
                    positive_count = (sentiment_subset['sentiment_label'] == 2).sum()
                    negative_count = (sentiment_subset['sentiment_label'] == 0).sum()
                    neutral_count = (sentiment_subset['sentiment_label'] == 1).sum()
                    
                    feature_row.update({
                        'sentiment_mean': float(np.mean(weighted_sentiment)),
                        'sentiment_sum': float(np.sum(weighted_sentiment)),
                        'sentiment_count': len(sentiment_subset),
                        'sentiment_positive_count': positive_count,
                        'sentiment_negative_count': negative_count,
                        'sentiment_neutral_count': neutral_count
                    })
                else:
                    feature_row.update({
                        'sentiment_mean': 1.0,
                        'sentiment_sum': 0.0,
                        'sentiment_count': 0,
                        'sentiment_positive_count': 0,
                        'sentiment_negative_count': 0,
                        'sentiment_neutral_count': 0
                    })
            
            features.append(feature_row)
    return pd.DataFrame(features)


def infer_news_to_candles_df(news_df: pd.DataFrame, candles_df: pd.DataFrame, artifacts_dir: str, 
                            p_threshold: float = 0.5, half_life_days: float = 0.5, max_days: int = 5, add_sentiment: bool = True) -> tuple:
    """Основная функция для инференса новостей с DataFrame входом"""
    ticker_to_idx, vocab, ckpt = load_artifacts(artifacts_dir)
    
    scores, sentiment_features = score_news(news_df, vocab, ckpt['state_dict'], num_labels=len(ticker_to_idx), 
                       max_len=ckpt['config'].get('max_len', 256), add_sentiment=add_sentiment)
    
    features_df = aggregate_to_candles(
        candles_df, news_df, scores, ticker_to_idx,
        sentiment_features=sentiment_features,
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
    if add_sentiment and sentiment_features is not None:
        feature_cols.extend(['sentiment_mean', 'sentiment_sum', 'sentiment_count', 
                            'sentiment_positive_count', 'sentiment_negative_count', 'sentiment_neutral_count'])
    
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
    args = parser.parse_args()

    ticker_to_idx, vocab, ckpt = load_artifacts(args.artifacts)

    df_news = pd.read_csv(args.news)
    df_candles = pd.read_csv(args.candles)

    scores = score_news(df_news, vocab, ckpt['state_dict'], num_labels=len(ticker_to_idx), max_len=ckpt['config'].get('max_len', 256))
    feats = aggregate_to_candles(
        df_candles, df_news, scores, ticker_to_idx,
        half_life_days=args.half_life_days,
        p_threshold=args.p_threshold,
        max_days=args.max_days,
    )

    if args.out.endswith('.parquet'):
        feats.to_parquet(args.out, index=False)
    else:
        feats.to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
