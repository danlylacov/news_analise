from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict, Any
import pandas as pd
import torch
import os
import numpy as np
from functools import lru_cache

from src.core.auto_label_tickers import build_aliases, assign_tickers_row
from src.core.infer_news_to_candles import infer_news_to_candles_df
from src.ml.nn_model import NewsTickerModel
from src.ml.nn_data import load_vocab

app = FastAPI(title="FORECAST API: JSON news + candles → features + join")

# Глобальные переменные для кэширования модели
_model_cache = {}
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@lru_cache(maxsize=1)
def load_cached_artifacts(artifacts_dir: str):
    """Кэшированная загрузка артефактов модели"""
    import json
    
    with open(os.path.join(artifacts_dir, 'tickers.json'), 'r', encoding='utf-8') as f:
        ticker_to_idx = json.load(f)['ticker_to_idx']
    
    vocab = load_vocab(os.path.join(artifacts_dir, 'vocab.json'))
    ckpt = torch.load(os.path.join(artifacts_dir, 'model.pt'), map_location='cpu')
    
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


class NewsItem(BaseModel):
    publish_date: str
    title: str
    publication: str


class CandleItem(BaseModel):
    begin: str
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class InferRequest(BaseModel):
    news: List[NewsItem] = Field(..., description='Список новостей')
    candles: List[CandleItem] = Field(..., description='Список свечей')
    artifacts_dir: str = Field('artifacts', description='Путь к артефактам модели')
    p_threshold: float = Field(0.5, description='Порог релевантности новостей')
    half_life_days: float = Field(0.5, description='Период полураспада влияния новостей')
    max_days: float = Field(5.0, description='Максимальный возраст учитываемых новостей')
    add_sentiment: bool = Field(True, description='Добавлять ли сентимент-анализ в результат')


class InferResponse(BaseModel):
    status: str
    rows_features: int
    rows_joined: int
    features_preview: Optional[List[Dict[str, Any]]] = None
    joined_preview: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None


def auto_label_news(news_dicts: List[Dict]) -> List[Dict]:
    """Автоматическая разметка тикеров для новостей"""
    # Конвертируем в DataFrame для обработки
    df_news = pd.DataFrame(news_dicts)
    
    # Строим словарь алиасов
    aliases = build_aliases(None)
    
    tickers = []
    for _, row in df_news.iterrows():
        tks = assign_tickers_row(row.get('title'), row.get('publication'), aliases)
        tickers.append(';'.join(tks))
    
    df_news['tickers'] = tickers
    
    return df_news.to_dict(orient='records')


def optimized_score_news(df_news: pd.DataFrame, artifacts_data: dict, batch_size: int = 512) -> np.ndarray:
    """Оптимизированная функция скоринга новостей с кэшированной моделью"""
    from src.ml.nn_data import encode_text
    
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


def optimized_aggregate_to_candles(
    df_candles: pd.DataFrame,
    df_news: pd.DataFrame,
    scores: np.ndarray,
    ticker_to_idx: dict,
    half_life_days: float = 2.0,
    p_threshold: float = 0.5,
    max_days: float = 20.0,
) -> pd.DataFrame:
    """Оптимизированная агрегация новостей к свечам"""
    import math
    
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


@app.post('/infer', response_model=InferResponse)
async def infer(request: InferRequest):
    """
    Основной эндпоинт для инференса новостей с сентимент-анализом
    
    Принимает:
    - news: список новостей (без тикеров)
    - candles: список свечей (OHLCV данные)
    - параметры модели и агрегации
    - add_sentiment: включить ли сентимент-анализ
    
    Возвращает:
    - features: агрегированные новостные фичи по тикерам и датам
    - joined: свечи с добавленными новостными фичами
    - sentiment колонки: sentiment_mean, sentiment_sum, sentiment_count, 
      sentiment_positive_count, sentiment_negative_count, sentiment_neutral_count
    """
    try:
        # Загружаем кэшированные артефакты
        artifacts_data = load_cached_artifacts(request.artifacts_dir)
        
        # Конвертируем Pydantic модели в словари
        news_dicts = [item.dict() for item in request.news]
        candles_dicts = [item.dict() for item in request.candles]
        
        # Автоматическая разметка тикеров для новостей
        labeled_news = auto_label_news(news_dicts)
        
        # Конвертируем в DataFrame
        df_news = pd.DataFrame(labeled_news)
        df_candles = pd.DataFrame(candles_dicts)
        
        # Используем функцию с сентимент-анализом
        features_df, joined_df = infer_news_to_candles_df(
            df_news, df_candles, request.artifacts_dir,
            p_threshold=request.p_threshold,
            half_life_days=request.half_life_days,
            max_days=request.max_days,
            add_sentiment=request.add_sentiment
        )
        
        return InferResponse(
            status="success",
            rows_features=len(features_df),
            rows_joined=len(joined_df),
            features_preview=features_df.head(50).to_dict(orient='records'),
            joined_preview=joined_df.head(50).to_dict(orient='records')
        )
        
    except Exception as e:
        return InferResponse(
            status="error",
            rows_features=0,
            rows_joined=0,
            message=f"Ошибка при обработке: {str(e)}"
        )


@app.get('/')
async def root():
    return {
        "message": "FORECAST API готов к работе", 
        "version": "2.0",
        "features": [
            "Анализ новостей с нейронной сетью",
            "Автоматическая разметка тикеров",
            "Агрегация новостных фич по свечам",
            "Сентимент-анализ новостей (0=негативный, 1=нейтральный, 2=позитивный)",
            "Временное затухание влияния новостей"
        ],
        "endpoints": [
            "/infer - основной эндпоинт для анализа",
            "/health - проверка состояния API"
        ]
    }


@app.post('/health')
async def health_check():
    """Проверка доступности артефактов модели"""
    try:
        import os
        artifacts_dir = "artifacts"
        required_files = ['model.pt', 'vocab.json', 'tickers.json', 'lexicon.json']
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(artifacts_dir, file)):
                missing_files.append(file)
        
        if missing_files:
            return {"status": "unhealthy", "missing_files": missing_files}
        else:
            return {"status": "healthy", "message": "Все артефакты найдены"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Для запуска напрямую: uvicorn app:app --host 0.0.0.0 --port 8000