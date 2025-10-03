from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict, Any
import pandas as pd

from auto_label_tickers import build_aliases, assign_tickers_row
from infer_news_to_candles import infer_news_to_candles_df

app = FastAPI(title="FORECAST API: JSON news + candles → features + join")


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


@app.post('/infer', response_model=InferResponse)
async def infer(request: InferRequest):
    """
    Основной эндпоинт для инференса новостей
    
    Принимает:
    - news: список новостей (без тикеров)
    - candles: список свечей (OHLCV данные)
    - параметры модели и агрегации
    
    Возвращает:
    - features: агрегированные новостные фичи по тикерам и датам
    - joined: свечи с добавленными новостными фичами
    """
    try:
        # Конвертируем Pydantic модели в словари
        news_dicts = [item.dict() for item in request.news]
        candles_dicts = [item.dict() for item in request.candles]
        
        # Автоматическая разметка тикеров для новостей
        labeled_news = auto_label_news(news_dicts)
        
        # Конвертируем в DataFrame
        df_news = pd.DataFrame(labeled_news)
        df_candles = pd.DataFrame(candles_dicts)
        
        # Инференс новостей к свечам
        features_df, joined_df = infer_news_to_candles_df(
            df_news, df_candles, 
            artifacts_dir=request.artifacts_dir,
            p_threshold=request.p_threshold,
            half_life_days=request.half_life_days,
            max_days=request.max_days
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
    return {"message": "FORECAST API готов к работе", "version": "1.0"}


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