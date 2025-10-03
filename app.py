from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, Literal
import os
import io
import pandas as pd

from auto_label_tickers import build_aliases, assign_tickers_row
from infer_news_to_candles import load_artifacts, score_news, aggregate_to_candles

app = FastAPI(title="FORECAST API: raw news + candles → features + join")


class InferParams(BaseModel):
    artifacts_dir: str = Field('artifacts')
    p_threshold: float = 0.5
    half_life_days: float = 0.5
    max_days: float = 5.0
    out_path: Optional[str] = None
    join_out_path: Optional[str] = None
    response_format: Literal['json','none'] = 'json'


def auto_label(df_news: pd.DataFrame) -> pd.DataFrame:
    aliases = build_aliases(None)
    tickers = []
    for _, r in df_news.iterrows():
        tks = assign_tickers_row(r.get('title'), r.get('publication'), aliases)
        tickers.append(';'.join(tks))
    df_news = df_news.copy()
    df_news['tickers'] = tickers
    return df_news


@app.post('/infer')
async def infer(
    news_file: UploadFile = File(..., description='CSV с publish_date,title,publication'),
    candles_file: UploadFile = File(..., description='CSV со свечами (begin,ticker,OHLCV)'),
    artifacts_dir: str = Form('artifacts'),
    p_threshold: float = Form(0.5),
    half_life_days: float = Form(0.5),
    max_days: float = Form(5.0),
    out_path: Optional[str] = Form(None),
    join_out_path: Optional[str] = Form(None),
    response_format: str = Form('json'),
):
    # читаем входные CSV из multipart
    news_bytes = await news_file.read()
    candles_bytes = await candles_file.read()
    df_news = pd.read_csv(io.BytesIO(news_bytes))
    df_candles = pd.read_csv(io.BytesIO(candles_bytes))

    # авторазметка тикеров
    df_news = auto_label(df_news)

    # артефакты/модель
    ticker_to_idx, vocab, ckpt = load_artifacts(artifacts_dir)

    # инференс
    scores = score_news(df_news, vocab, ckpt['state_dict'], num_labels=len(ticker_to_idx), max_len=ckpt['config'].get('max_len', 256))
    feats = aggregate_to_candles(
        df_candles, df_news, scores, ticker_to_idx,
        half_life_days=half_life_days,
        p_threshold=p_threshold,
        max_days=max_days,
    )

    # сохранить при необходимости
    if out_path:
        if out_path.endswith('.parquet'):
            feats.to_parquet(out_path, index=False)
        else:
            feats.to_csv(out_path, index=False)

    # джойн со свечами
    candles = df_candles.copy()
    if 'begin' in candles.columns:
        candles['begin'] = pd.to_datetime(candles['begin'], errors='coerce')
        candles['date'] = candles['begin'].dt.date
    joined = candles.merge(feats, on=['ticker', 'date'], how='left')
    for col in ['nn_news_sum', 'nn_news_mean', 'nn_news_max', 'nn_news_count']:
        if col in joined.columns:
            joined[col] = joined[col].fillna(0)

    if join_out_path:
        if join_out_path.endswith('.parquet'):
            joined.to_parquet(join_out_path, index=False)
        else:
            joined.to_csv(join_out_path, index=False)

    if response_format == 'none':
        return {"status": "ok", "rows_features": len(feats), "rows_joined": len(joined)}

    return {
        "rows_features": len(feats),
        "rows_joined": len(joined),
        "features_preview": feats.head(50).to_dict(orient='records'),
        "joined_preview": joined.head(50).to_dict(orient='records'),
    }

# uvicorn app:app --host 0.0.0.0 --port 8000
