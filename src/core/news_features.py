import argparse
import os
import re
from typing import List

import pandas as pd

from src.core.news_nlp import (
    SentimentLexicon,
    tokenize_lemmas,
    sentiment_score,
    keyword_flags,
    save_lexicon,
    load_lexicon,
)


def re_split_tickers(s: str) -> List[str]:
    return [p for p in re.split(r"[;,\s]+", str(s)) if p]


def explode_tickers(df: pd.DataFrame) -> pd.DataFrame:
    if 'tickers' not in df.columns:
        raise ValueError('Ожидается столбец tickers в новостях')
    df = df.copy()
    df['tickers'] = df['tickers'].fillna("")
    df['tickers_list'] = df['tickers'].apply(lambda s: [t.strip() for t in re_split_tickers(s) if t.strip()])
    df = df.explode('tickers_list').rename(columns={'tickers_list': 'ticker'})
    df = df[df['ticker'] != ""]
    return df


def deduplicate_by_title(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['title_norm'] = (
        df['title'].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df = df.drop_duplicates(subset=['ticker', 'date', 'title_norm'])
    df = df.drop(columns=['title_norm'])
    df['dup_count'] = 0
    return df


def compute_per_news_scores(df: pd.DataFrame, lex: SentimentLexicon) -> pd.DataFrame:
    lemmas_title = df['title'].astype(str).apply(tokenize_lemmas)
    lemmas_body = df['publication'].astype(str).apply(tokenize_lemmas)
    df['sent_title'] = lemmas_title.apply(lambda ls: sentiment_score(ls, lex))
    df['sent_body'] = lemmas_body.apply(lambda ls: sentiment_score(ls, lex))
    df['sent'] = 0.4 * df['sent_title'] + 0.6 * df['sent_body']
    kw_series = lemmas_title.combine(lemmas_body, lambda a, b: keyword_flags(a + b))
    kw_df = pd.DataFrame(list(kw_series.values), index=df.index)
    return pd.concat([df, kw_df], axis=1)


essential_cols = ['count_news', 'sentiment_sum', 'sentiment_mean', 'share_pos', 'share_neg', 'dup_count']


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(['ticker', 'date'])
    agg = grp.agg(
        count_news=('sent', 'size'),
        sentiment_sum=('sent', 'sum'),
        sentiment_mean=('sent', 'mean'),
    )
    # доли знаков
    sign = grp['sent'].apply(lambda s: pd.Series({
        'share_pos': (s > 0).mean() if len(s) else 0.0,
        'share_neg': (s < 0).mean() if len(s) else 0.0,
    }))
    agg = agg.join(sign)
    # дубликаты
    if 'dup_count' in df.columns:
        agg['dup_count'] = grp['dup_count'].sum()
    else:
        agg['dup_count'] = 0
    # ключевые слова
    kw_cols = [c for c in df.columns if c.startswith('kw_')]
    if kw_cols:
        agg = agg.join(grp[kw_cols].sum())
    agg = agg.reset_index()
    return agg


def add_rollings(feat: pd.DataFrame, windows=(1, 3, 5, 10, 20)) -> pd.DataFrame:
    feat = feat.sort_values(['ticker', 'date'])
    by = feat.groupby('ticker', group_keys=False)
    value_cols = [c for c in feat.columns if c not in ('ticker', 'date')]
    for w in windows:
        rolled = by[value_cols].rolling(window=w, min_periods=1).agg(['mean', 'sum'])
        rolled.columns = [f"{col}_w{w}_{stat}" for col, stat in rolled.columns]
        rolled = rolled.reset_index(level=0, drop=True)
        feat = pd.concat([feat, rolled], axis=1)
    return feat


def run_train(news_path: str, out_path: str, artifacts_dir: str) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    lex = SentimentLexicon.default()
    save_lexicon(os.path.join(artifacts_dir, 'lexicon.json'), lex)
    run_predict(news_path, out_path, artifacts_dir)


def run_predict(news_path: str, out_path: str, artifacts_dir: str) -> None:
    lex_path = os.path.join(artifacts_dir, 'lexicon.json')
    lex = load_lexicon(lex_path) if os.path.exists(lex_path) else SentimentLexicon.default()

    df = pd.read_csv(news_path)
    df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
    df['date'] = df['publish_date'].dt.date

    df = explode_tickers(df)
    df = deduplicate_by_title(df)
    df = compute_per_news_scores(df, lex)

    daily = aggregate_daily(df)
    feats = add_rollings(daily)

    if out_path.endswith('.parquet'):
        feats.to_parquet(out_path, index=False)
    else:
        feats.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Генерация признаков из новостей')
    parser.add_argument('mode', choices=['train', 'predict'])
    parser.add_argument('--news', required=True, help='CSV файл с новостями (publish_date,title,publication,tickers)')
    parser.add_argument('--out', required=True, help='Путь для записи признаков (parquet/csv)')
    parser.add_argument('--artifacts', default='artifacts', help='Папка для артефактов')
    args = parser.parse_args()

    if args.mode == 'train':
        run_train(args.news, args.out, args.artifacts)
    else:
        run_predict(args.news, args.out, args.artifacts)


if __name__ == '__main__':
    main()
