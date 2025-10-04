"""
Модуль для расчета сентимента новостей
Простая модель, которая считает среднее значение сентимента (0, 1, 2) по каждой новости
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SimpleSentimentAnalyzer:
    """
    Простой анализатор сентимента, который рассчитывает среднее значение
    сентимента по каждой новости для свечи
    """
    
    def __init__(self):
        """Инициализация анализатора сентимента"""
        self.sentiment_keywords = {
            'positive': [
                'рост', 'увеличение', 'прибыль', 'успех', 'развитие', 'инвестиции',
                'расширение', 'новые', 'прогресс', 'улучшение', 'повышение',
                'достижение', 'победа', 'лидерство', 'инновации', 'прорыв',
                'стабильность', 'надежность', 'качество', 'эффективность'
            ],
            'negative': [
                'падение', 'снижение', 'убыток', 'проблемы', 'кризис', 'риски',
                'сокращение', 'закрытие', 'увольнения', 'ухудшение', 'понижение',
                'неудача', 'поражение', 'слабость', 'застой', 'регресс',
                'нестабильность', 'ненадежность', 'плохое', 'неэффективность',
                'конфликт', 'спор', 'проблема', 'ошибка', 'недостаток'
            ],
            'neutral': [
                'отчет', 'данные', 'результаты', 'показатели', 'статистика',
                'анализ', 'обзор', 'информация', 'новости', 'сообщение',
                'заявление', 'комментарий', 'мнение', 'прогноз', 'планы'
            ]
        }
    
    def calculate_sentiment_score(self, text: str) -> float:
        """
        Рассчитывает сентимент-оценку для текста
        
        Args:
            text: Текст для анализа
            
        Returns:
            Сентимент-оценка от 0 до 2 (0=негативный, 1=нейтральный, 2=позитивный)
        """
        if not text or str(text).strip() == "":
            return 1.0  # нейтральный по умолчанию
        
        text_lower = str(text).lower()
        
        positive_count = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
        neutral_count = sum(1 for word in self.sentiment_keywords['neutral'] if word in text_lower)
        
        total_keywords = positive_count + negative_count + neutral_count
        
        if total_keywords == 0:
            return 1.0  # нейтральный если нет ключевых слов
        
        # Рассчитываем взвешенную оценку
        sentiment_score = (positive_count * 2 + neutral_count * 1 + negative_count * 0) / total_keywords
        
        return float(sentiment_score)
    
    def analyze_news_sentiment(self, df_news: pd.DataFrame) -> pd.DataFrame:
        """
        Анализирует сентимент новостей
        
        Args:
            df_news: Датафрейм с новостями (должен содержать колонки 'title' и 'publication')
            
        Returns:
            Датафрейм с добавленными колонками сентимента
        """
        df_result = df_news.copy()
        
        # Объединяем заголовок и текст публикации
        df_result['combined_text'] = (
            df_result['title'].fillna('') + ' ' + 
            df_result['publication'].fillna('')
        ).str.strip()
        
        # Рассчитываем сентимент для каждой новости
        logger.info(f"Анализ сентимента для {len(df_result)} новостей...")
        
        df_result['sentiment_score'] = df_result['combined_text'].apply(self.calculate_sentiment_score)
        
        # Добавляем категориальные метки
        df_result['sentiment_label'] = df_result['sentiment_score'].apply(
            lambda x: 0 if x < 0.7 else (2 if x > 1.3 else 1)
        )
        
        # Удаляем временную колонку
        df_result = df_result.drop(columns=['combined_text'])
        
        logger.info("Анализ сентимента завершен")
        return df_result


def add_sentiment_to_news(df_news: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет сентимент-анализ к новостям
    
    Args:
        df_news: Датафрейм с новостями
        
    Returns:
        Датафрейм с добавленными колонками сентимента
    """
    analyzer = SimpleSentimentAnalyzer()
    return analyzer.analyze_news_sentiment(df_news)


def aggregate_sentiment_to_candles(
    df_candles: pd.DataFrame,
    df_news: pd.DataFrame,
    sentiment_features: pd.DataFrame,
    half_life_days: float = 2.0,
    max_days: float = 20.0
) -> pd.DataFrame:
    """
    Агрегирует сентимент-фичи к свечам
    
    Args:
        df_candles: Датафрейм со свечами
        df_news: Датафрейм с новостями
        sentiment_features: Датафрейм с сентимент-фичами
        half_life_days: Период полураспада влияния новостей
        max_days: Максимальный возраст учитываемых новостей
        
    Returns:
        Датафрейм свечей с добавленными сентимент-фичами
    """
    df_result = df_candles.copy()
    
    # Конвертируем даты
    df_result['date'] = pd.to_datetime(df_result['date'])
    df_news['date'] = pd.to_datetime(df_news['date'])
    
    # Добавляем сентимент-фичи к новостям
    df_news_with_sentiment = df_news.merge(
        sentiment_features[['sentiment_score', 'sentiment_label']], 
        left_index=True, 
        right_index=True, 
        how='left'
    )
    
    # Заполняем пропущенные значения нейтральным сентиментом
    df_news_with_sentiment['sentiment_score'] = df_news_with_sentiment['sentiment_score'].fillna(1.0)
    df_news_with_sentiment['sentiment_label'] = df_news_with_sentiment['sentiment_label'].fillna(1)
    
    # Агрегируем сентимент по свечам
    sentiment_features_list = []
    
    for _, candle_row in df_result.iterrows():
        candle_date = candle_row['date']
        ticker = candle_row['ticker']
        
        # Фильтруем новости для данной свечи
        news_subset = df_news_with_sentiment[
            (df_news_with_sentiment['date'] <= candle_date) &
            (df_news_with_sentiment['date'] >= candle_date - pd.Timedelta(days=max_days))
        ]
        
        if len(news_subset) == 0:
            # Нет новостей - нейтральный сентимент
            sentiment_features_list.append({
                'sentiment_mean': 1.0,
                'sentiment_sum': 0.0,
                'sentiment_count': 0,
                'sentiment_positive_count': 0,
                'sentiment_negative_count': 0,
                'sentiment_neutral_count': 0
            })
            continue
        
        # Рассчитываем веса с временным затуханием
        days_ago = (candle_date - news_subset['date']).dt.days
        weights = np.exp(-np.log(2) * days_ago / half_life_days)
        
        # Взвешенные сентимент-оценки
        weighted_sentiment = news_subset['sentiment_score'] * weights
        
        # Подсчитываем категории
        positive_count = (news_subset['sentiment_label'] == 2).sum()
        negative_count = (news_subset['sentiment_label'] == 0).sum()
        neutral_count = (news_subset['sentiment_label'] == 1).sum()
        
        sentiment_features_list.append({
            'sentiment_mean': float(weighted_sentiment.mean()),
            'sentiment_sum': float(weighted_sentiment.sum()),
            'sentiment_count': len(news_subset),
            'sentiment_positive_count': positive_count,
            'sentiment_negative_count': negative_count,
            'sentiment_neutral_count': neutral_count
        })
    
    # Добавляем сентимент-фичи к свечам
    sentiment_df = pd.DataFrame(sentiment_features_list)
    df_result = pd.concat([df_result, sentiment_df], axis=1)
    
    return df_result
