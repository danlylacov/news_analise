import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd
from rapidfuzz import fuzz
from src.core.news_nlp import normalize_text


DEFAULT_ALIASES = {
    # банки
    "сбербанк": "SBER", "сбер": "SBER", "sber": "SBER",
    "втб": "VTBR", "vtb": "VTBR",
    # нефть и газ
    "газпром": "GAZP", "gazprom": "GAZP",
    "лукойл": "LKOH", "lukoil": "LKOH",
    "роснефть": "ROSN", "rosneft": "ROSN",
    "новатэк": "NVTK", "новатэк": "NVTK", "novatek": "NVTK",
    "сургутнефтегаз": "SNGS", "сургут": "SNGS",
    "газпром нефть": "SIBN", "gazprom neft": "SIBN",
    # металлурги и добытчики
    "магнитогорский металлургический комбинат": "MAGN", "ммк": "MAGN", "magnitogorsk": "MAGN",
    "нлмк": "NLMK", "липецкий металлургический комбинат": "NLMK",
    "северсталь": "CHMF",
    "норникель": "GMKN", "гмк норильский никель": "GMKN",
    "евраз": "EVR",
    "распадская": "RASP",
    "мечел": "MTLR",
    "полюс": "PLZL", "polyus": "PLZL",
    "фосагро": "PHOR",
    # ритейл/связь
    "магнит": "MGNT",
    "мтс": "MTSS", "mts": "MTSS",
}


def build_aliases(user_aliases_path: str = None) -> Dict[str, str]:
    aliases = DEFAULT_ALIASES.copy()
    if user_aliases_path and os.path.exists(user_aliases_path):
        with open(user_aliases_path, 'r', encoding='utf-8') as f:
            user = json.load(f)
        for k, v in user.items():
            aliases[normalize_text(k)] = v
    return aliases


def extract_upper_tickers(text: str) -> List[str]:
    # Находит потенциальные тикеры латиницей 2-6 символов
    cands = re.findall(r"\b[A-Z]{2,6}\b", text or "")
    return list({c for c in cands})


def assign_tickers_row(title: str, body: str, aliases: Dict[str, str], thresh: int = 80) -> List[str]:
    text = f"{title or ''} {body or ''}"
    norm = normalize_text(text)
    found = set()
    # точные/подстрочные совпадения
    for key, tkr in aliases.items():
        if key in norm:
            found.add(tkr)
    # фуззи по словам
    words = set(norm.split())
    for key, tkr in aliases.items():
        for w in words:
            if fuzz.partial_ratio(key, w) >= thresh:
                found.add(tkr)
    # извлечение латинских тикеров напрямую
    for cand in extract_upper_tickers(text):
        found.add(cand)
    return sorted(found)


def main():
    p = argparse.ArgumentParser(description='Авторазметка tickers для новостей')
    p.add_argument('--news', required=True, help='CSV с publish_date,title,publication')
    p.add_argument('--out', required=True, help='CSV с добавленным столбцом tickers')
    p.add_argument('--aliases', default=None, help='JSON алиасов {"alias": "TICKER"}')
    args = p.parse_args()

    aliases = build_aliases(args.aliases)
    df = pd.read_csv(args.news)
    if 'tickers' in df.columns:
        print('tickers уже есть — перезапишем по авторазметке')
    tickers_col = []
    for _, row in df.iterrows():
        tks = assign_tickers_row(row.get('title'), row.get('publication'), aliases)
        tickers_col.append(';'.join(tks))
    df['tickers'] = tickers_col
    df.to_csv(args.out, index=False)
    print(f'Сохранено: {args.out}')


if __name__ == '__main__':
    main()
