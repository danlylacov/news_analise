import re
import json
from dataclasses import dataclass
from typing import Dict, List

from razdel import tokenize
from text_unidecode import unidecode
import emoji

try:
    import pymorphy3
    _MORPH = pymorphy3.MorphAnalyzer()
except Exception:
    _MORPH = None


RUS_LETTERS_RE = re.compile(r"[^а-яёa-z0-9\s]+", flags=re.IGNORECASE)
MULTI_SPACE_RE = re.compile(r"\s+")


@dataclass
class SentimentLexicon:
    positive: Dict[str, float]
    negative: Dict[str, float]

    @staticmethod
    def default() -> "SentimentLexicon":
        positive = {
            "рост": 1.0, "повышение": 0.9, "улучша": 1.0, "рекорд": 0.8,
            "прибыль": 0.7, "дивиден": 0.6, "покупк": 0.5, "одобр": 0.7,
            "снижен_risk": 0.6, "сильн": 0.6, "выше": 0.4, "превыс": 0.7
        }
        negative = {
            "паден": -1.0, "сниж": -0.9, "ухудш": -1.0, "убыток": -0.8,
            "санкци": -0.9, "штраф": -0.7, "срыв": -0.7, "авар": -0.8,
            "риск": -0.6, "огранич": -0.6, "ниже": -0.4, "против": -0.3,
            "делистинг": -1.0, "банкрот": -1.0
        }
        return SentimentLexicon(positive=positive, negative=negative)


KEYWORD_PATTERNS: Dict[str, List[str]] = {
    "dividends": [r"дивиден"],
    "sanctions": [r"санкци"],
    "mna": [r"сделк", r"слиян", r"поглощен"],
    "guidance": [r"прогноз", r"guidance"],
    "production": [r"добыч", r"производств"],
    "lawsuit": [r"суд", r"исков", r"штраф"],
    "spo_ipo": [r"\bipo\b", r"\bspo\b", r"листинг", r"делистинг"],
}


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = emoji.replace_emoji(text, replace=" ")
    text = unidecode(text)
    text = text.lower()
    text = RUS_LETTERS_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def lemmatize_token(token: str) -> str:
    if not token:
        return token
    if _MORPH is None:
        return token
    try:
        p = _MORPH.parse(token)
        if p:
            return p[0].normal_form
    except Exception:
        pass
    return token


def tokenize_lemmas(text: str) -> List[str]:
    norm = normalize_text(text)
    tokens = [t.text for t in tokenize(norm)]
    lemmas = [lemmatize_token(tok) for tok in tokens]
    return [l for l in lemmas if l]


def sentiment_score(lemmas: List[str], lex: SentimentLexicon) -> float:
    score = 0.0
    for w in lemmas:
        for key, val in lex.positive.items():
            if w.startswith(key):
                score += val
                break
        for key, val in lex.negative.items():
            if w.startswith(key):
                score += val
                break
    if len(lemmas) > 0:
        score = score / (len(lemmas) ** 0.5)
    return score


def keyword_flags(lemmas: List[str]) -> Dict[str, int]:
    joined = " ".join(lemmas)
    flags: Dict[str, int] = {}
    for key, pats in KEYWORD_PATTERNS.items():
        found = any(re.search(p, joined) for p in pats)
        flags[f"kw_{key}"] = int(bool(found))
    return flags


def save_lexicon(path: str, lex: SentimentLexicon) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"positive": lex.positive, "negative": lex.negative}, f, ensure_ascii=False, indent=2)


def load_lexicon(path: str) -> SentimentLexicon:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SentimentLexicon(positive=data.get("positive", {}), negative=data.get("negative", {}))
