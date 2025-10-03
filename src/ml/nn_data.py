from typing import List, Dict, Tuple
import re
import json
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.core.news_nlp import normalize_text


def build_vocab(texts: List[str], min_freq: int = 3, max_size: int = 50000) -> Dict[str, int]:
    counter = Counter()
    for t in texts:
        nt = normalize_text(t)
        tokens = re.findall(r"[\w]+", nt)
        counter.update(tokens)
    # specials
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if tok in vocab:
            continue
        vocab[tok] = len(vocab)
        if len(vocab) >= max_size:
            break
    return vocab


def save_vocab(path: str, vocab: Dict[str, int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_text(text: str, vocab: Dict[str, int], max_len: int = 256) -> List[int]:
    nt = normalize_text(text)
    tokens = re.findall(r"[\w]+", nt)
    ids = [vocab.get(t, 1) for t in tokens][:max_len]
    return ids


class NewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Dict[str, int], ticker_to_idx: Dict[str, int], max_len: int = 256, mode: str = 'train'):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.ticker_to_idx = ticker_to_idx
        self.max_len = max_len
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = f"{row.get('title','')} [SEP] {row.get('publication','')}"
        ids = encode_text(text, self.vocab, self.max_len)
        if self.mode == 'train':
            labels = torch.zeros(len(self.ticker_to_idx), dtype=torch.float32)
            for t in re.split(r"[;,\s]+", str(row.get('tickers', ''))):
                t = t.strip()
                if t and t in self.ticker_to_idx:
                    labels[self.ticker_to_idx[t]] = 1.0
            return ids, labels
        return ids


def collate_batch(batch: List, pad_id: int = 0, mode: str = 'train') -> Dict[str, torch.Tensor]:
    if mode == 'train':
        ids_list, labels_list = zip(*batch)
    else:
        ids_list = batch
        labels_list = None
    max_len = max(len(ids) for ids in ids_list) if ids_list else 1
    input_ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(ids_list), max_len), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :L] = 1
    if mode == 'train':
        labels = torch.stack(labels_list)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return {"input_ids": input_ids, "attention_mask": attention_mask}
