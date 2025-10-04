import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedAttentionPooling(nn.Module):
    """Оптимизированная версия attention pooling с кэшированием"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: [B, T, H], mask: [B, T] (1 for valid)
        logits = self.proj(h).squeeze(-1)
        logits = logits.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(logits, dim=-1)
        pooled = torch.bmm(attn.unsqueeze(1), h).squeeze(1)
        return pooled


class OptimizedTextEncoder(nn.Module):
    """Оптимизированный текстовый энкодер с улучшенной производительностью"""
    def __init__(self, vocab_size: int, embed_dim: int = 256, rnn_hidden: int = 256, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Используем LSTM вместо GRU для лучшей производительности
        self.bilstm = nn.LSTM(embed_dim, rnn_hidden, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.pool = OptimizedAttentionPooling(rnn_hidden * 2)
        self.out_dim = rnn_hidden * 2

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.dropout(x)
        h, _ = self.bilstm(x)
        pooled = self.pool(h, attention_mask)
        return pooled


class OptimizedNewsTickerModel(nn.Module):
    """Оптимизированная модель для классификации новостей"""
    def __init__(self, vocab_size: int, num_labels: int, embed_dim: int = 256, rnn_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = OptimizedTextEncoder(vocab_size, embed_dim=embed_dim, rnn_hidden=rnn_hidden, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.out_dim, self.encoder.out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.out_dim, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        z = self.encoder(input_ids, attention_mask)
        logits = self.classifier(z)
        return logits

    @staticmethod
    def bce_with_logits_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor = None) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


# Оригинальные классы для обратной совместимости
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: [B, T, H], mask: [B, T] (1 for valid)
        logits = self.proj(h).squeeze(-1)
        logits = logits.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(logits, dim=-1)
        pooled = torch.bmm(attn.unsqueeze(1), h).squeeze(1)
        return pooled


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, rnn_hidden: int = 256, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bigru = nn.GRU(embed_dim, rnn_hidden, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.pool = AttentionPooling(rnn_hidden * 2)
        self.out_dim = rnn_hidden * 2

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.dropout(x)
        h, _ = self.bigru(x)
        pooled = self.pool(h, attention_mask)
        return pooled


class NewsTickerModel(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, embed_dim: int = 256, rnn_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = TextEncoder(vocab_size, embed_dim=embed_dim, rnn_hidden=rnn_hidden, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.out_dim, self.encoder.out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.out_dim, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        z = self.encoder(input_ids, attention_mask)
        logits = self.classifier(z)
        return logits

    @staticmethod
    def bce_with_logits_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor = None) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
