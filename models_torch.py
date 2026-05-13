from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """CNN for text: parallel convolution filters over word embeddings."""
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, pad_idx: int = 0,
                 filter_sizes: Iterable[int] = (3, 4, 5), num_filters: int = 96, dropout: float = 0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(list(filter_sizes)), num_classes)

    def forward(self, x):
        emb = self.embedding(x)              # [batch, seq, emb]
        emb = emb.transpose(1, 2)            # [batch, emb, seq]
        conv_outs = [F.relu(conv(emb)) for conv in self.convs]
        pooled = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]
        features = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(features))


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier for sequence modeling."""
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int,
                 pad_idx: int = 0, dropout: float = 0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=1
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        # last forward and last backward states
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(self.dropout(h))
