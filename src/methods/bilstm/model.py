from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        pad_idx: int,
    ) -> None:
        super().__init__()


        self.model_config = {
            "vocab_size": vocab_size,
            "embedding_size": embedding_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_classes": num_classes,
            "dropout": dropout,
            "pad_idx": pad_idx,
        }

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:

        embedded = self.embedding(input_ids)

        if lengths is not None:
            safe_lengths = lengths.detach().cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded,
                safe_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(embedded)


        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        final_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)

        final_hidden = self.dropout(final_hidden)
        logits = self.classifier(final_hidden)
        return logits
