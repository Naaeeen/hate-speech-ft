from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from .tokenizer import MODEL_NAME


class FrozenDistilBertClassifier(nn.Module):
    """Frozen DistilBERT backbone with a trainable classification head."""

    def __init__(
        self,
        *,
        model_name: str = MODEL_NAME,
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        hidden_size = int(self.backbone.config.hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        self.model_config = {
            "model_name": model_name,
            "num_classes": num_classes,
            "dropout": dropout,
            "hidden_size": hidden_size,
            "backbone_trainable": False,
            "pooling": "cls_token",
        }

    def train(self, mode: bool = True) -> "FrozenDistilBertClassifier":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        cls_representation = outputs.last_hidden_state[:, 0, :]
        cls_representation = self.dropout(cls_representation)
        logits = self.classifier(cls_representation)
        return logits
