"""Fixed image-only OutfitTransformer secondary baseline."""

from __future__ import annotations

import torch


class OutfitTransformer(torch.nn.Module):
    def __init__(self, input_dim: int = 2048, model_dim: int = 64):
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, model_dim)
        self.outfit_token = torch.nn.Parameter(torch.empty(1, 1, model_dim))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=16,
            dim_feedforward=256,
            dropout=0.1,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=6)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_dim, 1),
        )
        torch.nn.init.normal_(self.outfit_token, std=0.02)

    def forward(self, items: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        projected = self.projection(items)
        token = self.outfit_token.expand(len(items), -1, -1)
        sequence = torch.cat((token, projected), dim=1)
        token_mask = torch.zeros((len(items), 1), dtype=torch.bool, device=items.device)
        encoded = self.encoder(sequence, src_key_padding_mask=torch.cat((token_mask, padding_mask), dim=1))
        return self.classifier(encoded[:, 0]).squeeze(1)


def focal_binary_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    pt = torch.where(labels > 0.5, probabilities, 1.0 - probabilities)
    alpha_t = torch.where(labels > 0.5, alpha, 1.0 - alpha)
    return (-alpha_t * (1.0 - pt).pow(gamma) * torch.log(pt.clamp_min(1e-7))).mean()
