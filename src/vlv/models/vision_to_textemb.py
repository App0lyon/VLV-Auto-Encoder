from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torchvision import models


class VisionToTextEmb(nn.Module):
    def __init__(
        self,
        seq_len: int = 77,
        emb_dim: int = 1024,
        use_layernorm: bool = True,
        learned_pos_emb: bool = True,
    ) -> None:
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.seq_len = seq_len
        self.emb_dim = emb_dim

        in_dim = 2048
        hidden = 2048
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, seq_len * emb_dim),
        )
        self.ln = nn.LayerNorm(emb_dim) if use_layernorm else None
        self.pos_emb = nn.Embedding(seq_len, emb_dim) if learned_pos_emb else None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.backbone(images)
        x = self.proj(x)
        x = x.view(x.shape[0], self.seq_len, self.emb_dim)
        if self.pos_emb is not None:
            positions = torch.arange(self.seq_len, device=x.device)
            x = x + self.pos_emb(positions)[None, :, :]
        if self.ln is not None:
            x = self.ln(x)
        return x.float()


if __name__ == "__main__":
    model = VisionToTextEmb()
    images = torch.randn(2, 3, 512, 512)
    out = model(images)
    print("out", out.shape, out.dtype)
    assert out.shape == (2, 77, 1024)
    assert out.dtype == torch.float32
