"""Local PositionNet compatibility shim.

This module vendors the diffusion PositionNet implementation that
was removed from newer ``diffusers`` releases so that IDM-VTON can
run against modern dependency versions without relying on private
APIs. When ``diffusers`` exposes ``PositionNet`` we simply re-export
that type, otherwise we fall back to the copy bundled here.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

__all__ = ["PositionNet"]

try:  # pragma: no cover - exercised implicitly when diffusers ships PositionNet again
    from diffusers.models.embeddings import PositionNet as _DiffusersPositionNet  # type: ignore
except Exception:  # noqa: BLE001 - best-effort import, fall back below
    _DiffusersPositionNet = None


if _DiffusersPositionNet is not None:  # pragma: no cover - simple alias path
    PositionNet = _DiffusersPositionNet
else:

    class FourierEmbedder(nn.Module):
        def __init__(self, num_freqs: int = 64, temperature: int = 100) -> None:
            super().__init__()

            freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)
            freq_bands = freq_bands[None, None, None]
            self.register_buffer("freq_bands", freq_bands, persistent=False)

        def __call__(self, boxes: torch.Tensor) -> torch.Tensor:
            embedded = self.freq_bands * boxes.unsqueeze(-1)
            return torch.stack((embedded.sin(), embedded.cos()), dim=-1).permute(0, 1, 3, 4, 2).reshape(*boxes.shape[:2], -1)

    class PositionNet(nn.Module):
        """Copied from diffusers 0.25.0 (Apache-2.0)."""

        def __init__(
            self,
            positive_len: int,
            out_dim: Any,
            feature_type: str = "text-only",
            fourier_freqs: int = 8,
        ) -> None:
            super().__init__()
            self.positive_len = positive_len
            self.out_dim = out_dim

            self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
            self.position_dim = fourier_freqs * 2 * 4  # 2: sin/cos, 4: xyxy

            if isinstance(out_dim, tuple):
                out_dim = out_dim[0]

            if feature_type == "text-only":
                self.linears = nn.Sequential(
                    nn.Linear(self.positive_len + self.position_dim, 512),
                    nn.SiLU(),
                    nn.Linear(512, 512),
                    nn.SiLU(),
                    nn.Linear(512, out_dim),
                )
                self.null_positive_feature = nn.Parameter(torch.zeros(self.positive_len))
            elif feature_type == "text-image":
                self.linears_text = nn.Sequential(
                    nn.Linear(self.positive_len + self.position_dim, 512),
                    nn.SiLU(),
                    nn.Linear(512, 512),
                    nn.SiLU(),
                    nn.Linear(512, out_dim),
                )
                self.linears_image = nn.Sequential(
                    nn.Linear(self.positive_len + self.position_dim, 512),
                    nn.SiLU(),
                    nn.Linear(512, 512),
                    nn.SiLU(),
                    nn.Linear(512, out_dim),
                )
                self.null_text_feature = nn.Parameter(torch.zeros(self.positive_len))
                self.null_image_feature = nn.Parameter(torch.zeros(self.positive_len))

            self.null_position_feature = nn.Parameter(torch.zeros(self.position_dim))

        def forward(
            self,
            boxes: torch.Tensor,
            masks: torch.Tensor,
            positive_embeddings: torch.Tensor | None = None,
            phrases_masks: torch.Tensor | None = None,
            image_masks: torch.Tensor | None = None,
            phrases_embeddings: torch.Tensor | None = None,
            image_embeddings: torch.Tensor | None = None,
        ) -> torch.Tensor:
            masks = masks.unsqueeze(-1)

            xyxy_embedding = self.fourier_embedder(boxes)
            xyxy_null = self.null_position_feature.view(1, 1, -1)
            xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

            if positive_embeddings is not None:
                positive_null = self.null_positive_feature.view(1, 1, -1)
                positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null
                objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding], dim=-1))
            else:
                phrases_masks = phrases_masks.unsqueeze(-1)
                image_masks = image_masks.unsqueeze(-1)

                text_null = self.null_text_feature.view(1, 1, -1)
                image_null = self.null_image_feature.view(1, 1, -1)

                phrases_embeddings = phrases_embeddings * phrases_masks + (1 - phrases_masks) * text_null
                image_embeddings = image_embeddings * image_masks + (1 - image_masks) * image_null

                objs_text = self.linears_text(torch.cat([phrases_embeddings, xyxy_embedding], dim=-1))
                objs_image = self.linears_image(torch.cat([image_embeddings, xyxy_embedding], dim=-1))
                objs = torch.cat([objs_text, objs_image], dim=1)

            return objs
