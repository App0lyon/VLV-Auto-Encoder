from __future__ import annotations

import os
from typing import Optional

import torch
from diffusers import DiffusionPipeline
from torch import nn


SD21_LATENT_SCALING = 0.18215


class FrozenSD21(nn.Module):
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1-base",
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        fp16: bool = True,
        use_auth_token: Optional[str] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            if fp16 and torch.cuda.is_available():
                dtype = torch.float16
            else:
                dtype = torch.float32

        if use_auth_token is None:
            use_auth_token = os.environ.get("HF_TOKEN")
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_auth_token=use_auth_token,
        )
        self.vae = pipe.vae.eval()
        self.unet = pipe.unet.eval()
        self.scheduler = pipe.scheduler
        self._device = torch.device(device)
        self._dtype = dtype

        self.vae.to(self._device, dtype=self._dtype)
        self.unet.to(self._device, dtype=self._dtype)

        for module in (self.vae, self.unet):
            for p in module.parameters():
                p.requires_grad = False
                
        self.unet.set_attention_slice("auto")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @torch.no_grad()
    def encode_images_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        if images.device != self._device:
            images = images.to(self._device)
        images = images.to(dtype=self._dtype)
        images = images.clamp(0, 1) * 2 - 1
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * SD21_LATENT_SCALING
        return latents

    def add_noise(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        if latents.device != self._device:
            latents = latents.to(self._device)
        if noise.device != self._device:
            noise = noise.to(self._device)
        return self.scheduler.add_noise(latents, noise, timesteps)
