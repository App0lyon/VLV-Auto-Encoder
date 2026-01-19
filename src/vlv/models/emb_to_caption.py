from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class EmbeddingCaptioner(nn.Module):
    def __init__(
        self,
        model_name: str = "distilgpt2",
        prefix_len: int = 77,
        prefix_dim: int = 1024,
        max_length: int = 64,
        freeze_llm: bool = False,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.prefix_len = prefix_len
        self.prefix_dim = prefix_dim
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.llm_embed_dim = self.llm.get_input_embeddings().embedding_dim
        self.prefix_proj = nn.Linear(prefix_dim, self.llm_embed_dim)

        if freeze_llm:
            self._set_llm_requires_grad(False)

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except Exception as exc:
                raise RuntimeError("peft is required for LoRA but is not installed") from exc
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, config)

    def _set_llm_requires_grad(self, requires_grad: bool) -> None:
        for p in self.llm.parameters():
            p.requires_grad = requires_grad

    def forward(
        self,
        prefix_embeddings: torch.Tensor,
        captions: Optional[list[str]] = None,
    ):
        if prefix_embeddings.dim() != 3:
            raise ValueError("prefix_embeddings must have shape (B, 77, 1024)")

        prefix_embeddings = self.prefix_proj(prefix_embeddings)
        prefix_embeddings = prefix_embeddings.to(self.llm.device)

        if captions is None:
            raise ValueError("captions is required for teacher-forcing training")

        tokens = self.tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tokens.input_ids.to(self.llm.device)
        attention_mask = tokens.attention_mask.to(self.llm.device)

        tok_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_embeddings, tok_embeds], dim=1)

        prefix_mask = torch.ones(
            (input_ids.shape[0], self.prefix_len),
            device=self.llm.device,
            dtype=attention_mask.dtype,
        )
        full_attention = torch.cat([prefix_mask, attention_mask], dim=1)

        labels = torch.full(
            (input_ids.shape[0], self.prefix_len + input_ids.shape[1]),
            -100,
            device=self.llm.device,
            dtype=input_ids.dtype,
        )
        labels[:, self.prefix_len :] = input_ids

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            labels=labels,
        )
