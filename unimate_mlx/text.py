"""Frozen T5 text embeddings compatible with UniMate's reference conditioner.

PyTorch and Transformers are imported only when loading the pretrained model.
Callers may inject tokenizer/model objects, which keeps preprocessing tests
offline and lets applications manage model loading themselves.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Sequence

import numpy as np


DEFAULT_T5_MODEL = "google/flan-t5-base"


class FrozenT5Encoder:
    """T5 encoder using UniMate's padding mask and masked-mean pooling."""

    def __init__(
        self,
        model_name: str = DEFAULT_T5_MODEL,
        device: str = "cpu",
        *,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if (tokenizer is None) != (model is None):
            raise ValueError("inject both tokenizer and model, or neither")
        self.model_name = model_name
        self.device = device
        self._torch = None
        if model is None:
            try:
                import torch
                from transformers import T5EncoderModel, T5Tokenizer
            except ImportError as exc:
                raise ImportError(
                    "loading the pretrained text encoder requires torch and transformers"
                ) from exc
            self._torch = torch
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5EncoderModel.from_pretrained(model_name)
            model.to(device)
        self.tokenizer = tokenizer
        self.model = model
        if self._torch is None and hasattr(self.model, "parameters"):
            # An injected real PyTorch model still needs no-grad inference.
            try:
                import torch
            except ImportError as exc:
                raise ImportError("an injected PyTorch model requires torch") from exc
            self._torch = torch
        if hasattr(self.model, "eval"):
            self.model.eval()
        if hasattr(self.model, "parameters"):
            for parameter in self.model.parameters():
                if hasattr(parameter, "requires_grad_"):
                    parameter.requires_grad_(False)
                elif hasattr(parameter, "requires_grad"):
                    parameter.requires_grad = False

    def tokenize(self, texts: str | Sequence[str | None]) -> dict[str, Any]:
        """Tokenize like reference ``T5Conditioner.tokenize`` (dynamic padding)."""
        entries = [texts] if isinstance(texts, str) else list(texts)
        entries = [text if text is not None else "" for text in entries]
        if not entries:
            raise ValueError("texts must contain at least one item")
        # Reference uses tokenizer defaults for truncation/special tokens and
        # asks only for PyTorch tensors and batch padding.
        inputs = self.tokenizer(entries, return_tensors="pt", padding=True)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        empty_rows = [index for index, text in enumerate(entries) if text == ""]
        if empty_rows:
            inputs["attention_mask"][empty_rows, :] = 0
        return inputs

    def encode(self, texts: str | Sequence[str | None]) -> np.ndarray:
        """Return masked-mean pooled embeddings with shape ``(batch, dim)``.

        As in the reference, an empty/None input has an all-zero attention
        mask, and its pooled embedding is exactly a zero vector.
        """
        inputs = self.tokenize(texts)
        grad_context = self._torch.no_grad() if self._torch is not None else nullcontext()
        with grad_context:
            output = self.model(**inputs)
        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, dict):
            hidden = output["last_hidden_state"]
        else:
            hidden = output[0]
        mask = inputs["attention_mask"]
        if self._torch is not None:
            mask_for_hidden = mask.to(dtype=hidden.dtype)
            counts = mask.sum(dim=-1, keepdim=True).clamp(min=1).to(dtype=hidden.dtype)
            pooled = (hidden * mask_for_hidden.unsqueeze(-1)).sum(dim=1) / counts
            return pooled.detach().cpu().numpy().astype(np.float32, copy=False)
        hidden = np.asarray(hidden)
        mask = np.asarray(mask)
        counts = np.maximum(mask.sum(axis=-1, keepdims=True), 1)
        pooled = (hidden * mask[..., None]).sum(axis=1) / counts
        return np.asarray(pooled, dtype=np.float32)

    def prompt_embedding(self, prompt: str | None) -> np.ndarray:
        """Encode one prompt as a ``(dim,)`` vector."""
        return self.encode([prompt])[0]

    def joint_name_embeddings(self, names: Sequence[str | None]) -> np.ndarray:
        """Encode joint names as ``(joint_count, dim)`` vectors."""
        return self.encode(names)
