"""Checkpoint-compatible numerical primitives for the graph AdaLN denoiser.

The functions accept a flat mapping of PyTorch checkpoint names to MLX arrays.
PyTorch and MLX Linear weights both have ``(out_features, in_features)``
shape, so dense weights are used without transposition at load time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import mlx.core as mx
import mlx.nn as nn


def linear(x: mx.array, weights: Mapping[str, mx.array], prefix: str) -> mx.array:
    y = x @ mx.swapaxes(weights[f"{prefix}.weight"], -1, -2)
    bias = weights.get(f"{prefix}.bias")
    return y if bias is None else y + bias


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    """Match the reference LlamaRMSNorm's FP32 reduction and output cast."""
    dtype = x.dtype
    x32 = x.astype(mx.float32)
    normalized = x32 * mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
    return (normalized * weight).astype(dtype)


def swiglu(x: mx.array, weights: Mapping[str, mx.array], prefix: str) -> mx.array:
    gate, up = mx.split(linear(x, weights, f"{prefix}.w12"), 2, axis=-1)
    return linear(nn.silu(gate) * up, weights, f"{prefix}.w3")


def timestep_embedding(t: mx.array, dim: int = 256, max_period: int = 10000) -> mx.array:
    """Cosine then sine, matching UniMate's PyTorch embedding order."""
    half = dim // 2
    freq = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    phase = t.astype(mx.float32).reshape((-1, 1)) * freq.reshape((1, -1))
    emb = mx.concatenate((mx.cos(phase), mx.sin(phase)), axis=-1)
    if dim % 2:
        emb = mx.concatenate((emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)), axis=-1)
    return emb


def time_condition(t: mx.array, weights: Mapping[str, mx.array]) -> mx.array:
    x = timestep_embedding(t).astype(weights["time_embedder.mlp.0.weight"].dtype)
    x = nn.silu(linear(x, weights, "time_embedder.mlp.0"))
    return linear(x, weights, "time_embedder.mlp.2")


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return x * (1 + scale) + shift


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def apply_rope(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> tuple[mx.array, mx.array]:
    """Apply half rotation using FP32 arithmetic, as in reference UniMate."""
    dtype = q.dtype
    q32, k32 = q.astype(mx.float32), k.astype(mx.float32)
    q_rot = q32 * cos + rotate_half(q32) * sin
    k_rot = k32 * cos + rotate_half(k32) * sin
    return q_rot.astype(dtype), k_rot.astype(dtype)


def spectral_angles(coords: mx.array, weights: Mapping[str, mx.array]) -> mx.array:
    """SignNet: phi(v) + phi(-v), flattened across frequencies, then rho."""
    prefix = "rope_j.spectral_encoder"
    v = mx.expand_dims(coords, -1)

    def phi(value: mx.array) -> mx.array:
        h = nn.gelu(linear(value, weights, f"{prefix}.phi.0"))
        return linear(h, weights, f"{prefix}.phi.2")

    h = phi(v) + phi(-v)
    h = h.reshape((*h.shape[:-2], h.shape[-2] * h.shape[-1]))
    h = nn.gelu(linear(h, weights, f"{prefix}.rho.0"))
    return linear(h, weights, f"{prefix}.rho.2")


def spectral_rope(q: mx.array, k: mx.array, coords: mx.array) -> tuple[mx.array, mx.array]:
    """Apply spectral rotations to ``(B, F, H, J, Dh)`` queries and keys.

    This helper takes precomputed angles; call :func:`spectral_angles` first.
    """
    angles = mx.concatenate((coords, coords), axis=-1)
    cos = mx.cos(angles).astype(mx.float32)[:, None, None, :, :]
    sin = mx.sin(angles).astype(mx.float32)[:, None, None, :, :]
    return apply_rope(q, k, cos, sin)
