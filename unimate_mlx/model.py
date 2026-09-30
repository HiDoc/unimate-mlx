"""Functional MLX forward pass for the released graph AdaLN UniMate denoiser.

Inputs follow the official model: motion ``(B, J, 12, F)`` and conditioning
arrays with padded joints. Weights retain their Safetensors/PyTorch names.
This keeps checkpoint sharing explicit, especially the single SignNet used by
all spatial attention blocks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .primitives import (
    apply_rope,
    linear,
    modulate,
    rms_norm,
    spectral_angles,
    spectral_rope,
    swiglu,
    time_condition,
)


Weights = Mapping[str, mx.array]
Condition = Mapping[str, mx.array]


@dataclass(frozen=True)
class StaticCondition:
    """Skeleton-dependent rotations and attention biases reused in sampling."""

    spectral_angles: mx.array
    graph_biases: tuple[mx.array, ...]


def _mlp(x: mx.array, w: Weights, prefix: str) -> mx.array:
    return linear(nn.silu(linear(x, w, f"{prefix}.0")), w, f"{prefix}.2")


def _multihead_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    w: Weights,
    prefix: str,
    heads: int,
    key_valid: mx.array | None = None,
) -> mx.array:
    """PyTorch MultiheadAttention's packed in-projection, batch first."""
    dim = query.shape[-1]
    head_dim = dim // heads
    proj_weight = w[f"{prefix}.in_proj_weight"]
    proj_bias = w[f"{prefix}.in_proj_bias"]
    q_weight, k_weight, v_weight = mx.split(proj_weight, 3, axis=0)
    q_bias, k_bias, v_bias = mx.split(proj_bias, 3, axis=0)

    def project(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
        projected = x @ mx.swapaxes(weight, -1, -2) + bias
        return mx.transpose(projected.reshape((x.shape[0], x.shape[1], heads, head_dim)), (0, 2, 1, 3))

    q = project(query, q_weight, q_bias)
    k = project(key, k_weight, k_bias)
    v = project(value, v_weight, v_bias)
    logits = (q @ mx.swapaxes(k, -1, -2)).astype(mx.float32) * (head_dim**-0.5)
    if key_valid is not None:
        logits = mx.where(key_valid[:, None, None, :], logits, -mx.inf)
    probs = mx.softmax(logits, axis=-1).astype(v.dtype)
    result = mx.transpose(probs @ v, (0, 2, 1, 3)).reshape((query.shape[0], query.shape[1], dim))
    return linear(result, w, f"{prefix}.out_proj")


def _encode_input(x: mx.array, cond: Condition, w: Weights) -> tuple[mx.array, mx.array, mx.array]:
    batch, joints, _, frames = x.shape
    prefix = "input_layer"
    tpos = cond["tpos_first_frame"][:, None, :, :]
    tpos_root = _mlp(tpos[:, :, :1], w, f"{prefix}.root_tpos_embedder")
    tpos_joints = _mlp(tpos[:, :, 1:], w, f"{prefix}.joint_tpos_embedder")
    parent = cond["tpos_first_frame_parents"][:, None, 1:, :]
    parent_emb = _mlp(parent, w, f"{prefix}.parent_tpos_embedder")
    tpos_joints = _mlp(mx.concatenate((tpos_joints, parent_emb), axis=-1), w, f"{prefix}.tpos_fuse")
    tpos_emb = mx.concatenate((tpos_root, tpos_joints), axis=2)

    motion = mx.transpose(x, (0, 3, 1, 2))
    root = _mlp(motion[:, :, :1], w, f"{prefix}.root_x_embedder")
    joint = _mlp(motion[:, :, 1:], w, f"{prefix}.joint_x_embedder")
    tokens = mx.concatenate((tpos_emb, mx.concatenate((root, joint), axis=2)), axis=1)
    joints_valid = mx.arange(joints)[None, :] < cond["n_joints"][:, None]
    return tokens, tpos_emb[:, 0], joints_valid


def _tpos_pool(tpos: mx.array, valid: mx.array, w: Weights) -> mx.array:
    prefix = "tpos_pool.pool"
    batch = tpos.shape[0]
    queries = mx.broadcast_to(w[f"{prefix}.queries"], (batch, *w[f"{prefix}.queries"].shape[1:]))
    q_norm = rms_norm(queries, w[f"{prefix}.norm_q.weight"])
    kv_norm = rms_norm(tpos, w[f"{prefix}.norm_kv.weight"])
    queries = queries + _multihead_attention(q_norm, kv_norm, kv_norm, w, f"{prefix}.cross_attn", 4, valid)
    queries = queries + swiglu(rms_norm(queries, w[f"{prefix}.norm_ff.weight"]), w, f"{prefix}.ffn")
    pooled = mx.mean(queries, axis=1)
    return linear(rms_norm(pooled, w[f"{prefix}.norm_out.weight"]), w, f"{prefix}.out_proj")


def _graph_bias(cond: Condition, valid: mx.array, w: Weights, prefix: str) -> mx.array:
    distance = mx.take(w[f"{prefix}.graph_dist_embedding.weight"], cond["graph_dist"], axis=0)
    relation = mx.take(w[f"{prefix}.graph_rel_embedding.weight"], cond["joint_relations"], axis=0)
    dist_bias = linear(distance, w, f"{prefix}.graph_dist_proj") * w[f"{prefix}.graph_dist_scale"]
    rel_bias = linear(relation, w, f"{prefix}.graph_rel_proj") * w[f"{prefix}.graph_rel_scale"]
    bias = mx.transpose(dist_bias + rel_bias, (0, 3, 1, 2))
    return mx.where(valid[:, None, None, :], bias, -mx.inf)


def _spatial_attention(x: mx.array, graph_bias: mx.array, angles: mx.array, w: Weights, prefix: str, heads: int, backend: str) -> mx.array:
    batch, frames, joints, dim = x.shape
    head_dim = dim // heads
    packed = linear(x, w, f"{prefix}.qkv").reshape((batch, frames, joints, 3, heads, head_dim))
    q, k, v = (
        mx.transpose(mx.squeeze(part, axis=3), (0, 1, 3, 2, 4))
        for part in mx.split(packed, 3, axis=3)
    )
    q = rms_norm(q, w[f"{prefix}.q_norm.weight"])
    k = rms_norm(k, w[f"{prefix}.k_norm.weight"])
    q, k = spectral_rope(q, k, angles)
    if backend == "fast":
        mask = mx.broadcast_to(graph_bias[:, None], (batch, frames, heads, joints, joints))
        mask = mask.reshape((batch * frames, heads, joints, joints))
        attended = mx.fast.scaled_dot_product_attention(
            q.reshape((batch * frames, heads, joints, head_dim)),
            k.reshape((batch * frames, heads, joints, head_dim)),
            v.reshape((batch * frames, heads, joints, head_dim)),
            scale=head_dim**-0.5, mask=mask,
        ).reshape((batch, frames, heads, joints, head_dim))
    else:
        logits = (q @ mx.swapaxes(k, -1, -2)).astype(mx.float32) * (head_dim**-0.5)
        logits = logits + graph_bias[:, None]
        probs = mx.softmax(logits, axis=-1).astype(v.dtype)
        attended = probs @ v
    out = mx.transpose(attended, (0, 1, 3, 2, 4)).reshape((batch, frames, joints, dim))
    return linear(out, w, f"{prefix}.proj")


def _temporal_attention(x: mx.array, valid: mx.array, w: Weights, prefix: str, heads: int, base: int, backend: str) -> mx.array:
    batch, frames, joints, dim = x.shape
    head_dim = dim // heads
    per_joint = mx.transpose(x, (0, 2, 1, 3))
    packed = linear(per_joint, w, f"{prefix}.qkv").reshape((batch, joints, frames, 3, heads, head_dim))
    q, k, v = (
        mx.transpose(mx.squeeze(part, axis=3), (0, 1, 3, 2, 4))
        for part in mx.split(packed, 3, axis=3)
    )
    q = rms_norm(q, w[f"{prefix}.q_norm.weight"])
    k = rms_norm(k, w[f"{prefix}.k_norm.weight"])
    inv_freq = mx.exp(-math.log(base) * mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
    phases = mx.arange(frames, dtype=mx.float32)[:, None] * inv_freq[None, :]
    phases = mx.concatenate((phases, phases), axis=-1)
    q, k = apply_rope(q, k, mx.cos(phases)[None, None, None], mx.sin(phases)[None, None, None])
    if backend == "fast":
        mask = mx.broadcast_to(valid[:, None, None, None, :], (batch, joints, 1, 1, frames))
        mask = mask.reshape((batch * joints, 1, 1, frames))
        attended = mx.fast.scaled_dot_product_attention(
            q.reshape((batch * joints, heads, frames, head_dim)),
            k.reshape((batch * joints, heads, frames, head_dim)),
            v.reshape((batch * joints, heads, frames, head_dim)),
            scale=head_dim**-0.5, mask=mask,
        ).reshape((batch, joints, heads, frames, head_dim))
    else:
        logits = (q @ mx.swapaxes(k, -1, -2)).astype(mx.float32) * (head_dim**-0.5)
        logits = mx.where(valid[:, None, None, None, :], logits, -mx.inf)
        probs = mx.softmax(logits, axis=-1).astype(v.dtype)
        attended = probs @ v
    out = mx.transpose(attended, (0, 1, 3, 2, 4)).reshape((batch, joints, frames, dim))
    return linear(mx.transpose(out, (0, 2, 1, 3)), w, f"{prefix}.proj")


def _block(x: mx.array, y: mx.array, frame_valid: mx.array, angles: mx.array, graph_bias: mx.array, w: Weights, prefix: str, heads: int, temporal_base: int, attention_backend: str) -> mx.array:
    modulation = linear(nn.silu(y), w, f"{prefix}.adaLN_modulation.1")
    shift_s, scale_s, gate_s, shift_t, scale_t, gate_t, shift_m, scale_m, gate_m = mx.split(modulation, 9, axis=-1)

    def expand(z: mx.array) -> mx.array:
        return z[:, None, None, :]

    spatial = modulate(rms_norm(x, w[f"{prefix}.norm_s.weight"]), expand(shift_s), expand(scale_s))
    x = x + expand(gate_s) * _spatial_attention(spatial, graph_bias, angles, w, f"{prefix}.s_attn", heads, attention_backend)
    temporal = modulate(rms_norm(x, w[f"{prefix}.norm_t.weight"]), expand(shift_t), expand(scale_t))
    x = x + expand(gate_t) * _temporal_attention(temporal, frame_valid, w, f"{prefix}.t_attn", heads, temporal_base, attention_backend)
    mlp_input = modulate(rms_norm(x, w[f"{prefix}.norm_mlp.weight"]), expand(shift_m), expand(scale_m))
    return x + expand(gate_m) * swiglu(mlp_input, w, f"{prefix}.mlp")


def _final(x: mx.array, y: mx.array, valid: mx.array, w: Weights) -> mx.array:
    prefix = "final_layer"
    shift, scale = mx.split(linear(nn.silu(y), w, f"{prefix}.adaLN_modulation.1"), 2, axis=-1)
    x = modulate(rms_norm(x, w[f"{prefix}.norm_final.weight"]), shift[:, None, None], scale[:, None, None])
    batch, frames, joints, dim = x.shape
    root, other = x[:, :, :1], x[:, :, 1:]
    root_flat = root.reshape((batch * frames, 1, dim))
    other_flat = other.reshape((batch * frames, joints - 1, dim))
    q = rms_norm(root_flat, w[f"{prefix}.root_cross_norm_q.weight"])
    kv = rms_norm(other_flat, w[f"{prefix}.root_cross_norm_kv.weight"])
    kv_valid = mx.broadcast_to(valid[:, None, 1:], (batch, frames, joints - 1)).reshape((batch * frames, joints - 1))
    root_agg = _multihead_attention(q, kv, other_flat, w, f"{prefix}.root_cross_attn", 4, kv_valid)
    root = root + root_agg.reshape((batch, frames, 1, dim))
    out = mx.concatenate((_mlp(root, w, f"{prefix}.root_out"), _mlp(other, w, f"{prefix}.joint_out")), axis=2)
    return mx.transpose(out, (0, 2, 3, 1))


def prepare_static_condition(cond: Condition, weights: Weights, num_layers: int = 10) -> StaticCondition:
    """Compute per-character graph biases and SignNet angles once."""
    joints = cond["tpos_first_frame"].shape[1]
    valid = mx.arange(joints)[None, :] < cond["n_joints"][:, None]
    angles = spectral_angles(cond["spectral_feats"], weights)
    biases = tuple(
        _graph_bias(cond, valid, weights, f"transformer_blocks.{index}.s_attn")
        for index in range(num_layers)
    )
    mx.eval(angles, *biases)
    return StaticCondition(angles, biases)


def velocity(
    motion: mx.array,
    timesteps: mx.array,
    cond: Condition,
    weights: Weights,
    *,
    num_layers: int = 10,
    num_heads: int = 8,
    max_motion_length: int = 60,
    force_unconditional: bool = False,
    activations: dict[str, mx.array] | None = None,
    static_condition: StaticCondition | None = None,
    attention_backend: str = "fast",
) -> mx.array:
    """Predict velocity for the released graph/AdaLN config.

    The caller must supply reference-compatible graph and spectral features.
    ``motion_length`` and ``n_joints`` mark valid tokens in padded arrays.
    """
    if motion.ndim != 4 or motion.shape[2] != 12:
        raise ValueError("motion must have shape (B, J, 12, F)")
    batch, joints, _, frames = motion.shape
    if frames > max_motion_length:
        raise ValueError("motion exceeds configured max_motion_length")
    if not 2 <= joints <= 71:
        raise ValueError("supported checkpoints accept 2..71 joints")
    if cond["tpos_first_frame"].shape[1] != joints:
        raise ValueError("conditioning joint axis must match motion")
    if timesteps.shape != (batch,):
        raise ValueError("timesteps must have shape (B,)")
    if attention_backend not in {"manual", "fast"}:
        raise ValueError("attention_backend must be 'manual' or 'fast'")

    caption = cond.get("caption_emb")
    if caption is None or force_unconditional:
        caption = mx.zeros((batch, weights["cond_embedder.weight"].shape[1]), dtype=motion.dtype)
    y = time_condition(timesteps, weights) + linear(caption, weights, "cond_embedder")
    x, tpos, joint_valid = _encode_input(motion, cond, weights)
    y = y + _tpos_pool(tpos, joint_valid, weights)
    depth = mx.minimum(cond["joint_depths"], weights["depth_embedding.weight"].shape[0] - 1)
    x = x + mx.take(weights["depth_embedding.weight"], depth, axis=0)[:, None]
    x = x + linear(cond["joint_names_emb"], weights, "joint_name_embedder")[:, None]
    frame_valid = mx.arange(frames + 1)[None, :] < cond["motion_length"][:, None] + 1
    static = static_condition or prepare_static_condition(cond, weights, num_layers)
    if len(static.graph_biases) != num_layers:
        raise ValueError("static condition graph-bias count must match num_layers")
    temporal_base = (int(8 * (max_motion_length + 1) / math.pi) // 100 + 1) * 100
    for index in range(num_layers):
        x = _block(x, y, frame_valid, static.spectral_angles, static.graph_biases[index], weights, f"transformer_blocks.{index}", num_heads, temporal_base, attention_backend)
        if activations is not None:
            activations[f"block.{index}"] = x
    output = _final(x, y, joint_valid, weights)[:, :, :, 1:]
    if activations is not None:
        activations["velocity"] = output
    return output
