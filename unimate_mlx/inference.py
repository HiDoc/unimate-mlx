"""MLX flow-matching inference for precomputed UniMate conditioning."""

from __future__ import annotations

from collections.abc import Mapping

import mlx.core as mx
import numpy as np

from .model import prepare_static_condition, velocity


_MODEL_CONDITION_KEYS = (
    "caption_emb", "tpos_first_frame", "tpos_first_frame_parents", "n_joints",
    "motion_length", "joint_depths", "joint_names_emb", "spectral_feats",
    "graph_dist", "joint_relations",
)


def _batched_cfg_condition(condition: Mapping[str, mx.array]) -> dict[str, mx.array]:
    """Put conditioned and zero-caption branches in one model batch."""
    doubled = {
        name: mx.concatenate((condition[name], condition[name]), axis=0)
        for name in _MODEL_CONDITION_KEYS
        if name != "caption_emb"
    }
    caption = condition["caption_emb"]
    doubled["caption_emb"] = mx.concatenate((caption, mx.zeros_like(caption)), axis=0)
    return doubled


def _trim_joint_padding(noise: mx.array, condition: Mapping[str, mx.array]):
    """Crop joints that every sample masks out, retaining the original width."""
    original_joints = noise.shape[1]
    valid_joints = int(mx.max(condition["n_joints"]).item())
    if not 2 <= valid_joints <= original_joints:
        raise ValueError("n_joints must be within the motion joint axis")
    if valid_joints == original_joints:
        return noise, condition, original_joints
    matrix_keys = {"graph_dist", "joint_relations"}
    joint_keys = {
        "tpos_first_frame", "tpos_first_frame_parents", "joint_depths",
        "joint_names_emb", "spectral_feats",
    }
    cropped = {}
    for name in _MODEL_CONDITION_KEYS:
        value = condition[name]
        if name in matrix_keys:
            value = value[:, :valid_joints, :valid_joints]
        elif name in joint_keys:
            value = value[:, :valid_joints]
        cropped[name] = value
    return noise[:, :valid_joints], cropped, original_joints


def _restore_joint_padding(motion: mx.array, original_joints: int) -> mx.array:
    missing = original_joints - motion.shape[1]
    return motion if missing == 0 else mx.pad(motion, ((0, 0), (0, missing), (0, 0), (0, 0)))


def initial_noise(shape: tuple[int, int, int, int], seed: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Create repeatable MLX noise; PyTorch uses a different RNG sequence."""
    if len(shape) != 4 or any(size <= 0 for size in shape):
        raise ValueError("shape must be positive (B, J, 12, F) dimensions")
    if shape[2] != 12:
        raise ValueError("UniMate motion has 12 features per joint")
    mx.random.seed(seed)
    return mx.random.normal(shape, dtype=dtype)


def _guided_predictor(condition, weights, guidance_scale, force_unconditional, cfg_mode, attention_backend, compile_model):
    static = None if guidance_scale > 1 and cfg_mode == "batched" else prepare_static_condition(condition, weights)
    batched_condition = None
    batched_static = None
    if guidance_scale > 1 and cfg_mode == "batched":
        batched_condition = _batched_cfg_condition(condition)
        batched_static = prepare_static_condition(batched_condition, weights)

    conditioned_call = lambda state, timestep: velocity(
        state, timestep, condition, weights,
        static_condition=static, attention_backend=attention_backend,
    )
    unconditioned_call = lambda state, timestep: velocity(
        state, timestep, condition, weights, force_unconditional=True,
        static_condition=static, attention_backend=attention_backend,
    )
    batched_call = None
    if batched_condition is not None:
        batched_call = lambda state, timestep: velocity(
            state, timestep, batched_condition, weights,
            static_condition=batched_static, attention_backend=attention_backend,
        )
    if compile_model:
        conditioned_call = mx.compile(conditioned_call)
        unconditioned_call = mx.compile(unconditioned_call)
        if batched_call is not None:
            batched_call = mx.compile(batched_call)

    def predict(state: mx.array, time: float) -> mx.array:
        timestep = mx.full((state.shape[0],), float(time), dtype=mx.float32)
        if force_unconditional:
            return unconditioned_call(state, timestep)
        if batched_condition is not None:
            paired_motion = mx.concatenate((state, state), axis=0)
            paired_time = mx.concatenate((timestep, timestep), axis=0)
            paired_velocity = batched_call(paired_motion, paired_time)
            conditioned, unconditioned = mx.split(paired_velocity, 2, axis=0)
            return unconditioned + guidance_scale * (conditioned - unconditioned)
        conditioned = conditioned_call(state, timestep)
        if guidance_scale > 1:
            unconditioned = unconditioned_call(state, timestep)
            return unconditioned + guidance_scale * (conditioned - unconditioned)
        return conditioned

    return predict


def sample_fixed(
    noise: mx.array,
    condition: Mapping[str, mx.array],
    weights: Mapping[str, mx.array],
    *,
    num_steps: int = 50,
    guidance_scale: float = 3.0,
    force_unconditional: bool = False,
    evaluate_each_step: bool = True,
    cfg_mode: str = "separate",
    attention_backend: str = "fast",
    method: str = "euler",
    compile_model: bool = False,
    trim_joints: bool = False,
) -> mx.array:
    """Integrate the released velocity model from t=0 to t=1.

    ``num_steps`` is the number of time grid points, matching the official
    fixed Euler path. Thus the model is evaluated ``num_steps - 1`` times.
    The official CLI defaults to adaptive Dormand–Prince; this fixed path is
    for reproducible first-stage MLX parity tests.
    """
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if guidance_scale < 1:
        raise ValueError("guidance_scale must be at least 1")
    if noise.ndim != 4 or noise.shape[2] != 12:
        raise ValueError("noise must have shape (B, J, 12, F)")
    if guidance_scale > 1 and force_unconditional:
        raise ValueError("force_unconditional conflicts with guidance_scale > 1")
    if cfg_mode not in {"separate", "batched"}:
        raise ValueError("cfg_mode must be 'separate' or 'batched'")
    if method not in {"euler", "rk4"}:
        raise ValueError("method must be 'euler' or 'rk4'")

    original_joints = noise.shape[1]
    if trim_joints:
        noise, condition, original_joints = _trim_joint_padding(noise, condition)
    x = noise
    grid = np.linspace(0, 1, num_steps, dtype=np.float32)
    predict = _guided_predictor(condition, weights, guidance_scale, force_unconditional, cfg_mode, attention_backend, compile_model)

    for index in range(num_steps - 1):
        start = float(grid[index])
        dt = float(grid[index + 1] - grid[index])
        if method == "euler":
            x = x + dt * predict(x, start)
        else:
            one_third = 1.0 / 3.0
            first_stage = float(np.float32(grid[index] + np.float32(dt * one_third)))
            second_stage = float(np.float32(grid[index] + np.float32(dt * (2.0 * one_third))))
            end = float(grid[index + 1])
            k1 = predict(x, start)
            if evaluate_each_step:
                mx.eval(k1)
            k2 = predict(x + dt * k1 * one_third, first_stage)
            if evaluate_each_step:
                mx.eval(k2)
            k3 = predict(x + dt * (k2 - k1 * one_third), second_stage)
            if evaluate_each_step:
                mx.eval(k3)
            k4 = predict(x + dt * (k1 - k2 + k3), end)
            x = x + (dt / 8.0) * (k1 + 3 * (k2 + k3) + k4)
        if evaluate_each_step:
            mx.eval(x)
    return _restore_joint_padding(x, original_joints)


def sample_euler(*args, **kwargs) -> mx.array:
    """Backward-compatible fixed Euler entry point."""
    return sample_fixed(*args, method="euler", **kwargs)


def sample_rk4(*args, **kwargs) -> mx.array:
    """Torchdiffeq's 3/8-rule RK4 on the reference time grid."""
    return sample_fixed(*args, method="rk4", **kwargs)


def sample_adaptive(
    noise: mx.array,
    condition: Mapping[str, mx.array],
    weights: Mapping[str, mx.array],
    *,
    guidance_scale: float = 3.0,
    force_unconditional: bool = False,
    cfg_mode: str = "separate",
    attention_backend: str = "fast",
    rtol: float = 1e-3,
    atol: float = 1e-6,
    diagnostics: dict[str, int] | None = None,
    compile_model: bool = False,
    trim_joints: bool = False,
) -> mx.array:
    """Adaptive Dormand–Prince sampling with official tolerances."""
    if noise.ndim != 4 or noise.shape[2] != 12:
        raise ValueError("noise must have shape (B, J, 12, F)")
    if noise.dtype != mx.float32:
        raise ValueError("adaptive Dopri5 currently requires float32 motion and weights")
    if guidance_scale < 1 or (guidance_scale > 1 and force_unconditional):
        raise ValueError("invalid guidance_scale and force_unconditional combination")
    if cfg_mode not in {"separate", "batched"}:
        raise ValueError("cfg_mode must be 'separate' or 'batched'")
    from .solvers import dopri5

    original_joints = noise.shape[1]
    if trim_joints:
        noise, condition, original_joints = _trim_joint_padding(noise, condition)
    predict = _guided_predictor(condition, weights, guidance_scale, force_unconditional, cfg_mode, attention_backend, compile_model)
    result = dopri5(lambda time, state: predict(state, time), noise, rtol=rtol, atol=atol, diagnostics=diagnostics)
    return _restore_joint_padding(result, original_joints)
