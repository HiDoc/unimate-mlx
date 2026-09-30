"""Small NumPy flow-matching sampler matching UniMate's linear-flow defaults.

The official reference defaults to ``torchdiffeq``'s adaptive ``dopri5`` ODE
solver. It also supports fixed Euler integration with 50 saved grid points.
This module implements that Euler path explicitly so it can be used without
the reference PyTorch stack or a full UniMate model.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


DEFAULT_NUM_STEPS = 50


def initial_noise(
    shape: Sequence[int], seed: int | None = None, dtype: np.dtype | type = np.float32
) -> np.ndarray:
    """Create deterministic standard-normal initial state when ``seed`` is set."""
    if not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise ValueError("shape must contain positive integer dimensions")
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")
    return np.random.default_rng(seed).standard_normal(tuple(shape), dtype=dtype)


def classifier_free_guidance(
    conditioned: np.ndarray, unconditioned: np.ndarray, scale: float
) -> np.ndarray:
    """Apply UniMate CFG: ``v_u + scale * (v_c - v_u)``."""
    if conditioned.shape != unconditioned.shape:
        raise ValueError("conditioned and unconditioned predictions must have equal shapes")
    return unconditioned + scale * (conditioned - unconditioned)


def euler_sample(
    model: Any,
    noise: np.ndarray,
    condition: Any = None,
    *,
    unconditional_condition: Any = None,
    cfg_scale: float = 1.0,
    force_unconditional: bool = False,
    num_steps: int = DEFAULT_NUM_STEPS,
    t0: float = 0.0,
    t1: float = 1.0,
) -> np.ndarray:
    """Integrate ``dx/dt = model(x,t,condition)`` using reference-grid Euler.

    ``num_steps`` is the number of points in the reference ``linspace`` grid,
    so Euler evaluates the model ``num_steps - 1`` times. The model signature
    is ``model(x, batch_timesteps, condition, force_unconditional=False)``;
    batch timesteps have shape ``(batch,)``. If an unconditional condition is
    supplied and ``cfg_scale > 1``, two predictions are combined via CFG.

    Reference inference uses ``cfg_scale == 1`` as a separate unconditional
    sampling mode. Set ``force_unconditional=True`` to reproduce that branch;
    ordinary CFG scale 1 with conditional inputs returns the conditional
    prediction, as implied by the standard CFG equation.
    """
    x = np.asarray(noise).copy()
    if x.ndim < 1 or x.shape[0] < 1:
        raise ValueError("noise must have a nonempty batch dimension")
    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError("noise must have a floating-point dtype")
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 2:
        raise ValueError("num_steps must be an integer >= 2")
    if not np.isfinite(cfg_scale) or cfg_scale < 1.0:
        raise ValueError("cfg_scale must be finite and >= 1.0")
    grid = np.linspace(t0, t1, num_steps, dtype=np.float64)
    batch = x.shape[0]

    def predict(state: np.ndarray, time: float) -> np.ndarray:
        times = np.full((batch,), time, dtype=state.dtype)
        if force_unconditional:
            result = model(state, times, condition, force_unconditional=True)
        elif unconditional_condition is not None and cfg_scale > 1.0:
            conditioned = np.asarray(model(state, times, condition))
            unconditioned = np.asarray(model(state, times, unconditional_condition))
            result = classifier_free_guidance(conditioned, unconditioned, cfg_scale)
        else:
            result = model(state, times, condition)
        result = np.asarray(result)
        if result.shape != state.shape:
            raise ValueError(f"model returned shape {result.shape}, expected {state.shape}")
        return result

    for index in range(num_steps - 1):
        dt = grid[index + 1] - grid[index]
        x = x + np.asarray(dt, dtype=x.dtype) * predict(x, float(grid[index]))
    return x
