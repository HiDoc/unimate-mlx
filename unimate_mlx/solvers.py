"""Adaptive Dormand–Prince 5(4) integration for MLX velocity fields.

The coefficients and RMS error controller follow torchdiffeq's ``dopri5``
settings. Model states remain MLX arrays; only the scalar error ratio crosses
to Python to choose the next step. The final step lands exactly on ``t1``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mlx.core as mx


_A = (
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_C = (1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
_B = (35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0)
_E = (
    35 / 384 - 1951 / 21600,
    0,
    500 / 1113 - 22642 / 50085,
    125 / 192 - 451 / 720,
    -2187 / 6784 + 12231 / 42400,
    11 / 84 - 649 / 6300,
    -1 / 60,
)


def _rms(value: mx.array) -> float:
    value = value.astype(mx.float32)
    return float(mx.sqrt(mx.mean(mx.square(value))).item())


def _weighted_sum(weights: tuple[float, ...], values: list[mx.array]) -> mx.array:
    total = weights[0] * values[0]
    for weight, value in zip(weights[1:], values[1:]):
        if weight:
            total = total + weight * value
    return total


def dopri5(
    field: Callable[[float, mx.array], mx.array],
    initial: mx.array,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    max_steps: int = 1000,
    diagnostics: dict[str, int] | None = None,
) -> mx.array:
    """Integrate ``dy/dt = field(t,y)`` with adaptive 5(4) error control."""
    if not t0 < t1 or rtol <= 0 or atol <= 0 or max_steps < 1:
        raise ValueError("require t0 < t1, positive tolerances and positive max_steps")
    y = initial
    t = float(t0)
    f = field(t, y)
    mx.eval(f)
    evaluations = 1

    # Hairer's empirical initial step, as used by torchdiffeq.
    scale = atol + rtol * mx.abs(y)
    d0 = _rms(y / scale)
    d1 = _rms(f / scale)
    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
    h0 = min(h0, t1 - t)
    probe = field(t + h0, y + h0 * f)
    mx.eval(probe)
    evaluations += 1
    d2 = _rms((probe - f) / scale) / h0
    h1 = max(1e-6, h0 * 1e-3) if max(d1, d2) <= 1e-15 else (0.01 / max(d1, d2)) ** (1 / 5)
    h = min(100 * h0, h1, t1 - t)

    accepted = 0
    rejected = 0
    for _ in range(max_steps):
        if t >= t1:
            break
        h = min(h, t1 - t)
        if h <= 0 or t + h <= t:
            raise RuntimeError("adaptive ODE step underflow")
        stages = [f]
        for index, coefficients in enumerate(_A):
            stage_y = y + h * _weighted_sum(coefficients, stages)
            stage_t = t + _C[index] * h
            stage_f = field(stage_t, stage_y)
            mx.eval(stage_f)
            stages.append(stage_f)
            evaluations += 1
        next_y = y + h * _weighted_sum(_B, stages)
        error = h * _weighted_sum(_E, stages)
        error_scale = atol + rtol * mx.maximum(mx.abs(y), mx.abs(next_y))
        ratio = _rms(error / error_scale)
        if not math.isfinite(ratio):
            raise RuntimeError("nonfinite adaptive ODE error ratio")
        if ratio <= 1:
            y = next_y
            mx.eval(y)
            f = stages[-1]  # FSAL: final stage is the next step's first.
            t += h
            accepted += 1
        else:
            rejected += 1
        if ratio == 0:
            factor = 10.0
        else:
            factor = max(0.2, min(10.0, 0.9 * ratio ** (-1 / 5)))
            if ratio > 1:
                factor = min(1.0, factor)
        h *= factor
    else:
        raise RuntimeError(f"adaptive ODE exceeded {max_steps} trial steps")

    if diagnostics is not None:
        diagnostics.update(evaluations=evaluations, accepted_steps=accepted, rejected_steps=rejected)
    return y
