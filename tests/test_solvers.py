import math

import numpy as np
import pytest

try:
    import mlx.core as mx
    from unimate_mlx.solvers import dopri5
except ImportError as error:
    pytest.skip(f"MLX device unavailable: {error}", allow_module_level=True)


def test_dopri5_exponential_decay_and_diagnostics():
    initial = mx.array([1.0, 2.0], dtype=mx.float32)
    diagnostics = {}
    result = dopri5(lambda time, state: -state, initial, rtol=1e-6, atol=1e-8, diagnostics=diagnostics)
    np.testing.assert_allclose(np.array(result), np.array([1.0, 2.0]) * math.exp(-1), atol=1e-4)
    assert diagnostics["evaluations"] >= 8
    assert diagnostics["accepted_steps"] > 0


def test_dopri5_constant_velocity():
    result = dopri5(lambda time, state: mx.ones_like(state) * 2, mx.zeros((2, 3)))
    np.testing.assert_allclose(np.array(result), np.full((2, 3), 2), atol=1e-5)
