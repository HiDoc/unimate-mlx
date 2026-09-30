import numpy as np
import pytest

try:
    import mlx.core as mx
    from unimate_mlx.primitives import apply_rope, rms_norm, spectral_angles, timestep_embedding
except ImportError as error:
    pytest.skip(f"MLX device unavailable: {error}", allow_module_level=True)


def test_rms_norm_uses_fp32_reduction():
    values = np.array([[0.25, -1.5, 2.0, 3.0]], dtype=np.float16)
    weight = np.array([1.0, 0.5, -2.0, 1.25], dtype=np.float32)
    actual = np.array(rms_norm(mx.array(values), mx.array(weight)))
    expected = (values.astype(np.float32) / np.sqrt(np.mean(values.astype(np.float32) ** 2, axis=-1, keepdims=True) + 1e-6) * weight).astype(np.float16)
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=0)


def test_timestep_embedding_cos_then_sin():
    actual = np.array(timestep_embedding(mx.array([0.0, 1.0]), dim=5))
    np.testing.assert_allclose(actual[0], [1, 1, 0, 0, 0], atol=1e-7)
    np.testing.assert_allclose(actual[1, [0, 2]], [np.cos(1), np.sin(1)], atol=1e-6)


def test_rope_preserves_norm():
    q = mx.array(np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4))
    angles = mx.array(np.array([0.2, 0.5, 0.2, 0.5], dtype=np.float32).reshape(1, 1, 1, 4))
    rotated, _ = apply_rope(q, q, mx.cos(angles), mx.sin(angles))
    np.testing.assert_allclose(np.sum(np.array(rotated) ** 2, axis=-1), np.sum(np.array(q) ** 2, axis=-1), atol=1e-5)


def test_spectral_sign_invariance():
    rng = np.random.default_rng(42)
    shapes = {"phi.0": (64, 1), "phi.2": (64, 64), "rho.0": (64, 512), "rho.2": (32, 64)}
    weights = {}
    for name, shape in shapes.items():
        weights[f"rope_j.spectral_encoder.{name}.weight"] = mx.array(rng.normal(0, 0.02, shape).astype(np.float32))
        weights[f"rope_j.spectral_encoder.{name}.bias"] = mx.array(rng.normal(0, 0.02, shape[0]).astype(np.float32))
    coords = rng.normal(size=(2, 5, 8)).astype(np.float32)
    signs = rng.choice([-1, 1], size=(2, 1, 8)).astype(np.float32)
    actual = np.array(spectral_angles(mx.array(coords), weights))
    flipped = np.array(spectral_angles(mx.array(coords * signs), weights))
    np.testing.assert_allclose(actual, flipped, atol=1e-5, rtol=0)
