import unittest

import numpy as np

from unimate_mlx.sampler import (
    classifier_free_guidance,
    euler_sample,
    initial_noise,
)


class ConstantVelocity:
    def __init__(self, velocity):
        self.velocity = np.asarray(velocity, dtype=np.float32)
        self.calls = []

    def __call__(self, x, times, condition, force_unconditional=False):
        self.calls.append((times.copy(), condition, force_unconditional))
        if force_unconditional:
            return np.ones_like(x) * 2
        return np.ones_like(x) * (self.velocity + float(condition))


class SamplerTests(unittest.TestCase):
    def test_reference_grid_euler_update(self):
        model = ConstantVelocity(0)
        noise = np.zeros((2, 3), dtype=np.float32)
        result = euler_sample(model, noise, condition=2, num_steps=5)
        self.assertTrue(np.allclose(result, 2.0))  # four updates, each dt = 1/4
        self.assertEqual(len(model.calls), 4)
        self.assertEqual([float(call[0][0]) for call in model.calls], [0, .25, .5, .75])

    def test_cfg_formula_and_unconditional_branch(self):
        self.assertTrue(np.allclose(
            classifier_free_guidance(np.array([5.]), np.array([2.]), 3), [11.]
        ))
        model = ConstantVelocity(0)
        result = euler_sample(
            model, np.zeros((1, 2), dtype=np.float32), condition=7,
            unconditional_condition=0, cfg_scale=3, num_steps=2,
        )
        self.assertTrue(np.allclose(result, 21.0))
        self.assertEqual(len(model.calls), 2)
        uncond = ConstantVelocity(0)
        result = euler_sample(
            uncond, np.zeros((1, 2), dtype=np.float32), condition=7,
            force_unconditional=True, num_steps=2,
        )
        self.assertTrue(np.allclose(result, 2.0))
        self.assertEqual(uncond.calls[0][2], True)

    def test_seeded_initial_noise_is_repeatable(self):
        a = initial_noise((2, 3), seed=42)
        b = initial_noise((2, 3), seed=42)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, initial_noise((2, 3), seed=43)))


if __name__ == "__main__":
    unittest.main()
