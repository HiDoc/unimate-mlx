import tempfile
import unittest
from pathlib import Path

import numpy as np

from unimate_mlx.cache_text import precompute_embeddings


class FakeEncoder:
    def prompt_embedding(self, prompt):
        return np.array([len(prompt), 1], dtype=np.float32)

    def joint_name_embeddings(self, names):
        return np.array([[len(name), 2] for name in names], dtype=np.float32)


class CacheTextTests(unittest.TestCase):
    def test_writes_prompt_and_joint_embeddings_for_single_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            cond_path = Path(tmp) / "cond.npy"
            output_path = Path(tmp) / "cache" / "embeddings.npz"
            np.save(cond_path, {"character": {"clean_joint_names": ["Hip", "Knee"]}})
            seen = []

            def resolve(condition, object_type):
                seen.append((condition, object_type))
                return condition["clean_joint_names"]

            result = precompute_embeddings(
                cond_path,
                "walk forward",
                output_path,
                encoder=FakeEncoder(),
                joint_name_resolver=resolve,
            )
            self.assertEqual(result, output_path)
            with np.load(output_path) as cache:
                self.assertEqual(set(cache.files), {"caption_emb", "joint_names_emb"})
                self.assertTrue(np.array_equal(cache["caption_emb"], [12, 1]))
                self.assertTrue(np.array_equal(cache["joint_names_emb"], [[3, 2], [4, 2]]))
            self.assertEqual(seen[0][1], "character")

    def test_multi_entry_condition_requires_object_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            cond_path = Path(tmp) / "cond.npy"
            np.save(cond_path, {"a": {}, "b": {}})
            with self.assertRaisesRegex(ValueError, "object_type is required"):
                precompute_embeddings(
                    cond_path,
                    "walk",
                    Path(tmp) / "embeddings.npz",
                    encoder=FakeEncoder(),
                    joint_name_resolver=lambda cond, name: ["Hip"],
                )


if __name__ == "__main__":
    unittest.main()
