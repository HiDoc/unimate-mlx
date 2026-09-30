import unittest
from types import SimpleNamespace

import numpy as np

from unimate_mlx.text import DEFAULT_T5_MODEL, FrozenT5Encoder


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        # A predictable two-token encoding. Empty strings still receive
        # placeholder tokens; the encoder must zero their attention mask.
        ids = np.array([[len(text), len(text) + 1] for text in texts], dtype=np.int64)
        return {"input_ids": ids, "attention_mask": np.ones_like(ids)}


class FakeModel:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def __call__(self, input_ids, attention_mask):
        hidden = np.stack([input_ids, input_ids * 2], axis=-1).astype(np.float32)
        return SimpleNamespace(last_hidden_state=hidden)


class FrozenT5EncoderTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()
        self.encoder = FrozenT5Encoder(
            tokenizer=self.tokenizer, model=self.model, device="cpu"
        )

    def test_reference_tokenizer_options_and_masked_mean(self):
        embeddings = self.encoder.encode(["walk", "", None])
        self.assertEqual(embeddings.shape, (3, 2))
        self.assertTrue(np.allclose(embeddings[0], [4.5, 9.0]))
        self.assertTrue(np.array_equal(embeddings[1], [0.0, 0.0]))
        self.assertTrue(np.array_equal(embeddings[2], [0.0, 0.0]))
        texts, kwargs = self.tokenizer.calls[0]
        self.assertEqual(texts, ["walk", "", ""])
        self.assertEqual(kwargs, {"return_tensors": "pt", "padding": True})
        self.assertTrue(self.model.eval_called)

    def test_single_prompt_and_joint_name_shapes(self):
        self.assertEqual(self.encoder.prompt_embedding("run").shape, (2,))
        self.assertEqual(self.encoder.joint_name_embeddings(["Hip", "Knee"]).shape, (2, 2))

    def test_default_model_matches_reference_and_rejects_partial_injection(self):
        self.assertEqual(DEFAULT_T5_MODEL, "google/flan-t5-base")
        with self.assertRaises(ValueError):
            FrozenT5Encoder(tokenizer=self.tokenizer)


if __name__ == "__main__":
    unittest.main()
