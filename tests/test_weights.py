import json
import struct
import tempfile
import unittest
from pathlib import Path

from unimate_mlx.weights import (
    SafetensorsFormatError,
    inspect_safetensors_header,
    validate_manifest,
)


def write_safetensors(path: Path, tensors: dict) -> None:
    header = json.dumps(tensors, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00\x80?\x00\x00\x00@")


class HeaderParserTests(unittest.TestCase):
    def test_parses_tensor_metadata_and_validates_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.safetensors"
            write_safetensors(path, {
                "weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                "__metadata__": {"format": "pt"},
            })
            header = inspect_safetensors_header(path)
            self.assertEqual(header.tensors[0].name, "weight")
            self.assertEqual(header.tensors[0].shape, (2,))
            self.assertEqual(header.metadata["format"], "pt")
            validate_manifest(header, {"tensors": {"weight": {"shape": [2], "dtype": "F32"}}})

    def test_rejects_invalid_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.safetensors"
            write_safetensors(path, {
                "weight": {"dtype": "F32", "shape": [3], "data_offsets": [0, 12]},
            })
            with self.assertRaises(SafetensorsFormatError):
                inspect_safetensors_header(path)

    def test_manifest_requires_exact_names_and_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.safetensors"
            write_safetensors(path, {
                "weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            })
            header = inspect_safetensors_header(path)
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                validate_manifest(header, {"other": {"shape": [2]}})


if __name__ == "__main__":
    unittest.main()
