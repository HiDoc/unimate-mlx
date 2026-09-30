"""Safetensors inspection and loading helpers.

The header parser intentionally uses only the Python standard library so a
checkpoint can be audited without importing either Safetensors or MLX.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}
_MAX_HEADER_SIZE = 100 * 1024 * 1024


class SafetensorsFormatError(ValueError):
    """Raised when a Safetensors file has an invalid header or data layout."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "data_offsets": list(self.data_offsets),
        }


@dataclass(frozen=True)
class SafetensorsHeader:
    tensors: tuple[TensorInfo, ...]
    metadata: Mapping[str, str]
    header_size: int
    data_size: int


def inspect_safetensors_header(path: str | Path) -> SafetensorsHeader:
    """Read and validate tensor names, shapes, dtypes, and data offsets."""
    path = Path(path)
    file_size = path.stat().st_size
    if file_size < 8:
        raise SafetensorsFormatError("file is shorter than the 8-byte header length")
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        header_size = struct.unpack("<Q", raw_size)[0]
        if header_size == 0 or header_size > _MAX_HEADER_SIZE:
            raise SafetensorsFormatError(f"invalid header size: {header_size}")
        if 8 + header_size > file_size:
            raise SafetensorsFormatError("header extends past end of file")
        try:
            header_obj = json.loads(stream.read(header_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafetensorsFormatError(f"invalid JSON header: {exc}") from exc

    if not isinstance(header_obj, dict):
        raise SafetensorsFormatError("header JSON must be an object")
    data_size = file_size - 8 - header_size
    metadata = header_obj.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()
    ):
        raise SafetensorsFormatError("__metadata__ must map strings to strings")

    tensors: list[TensorInfo] = []
    spans: list[tuple[int, int, str]] = []
    for name, entry in header_obj.items():
        if not isinstance(name, str) or not name or not isinstance(entry, dict):
            raise SafetensorsFormatError(f"invalid tensor entry {name!r}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype not in _DTYPE_BYTES:
            raise SafetensorsFormatError(f"{name}: unsupported or invalid dtype {dtype!r}")
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
        ):
            raise SafetensorsFormatError(f"{name}: shape must be a list of nonnegative integers")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(x, int) or isinstance(x, bool) for x in offsets)
        ):
            raise SafetensorsFormatError(f"{name}: data_offsets must contain two integers")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise SafetensorsFormatError(
                f"{name}: offsets [{start}, {end}] outside data section of {data_size} bytes"
            )
        elements = 1
        for dim in shape:
            elements *= dim
        expected_nbytes = elements * _DTYPE_BYTES[dtype]
        if end - start != expected_nbytes:
            raise SafetensorsFormatError(
                f"{name}: offsets span {end - start} bytes, expected {expected_nbytes}"
            )
        tensors.append(TensorInfo(name, dtype, tuple(shape), (start, end)))
        if end > start:
            spans.append((start, end, name))

    spans.sort()
    for previous, current in zip(spans, spans[1:]):
        if current[0] < previous[1]:
            raise SafetensorsFormatError(
                f"overlapping data offsets for {previous[2]!r} and {current[2]!r}"
            )
        if current[0] != previous[1]:
            raise SafetensorsFormatError("tensor offsets contain a gap in the data section")
    if spans and (spans[0][0] != 0 or spans[-1][1] != data_size):
        raise SafetensorsFormatError("tensor offsets do not cover the complete data section")
    if not spans and data_size:
        raise SafetensorsFormatError("data section is nonempty but header has no tensors")
    return SafetensorsHeader(tuple(tensors), metadata, header_size, data_size)


def validate_manifest(header: SafetensorsHeader, manifest: str | Path | Mapping[str, Any]) -> None:
    """Require exact tensor names and shapes from a JSON manifest.

    The manifest may be ``{"tensors": {name: {"shape": [...], "dtype": "F32"}}}``
    or a direct ``{name: {"shape": [...], "dtype": "F32"}}`` mapping. Dtype
    is checked when specified.
    """
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    expected = manifest.get("tensors", manifest)
    if not isinstance(expected, Mapping):
        raise ValueError("manifest must be a mapping of tensor names to specifications")
    actual = {tensor.name: tensor for tensor in header.tensors}
    expected_names = set(expected)
    actual_names = set(actual)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    problems = []
    if missing:
        problems.append(f"missing tensors: {', '.join(missing)}")
    if extra:
        problems.append(f"unexpected tensors: {', '.join(extra)}")
    for name in sorted(expected_names & actual_names):
        spec = expected[name]
        if not isinstance(spec, Mapping) or "shape" not in spec:
            raise ValueError(f"manifest entry {name!r} must specify shape")
        wanted_shape = tuple(spec["shape"])
        if actual[name].shape != wanted_shape:
            problems.append(f"{name}: shape {actual[name].shape} != expected {wanted_shape}")
        if "dtype" in spec and actual[name].dtype != spec["dtype"]:
            problems.append(f"{name}: dtype {actual[name].dtype} != expected {spec['dtype']}")
    if problems:
        raise ValueError("manifest mismatch: " + "; ".join(problems))


def load_safetensors(path: str | Path, *, backend: str = "numpy") -> dict[str, Any]:
    """Load tensors with Safetensors NumPy; optionally convert them to MLX arrays."""
    if backend not in {"numpy", "mlx"}:
        raise ValueError("backend must be 'numpy' or 'mlx'")
    # Validate structure before the dependency loader maps tensor contents.
    inspect_safetensors_header(path)
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise ImportError("loading weights requires the 'safetensors' package") from exc
    weights = load_file(str(path))
    if backend == "numpy":
        return weights
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise ImportError("backend='mlx' requires MLX on Apple Silicon") from exc
    return {name: mx.array(value) for name, value in weights.items()}
