"""Conservative facing-pair inference for named GLB skeleton joints."""

from __future__ import annotations

import json
import struct
from pathlib import Path


_FRONT_HIP_PAIRS = (
    ("FR_hip", "FL_hip"),
    ("front_right_hip", "front_left_hip"),
    ("FrontRightHip", "FrontLeftHip"),
)


def skin_joint_names(path: str | Path) -> set[str]:
    """Read skin joint names from a GLB without loading geometry or animation."""
    path = Path(path)
    if path.suffix.lower() != ".glb":
        return set()
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError("GLB header is truncated")
        magic, version, _ = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2:
            raise ValueError("expected a GLB version 2 asset")
        chunk_header = stream.read(8)
        if len(chunk_header) != 8:
            raise ValueError("GLB JSON chunk is truncated")
        length, kind = struct.unpack("<I4s", chunk_header)
        if kind != b"JSON":
            raise ValueError("GLB first chunk must contain JSON")
        document = json.loads(stream.read(length))
    nodes = document.get("nodes", [])
    joint_indices = {index for skin in document.get("skins", []) for index in skin.get("joints", [])}
    return {nodes[index].get("name") for index in joint_indices if 0 <= index < len(nodes) and nodes[index].get("name")}


def infer_front_hip_pair(path: str | Path) -> tuple[str, str] | None:
    """Return an unambiguous front right/left hip pair from a rigged GLB."""
    names = skin_joint_names(path)
    matches = [pair for pair in _FRONT_HIP_PAIRS if pair[0] in names and pair[1] in names]
    return matches[0] if len(matches) == 1 else None
