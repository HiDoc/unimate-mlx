"""Audit UniMate character runs and their exported animation artifacts."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def _glb_document(path: Path) -> dict:
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError("GLB header is truncated")
        magic, version, total_length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2 or total_length != path.stat().st_size:
            raise ValueError("GLB header or declared length is invalid")
        chunk_length, kind = struct.unpack("<I4s", stream.read(8))
        if kind != b"JSON":
            raise ValueError("GLB first chunk is not JSON")
        return json.loads(stream.read(chunk_length))


def _tree_depth(parents: np.ndarray) -> int:
    count = len(parents)
    roots = [index for index, parent in enumerate(parents) if parent < 0 or parent == index]
    if len(roots) != 1:
        raise ValueError(f"expected one root, found {len(roots)}")
    root = roots[0]
    maximum = 0
    for joint in range(count):
        visited = set()
        cursor = joint
        depth = 0
        while cursor != root:
            if cursor in visited or not 0 <= cursor < count:
                raise ValueError("skeleton has a cycle or invalid parent")
            visited.add(cursor)
            cursor = int(parents[cursor])
            depth += 1
        maximum = max(maximum, depth)
    return maximum


def audit(run: Path) -> dict:
    manifest_path = run / "run.json"
    if manifest_path.is_file():
        preprocessing = run / json.loads(manifest_path.read_text())["preprocessed"]
    else:
        candidates = list(run.glob("preprocessed*/cond.npy"))
        if len(candidates) != 1:
            raise ValueError("cannot identify one preprocessing directory; run.json is required")
        preprocessing = candidates[0].parent
    topology_map = np.load(preprocessing / "cond.npy", allow_pickle=True).item()
    if "parents" in topology_map:
        topology = topology_map
    elif len(topology_map) == 1:
        topology = next(iter(topology_map.values()))
    else:
        raise ValueError("topology has multiple entries")
    parents = np.asarray(topology["parents"], dtype=np.int64)
    joints = len(parents)
    depth = _tree_depth(parents)
    if np.asarray(topology["spectral_feats"]).shape != (joints, 8):
        raise ValueError("spectral feature shape does not match checkpoint")

    with np.load(run / "conditioning.npz", allow_pickle=False) as condition:
        max_joints = int(condition["tpos_first_frame"].shape[1])
        if not 2 <= joints <= max_joints or max_joints not in (61, 71):
            raise ValueError(f"joint count {joints} outside selected checkpoint range 2..{max_joints}")
        if int(condition["n_joints"][0]) != joints:
            raise ValueError("conditioning joint count differs from topology")
        if condition["graph_dist"].shape != (1, max_joints, max_joints):
            raise ValueError(f"graph distances are not padded to {max_joints} joints")
        if not np.isfinite(condition["tpos_first_frame"]).all():
            raise ValueError("T-pose conditioning contains nonfinite values")

    motion = np.load(run / "motion_normalized.npy", allow_pickle=False)
    features = np.load(run / "motion_features.npy", allow_pickle=False)
    if motion.ndim != 4 or motion.shape[:3] != (1, max_joints, 12):
        raise ValueError("normalized motion has wrong shape")
    frames = motion.shape[3]
    if features.shape != (frames, joints, 12):
        raise ValueError("denormalized feature shape differs from motion/topology")
    if not np.isfinite(motion).all() or not np.isfinite(features).all():
        raise ValueError("generated motion contains nonfinite values")
    temporal_std = float(np.mean(np.std(features.astype(np.float32), axis=0)))
    if temporal_std <= 0:
        raise ValueError("generated motion has no temporal variation")

    animated = run / "animated"
    glb = animated / "motion_features.glb"
    fbx = animated / "motion_features.fbx"
    document = _glb_document(glb)
    channels = sum(len(animation.get("channels", [])) for animation in document.get("animations", []))
    skins = len(document.get("skins", []))
    if not channels or not skins:
        raise ValueError("GLB has no animation channels or skin")
    if fbx.stat().st_size <= 1024:
        raise ValueError("FBX export is missing or too small")

    return {
        "run": str(run),
        "joints": joints,
        "padded_joints": max_joints,
        "max_depth": depth,
        "frames": frames,
        "temporal_std": temporal_std,
        "glb_channels": channels,
        "glb_skins": skins,
        "glb_bytes": glb.stat().st_size,
        "fbx_bytes": fbx.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, help="optional JSON report")
    args = parser.parse_args()
    results = []
    failed = False
    for run in args.runs:
        try:
            results.append(audit(run))
        except (OSError, ValueError, KeyError, TypeError) as error:
            failed = True
            results.append({"run": str(run), "error": str(error)})
    report = {"passed": sum("error" not in item for item in results), "total": len(results), "results": results}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
