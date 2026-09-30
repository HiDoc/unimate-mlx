"""Prepare model conditioning from the official character preprocessing output.

``preprocess_char.py`` writes a topology ``cond.npy`` entry with raw T-pose
positions and graph features. This module applies the released checkpoint's
dataset statistics and padding before MLX inference. Text embeddings are
provided by the frozen FLAN-T5 encoder or a cache.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


PREVIEW_MAX_JOINTS = 61
FEATURES = 12
TEXT_DIM = 768
SPECTRAL_FREQS = 8
IDENTITY_6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


def load_topology(path: str | Path, object_type: str | None = None) -> Mapping[str, Any]:
    """Read one trusted official ``cond.npy`` topology entry."""
    payload = np.load(path, allow_pickle=True).item()
    if "parents" in payload and "tpos_first_frame" in payload:
        return payload
    if object_type is None:
        if len(payload) != 1:
            raise ValueError("cond.npy contains multiple objects; specify object_type")
        return next(iter(payload.values()))
    try:
        return payload[object_type]
    except KeyError as error:
        raise ValueError(f"object_type {object_type!r} absent from cond.npy") from error


def model_joint_names(topology: Mapping[str, Any]) -> list[str]:
    """Select the complete cleaned name list used by official inference."""
    count = len(topology["parents"])
    cleaned = topology.get("clean_joint_names")
    if cleaned is not None and len(cleaned) == count and all(str(name).strip() for name in cleaned):
        return [str(name) for name in cleaned]
    raw = topology.get("joint_names")
    if raw is None or len(raw) != count:
        raise ValueError("topology must contain one joint name per parent")
    return [str(name) for name in raw]


def prepare_condition(
    topology: Mapping[str, Any],
    dataset_stats: Mapping[str, Any],
    *,
    dataset_type: str,
    caption_embedding: np.ndarray,
    joint_name_embeddings: np.ndarray,
    motion_length: int = 60,
    ground_tpose: bool = True,
    max_joints: int = PREVIEW_MAX_JOINTS,
) -> dict[str, np.ndarray]:
    """Produce a one-character batch padded to the selected checkpoint limit."""
    parents = np.asarray(topology["parents"], dtype=np.int64)
    count = len(parents)
    if not 2 <= count <= max_joints:
        raise ValueError(f"joint count must be 2..{max_joints}")
    if not 1 <= motion_length <= 60:
        raise ValueError("motion_length must be 1..60")
    tpos = np.asarray(topology["tpos_first_frame"], dtype=np.float32).copy()
    if tpos.shape != (count, 3):
        raise ValueError("tpos_first_frame must have shape (J, 3)")
    if ground_tpose:
        tpos[:, 1] -= np.min(tpos[:, 1])
    padded_features = np.zeros((count, FEATURES), dtype=np.float32)
    padded_features[:, :3] = tpos
    padded_features[:, 3:9] = IDENTITY_6D

    try:
        stats = dataset_stats[dataset_type]
    except KeyError as error:
        raise ValueError(f"dataset statistics missing {dataset_type!r}") from error
    mean = np.empty((count, FEATURES), dtype=np.float32)
    std = np.empty_like(mean)
    mean[0], std[0] = stats["mean_root"], stats["std_root"]
    mean[1:], std[1:] = stats["mean_local"], stats["std_local"]
    if np.any(std <= 0):
        raise ValueError("dataset standard deviations must be positive")
    normalized = np.nan_to_num((padded_features - mean) / std)
    parent_features = normalized.copy()
    for child, parent in enumerate(parents):
        if parent != -1:
            if parent < 0 or parent >= count:
                raise ValueError(f"invalid parent index {parent} for joint {child}")
            parent_features[child] = normalized[parent]

    caption = np.asarray(caption_embedding, dtype=np.float32)
    names = np.asarray(joint_name_embeddings, dtype=np.float32)
    if caption.shape != (TEXT_DIM,) or names.shape != (count, TEXT_DIM):
        raise ValueError("caption_embedding and joint_name_embeddings must be (768,) and (J,768)")

    graph = np.asarray(topology.get("joint_graph_dists", topology.get("joint_graph_dist")), dtype=np.int64)
    relations = np.asarray(topology["joint_relations"], dtype=np.int64)
    depths = np.asarray(topology["joint_depths"], dtype=np.int64)
    spectral = np.asarray(topology["spectral_feats"], dtype=np.float32)
    if graph.shape != (count, count) or relations.shape != (count, count):
        raise ValueError("graph distances and relations must have shape (J,J)")
    if depths.shape != (count,) or spectral.shape != (count, SPECTRAL_FREQS):
        raise ValueError("joint depths or spectral feature shape does not match checkpoint")
    if np.any(graph < 0) or np.any(graph > 5) or np.any(relations < 0) or np.any(relations > 5):
        raise ValueError("graph distances must be 0..5 and relations 0..5")

    def pad(array: np.ndarray, shape: tuple[int, ...], *, fill: int | float = 0) -> np.ndarray:
        result = np.full(shape, fill, dtype=array.dtype)
        index = tuple(slice(0, size) for size in array.shape)
        result[index] = array
        return result[None]

    return {
        "caption_emb": caption[None],
        "tpos_first_frame": pad(normalized, (max_joints, FEATURES)),
        "tpos_first_frame_parents": pad(parent_features, (max_joints, FEATURES)),
        "n_joints": np.array([count], dtype=np.int64),
        "motion_length": np.array([motion_length], dtype=np.int64),
        "joint_depths": pad(depths, (max_joints,)),
        "joint_names_emb": pad(names, (max_joints, TEXT_DIM)),
        "spectral_feats": pad(spectral, (max_joints, SPECTRAL_FREQS)),
        "graph_dist": pad(graph, (max_joints, max_joints)),
        "joint_relations": pad(relations, (max_joints, max_joints)),
        "mean": pad(mean, (max_joints, FEATURES)),
        "std": pad(std, (max_joints, FEATURES), fill=1),
        "parents": pad(parents, (max_joints,), fill=-1),
    }


def denormalize_motion(motion: np.ndarray, condition: Mapping[str, np.ndarray]) -> np.ndarray:
    """Convert one generated padded ``(1,Jmax,12,F)`` sample to ``(F,J,12)``."""
    motion = np.asarray(motion)
    padded_joints = np.asarray(condition["mean"]).shape[1]
    if motion.ndim != 4 or motion.shape[:3] != (1, padded_joints, FEATURES):
        raise ValueError(f"motion must have shape (1, {padded_joints}, 12, F)")
    count = int(np.asarray(condition["n_joints"])[0])
    mean = np.asarray(condition["mean"])[0, :count]
    std = np.asarray(condition["std"])[0, :count]
    return np.transpose(motion[0, :count], (2, 0, 1)) * std[None] + mean[None]
