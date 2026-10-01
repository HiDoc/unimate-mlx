"""Helpers for joining denormalized UniMate motion clips."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def chain_motion_clips(
    clips: Sequence[np.ndarray], *, blend_frames: int = 8
) -> np.ndarray:
    """Join ``(frames, joints, 12)`` feature clips with smooth transitions.

    The transition occupies ``blend_frames`` frames on each side of each
    boundary. A cubic Hermite bridge connects the poses at the ends of that
    window, using neighboring poses to estimate endpoint velocities. This
    keeps every input frame slot (and therefore the total duration) while
    replacing only the local seam region. The input arrays are never modified.

    Features are interpolated component-wise in denormalized feature space.
    """
    if not clips:
        raise ValueError("clips must contain at least one motion clip")
    if not isinstance(blend_frames, int) or isinstance(blend_frames, bool) or blend_frames < 0:
        raise ValueError("blend_frames must be a non-negative integer")

    arrays = [np.asarray(clip) for clip in clips]
    first_shape = arrays[0].shape
    if len(first_shape) != 3 or first_shape[2] != 12:
        raise ValueError("each clip must have shape (60, joints, 12)")
    if first_shape[0] != 60:
        raise ValueError("each clip must contain exactly 60 frames")
    if first_shape[0] < 1 or first_shape[1] < 1:
        raise ValueError("clips must have at least one frame and one joint")
    if blend_frames and first_shape[0] < 2 * blend_frames:
        raise ValueError("each clip must have at least 2 * blend_frames frames")

    for clip in arrays:
        if clip.shape != first_shape:
            raise ValueError("all clips must have matching frame, joint, and feature dimensions")
        if not np.issubdtype(clip.dtype, np.number):
            raise ValueError("clips must contain numeric features")
        if not np.isfinite(clip).all():
            raise ValueError("clips must contain only finite features")

    result = np.concatenate(arrays, axis=0).astype(
        np.result_type(*(clip.dtype for clip in arrays), np.float32), copy=True
    )
    if blend_frames == 0 or len(arrays) == 1:
        return result

    frames = first_shape[0]
    window = 2 * blend_frames
    # The bridge is cubic Hermite with velocities estimated from the adjacent
    # samples. Its endpoint pose and slope match the untouched clip regions.
    u = np.linspace(0.0, 1.0, window, dtype=result.dtype).reshape(window, 1, 1)
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2

    for index in range(1, len(arrays)):
        boundary = index * frames
        start = boundary - blend_frames
        end = boundary + blend_frames - 1
        p0 = result[start].copy()
        p1 = result[end].copy()
        v0 = result[start] - result[start - 1]
        v1 = result[end + 1] - result[end]
        span = window - 1
        bridge = (
            h00 * p0
            + h10 * span * v0
            + h01 * p1
            + h11 * span * v1
        )
        result[start : end + 1] = bridge

    return result
