"""Measure root travel and foot heights in animated quadruped GLBs.

Run with ``blender -b -P this_script.py -- clip1.glb clip2.glb``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
import numpy as np


if "--" not in sys.argv:
    raise SystemExit("pass animated GLB paths after --")
paths = [Path(item) for item in sys.argv[sys.argv.index("--") + 1 :]]
foot_names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
for path in paths:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path.resolve()))
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    armature = next((obj for obj in armatures if all(name in obj.pose.bones for name in (*foot_names, "base"))), None)
    if armature is None:
        raise ValueError(f"{path}: Go2 root and foot bones not found")
    points = np.empty((60, 5, 3), dtype=np.float64)
    for frame in range(1, 61):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for index, name in enumerate(("base", *foot_names)):
            location = armature.matrix_world @ armature.pose.bones[name].matrix.translation
            points[frame - 1, index] = tuple(location)
    root = points[:, 0]
    feet = points[:, 1:]
    ground = float(feet[..., 2].min())
    heights = feet[..., 2] - ground
    contact = heights < 0.025
    frame_step = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=-1)
    stance_pairs = contact[1:] & contact[:-1]
    report = {
        "path": str(path),
        "root_horizontal_travel": float(np.linalg.norm(root[-1, :2] - root[0, :2])),
        "root_vertical_std": float(root[:, 2].std()),
        "foot_height_max": float(heights.max()),
        "foot_height_mean": float(heights.mean()),
        "stance_fraction": float(contact.mean()),
        "stance_slip_mean_per_frame": float(frame_step[stance_pairs].mean()) if stance_pairs.any() else None,
    }
    print("GAIT_AUDIT", json.dumps(report), flush=True)
