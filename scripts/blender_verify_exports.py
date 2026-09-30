"""Reimport exported GLB/FBX animations in Blender and check their rigs.

Run with ``blender -b --python-exit-code 1 -P this_script.py -- files...``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy


if "--" not in sys.argv:
    raise SystemExit("pass one or more GLB/FBX paths after --")
paths = [Path(item) for item in sys.argv[sys.argv.index("--") + 1 :]]
if not paths:
    raise SystemExit("no paths supplied")

for path in paths:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if path.suffix.lower() == ".glb":
        bpy.ops.import_scene.gltf(filepath=str(path.resolve()))
    elif path.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path.resolve()))
    else:
        raise ValueError(f"unsupported asset format: {path}")
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    actions = list(bpy.data.actions)
    report = {
        "path": str(path), "armatures": len(armatures),
        "meshes": len(meshes), "actions": len(actions),
        "bones": max((len(obj.data.bones) for obj in armatures), default=0),
    }
    print("COMPATIBILITY_IMPORT", json.dumps(report), flush=True)
    if not armatures or not meshes or not actions or report["bones"] < 2:
        raise ValueError(f"export did not reimport as an animated skinned rig: {path}")
