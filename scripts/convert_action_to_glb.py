"""Keep one animation from an embedded glTF and export a rigged GLB.

Run with ``blender -b -P this_script.py -- input.gltf output.glb Walk``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy


if "--" not in sys.argv or len(sys.argv) < sys.argv.index("--") + 4:
    raise SystemExit("pass input glTF, output GLB, and action name after --")
source, destination, action_name = sys.argv[sys.argv.index("--") + 1 :][:3]
destination = Path(destination).resolve()
destination.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(Path(source).resolve()))
action = bpy.data.actions.get(action_name)
if action is None:
    raise ValueError(f"action {action_name!r} not found")
for armature in [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]:
    armature.animation_data_create()
    armature.animation_data.action = action
    if action.slots:
        armature.animation_data.action_slot = action.slots[0]
for other in list(bpy.data.actions):
    if other != action:
        bpy.data.actions.remove(other)
bpy.ops.export_scene.gltf(filepath=str(destination), export_format="GLB", export_animations=True)
print("ACTION_EXPORT", destination, action.name, flush=True)
