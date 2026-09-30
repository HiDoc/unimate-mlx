"""Build an actionless 61/62/71/72-bone rigged GLB for checkpoint-limit testing.

Run with ``blender -b --python-exit-code 1 -P this_script.py -- /path/to/output.glb [61|62|71|72]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
from mathutils import Vector


if "--" not in sys.argv or len(sys.argv) <= sys.argv.index("--") + 1:
    raise SystemExit("pass an output GLB path after --")
output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
joints = int(sys.argv[sys.argv.index("--") + 2]) if len(sys.argv) > sys.argv.index("--") + 2 else 61
if joints not in (61, 62, 71, 72):
    raise SystemExit("this fixture supports 61, 62, 71 or 72 joints")
output.parent.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
armature_data = bpy.data.armatures.new("Limit61 bones")
armature = bpy.data.objects.new("Limit61 armature", armature_data)
bpy.context.collection.objects.link(armature)
bpy.context.view_layer.objects.active = armature
armature.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
root = armature_data.edit_bones.new("root")
root.head = (0, 0, 0.4)
root.tail = (0, 0, 0.6)
directions = (Vector((1, 1, 0)), Vector((1, -1, 0)), Vector((-1, 1, 0)), Vector((-1, -1, 0)))
for branch, direction in enumerate(directions):
    direction.normalize()
    parent = root
    count = (joints - 1) // 4 + (branch < (joints - 1) % 4)
    for joint in range(1, count + 1):
        bone = armature_data.edit_bones.new(f"branch_{branch}_joint_{joint:02d}")
        bone.head = parent.tail
        bone.tail = Vector(parent.tail) + 0.09 * direction + Vector((0, 0, 0.015))
        bone.parent = parent
        parent = bone
bpy.ops.object.mode_set(mode="OBJECT")

bpy.ops.mesh.primitive_cube_add(size=0.4, location=(0, 0, 0.5))
mesh = bpy.context.object
mesh.name = "Skinned core"
group = mesh.vertex_groups.new(name="root")
group.add(list(range(len(mesh.data.vertices))), 1.0, "REPLACE")
modifier = mesh.modifiers.new("Armature skin", "ARMATURE")
modifier.object = armature
mesh.parent = armature

bpy.ops.export_scene.gltf(filepath=str(output), export_format="GLB", export_animations=False)
print("LIMIT61_EXPORT", output, len(armature_data.bones), flush=True)
