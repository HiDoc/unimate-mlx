"""Rebind vertices from a synthetic neutral_bone to the nearest retained bone.

Run with ``blender -b --python-exit-code 1 -P this_script.py -- input.glb output.glb``.
The source GLB is never modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
from mathutils import Vector


if "--" not in sys.argv or len(sys.argv) < sys.argv.index("--") + 3:
    raise SystemExit("pass input and output GLB paths after --")
input_path, output_path = [Path(value).resolve() for value in sys.argv[sys.argv.index("--") + 1 :][:2]]
output_path.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(input_path))


def segment_distance(point: Vector, head: Vector, tail: Vector) -> float:
    direction = tail - head
    length_squared = direction.length_squared
    if length_squared <= 1e-12:
        return (point - head).length_squared
    fraction = max(0.0, min(1.0, (point - head).dot(direction) / length_squared))
    return (point - (head + fraction * direction)).length_squared


reassigned = 0
removed = 0
for armature in [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]:
    if "neutral_bone" not in armature.data.bones:
        continue
    retained = [bone for bone in armature.data.bones if bone.name != "neutral_bone" and bone.use_deform]
    if not retained:
        raise ValueError(f"{armature.name}: no retained deform bones")
    inverse_armature = armature.matrix_world.inverted()
    for mesh in [obj for obj in bpy.data.objects if obj.type == "MESH"]:
        neutral = mesh.vertex_groups.get("neutral_bone")
        if neutral is None:
            continue
        for vertex in mesh.data.vertices:
            weight = next((entry.weight for entry in vertex.groups if entry.group == neutral.index), 0.0)
            if weight <= 0:
                continue
            world_point = mesh.matrix_world @ vertex.co
            local_point = inverse_armature @ world_point
            closest = min(retained, key=lambda bone: segment_distance(local_point, bone.head_local, bone.tail_local))
            target = mesh.vertex_groups.get(closest.name) or mesh.vertex_groups.new(name=closest.name)
            try:
                existing = target.weight(vertex.index)
            except RuntimeError:
                existing = 0.0
            target.add([vertex.index], min(1.0, existing + weight), "REPLACE")
            reassigned += 1
        mesh.vertex_groups.remove(neutral)
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    armature.data.edit_bones.remove(armature.data.edit_bones["neutral_bone"])
    bpy.ops.object.mode_set(mode="OBJECT")
    removed += 1

bpy.ops.export_scene.gltf(filepath=str(output_path), export_format="GLB", export_animations=False)
print("NEUTRAL_REPAIR", reassigned, "weighted vertices", removed, "bones removed", output_path, flush=True)
