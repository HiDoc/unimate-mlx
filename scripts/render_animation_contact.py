"""Render diagnostic animation frames with Blender Workbench.

Run with ``blender -b --python-exit-code 1 -P this_script.py -- --asset clip.glb --output-dir frames``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--frames", default="1,15,30,45,60")
parser.add_argument("--action", help="select a named source action when a GLB has multiple clips")
parser.add_argument("--size", type=int, default=512)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:])
frames = [int(value) for value in args.frames.split(",")]
args.output_dir.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(args.asset.resolve()))
scene = bpy.context.scene
if args.action:
    action = bpy.data.actions.get(args.action)
    if action is None:
        raise ValueError(f"action {args.action!r} not found")
    for obj in scene.objects:
        if obj.type == "ARMATURE":
            obj.animation_data_create()
            obj.animation_data.action = action
            if action.slots:
                obj.animation_data.action_slot = action.slots[0]
all_meshes = [obj for obj in scene.objects if obj.type == "MESH"]
meshes = [obj for obj in all_meshes if obj.vertex_groups]
if meshes:
    for mesh in all_meshes:
        if mesh not in meshes:
            mesh.hide_render = True
else:
    meshes = all_meshes
if not meshes:
    raise ValueError("asset has no meshes")

def frame_bounds(frame):
    low = Vector((math.inf, math.inf, math.inf))
    high = Vector((-math.inf, -math.inf, -math.inf))
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for mesh in meshes:
        evaluated = mesh.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            point = evaluated.matrix_world @ Vector(corner)
            for axis in range(3):
                low[axis] = min(low[axis], point[axis])
                high[axis] = max(high[axis], point[axis])
    return (low + high) / 2, max((high - low))

camera_data = bpy.data.cameras.new("Audit camera")
camera = bpy.data.objects.new("Audit camera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
camera_data.type = "ORTHO"

scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "MATERIAL"
scene.display.shading.show_shadows = True
scene.render.resolution_x = args.size
scene.render.resolution_y = args.size
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
if scene.world is None:
    scene.world = bpy.data.worlds.new("Audit background")
scene.world.color = (0.11, 0.12, 0.14)

for frame in frames:
    center, extent = frame_bounds(frame)
    bone_points = [
        obj.matrix_world @ bone.head
        for obj in scene.objects if obj.type == "ARMATURE"
        for bone in obj.pose.bones
    ]
    bone_extent = max(
        (max(point[axis] for point in bone_points) - min(point[axis] for point in bone_points)
         for axis in range(3)),
        default=0,
    )
    print("AUDIT_BOUNDS", frame, "mesh", round(extent, 4), "bones", round(bone_extent, 4), flush=True)
    camera.location = center + Vector((extent * 1.8, -extent * 2.5, extent * 1.35))
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera_data.ortho_scale = extent * 1.2
    scene.render.filepath = str((args.output_dir / f"frame_{frame:03d}.png").resolve())
    bpy.ops.render.render(write_still=True)
    print("AUDIT_FRAME", frame, scene.render.filepath, flush=True)
