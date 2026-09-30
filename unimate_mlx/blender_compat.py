"""Execute official UniMate asset scripts with Blender 5.2 Action support.

Blender 4.4 moved F-curves to layered Action channelbags. The official
UniMate scripts use the older ``action.fcurves`` collection API. This adapter
provides the small subset they use without modifying the reference checkout.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import bpy


class ActionFCurves:
    def __init__(self, action):
        self.action = action

    def _bags(self):
        for layer in self.action.layers:
            for strip in layer.strips:
                if strip.type == "KEYFRAME":
                    yield from strip.channelbags

    def __iter__(self):
        for bag in self._bags():
            yield from bag.fcurves

    def __len__(self):
        return sum(len(bag.fcurves) for bag in self._bags())

    def _writable_bag(self):
        bag = next(self._bags(), None)
        if bag is not None:
            return bag
        slot = self.action.slots[0] if self.action.slots else self.action.slots.new(id_type="OBJECT", name="Object")
        layer = self.action.layers[0] if self.action.layers else self.action.layers.new("Animation")
        strip = next((item for item in layer.strips if item.type == "KEYFRAME"), None)
        if strip is None:
            strip = layer.strips.new(type="KEYFRAME")
        bag = strip.channelbags.new(slot=slot)
        for obj in bpy.data.objects:
            data = obj.animation_data
            if data is not None and data.action == self.action:
                data.action_slot = slot
        return bag

    def new(self, *, data_path, index=0):
        return self._writable_bag().fcurves.new(data_path=data_path, index=index)

    def remove(self, curve):
        for bag in self._bags():
            if curve in bag.fcurves:
                bag.fcurves.remove(curve)
                return
        raise ValueError("F-curve does not belong to this Action")


if not hasattr(bpy.types.Action, "fcurves"):
    bpy.types.Action.fcurves = property(lambda action: ActionFCurves(action))

if "--" not in sys.argv:
    raise SystemExit("expected official script path after --")
arguments = sys.argv[sys.argv.index("--") + 1 :]
if not arguments:
    raise SystemExit("expected official script path after --")
target, *target_args = arguments

# The reference palette uses three PCA columns when K>=3, but its SVD can
# return only J columns. Tiny 2-joint rigs therefore fail during a preview
# generated inside preprocessing even when --save_vis is not requested.
if Path(target).name == "preprocess_char.py":
    reference_root = Path(target).resolve().parents[2]
    sys.path.insert(0, str(reference_root))
    from data_process.utils import plotting
    import numpy as np

    original_palette = plotting._spectral_to_rgb

    def _small_rig_palette(features):
        if features.shape[0] < 3 and features.shape[1] >= 3:
            extended = np.concatenate((features, np.repeat(features[-1:], 3 - features.shape[0], axis=0)))
            return original_palette(extended)[: features.shape[0]]
        return original_palette(features)

    plotting._spectral_to_rgb = _small_rig_palette

sys.argv = [target, *target_args]
if Path(target).name == "animate_motion.py" and "--anim_mode" in target_args and target_args[target_args.index("--anim_mode") + 1] == "ik":
    # Upstream IK reorders per-joint rotations/positions, then mistakenly
    # indexes the root-orientation time series by joint indices. Keep the
    # source checkout untouched and correct that one expression at execution.
    source = Path(target).read_text()
    original = "orients=ik_anim.orients[:, inverse_order],"
    if source.count(original) != 1:
        raise RuntimeError("official IK orientation expression changed; inspect the reference source")
    patched = source.replace(original, "orients=ik_anim.orients,", 1)
    namespace = {"__name__": "__main__", "__file__": str(target), "__package__": None}
    exec(compile(patched, str(target), "exec"), namespace)
else:
    runpy.run_path(target, run_name="__main__")
