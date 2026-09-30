# Character compatibility audit

Eight public rigs, an actionless variant, and synthetic 61- and 71-bone boundary rigs were tested with the official UniMate preprocessing and animation export scripts, Blender 5.2, and the MLX denoiser. The 61-bone rig uses the preview checkpoint; the 71-bone rig uses the official recommended v2 graph/AdaLN checkpoint.

| Rig | Source | Valid joints | Max depth | GLB animation channels |
|---|---|---:|---:|---:|
| RiggedSimple | [Khronos sample](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/RiggedSimple) | 2 | 1 | 4 |
| EVE robot | [Demo GLB](https://linzhanmou.com/unimate/resources/glbs/eve-raise_arm.glb) | 5 | 2 | 10 |
| Shark | [Demo GLB](https://linzhanmou.com/unimate/resources/glbs/jaws-swim_right.glb) | 16 | 7 | 32 |
| Go2 quadruped | [Demo GLB](https://linzhanmou.com/unimate/resources/glbs/go2-walk.glb) | 17 | 4 | 34 |
| RiggedFigure | [Khronos sample](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/RiggedFigure) | 19 | 5 | 38 |
| RiggedFigure without actions | Derived from the Khronos sample by removing animation actions | 19 | 5 | 38 |
| Mixamo humanoid | [Demo GLB](https://linzhanmou.com/unimate/resources/glbs/mixamo-backflip.glb) | 22 | 7 | 44 |
| Flower | [Demo GLB](https://linzhanmou.com/unimate/resources/glbs/flower-close.glb) | 23 | 7 | 46 |
| Eagle | [Demo GLB](https://linzhanmou.com/unimate/resources/glbs/eagle-take_off.glb) | 51 | 7 | 102 |
| Synthetic four-branch rig | Generated with `scripts/make_test_rig_61.py` | 61 | 15 | 122 |
| Synthetic four-branch rig, v2 | Generated with `scripts/make_test_rig_61.py` | 71 | 18 | 142 |

For each rig, the character CLI generated 60 frames using a three-point Euler grid to exercise the entire preprocessing → T5 conditioning → MLX forward pass → retargeting → GLB/FBX path. Mixamo used Mixamo statistics; the others used Objaverse statistics. These short runs test compatibility and artifact integrity, not motion quality or convergence of the sampler. The Go2 rig was also generated with the default adaptive sampler and a longer fixed RK4 run; see the [README](README.md) for parity and speed results.

All eleven runs passed `scripts/audit_compatibility.py`: each has one connected rooted parent tree, finite normalized and exported features, a 60-frame motion, a skinned GLB with animation channels, and a nonempty FBX. All **22** exported files reimported in Blender 5.2 with an armature, at least one mesh, at least one action, and the expected bones. `scripts/audit_model_parity.py` compared one full-width FP32 velocity prediction per rig against official PyTorch using identical noise and timestep; all passed a `1e-4` maximum absolute tolerance. The worst preview-checkpoint valid-joint error was **1.16e-5** on the shark; the 71-bone v2 rig differed by **2.17e-5**. The preview checkpoint accepted 61 bones and rejected 62; the v2 checkpoint accepted 71 and rejected 72 before T5 encoding.

The two-joint RiggedSimple initially exposed an official plotting bug: its spectral palette requests three PCA columns even when SVD returns only two. The local Blender adapter pads the palette input for that preview. It leaves graph features and model inputs unchanged.

The audit scripts are reusable for further assets:

```sh
python scripts/audit_compatibility.py outputs/compat/eve outputs/compat/eagle
python scripts/audit_model_parity.py --reference /path/to/UniMate \
  outputs/compat/eve outputs/compat/eagle
blender -b --python-exit-code 1 -P scripts/blender_verify_exports.py -- \
  outputs/compat/eve/animated/motion_features.glb \
  outputs/compat/eve/animated/motion_features.fbx
```

This coverage does not establish support for every production asset. Remaining useful cases include multiple armatures/skins and unusual root transforms. For the Go2 quadruped, visual testing found that exported animation integrity and model parity do not guarantee convincing foot contact; the investigation is in [QUALITY.md](QUALITY.md).
