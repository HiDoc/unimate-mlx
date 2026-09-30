# Free rigged assets for quality checks

These are already skinned, animated assets from sources outside the UniMate demo. Licenses were checked in the model or pack pages before testing. Downloaded assets and generated clips stay under ignored local directories; no third-party mesh is included in this repository.

| Asset | License and attribution | Animation source | Test result |
|---|---|---|---|
| [Khronos Fox GLB](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/Fox/glTF-Binary/Fox.glb) | [Fox model page](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/Fox): mesh CC0 by PixelMannen; rig and animations **CC BY 4.0** by @tomkranis | Walk, Run, Survey; 24 source skin joints | 22 model joints after preprocessing; GLB/FBX pass. Preview walk looks subdued relative to the authored Walk action. The recommended v2 checkpoint moves limbs more, but still differs visibly. |
| [Khronos CesiumMan GLB](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/CesiumMan/glTF-Binary/CesiumMan.glb) | [Model page](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/CesiumMan): **CC BY 4.0**, with Cesium trademark terms | One humanoid action; 19 skin joints | 19 model joints; GLB/FBX pass. Generated walk is recognizable but more subdued than the source motion. |
| [Quaternius Wolf glTF](https://drive.google.com/file/d/1lFQoQ9ln2Z2wGuFFWObj9i5jHqUl_ftG/view) | [Pack page](https://quaternius.com/packs/ultimateanimatedanimals.html) and included `License.txt`: **CC0 1.0**; 12 animals with FBX, Blend and embedded glTF | Wolf has 12 actions including Walk; 51 source skin joints | 38 model joints after preprocessing. Initially failed visually because canonicalization introduced a stationary `neutral_bone` weighted to 386 vertices. Automatic repair rebinds those vertices and removes the bone; mesh bounds now remain stable. Gait still differs from authored Walk. |
| [Quaternius Horse glTF](https://drive.google.com/file/d/1hbtY8kxnXiPdwYGVY7rWRgU0jl_-Q-LG/view) | Same **CC0 1.0** pack license | Horse has 13 actions including Walk; 50 source skin joints | 42 model joints. Automatic repair rebinds 456 vertices from `neutral_bone`; exported mesh bounds stay stable. Generated motion is lively but unlike the authored Walk cycle. |

The Quaternius files are served from the pack page's [public download folder](https://drive.google.com/drive/folders/1uJ3N5HfB7jKTseJUNQr3N4YaN0UuEtHk). Its `glTF/Wolf.gltf` and `glTF/Horse.gltf` embed their buffers; `scripts/convert_action_to_glb.py` can isolate the Walk action and export each as a GLB for this pipeline. The same CC0 pack also contains Alpaca, Bull, Cow, Deer, Donkey, another Fox, two Horses, Husky, ShibaInu and Stag for further character checks.

For example, after downloading `Wolf.gltf`:

```sh
blender -b --python-exit-code 1 -P scripts/convert_action_to_glb.py -- \
  Wolf.gltf Wolf_Walk.glb Walk
python -m unimate_mlx.character \
  --reference /path/to/UniMate --character Wolf_Walk.glb \
  --face-r FrontShoulder.R --face-l FrontShoulder.L \
  --prompt 'A wolf walks forward.' --dataset-type truebones \
  --output-dir outputs/wolf
```

The character CLI detects `neutral_bone` in the canonical rig and repairs its vertex weights automatically. Use `--no-repair-neutral-skin` only when intentionally preserving such a bone.

All four tested assets passed `scripts/audit_compatibility.py` and same-input FP32 PyTorch versus MLX velocity comparison; the worst valid-joint error was **1.14e-5**. All eight exported GLB/FBX files reimported in Blender 5.2 with a mesh, armature and action. The local quality contact sheets are `outputs/diagnostics/fox_preview_v2_comparison.png`, `outputs/diagnostics/cesium_quality_comparison.png`, `outputs/diagnostics/wolf_repaired_comparison.png`, and `outputs/diagnostics/horse_quality_comparison.png`.

**Useful negative test:** [Khronos RecursiveSkeletons](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/RecursiveSkeletons) is CC BY 4.0 and contains many skins. Official preprocessing combines it into 201 joints, beyond either the preview's 61-joint or v2's 71-joint limit. The CLI rejects it explicitly before text encoding. [Khronos SimpleSkin](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/SimpleSkin) is CC0 and useful for a minimal skinning edge case, but too simple for judging generated motion quality.

The independent quadrupeds reinforce the [Go2 quality finding](QUALITY.md): successful export and numerical model parity do not guarantee a convincing gait. The neutral-bone repair fixes one concrete asset distortion; it does not supply missing foot-contact behavior.
