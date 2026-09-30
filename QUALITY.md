# Go2 motion quality investigation

The public [Go2 walk](https://linzhanmou.com/unimate/resources/glbs/go2-walk.glb) was used to investigate an odd-looking quadruped result. Its prompt in the site's catalog is “A quadruped robot walks forward.” The site publishes the animated clip, but not its noise seed or sampler settings. Comparisons with the site's clip therefore assess appearance and gait; numerical parity is established separately against the official PyTorch model with identical input tensors.

## Findings

1. **Facing was missing.** The first preprocessing pass recorded `face_joint_idxs` as `-1/-1`, although the robot has clear `FR_hip` and `FL_hip` bones and its physical front lies along the source rig's +X axis. The character CLI now conservatively detects that front-hip pair in GLBs and passes it to official preprocessing. The resulting canonical T-pose points its front along +Z. Explicit `--face-r/--face-l` still override this choice; `--no-auto-facing` disables inference. Facing-aware preprocessing uses a distinct directory so older identity-facing caches cannot be silently reused.
2. **The first noise sample was nearly stationary.** Its exported root moved about 0.02 m in 60 frames; the demo clip moves about 1.83 m in Blender units. On the facing-aware rig, a different MLX seed and guidance 5 produced forward movement near the demo distance. Several other tested seeds remained nearly still or lifted feet too far. The denoiser and adaptive sampler match official PyTorch numerically, so this variation is in generation, not the MLX port's arithmetic.
3. **Statistics affect pace, but switching datasets is not a general fix.** With the same seed, TrueBones statistics produced roughly 8–10 m of canonical forward travel, and Mixamo statistics roughly 5–6 m. Objaverse statistics with guidance 5 produced about 2.5 m, near the demo's 2.6 m canonical trajectory. The result supports retaining Objaverse for this particular Go2 run.
4. **IK uses the model's position channels and changes foot pose.** The official `animate_motion.py` IK branch had a root-orientation indexing error for this rig. The local Blender adapter corrects that expression in memory and exposes `--anim-mode ik`; it leaves the official checkout untouched. IK can reduce some high foot lifts compared with FK, but it does not reliably plant the feet.
5. **The older checkpoint matters, but v2 is not an automatic Go2 cure.** The original Safetensors checkpoint is converted from a superseded preview. The official recommended v2 graph/AdaLN checkpoint was converted and validated against PyTorch, and it supports 71 padded joints. On the same Go2 skeleton and prompt, tested v2 seeds gave steadier rotation-based feet but generally shorter forward travel. One full adaptive v2 sample with IK moved `0.82` Blender units and had a `0.21` highest-foot range and `0.075` near-ground fraction. Those values did not beat the selected preview candidate for this walk. The demo does not disclose which checkpoint, seed, or settings produced its curated clip.

## Measured visual-quality gap

The table compares the published demo animation with a facing-aware, guidance-5, seed-5 MLX candidate exported through IK. Metrics come from `scripts/analyze_exported_gait.py`, sampling the imported armature at frames 1–60. A foot is counted as near ground when its bone head is within 0.025 Blender units of the lowest foot bone head over the clip.

| Metric | Published demo | Selected MLX candidate |
|---|---:|---:|
| Root horizontal travel | 1.83 | 2.09 |
| Highest foot above clip ground | 0.065 | 0.156 |
| Near-ground foot fraction | 0.696 | 0.058 |
| Mean per-frame slip during near-ground pairs | 0.014 | 0.065 |

The selected clip has better forward travel and less extreme lift than the initial output, but foot contact and sliding are **still visibly worse** than the curated demo. The foot-bone metric is a proxy, not a mesh collision test. We have not declared quadruped motion quality solved.

An experimental stance-target correction was also tested through the official IK exporter. It reduced measured near-ground slip from `0.065` to `0.026` per frame, but the highest foot rose from `0.156` to `0.204` and the near-ground fraction stayed near `0.06`. That trade-off was not promoted into the production pipeline.

A smoothed per-frame vertical grounding correction raised the near-ground fraction only from `0.058` to `0.096`; measured sliding stayed at `0.065`. It was also left out. Correcting a single root height or foot target does not reproduce the demo's sustained stance contacts.

Reproduce the selected clip with:

```sh
python -m unimate_mlx.character \
  --reference /path/to/UniMate \
  --character /path/to/go2-walk.glb \
  --prompt 'A quadruped robot walks forward.' \
  --dataset-type objaverse --guidance 5 --seed 5 \
  --solver rk4 --steps 16 --trim-joints --anim-mode ik \
  --output-dir outputs/go2_quality
```

`scripts/render_animation_contact.py` renders diagnostic frames, and `scripts/analyze_exported_gait.py` measures root travel and foot-bone motion. The selected clip in this workspace is `outputs/demo_go2_seed5_quality/animated/motion_features.glb`; its comparative contact sheet is `outputs/diagnostics/go2_seed5_comparison.png` (generated outputs are ignored by git).

## Joint-limit stress test

`scripts/make_test_rig_61.py` created an actionless 61-bone, four-branch skinned GLB. It completed preprocessing, FLAN-T5 conditioning, MLX inference, GLB/FBX export, and reimport with all 61 bones. Its full-width MLX velocity differed from PyTorch by at most `8.2e-6`. A 62-bone control GLB was rejected after preprocessing with a clear checkpoint-limit error before text encoding.

Next quality work should focus on stance-aware foot locking or contact constraints, plus review of multiple generated candidates. Adjusting root height alone cannot fix the low near-ground foot fraction or foot sliding.

## Independent rig checks

The [free rig set](FREE_RIGS.md) confirms the quadruped quality gap extends beyond the UniMate demo. Khronos Fox and CC0 Quaternius Wolf/Horse all generated structurally valid animations with FP32 PyTorch parity, but their gait and posture differ visibly from the authored Walk cycles. CesiumMan provided a humanoid control whose generated walk was recognizable though subdued. The Wolf also exposed a separate asset distortion: official canonicalization created a stationary `neutral_bone` with 386 weighted vertices. As the body moved, those vertices stretched into long trails. The pipeline now reassigns such weights to nearby retained bones and removes `neutral_bone` before export. On Wolf, the generated motion tensor remained identical while mesh extent stayed near 2.1 units at frames 1, 30 and 60, instead of growing to about 12 units.
