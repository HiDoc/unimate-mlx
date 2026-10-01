# UniMate MLX

UniMate motion inference on Apple Silicon. The released UniMate EMA denoiser and adaptive Dormand–Prince sampler run in [MLX](https://github.com/ml-explore/mlx). The hybrid pipeline keeps the [official UniMate](https://github.com/Friedrich-M/UniMate) Blender preprocessing and animation export scripts, and frozen PyTorch FLAN-T5 text encoding. See [ROADMAP.md](ROADMAP.md) for the longer plan.

The original port used [`tarn59/UniMate-Weights`](https://huggingface.co/tarn59/UniMate-Weights), configuration `uniml3d_60frames_graph_adaln`: 60 frames, 61 padded joints, 12 features per joint, 512 hidden channels, 10 transformer blocks, and 8 heads. This is the **superseded preview** model according to the [official model card](https://huggingface.co/Linzhan/UniMate). The same MLX architecture now also supports the official recommended v2 graph/AdaLN EMA with 71 padded joints. Both have 74,077,328 parameters and tied spectral encoder weights. The checkpoint directory's `config.json` and `dataset_stats.npy` are used automatically.

Validation used official UniMate commit `5d6aabedd947297b5ba6706d8e9113e68c0c3e4f` and weights repository revision `518325e09a555f059ba0efe6face299e9b853d8c`.

## Setup

Use an Apple Silicon Mac with Python 3.13 and Blender 5.2. The Blender 5.2 adapter uses the project's Python packages through `--python-use-system-env`, so the Python minor version should match Blender's embedded Python.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[reference,studio,test]'
pip install 'setuptools<81' loguru matplotlib imageio tqdm
pip install --no-build-isolation 'Motion @ git+https://github.com/inbar-2344/Motion.git'
git clone https://github.com/Friedrich-M/UniMate.git /path/to/UniMate
```

Download `model_ema.safetensors`, `config.json`, and `dataset_stats.npy` from the [weights repository](https://huggingface.co/tarn59/UniMate-Weights/tree/main) into `weights/`. The checkpoint is about 283 MiB. The `weights/` and `outputs/` directories are excluded from git.

## UniMate Studio

The local Vue/TypeScript studio accepts a rigged GLB, an ordered queue of preset or written motion prompts, and generates one 60-frame (2-second) animation per prompt. It previews each finished clip and exports both individual GLB/FBX files and one chained GLB/FBX sequence. The 3D viewer supports orbiting, playback, and scrubbing. Progress and remaining time update as clips finish. Use Node.js 22 or newer for the frontend tests.

Install the frontend once, then start the API and Vite together:

```sh
npm --prefix web install
make
```

Open `http://127.0.0.1:5173`. `make` (or `make studio`) finds the official source at `../UniMate` or `/private/tmp/unimate-reference`. If it lives elsewhere, run `make UNIMATE_REFERENCE_DIR=/path/to/UniMate`. Ctrl-C stops both processes. The API listens on `127.0.0.1:8000` and Vite proxies `/api` there. For a built UI served by the Python API, run `make studio-build` before starting `python -m unimate_mlx.studio_api`.

The **Fast** mode uses a 24-point RK4 sampler with valid-joint trimming. On the 17-joint Go2 test rig, sampling took 10.6 seconds on Apple M4; the reference adaptive mode took 58.8 seconds. Those are sampling measurements, so first-run text encoding, Blender preprocessing, and export add time. The UI estimates the full job and adjusts after completed clips. For example, a two-clip 2-joint RiggedSimple run took 46 seconds overall, with roughly 1 second of sampling per clip. Sequence blending keeps 60 frames per clip; a two-clip run produced one 120-frame GLB animation. Use `make studio-test` for the focused backend and frontend unit tests.

For v2, download `config.json`, `dataset_stats.npy`, and `checkpoints/checkpoint_step_100000.pt` from the [official recommended checkpoint](https://huggingface.co/Linzhan/UniMate/tree/main/unimate_uniml3d_f60_v2). Extract its EMA into `weights/v2/`:

```sh
python scripts/convert_official_checkpoint.py \
  --reference /path/to/UniMate \
  --checkpoint /path/to/checkpoint_step_100000.pt \
  --config weights/v2/config.json \
  --output weights/v2/model_ema.safetensors
```

Pass `--checkpoint weights/v2/model_ema.safetensors` to the character CLI to use v2. The CLI then pads to 71 joints and loads the matching v2 statistics. The preview stays the default until broader v2 motion-quality validation is complete.

```sh
python scripts/inspect_checkpoint.py weights/model_ema.safetensors \
  --manifest unimate_mlx/checkpoint_manifest.json
```

## Generate an animated character

```sh
python -m unimate_mlx.character \
  --reference /path/to/UniMate \
  --character /path/to/rigged_character.glb \
  --prompt 'walk forward' \
  --dataset-type objaverse \
  --solver dopri5 --guidance 3 --seed 42 \
  --output-dir outputs/walk
```

This writes `outputs/walk/animated/motion_features.glb` and `.fbx`. It also keeps the canonical rig, topology, model conditioning, normalized motion, and denormalized 12-feature motion under the output directory. For rigs with a known facing pair, add `--face-r <bone>` and `--face-l <bone>`. The input should be a rigged GLB or FBX with a usable armature and no more than 61 joints after reference preprocessing.

For GLBs with an unambiguous front-right/front-left hip pair, the CLI now infers facing automatically; use `--no-auto-facing` to retain identity facing. It prints the preprocessing directory, which is keyed by facing choice so a stale canonical rig cannot be reused silently. `--anim-mode ik` selects the official position-fitting export path; FK remains the default. See the [Go2 quality investigation](QUALITY.md) for their measured trade-offs.

After one run, cache its T5 embeddings and reuse them without loading the text encoder again:

```sh
python -m unimate_mlx.cache_text \
  --cond outputs/walk/preprocessed/cond.npy \
  --prompt 'walk forward' --output outputs/walk/text_cache.npz
python -m unimate_mlx.character \
  --reference /path/to/UniMate --character /path/to/rigged_character.glb \
  --text-embeddings outputs/walk/text_cache.npz --output-dir outputs/walk_cached
```

The hybrid pipeline runs four stages:

1. Official Blender preprocessing extracts the skeleton and writes `cond.npy` plus a canonical GLB.
2. Frozen FLAN-T5 encodes the prompt and cleaned joint names. The checkpoint's root and local statistics normalize the T-pose; graph and spectral features are padded to 61 joints.
3. MLX runs the EMA denoiser with classifier-free guidance and adaptive Dormand–Prince integration. Attention uses MLX's optimized kernel and repeated denoiser calls are compiled by default.
4. Official Blender animation recovery and retargeting exports GLB and FBX.

The default solver uses the reference's `dopri5` method and tolerances (`rtol=1e-3`, `atol=1e-6`). `--solver euler` and `--solver rk4` use fixed grids; `--steps` sets the number of grid points for those solvers. PyTorch and MLX random seeds generate different noise sequences, so pass the same `--initial-noise` NPY to compare outputs.

For a faster approximation on rigs with many padded joints:

```sh
python -m unimate_mlx.character \
  --reference /path/to/UniMate --character /path/to/rigged_character.glb \
  --prompt 'walk forward' --solver rk4 --steps 24 --trim-joints \
  --output-dir outputs/walk_fast
```

Joint trimming preserves valid-joint predictions in fixed-grid sampling, while adaptive trimming changes the solver's error control and its step schedule. Keep trimming off when comparing with the official adaptive sampler.

## Component and parity checks

```sh
python -m pytest -q
python scripts/compare_pytorch_mlx.py --reference /path/to/UniMate --frames 60
python scripts/compare_pytorch_mlx.py --reference /path/to/UniMate \
  --solver dopri5 --sampler-steps 50 --guidance-scale 3 --compile
python scripts/benchmark.py --conditioning outputs/walk/conditioning.npz \
  --frames 60 --solver dopri5 --guidance 3
```

With the released FP32 EMA weights, the maximum denoiser velocity difference on identical synthetic inputs is about `2.2e-6` for 60 frames. A complete 49-update Euler CFG trajectory differs by at most `1.4e-6`. The adaptive solver differs from official torchdiffeq by at most `4.9e-5` on the same synthetic noise.

The [interactive demo](https://linzhanmou.com/unimate/interactive.html#welcome) publishes animated GLBs and prompts, but no sampling seeds or intermediate tensors. We used its Go2 quadruped rig and prompt, "A quadruped robot walks forward," to build real conditioning. Numerical comparison uses the official checkpoint with the same noise supplied to both frameworks; it does not compare against the website's unknown sample. On the 17 valid Go2 joints, compiled full-width MLX adaptive sampling differs from official PyTorch by `8.6e-5` maximum absolute error and `3.6e-6` relative L2 error.

| Go2 mode on Apple M4 | Sampler time | Valid-joint relative L2 vs official adaptive |
|---|---:|---:|
| FP32 adaptive, compiled, full 61 joints | 58.8 s | `3.6e-6` |
| FP32 RK4, 24 points, compiled, 17 joints | 10.6 s | `4.7e-3` |

These are single runs of numerical sampling only; they exclude T5 and Blender. Compilation reduced the full adaptive run from 75.4 s to 58.8 s without changing its output. The faster RK4 setting preserves the same valid-joint result as full-width RK4 within `1.3e-5`, but it is an approximation of the adaptive reference.

With cached text embeddings, the complete Go2 character command took 66.7 s in adaptive mode and 17.5 s in the trimmed RK4 mode, including preprocessing and GLB/FBX export. Both outputs used identical supplied noise.

The one-command path was tested on the public Khronos `RiggedFigure.glb` with Blender 5.2. It produced a 60-frame GLB with one animation, 38 channels and one skin, plus an FBX. The bundled Blender adapter exposes the legacy `Action.fcurves` operations used by the official scripts through Blender 5.2's layered Action channelbags. Other Blender API differences may appear with different rigs.

The broader [character compatibility audit](COMPATIBILITY.md) covers eleven cases from 2 to 71 joints, including actionless rigs and both checkpoint boundaries. All eleven passed artifact and same-input FP32 PyTorch versus MLX velocity checks; all 22 exported GLB/FBX files reimported in Blender 5.2. The [Go2 visual-quality investigation](QUALITY.md) explains why a structurally valid animation can still have an unnatural gait.

An additional [free rig quality set](FREE_RIGS.md) includes Khronos Fox and CesiumMan plus CC0 Quaternius Wolf and Horse. Those four independent assets passed export and parity checks, and exposed a synthetic neutral-bone skinning bug that is now repaired automatically. Their visual gait quality still varies.

The benchmark script records SoC, RAM, precision, frame and joint counts, sampling grid, latency and peak MLX allocator memory. Its measurements exclude T5 encoding and Blender preprocessing/export.

FP16 fixed-grid sampling is experimental. It reduced one 60-frame, 50-point Euler run from 20.7 s to 18.9 s and peak MLX memory from 1.80 GB to 1.28 GB. The maximum denoiser velocity difference versus PyTorch MPS is about `3.9e-3` at 60 frames. Adaptive sampling currently requires FP32.

The roadmap remains open for FP16/BF16 parity, a pure MLX text encoder, broader character coverage, and further performance work. The checkpoint and original code retain their own licenses and usage conditions.
