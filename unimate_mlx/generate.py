"""Generate raw UniMate motion from reference-compatible precomputed features.

This is a numerical inference entry point. It writes the 12-feature motion
representation as NumPy; GLB/FBX retargeting requires the asset pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_CONDITION = (
    "caption_emb", "tpos_first_frame", "tpos_first_frame_parents", "n_joints",
    "motion_length", "joint_depths", "joint_names_emb", "spectral_feats",
    "graph_dist", "joint_relations",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/model_ema.safetensors"))
    parser.add_argument("--config", type=Path, help="matching config.json; defaults to checkpoint directory")
    parser.add_argument("--conditioning", type=Path, required=True, help="NPZ with reference-compatible conditioning arrays")
    parser.add_argument("--initial-noise", type=Path, help="optional NPY noise for exact cross-framework comparison")
    parser.add_argument("--output", type=Path, required=True, help="output raw motion NPY")
    parser.add_argument("--features-output", type=Path, help="optional denormalized (F,J,12) NPY for official animate_motion.py")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--steps", type=int, default=50, help="time grid points; Euler uses steps minus one evaluations, RK4 four times as many")
    parser.add_argument("--solver", choices=("euler", "rk4", "dopri5"), default="dopri5")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Dopri5 relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-6, help="Dopri5 absolute tolerance")
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--cfg-mode", choices=("separate", "batched"), default="separate")
    parser.add_argument("--attention-backend", choices=("manual", "fast"), default="fast")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="compile repeated MLX denoiser calls")
    parser.add_argument("--trim-joints", action=argparse.BooleanOptionalAction, default=False, help="skip computation for joints masked out in every sample")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32", help="experimental FP16 may differ from reference")
    args = parser.parse_args(argv)
    if args.solver == "dopri5" and args.dtype != "float32":
        parser.error("adaptive Dopri5 currently requires --dtype float32")
    config_path = args.config or args.checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text())
    if config["model"]["attention"] != "graph" or config["model"]["text_cond"] != "adaln":
        parser.error("this MLX model supports the graph/AdaLN UniMate checkpoint")
    max_joints = int(config["dataset"]["max_joints"])
    max_frames = int(config["dataset"]["max_motion_length"])

    import mlx.core as mx
    dtype = mx.float16 if args.dtype == "float16" else mx.float32

    from .inference import initial_noise, sample_adaptive, sample_fixed
    from .weights import inspect_safetensors_header, load_safetensors, validate_manifest

    header = inspect_safetensors_header(args.checkpoint)
    validate_manifest(header, Path(__file__).with_name("checkpoint_manifest.json"))
    with np.load(args.conditioning, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_CONDITION if name not in archive]
        if missing:
            parser.error("conditioning NPZ is missing: " + ", ".join(missing))
        condition = {
            name: mx.array(archive[name]).astype(dtype) if archive[name].dtype.kind == "f" else mx.array(archive[name])
            for name in REQUIRED_CONDITION
        }
        normalizers = {name: archive[name] for name in ("mean", "std", "n_joints") if name in archive}
    if args.features_output and set(normalizers) != {"mean", "std", "n_joints"}:
        parser.error("conditioning NPZ needs mean and std for --features-output")
    batch = condition["n_joints"].shape[0]
    if args.frames < 1 or args.frames > max_frames:
        parser.error(f"frames must be 1..{max_frames} for this checkpoint")
    if condition["tpos_first_frame"].shape[1] != max_joints:
        parser.error(f"conditioning must be padded to {max_joints} joints for this checkpoint")
    shape = (batch, max_joints, 12, args.frames)
    if args.initial_noise:
        noise_array = np.load(args.initial_noise, allow_pickle=False)
        if noise_array.shape != shape:
            parser.error(f"initial noise shape {noise_array.shape} must be {shape}")
        noise = mx.array(noise_array).astype(dtype)
    else:
        noise = initial_noise(shape, args.seed, dtype=dtype)
    weights = load_safetensors(args.checkpoint, backend="mlx")
    if args.dtype == "float16":
        weights = {name: value.astype(dtype) for name, value in weights.items()}
    if args.solver == "dopri5":
        diagnostics = {}
        motion = sample_adaptive(
            noise, condition, weights, guidance_scale=args.guidance,
            cfg_mode=args.cfg_mode, attention_backend=args.attention_backend,
            rtol=args.rtol, atol=args.atol, diagnostics=diagnostics,
            compile_model=args.compile,
            trim_joints=args.trim_joints,
        )
        print(f"Adaptive solver: {diagnostics}")
    else:
        motion = sample_fixed(
            noise, condition, weights, num_steps=args.steps,
            guidance_scale=args.guidance, cfg_mode=args.cfg_mode,
            attention_backend=args.attention_backend, method=args.solver,
            compile_model=args.compile,
            trim_joints=args.trim_joints,
        )
    mx.eval(motion)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.array(motion))
    print(f"Saved raw motion {args.output} with shape {motion.shape}")
    if args.features_output:
        from .conditioning import denormalize_motion

        args.features_output.parent.mkdir(parents=True, exist_ok=True)
        features = denormalize_motion(np.array(motion), normalizers)
        np.save(args.features_output, features)
        print(f"Saved denormalized features {args.features_output} with shape {features.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
