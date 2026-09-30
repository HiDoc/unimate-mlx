"""Measure parity-tested FP32 MLX Euler generation on precomputed conditioning."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.check_output(("sysctl", "-n", name), text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/model_ema.safetensors"))
    parser.add_argument("--config", type=Path, help="matching config.json; defaults to checkpoint directory")
    parser.add_argument("--conditioning", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--solver", choices=("euler", "rk4", "dopri5"), default="dopri5")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--guidance", type=float, default=3)
    parser.add_argument("--cfg-mode", choices=("separate", "batched"), default="separate")
    parser.add_argument("--attention-backend", choices=("manual", "fast"), default="fast")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trim-joints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--defer-sync", action="store_true", help="evaluate only after all Euler steps")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--initial-noise", type=Path, help="optional shared NPY noise for parity benchmarks")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup nonnegative")
    if args.solver == "dopri5" and args.defer_sync:
        parser.error("dopri5 requires evaluation at each adaptive stage")
    if args.solver == "dopri5" and args.dtype != "float32":
        parser.error("adaptive Dopri5 currently requires --dtype float32")
    config_path = args.config or args.checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text())
    max_joints = int(config["dataset"]["max_joints"])

    import mlx.core as mx
    dtype = mx.float16 if args.dtype == "float16" else mx.float32

    from unimate_mlx.inference import initial_noise, sample_adaptive, sample_fixed
    from unimate_mlx.weights import inspect_safetensors_header, load_safetensors, validate_manifest

    header = inspect_safetensors_header(args.checkpoint)
    validate_manifest(header, Path(__file__).resolve().parents[1] / "unimate_mlx/checkpoint_manifest.json")
    with np.load(args.conditioning, allow_pickle=False) as archive:
        condition = {key: mx.array(archive[key]).astype(dtype) if archive[key].dtype.kind == "f" else mx.array(archive[key]) for key in archive.files if key in (
            "caption_emb", "tpos_first_frame", "tpos_first_frame_parents", "n_joints",
            "motion_length", "joint_depths", "joint_names_emb", "spectral_feats",
            "graph_dist", "joint_relations",
        )}
    weights = load_safetensors(args.checkpoint, backend="mlx")
    if args.dtype == "float16":
        weights = {name: value.astype(dtype) for name, value in weights.items()}
    mx.eval(*weights.values())
    if condition["tpos_first_frame"].shape[1] != max_joints:
        parser.error(f"conditioning must be padded to {max_joints} joints")
    shape = (condition["n_joints"].shape[0], max_joints, 12, args.frames)
    supplied_noise = None
    if args.initial_noise:
        array = np.load(args.initial_noise, allow_pickle=False)
        if array.shape != shape:
            parser.error(f"initial noise shape {array.shape} must be {shape}")
        supplied_noise = mx.array(array).astype(dtype)
    latencies = []
    peaks = []
    evaluations = []
    for iteration in range(args.warmup + args.repeats):
        noise = supplied_noise if supplied_noise is not None else initial_noise(shape, args.seed, dtype=dtype)
        mx.eval(noise)
        mx.reset_peak_memory()
        started = time.perf_counter()
        solver_stats = {}
        if args.solver == "dopri5":
            generated = sample_adaptive(
                noise, condition, weights, guidance_scale=args.guidance,
                cfg_mode=args.cfg_mode, attention_backend=args.attention_backend,
                rtol=args.rtol, atol=args.atol, diagnostics=solver_stats,
                compile_model=args.compile,
                trim_joints=args.trim_joints,
            )
        else:
            generated = sample_fixed(
                noise, condition, weights, num_steps=args.steps,
                guidance_scale=args.guidance, cfg_mode=args.cfg_mode,
                evaluate_each_step=not args.defer_sync,
                attention_backend=args.attention_backend, method=args.solver,
                compile_model=args.compile,
                trim_joints=args.trim_joints,
            )
            solver_stats["evaluations"] = (args.steps - 1) * (4 if args.solver == "rk4" else 1)
        mx.eval(generated)
        duration = time.perf_counter() - started
        if iteration >= args.warmup:
            latencies.append(duration)
            peaks.append(mx.get_peak_memory())
            evaluations.append(solver_stats["evaluations"])

    report = {
        "machine": platform.machine(),
        "soc": _sysctl("machdep.cpu.brand_string"),
        "ram_bytes": int(_sysctl("hw.memsize") or 0),
        "dtype": args.dtype,
        "initial_noise": str(args.initial_noise) if args.initial_noise else None,
        "frames": args.frames,
        "padded_joints": max_joints,
        "valid_joints": int(np.array(condition["n_joints"])[0]),
        "grid_points": args.steps,
        "solver": args.solver,
        "rtol": args.rtol if args.solver == "dopri5" else None,
        "atol": args.atol if args.solver == "dopri5" else None,
        "velocity_evaluations": evaluations,
        "guidance_scale": args.guidance,
        "cfg_mode": args.cfg_mode,
        "attention_backend": args.attention_backend,
        "compiled": args.compile,
        "trim_joints": args.trim_joints,
        "compute_joints": int(np.array(condition["n_joints"]).max()) if args.trim_joints else max_joints,
        "sync_each_step": not args.defer_sync,
        "seconds": latencies,
        "peak_mlx_memory_bytes": peaks,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
