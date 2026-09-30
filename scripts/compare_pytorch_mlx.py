"""Compare the released UniMate EMA model against the MLX port.

Example:
    python scripts/compare_pytorch_mlx.py --reference /path/to/UniMate

Both models receive identical deterministic synthetic inputs and the same
Safetensors weights. No text encoder or asset preprocessing is involved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def make_inputs(seed: int, frames: int, joints: int = 61, valid_joints: int = 8, motion_length: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    motion = rng.normal(0, 0.1, (1, joints, 12, frames)).astype(np.float32)
    timestep = np.array([0.5], dtype=np.float32)
    graph_dist = np.zeros((1, joints, joints), dtype=np.int64)
    relations = np.zeros((1, joints, joints), dtype=np.int64)
    for i in range(valid_joints):
        for j in range(valid_joints):
            graph_dist[0, i, j] = min(abs(i - j), 5)
            relations[0, i, j] = 1 if j == i - 1 else 2 if j == i + 1 else 0 if i == j else 4
    tpos = rng.normal(0, 0.1, (1, joints, 12)).astype(np.float32)
    parents = np.concatenate((tpos[:, :1], tpos[:, :-1]), axis=1)
    cond = {
        "caption_emb": rng.normal(0, 0.1, (1, 768)).astype(np.float32),
        "tpos_first_frame": tpos,
        "tpos_first_frame_parents": parents,
        "n_joints": np.array([valid_joints], dtype=np.int64),
        "motion_length": np.array([frames if motion_length is None else motion_length], dtype=np.int64),
        "joint_depths": np.minimum(np.arange(joints)[None, :], 19).astype(np.int64),
        "joint_names_emb": rng.normal(0, 0.1, (1, joints, 768)).astype(np.float32),
        "spectral_feats": rng.normal(0, 0.1, (1, joints, 8)).astype(np.float32),
        "graph_dist": graph_dist,
        "joint_relations": relations,
    }
    return motion, timestep, cond


def build_reference_model(reference_path: Path, config: dict):
    sys.path.insert(0, str(reference_path))
    from unimate.models.denoiser.graph_adaln import UniMateGraphAdaLN

    dataset, model = config["dataset"], config["model"]
    keys = (
        "latent_dim", "ff_size", "num_layers", "num_heads", "dropout", "use_spectral_rope",
        "max_freqs", "use_signnet", "cond_mode", "cond_mask_prob", "use_joint_name_emb",
        "use_graph_emb", "use_depth_emb", "concat_parent_features", "num_tpos_queries",
        "inject_tpos_to_adaln", "use_graph_attn_bias", "share_graph_attn_bias",
    )
    kwargs = {key: model[key] for key in keys}
    kwargs.update(
        feature_len=dataset["feature_len"],
        max_motion_length=dataset["max_motion_length"],
        max_joints=dataset["max_joints"],
        max_depth=dataset["max_depth"],
        text_dim=768,
    )
    return UniMateGraphAdaLN(**kwargs)


def reference_model(reference_path: Path, config: dict, checkpoint: Path):
    from safetensors import safe_open

    network = build_reference_model(reference_path, config)
    with safe_open(str(checkpoint), framework="pt") as handle:
        state = {key: handle.get_tensor(key) for key in handle.keys()}
        aliases = handle.metadata() or {}
    for alias, original in aliases.items():
        if alias.startswith("transformer_blocks."):
            state[alias] = state[original]
    network.load_state_dict(state, strict=True)
    return network.eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="official UniMate checkout")
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/model_ema.safetensors"))
    parser.add_argument("--config", type=Path, default=Path("weights/config.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--motion-length", type=int, help="valid frames within padded input")
    parser.add_argument("--valid-joints", type=int, default=8)
    parser.add_argument("--conditioning", type=Path, help="real model-ready NPZ instead of synthetic conditioning")
    parser.add_argument("--initial-noise", type=Path, help="NPY motion/noise shared by PyTorch and MLX")
    parser.add_argument("--save-baseline", type=Path)
    parser.add_argument("--save-conditioning", type=Path, help="save a matching NPZ for the MLX generation CLI")
    parser.add_argument("--sampler-steps", type=int, default=0, help="compare fixed-grid sampling on shared initial motion")
    parser.add_argument("--solver", choices=("euler", "rk4", "dopri5"), default="euler")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--cfg-mode", choices=("separate", "batched"), default="separate")
    parser.add_argument("--attention-backend", choices=("manual", "fast"), default="fast")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--trim-joints", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--torch-device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()

    import mlx.core as mx
    import torch
    from unimate_mlx.model import velocity
    from unimate_mlx.weights import load_safetensors

    config = json.loads(args.config.read_text())
    if not 2 <= args.valid_joints <= config["dataset"]["max_joints"]:
        parser.error("valid-joints must be between 2 and the configured max_joints")
    if args.motion_length is not None and not 1 <= args.motion_length <= args.frames:
        parser.error("motion-length must be between 1 and frames")
    if args.conditioning:
        required = (
            "caption_emb", "tpos_first_frame", "tpos_first_frame_parents", "n_joints",
            "motion_length", "joint_depths", "joint_names_emb", "spectral_feats",
            "graph_dist", "joint_relations",
        )
        with np.load(args.conditioning, allow_pickle=False) as archive:
            cond = {name: archive[name] for name in required}
        args.valid_joints = int(cond["n_joints"].max())
        if args.initial_noise:
            motion = np.load(args.initial_noise, allow_pickle=False)
        else:
            motion = np.random.default_rng(args.seed).normal(
                size=(cond["n_joints"].shape[0], config["dataset"]["max_joints"], 12, int(cond["motion_length"].max()))
            ).astype(np.float32)
        timestep = np.full((motion.shape[0],), 0.5, dtype=np.float32)
    else:
        motion, timestep, cond = make_inputs(
            args.seed, args.frames, config["dataset"]["max_joints"],
            args.valid_joints, args.motion_length,
        )
        if args.initial_noise:
            motion = np.load(args.initial_noise, allow_pickle=False)
    torch_model = reference_model(args.reference, config, args.checkpoint)
    if args.dtype == "float16":
        motion = motion.astype(np.float16)
        timestep = timestep.astype(np.float16)
        cond = {key: value.astype(np.float16) if np.issubdtype(value.dtype, np.floating) else value for key, value in cond.items()}
        torch_model = torch_model.half()
    torch_model = torch_model.to(args.torch_device)
    torch_activations: dict[str, np.ndarray] = {}
    hooks = [
        block.register_forward_hook(
            lambda module, inputs, output, index=index: torch_activations.__setitem__(f"block.{index}", output.detach().cpu().numpy())
        )
        for index, block in enumerate(torch_model.transformer_blocks)
    ]
    with torch.no_grad():
        torch_output = torch_model(
            torch.from_numpy(motion).to(args.torch_device), torch.from_numpy(timestep).to(args.torch_device),
            {key: torch.from_numpy(value).to(args.torch_device) for key, value in cond.items()},
        ).cpu().numpy()
    for hook in hooks:
        hook.remove()
    torch_activations["velocity"] = torch_output

    weights = load_safetensors(args.checkpoint, backend="mlx")
    if args.dtype == "float16":
        weights = {name: value.astype(mx.float16) for name, value in weights.items()}
    mlx_activations: dict[str, mx.array] = {}
    mlx_output = velocity(
        mx.array(motion), mx.array(timestep), {key: mx.array(value) for key, value in cond.items()},
        weights, num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        max_motion_length=config["dataset"]["max_motion_length"],
        activations=mlx_activations,
        attention_backend=args.attention_backend,
    )
    mx.eval(mlx_output)
    failed = False
    for key, expected in torch_activations.items():
        actual = np.array(mlx_activations[key])
        max_error = float(np.max(np.abs(expected - actual)))
        mean_error = float(np.mean(np.abs(expected - actual)))
        print(f"{key:12} max={max_error:.8g} mean={mean_error:.8g}")
        tolerance = 1e-3 if args.dtype == "float16" else 1e-4
        if not np.allclose(expected, actual, atol=tolerance, rtol=tolerance):
            failed = True
    if args.sampler_steps:
        if args.sampler_steps < 2 or args.guidance_scale < 1:
            parser.error("sampler steps must be >=2 and guidance scale must be >=1")
        from unimate_mlx.inference import sample_adaptive, sample_fixed

        torch_cond = {key: torch.from_numpy(value).to(args.torch_device) for key, value in cond.items()}
        state = torch.from_numpy(motion.copy()).to(args.torch_device)
        grid = np.linspace(0, 1, args.sampler_steps, dtype=np.float32)
        def field(time, state):
            t = torch.full((state.shape[0],), float(time), dtype=state.dtype, device=args.torch_device)
            prediction = torch_model(state, t, torch_cond)
            if args.guidance_scale > 1:
                unconditioned = torch_model(state, t, torch_cond, force_mask=True)
                prediction = unconditioned + args.guidance_scale * (prediction - unconditioned)
            return prediction
        with torch.no_grad():
            if args.solver in {"rk4", "dopri5"}:
                from torchdiffeq import odeint

                state = odeint(
                    field, state, torch.from_numpy(grid).to(args.torch_device),
                    method=args.solver, atol=1e-6, rtol=1e-3,
                )[-1]
            else:
                for index in range(args.sampler_steps - 1):
                    state = state + float(grid[index + 1] - grid[index]) * field(float(grid[index]), state)
        mlx_condition = {key: mx.array(value) for key, value in cond.items()}
        if args.solver == "dopri5":
            sampled = sample_adaptive(
                mx.array(motion), mlx_condition, weights,
                guidance_scale=args.guidance_scale, cfg_mode=args.cfg_mode,
                attention_backend=args.attention_backend,
                compile_model=args.compile,
                trim_joints=args.trim_joints,
            )
        else:
            sampled = sample_fixed(
                mx.array(motion), mlx_condition, weights,
                num_steps=args.sampler_steps, guidance_scale=args.guidance_scale,
                cfg_mode=args.cfg_mode, method=args.solver,
                attention_backend=args.attention_backend,
                compile_model=args.compile,
                trim_joints=args.trim_joints,
            )
        mx.eval(sampled)
        compared_joints = args.valid_joints if args.trim_joints else state.shape[1]
        reference_sample = state.cpu().numpy()[:, :compared_joints]
        mlx_sample = np.array(sampled)[:, :compared_joints]
        sample_error = float(np.max(np.abs(reference_sample - mlx_sample)))
        print(f"{args.solver}.{args.sampler_steps}    max={sample_error:.8g}")
        failed |= not np.allclose(reference_sample, mlx_sample, atol=tolerance, rtol=tolerance)
    if args.save_baseline:
        args.save_baseline.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_baseline,
            motion=motion, timestep=timestep,
            **{f"cond_{key}": value for key, value in cond.items()},
            **{f"torch_{key}": value for key, value in torch_activations.items()},
        )
        print(f"Saved {args.save_baseline}")
    if args.save_conditioning:
        args.save_conditioning.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_conditioning, **cond)
        print(f"Saved {args.save_conditioning}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
