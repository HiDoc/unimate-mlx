"""Compare official PyTorch and MLX EMA velocity on real prepared characters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_pytorch_mlx import reference_model


MODEL_KEYS = (
    "caption_emb", "tpos_first_frame", "tpos_first_frame_parents", "n_joints",
    "motion_length", "joint_depths", "joint_names_emb", "spectral_feats",
    "graph_dist", "joint_relations",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/model_ema.safetensors"))
    parser.add_argument("--config", type=Path, default=Path("weights/config.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, help="optional JSON report")
    parser.add_argument("runs", nargs="+", type=Path, help="character output directories with conditioning.npz")
    args = parser.parse_args()

    import mlx.core as mx
    import torch

    from unimate_mlx.model import velocity
    from unimate_mlx.weights import load_safetensors

    torch.set_num_threads(4)
    config = json.loads(args.config.read_text())
    pytorch_model = reference_model(args.reference, config, args.checkpoint)
    weights = load_safetensors(args.checkpoint, backend="mlx")
    reports = []
    for run in args.runs:
        with np.load(run / "conditioning.npz", allow_pickle=False) as archive:
            condition = {key: archive[key] for key in MODEL_KEYS}
        batch = condition["n_joints"].shape[0]
        frames = int(condition["motion_length"].max())
        noise = np.random.default_rng(args.seed).normal(
            size=(batch, config["dataset"]["max_joints"], 12, frames)
        ).astype(np.float32)
        timestep = np.full((batch,), args.timestep, dtype=np.float32)
        with torch.no_grad():
            expected = pytorch_model(
                torch.from_numpy(noise), torch.from_numpy(timestep),
                {key: torch.from_numpy(value) for key, value in condition.items()},
            ).numpy()
        actual = velocity(
            mx.array(noise), mx.array(timestep),
            {key: mx.array(value) for key, value in condition.items()}, weights,
        )
        mx.eval(actual)
        actual = np.array(actual)
        valid_joints = int(condition["n_joints"].max())
        difference = actual[:, :valid_joints] - expected[:, :valid_joints]
        max_error = float(np.max(np.abs(difference)))
        report = {
            "run": str(run), "valid_joints": valid_joints, "frames": frames,
            "max_abs_error": max_error,
            "mean_abs_error": float(np.mean(np.abs(difference))),
            "passed": bool(np.isfinite(difference).all() and max_error <= args.atol),
        }
        reports.append(report)
        print(f"{run}: J={valid_joints} max={max_error:.8g} pass={report['passed']}")
    result = {"passed": sum(item["passed"] for item in reports), "total": len(reports), "results": reports}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
