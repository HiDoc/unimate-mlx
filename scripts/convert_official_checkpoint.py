"""Extract UniMate EMA parameters from an official PyTorch snapshot to Safetensors.

Uses weights-only checkpoint loading and validates every tensor against the
official model's named-parameter order before saving tied modules once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_pytorch_mlx import build_reference_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="official UniMate checkout")
    parser.add_argument("--checkpoint", type=Path, required=True, help="official checkpoint_step_*.pt")
    parser.add_argument("--config", type=Path, required=True, help="matching config.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from safetensors.torch import save_model

    config = json.loads(args.config.read_text())
    model = build_reference_model(args.reference, config)
    snapshot = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    ema = snapshot["ema_state_dict"]["shadow_params"]
    parameters = list(model.named_parameters())
    if len(ema) != len(parameters):
        raise ValueError(f"EMA has {len(ema)} tensors, model has {len(parameters)} unique parameters")
    with torch.no_grad():
        for (name, parameter), shadow in zip(parameters, ema):
            if parameter.shape != shadow.shape or parameter.dtype != shadow.dtype:
                raise ValueError(f"EMA tensor mismatch at {name}: {tuple(shadow.shape)} vs {tuple(parameter.shape)}")
            parameter.copy_(shadow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(args.output), metadata={
        "source": args.checkpoint.name,
        "step": str(snapshot.get("step", "unknown")),
        "config": config["experiment"]["name"],
    })
    print(f"Saved {len(parameters)} unique EMA tensors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
