#!/usr/bin/env python3
"""Inspect the local UniMate Safetensors checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unimate_mlx.weights import inspect_safetensors_header, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", nargs="?", default="weights/model_ema.safetensors",
        help="Safetensors file (default: weights/model_ema.safetensors)",
    )
    parser.add_argument("--manifest", help="JSON tensor name/shape manifest for exact validation")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    args = parser.parse_args()

    header = inspect_safetensors_header(args.checkpoint)
    if args.manifest:
        validate_manifest(header, args.manifest)
    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "header_size": header.header_size,
        "data_size": header.data_size,
        "tensor_count": len(header.tensors),
        "metadata": dict(header.metadata),
        "tensors": [tensor.as_dict() for tensor in header.tensors],
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Checkpoint: {result['checkpoint']}")
        print(f"Tensors: {len(header.tensors)}  Data: {header.data_size:,} bytes")
        for tensor in header.tensors:
            print(f"{tensor.name}\t{tensor.dtype}\t{list(tensor.shape)}\t{list(tensor.data_offsets)}")
        if header.metadata:
            print("Metadata:")
            for key, value in header.metadata.items():
                print(f"  {key}: {value}")
        if args.manifest:
            print(f"Manifest matches: {args.manifest}")


if __name__ == "__main__":
    main()
