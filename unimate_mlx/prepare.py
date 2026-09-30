"""Build a model-ready condition NPZ from official UniMate topology output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .conditioning import load_topology, model_joint_names, prepare_condition


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True, help="cond.npy from official preprocess_char.py")
    parser.add_argument("--object-type", help="entry key when cond.npy has more than one character")
    parser.add_argument("--dataset-type", required=True, choices=("objaverse", "mixamo", "truebones"))
    parser.add_argument("--stats", type=Path, default=Path("weights/dataset_stats.npy"))
    parser.add_argument("--config", type=Path, default=Path("weights/config.json"), help="matching checkpoint config.json")
    parser.add_argument("--prompt", help="text prompt to encode with frozen FLAN-T5")
    parser.add_argument("--text-embeddings", type=Path, help="NPZ with cached caption_emb (768,) and joint_names_emb (J,768)")
    parser.add_argument("--motion-length", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.prompt is None) == (args.text_embeddings is None):
        parser.error("provide exactly one of --prompt and --text-embeddings")

    topology = load_topology(args.topology, args.object_type)
    stats = np.load(args.stats, allow_pickle=True).item()
    config = json.loads(args.config.read_text())
    if args.text_embeddings:
        with np.load(args.text_embeddings, allow_pickle=False) as data:
            caption = data["caption_emb"]
            names = data["joint_names_emb"]
    else:
        from .text import FrozenT5Encoder

        encoder = FrozenT5Encoder()
        caption = encoder.prompt_embedding(args.prompt)
        names = encoder.joint_name_embeddings(model_joint_names(topology))
    prepared = prepare_condition(
        topology, stats, dataset_type=args.dataset_type,
        caption_embedding=caption, joint_name_embeddings=names,
        motion_length=args.motion_length,
        max_joints=int(config["dataset"]["max_joints"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **prepared)
    print(f"Saved conditioning {args.output} for {int(prepared['n_joints'][0])} joints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
