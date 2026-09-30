"""Precompute frozen T5 embeddings for one prompt and a skeleton condition."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .text import DEFAULT_T5_MODEL, FrozenT5Encoder


JointNameResolver = Callable[[dict[str, Any], str], Sequence[str]]


def _official_joint_name_resolver() -> JointNameResolver:
    """Use the local equivalent of the reference cleaned-name selection."""
    from .conditioning import model_joint_names

    return lambda condition, object_type: model_joint_names(condition)


def precompute_embeddings(
    cond_path: str | Path,
    prompt: str,
    output_path: str | Path,
    *,
    object_type: str | None = None,
    model_name: str = DEFAULT_T5_MODEL,
    device: str = "cpu",
    encoder: Any | None = None,
    joint_name_resolver: JointNameResolver | None = None,
) -> Path:
    """Encode a prompt and the selected topology's joint names into an NPZ.

    The output contains exactly ``caption_emb`` (``D``) and
    ``joint_names_emb`` (``J,D``), suitable for loading into model conditioning.
    A sole entry may omit ``object_type``; multi-entry condition files require
    an explicit key.
    """
    cond_dict = np.load(cond_path, allow_pickle=True).item()
    if not isinstance(cond_dict, dict) or not cond_dict:
        raise ValueError(f"{cond_path}: expected a nonempty object_type → condition mapping")
    if object_type is None:
        if len(cond_dict) != 1:
            raise ValueError("object_type is required when cond.npy has multiple entries")
        object_type = next(iter(cond_dict))
    if object_type not in cond_dict:
        raise KeyError(
            f"object_type {object_type!r} not found; available keys: {sorted(cond_dict)}"
        )
    condition = cond_dict[object_type]
    if not isinstance(condition, dict):
        raise ValueError(f"condition entry for {object_type!r} must be a dictionary")

    resolver = joint_name_resolver or _official_joint_name_resolver()
    joint_names = list(resolver(condition, object_type))
    if not joint_names:
        raise ValueError(f"condition for {object_type!r} contains no joint names")
    encoder = encoder or FrozenT5Encoder(model_name=model_name, device=device)
    caption_emb = np.asarray(encoder.prompt_embedding(prompt), dtype=np.float32)
    joint_names_emb = np.asarray(encoder.joint_name_embeddings(joint_names), dtype=np.float32)
    if caption_emb.ndim != 1 or joint_names_emb.ndim != 2:
        raise ValueError("encoder must return caption (D,) and joint-name (J,D) embeddings")
    if joint_names_emb.shape != (len(joint_names), caption_emb.shape[0]):
        raise ValueError(
            f"embedding shape mismatch: caption {caption_emb.shape}, "
            f"joint names {joint_names_emb.shape}, names={len(joint_names)}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        caption_emb=caption_emb,
        joint_names_emb=joint_names_emb,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cond", required=True, help="Official cond.npy topology dictionary")
    parser.add_argument("--prompt", required=True, help="Prompt text to encode")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--object-type", help="Condition key (optional for a single-entry file)")
    parser.add_argument("--model", default=DEFAULT_T5_MODEL, help="Hugging Face T5 model name")
    parser.add_argument("--device", default="cpu", help="PyTorch device for the text encoder")
    args = parser.parse_args()
    output = precompute_embeddings(
        args.cond,
        args.prompt,
        args.output,
        object_type=args.object_type,
        model_name=args.model,
        device=args.device,
    )
    print(f"Saved text embeddings: {output}")


if __name__ == "__main__":
    main()
