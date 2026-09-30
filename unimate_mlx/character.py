"""Run the hybrid UniMate character pipeline for a rigged GLB/FBX.

Blender and the official UniMate checkout handle character preprocessing and
animation export. FLAN-T5 runs through PyTorch; only the denoiser and fixed
Euler flow sampler run in MLX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sysconfig
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="official UniMate checkout")
    parser.add_argument("--character", type=Path, required=True, help="rigged GLB or FBX")
    parser.add_argument("--prompt", help="motion description; optional with cached text embeddings")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-type", choices=("objaverse", "mixamo", "truebones"), default="objaverse")
    parser.add_argument("--anim-mode", choices=("fk", "ik"), default="fk", help="official rotation FK or position-fitting IK export")
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/model_ema.safetensors"))
    parser.add_argument("--config", type=Path, help="matching config.json; defaults to checkpoint directory")
    parser.add_argument("--stats", type=Path, help="matching dataset_stats.npy; defaults to checkpoint directory")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--face-r", help="raw right hip bone name for facing canonicalization")
    parser.add_argument("--face-l", help="raw left hip bone name for facing canonicalization")
    parser.add_argument("--auto-facing", action=argparse.BooleanOptionalAction, default=True,
                        help="infer a front right/left hip pair when the GLB has unambiguous names")
    parser.add_argument("--repair-neutral-skin", action=argparse.BooleanOptionalAction, default=True,
                        help="rebind vertices from a synthetic neutral_bone before animation")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--solver", choices=("euler", "rk4", "dopri5"), default="dopri5")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--cfg-mode", choices=("separate", "batched"), default="separate")
    parser.add_argument("--attention-backend", choices=("manual", "fast"), default="fast")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trim-joints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--initial-noise", type=Path, help="NPY noise shared with the reference sampler")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--text-embeddings", type=Path, help="cached caption and joint name embedding NPZ")
    args = parser.parse_args()
    reference = args.reference.resolve()
    character = args.character.resolve()
    output = args.output_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config_path = (args.config or args.checkpoint.parent / "config.json").resolve()
    stats_path = (args.stats or args.checkpoint.parent / "dataset_stats.npy").resolve()
    config = json.loads(config_path.read_text())
    max_joints = int(config["dataset"]["max_joints"])
    if not reference.is_dir() or not character.is_file():
        parser.error("reference checkout and character file must exist")
    if bool(args.face_r) != bool(args.face_l):
        parser.error("provide both --face-r and --face-l, or neither")
    if (args.prompt is None) == (args.text_embeddings is None):
        parser.error("provide exactly one of --prompt and --text-embeddings")
    if args.solver == "dopri5" and args.dtype != "float32":
        parser.error("adaptive Dopri5 currently requires --dtype float32")
    if args.auto_facing and not args.face_r:
        from .asset_facing import infer_front_hip_pair

        inferred = infer_front_hip_pair(character)
        if inferred is not None:
            args.face_r, args.face_l = inferred
            print(f"Using inferred facing pair: right={args.face_r}, left={args.face_l}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    if args.face_r:
        face_key = hashlib.sha256(f"{args.face_r}|{args.face_l}".encode()).hexdigest()[:8]
        preprocessed = output / f"preprocessed_face_{face_key}"
    else:
        preprocessed = output / "preprocessed"
    preprocessed.mkdir(exist_ok=True)
    print(f"Character preprocessing: {preprocessed}", flush=True)
    topology = preprocessed / "cond.npy"
    canonical = preprocessed / f"{character.stem}_canonical.glb"
    condition_path = output / "conditioning.npz"
    raw_path = output / "motion_normalized.npy"
    features_path = output / "motion_features.npy"
    animated = output / "animated"

    preprocess_script = reference / "data_process/mesh_animation/preprocess_char.py"
    animate_script = reference / "data_process/mesh_animation/animate_motion.py"
    if not preprocess_script.is_file() or not animate_script.is_file():
        parser.error("reference checkout lacks the character preprocessing/export scripts")

    def run(command: list[str]) -> None:
        print("Running:", " ".join(command), flush=True)
        environment = os.environ.copy()
        purelib = sysconfig.get_paths()["purelib"]
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (purelib, environment.get("PYTHONPATH"))))
        subprocess.run(command, cwd=reference, env=environment, check=True)

    blender_command = [
        args.blender, "-b", "--python-use-system-env", "--python-exit-code", "1",
        "-P", str(Path(__file__).with_name("blender_compat.py")), "--",
    ]

    if not topology.is_file() or not canonical.is_file():
        command = [
            *blender_command, str(preprocess_script),
            "--char_path", str(character), "--output_dir", str(preprocessed),
        ]
        if args.face_r:
            command += ["--face_r", args.face_r, "--face_l", args.face_l]
        run(command)
    if not topology.is_file() or not canonical.is_file():
        raise RuntimeError("reference preprocessing did not produce cond.npy and canonical GLB")

    from .conditioning import load_topology

    joint_count = len(load_topology(topology)["parents"])
    if joint_count > max_joints:
        parser.error(f"character has {joint_count} joints after preprocessing; this EMA checkpoint supports at most {max_joints}")
    if args.repair_neutral_skin:
        from .asset_facing import skin_joint_names

        if "neutral_bone" in skin_joint_names(canonical):
            repaired = preprocessed / f"{character.stem}_canonical_repaired.glb"
            repair_script = Path(__file__).with_name("repair_neutral_skin.py")
            if not repaired.is_file() or repaired.stat().st_mtime < max(canonical.stat().st_mtime, repair_script.stat().st_mtime):
                run([
                    args.blender, "-b", "--python-exit-code", "1",
                    "-P", str(repair_script), "--",
                    str(canonical), str(repaired),
                ])
            if "neutral_bone" in skin_joint_names(repaired):
                raise RuntimeError("neutral_bone remains in repaired canonical GLB")
            print(f"Using repaired skin: {repaired}", flush=True)
            canonical = repaired

    from .prepare import main as prepare_main
    from .generate import main as generate_main

    prepare_args = [
        "--topology", str(topology), "--dataset-type", args.dataset_type,
        "--stats", str(stats_path), "--config", str(config_path),
        "--output", str(condition_path),
    ]
    if args.text_embeddings:
        prepare_args += ["--text-embeddings", str(args.text_embeddings.resolve())]
    else:
        prepare_args += ["--prompt", args.prompt]
    prepare_main(prepare_args)
    generate_args = [
        "--checkpoint", str(checkpoint_path), "--config", str(config_path),
        "--conditioning", str(condition_path), "--output", str(raw_path),
        "--features-output", str(features_path), "--steps", str(args.steps),
        "--solver", args.solver, "--rtol", str(args.rtol), "--atol", str(args.atol),
        "--guidance", str(args.guidance), "--cfg-mode", args.cfg_mode,
        "--attention-backend", args.attention_backend,
        "--seed", str(args.seed),
        "--dtype", args.dtype,
        "--compile" if args.compile else "--no-compile",
        "--trim-joints" if args.trim_joints else "--no-trim-joints",
    ]
    if args.initial_noise:
        generate_args += ["--initial-noise", str(args.initial_noise.resolve())]
    generate_main(generate_args)
    run([
        *blender_command, str(animate_script),
        "--dataset_type", args.dataset_type, "--char_path", str(canonical),
        "--anim_path", str(features_path), "--cond_path", str(topology),
        "--output_dir", str(animated), "--anim_mode", args.anim_mode,
    ])
    for extension in ("glb", "fbx"):
        exported = animated / f"{features_path.stem}.{extension}"
        if not exported.is_file() or exported.stat().st_size == 0:
            raise RuntimeError(f"Blender did not produce a nonempty {extension.upper()} export: {exported}")
    manifest = {
        "character": str(character),
        "reference": str(reference),
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "stats": str(stats_path),
        "prompt": args.prompt,
        "text_embeddings": str(args.text_embeddings.resolve()) if args.text_embeddings else None,
        "preprocessed": str(preprocessed.relative_to(output)),
        "face_r": args.face_r,
        "face_l": args.face_l,
        "dataset_type": args.dataset_type,
        "anim_mode": args.anim_mode,
        "solver": args.solver,
        "steps": args.steps if args.solver != "dopri5" else None,
        "guidance": args.guidance,
        "seed": args.seed if not args.initial_noise else None,
        "initial_noise": str(args.initial_noise.resolve()) if args.initial_noise else None,
        "trim_joints": args.trim_joints,
        "repair_neutral_skin": args.repair_neutral_skin,
        "canonical_asset": str(canonical.relative_to(output)),
        "dtype": args.dtype,
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Animation exports: {animated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
