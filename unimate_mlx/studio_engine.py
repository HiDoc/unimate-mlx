"""Local batch orchestration for the UniMate motion studio."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sysconfig
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .asset_facing import infer_front_hip_pair, skin_joint_names
from .conditioning import denormalize_motion, load_topology, model_joint_names, prepare_condition
from .sequence import chain_motion_clips


PRESETS = [
    {"id": "idle", "label": "Idle", "prompt": "Stands calmly, breathing and shifting weight slightly."},
    {"id": "walk", "label": "Walk", "prompt": "Walks forward at a steady, relaxed pace."},
    {"id": "run", "label": "Run", "prompt": "Runs forward with purposeful strides."},
    {"id": "turn", "label": "Turn", "prompt": "Turns to the left and settles into a new facing direction."},
    {"id": "jump", "label": "Jump", "prompt": "Jumps once, lands naturally, and regains balance."},
    {"id": "attack", "label": "Attack", "prompt": "Makes a quick forward attack and returns to guard."},
    {"id": "build", "label": "Build", "prompt": "Works carefully on something in front of the body."},
    {"id": "gather", "label": "Gather", "prompt": "Reaches down, gathers an item, and stands back up."},
    {"id": "carry", "label": "Carry", "prompt": "Walks forward while carrying something carefully."},
    {"id": "celebrate", "label": "Celebrate", "prompt": "Celebrates with an excited gesture."},
]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    directory: Path
    config: dict[str, Any]

    @property
    def checkpoint(self) -> Path:
        return self.directory / "model_ema.safetensors"

    @property
    def stats(self) -> Path:
        return self.directory / "dataset_stats.npy"

    @property
    def max_joints(self) -> int:
        return int(self.config["dataset"]["max_joints"])


@dataclass
class Job:
    id: str
    folder: Path
    filename: str
    prompts: list[dict[str, str]]
    options: dict[str, Any]
    status: str = "queued"
    stage: str = "queued"
    message: str = "Waiting for the local renderer"
    error: str | None = None
    base_progress: float = 0.0
    stage_span: float = 0.0
    stage_started_at: float = field(default_factory=time.monotonic)
    stage_estimate_seconds: float = 0.0
    estimated_remaining_at_stage: float | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    current_clip: int = 0
    clips: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Path] = field(default_factory=dict)
    sample_seconds: list[float] = field(default_factory=list)
    joint_count: int | None = None
    cancel_requested: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def stage_update(self, stage: str, message: str, progress: float, span: float, estimate: float, remaining: float) -> None:
        with self.lock:
            self.status = "running"
            self.stage = stage
            self.message = message
            self.base_progress = progress
            self.stage_span = span
            self.stage_started_at = time.monotonic()
            self.stage_estimate_seconds = estimate
            self.estimated_remaining_at_stage = remaining
            if self.started_at is None:
                self.started_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            stage_elapsed = time.monotonic() - self.stage_started_at
            fraction = min(0.92, stage_elapsed / self.stage_estimate_seconds) if self.status == "running" and self.stage_estimate_seconds > 0 else 0
            progress = min(0.99, self.base_progress + self.stage_span * fraction) if self.status == "running" else (1.0 if self.status == "ready" else self.base_progress)
            remaining = None if self.estimated_remaining_at_stage is None else max(0, self.estimated_remaining_at_stage - stage_elapsed)
            if self.status in {"ready", "failed", "cancelled"}:
                remaining = 0
            return {
                "id": self.id,
                "filename": self.filename,
                "status": self.status,
                "stage": self.stage,
                "message": self.message,
                "error": self.error,
                "progress": progress,
                "currentClip": self.current_clip,
                "totalClips": len(self.prompts),
                "clips": [item.copy() for item in self.clips],
                "artifacts": list(self.artifacts),
                "jointCount": self.joint_count,
                "estimatedRemainingSeconds": round(remaining) if remaining is not None else None,
                "stageEstimateSeconds": round(self.stage_estimate_seconds),
                "elapsedSeconds": round(time.time() - (self.started_at or self.created_at)),
                "sampleSeconds": [round(value, 1) for value in self.sample_seconds],
                "cancelRequested": self.cancel_requested,
                "createdAt": self.created_at,
            }


class StudioEngine:
    def __init__(self, *, root: Path, reference: Path | None = None, blender: str = "blender", output_root: Path | None = None):
        self.root = root.resolve()
        self.reference = (reference or Path(os.environ.get("UNIMATE_REFERENCE_DIR", self.root.parent / "UniMate"))).resolve()
        self.blender = blender
        self.output_root = (output_root or self.root / "outputs" / "studio").resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="unimate-studio")
        self._encoder = None
        self._weight_cache: dict[Path, dict[str, Any]] = {}
        self._models = self._discover_models()
        self._jobs_lock = threading.Lock()

    def _discover_models(self) -> dict[str, ModelSpec]:
        models = {}
        for name, label, directory in (
            ("preview", "UniMate preview · 61 joints", self.root / "weights"),
            ("v2", "UniMate v2 · 71 joints", self.root / "weights" / "v2"),
        ):
            config_path = directory / "config.json"
            if not config_path.is_file():
                continue
            try:
                config = json.loads(config_path.read_text())
            except (OSError, ValueError):
                continue
            models[name] = ModelSpec(name, label, directory, config)
        return models

    def configuration(self) -> dict[str, Any]:
        models = [
            {"id": item.id, "label": item.label, "maxJoints": item.max_joints,
             "available": item.checkpoint.is_file() and item.stats.is_file()}
            for item in self._models.values()
        ]
        return {
            "ready": self.reference.is_dir() and any(item["available"] for item in models),
            "referenceAvailable": self.reference.is_dir(),
            "referencePath": str(self.reference),
            "models": models,
            "presets": PRESETS,
            "clipSeconds": 2,
            "framesPerClip": 60,
            "framesPerSecond": 30,
            "timingNote": "Initial estimates use Apple M4 runs; time remaining adjusts after the first clip.",
        }

    def create_job(self, *, job_id: str, upload_path: Path, filename: str, prompts: list[dict[str, str]], options: dict[str, Any]) -> Job:
        folder = self.output_root / job_id
        job = Job(job_id, folder, filename, prompts, options)
        job.clips = [
            {"index": index, "label": item.get("label") or f"Clip {index + 1}",
             "prompt": item["text"], "status": "queued", "durationSeconds": 2,
             "seed": int(options.get("seed", 42)) + index}
            for index, item in enumerate(prompts)
        ]
        job.artifacts["original"] = upload_path
        with self._jobs_lock:
            self.jobs[job_id] = job
        self.executor.submit(self._run, job)
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self._jobs_lock:
            return self.jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._jobs_lock:
            jobs = list(self.jobs.values())
        return [job.snapshot() for job in reversed(jobs[-20:])]

    def _blender_env(self) -> dict[str, str]:
        environment = os.environ.copy()
        purelib = sysconfig.get_paths()["purelib"]
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (purelib, environment.get("PYTHONPATH"))))
        return environment

    def _run_blender(self, job: Job, command: list[str], log_name: str) -> None:
        log_path = job.folder / "logs" / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=self.reference, env=self._blender_env(), stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-12:])
            raise RuntimeError(f"Blender failed during {log_name}:\n{tail}")

    def _blender_script(self, script: Path, *arguments: str) -> list[str]:
        return [
            self.blender, "-b", "--python-use-system-env", "--python-exit-code", "1",
            "-P", str(Path(__file__).with_name("blender_compat.py")), "--", str(script), *arguments,
        ]

    def _sample_estimate(self, model: ModelSpec, mode: str, joints: int) -> float:
        if mode == "reference":
            return (59 if model.id == "preview" else 150) * model.max_joints / (61 if model.id == "preview" else 71)
        return max(6, 10.6 * (max(joints, 2) / 17) ** 0.7)

    def _remaining(self, job: Job, sample_estimate: float, clips_remaining: int, stage_extra: float = 0) -> float:
        measured = float(np.mean(job.sample_seconds)) if job.sample_seconds else sample_estimate
        return stage_extra + clips_remaining * (measured + 2.5) + 5

    def _encoder_instance(self):
        if self._encoder is None:
            from .text import FrozenT5Encoder

            self._encoder = FrozenT5Encoder()
        return self._encoder

    def _weights(self, model: ModelSpec):
        checkpoint = model.checkpoint.resolve()
        if checkpoint not in self._weight_cache:
            import mlx.core as mx

            from .weights import inspect_safetensors_header, load_safetensors, validate_manifest

            validate_manifest(inspect_safetensors_header(checkpoint), Path(__file__).with_name("checkpoint_manifest.json"))
            values = load_safetensors(checkpoint, backend="mlx")
            mx.eval(*values.values())
            self._weight_cache[checkpoint] = values
        return self._weight_cache[checkpoint]

    def _run(self, job: Job) -> None:
        try:
            self._execute(job)
        except Exception as error:
            with job.lock:
                job.status = "failed"
                job.stage = "failed"
                job.message = "Generation stopped"
                job.error = str(error)
                job.finished_at = time.time()
            (job.folder / "error.txt").write_text(str(error) + "\n")

    def _execute(self, job: Job) -> None:
        import mlx.core as mx

        from .inference import initial_noise, sample_adaptive, sample_fixed

        model = self._models[job.options["model"]]
        if not model.checkpoint.is_file() or not model.stats.is_file():
            raise FileNotFoundError(f"Model files are incomplete in {model.directory}")
        if not self.reference.is_dir():
            raise FileNotFoundError(f"Official UniMate checkout not found at {self.reference}")
        job.folder.mkdir(parents=True, exist_ok=True)
        sample_estimate = self._sample_estimate(model, job.options["mode"], 17)
        job.stage_update("preparing", "Reading skeleton and preparing the rig", 0.02, 0.1, 12, self._remaining(job, sample_estimate, len(job.prompts), 22))
        source = job.artifacts["original"]
        right = job.options.get("faceRight") or None
        left = job.options.get("faceLeft") or None
        if not right and job.options.get("autoFacing", True):
            inferred = infer_front_hip_pair(source)
            if inferred:
                right, left = inferred
        face_key = hashlib.sha256(f"{right}|{left}".encode()).hexdigest()[:8] if right else None
        prepared_dir = job.folder / (f"prepared_face_{face_key}" if face_key else "prepared")
        prepared_dir.mkdir(exist_ok=True)
        topology_path = prepared_dir / "cond.npy"
        canonical = prepared_dir / f"{source.stem}_canonical.glb"
        preprocess = self.reference / "data_process/mesh_animation/preprocess_char.py"
        arguments = ["--char_path", str(source), "--output_dir", str(prepared_dir)]
        if right and left:
            arguments += ["--face_r", right, "--face_l", left]
        self._run_blender(job, self._blender_script(preprocess, *arguments), "preprocess.log")
        if not topology_path.is_file() or not canonical.is_file():
            raise RuntimeError("Preprocessing did not produce topology and a canonical GLB")
        topology = load_topology(topology_path)
        joint_count = len(topology["parents"])
        if not 2 <= joint_count <= model.max_joints:
            raise ValueError(f"Rig has {joint_count} joints; {model.label} accepts 2–{model.max_joints}")
        with job.lock:
            job.joint_count = joint_count
            job.artifacts["canonical"] = canonical
        if "neutral_bone" in skin_joint_names(canonical):
            repaired = prepared_dir / f"{source.stem}_canonical_repaired.glb"
            self._run_blender(job, [
                self.blender, "-b", "--python-exit-code", "1", "-P",
                str(Path(__file__).with_name("repair_neutral_skin.py")), "--", str(canonical), str(repaired),
            ], "skin_repair.log")
            canonical = repaired
            with job.lock:
                job.artifacts["canonical"] = canonical

        sample_estimate = self._sample_estimate(model, job.options["mode"], joint_count)
        job.stage_update("encoding", "Encoding prompt queue and joint names", 0.12, 0.07, 10, self._remaining(job, sample_estimate, len(job.prompts), 14))
        encoder = self._encoder_instance()
        names = model_joint_names(topology)
        joint_embeddings = encoder.joint_name_embeddings(names)
        caption_embeddings = encoder.encode([item["text"] for item in job.prompts])
        stats = np.load(model.stats, allow_pickle=True).item()

        job.stage_update("loading", "Loading EMA weights into MLX", 0.19, 0.04, 4, self._remaining(job, sample_estimate, len(job.prompts), 8))
        weights = self._weights(model)
        clips: list[np.ndarray] = []
        export_script = self.reference / "data_process/mesh_animation/animate_motion.py"
        clip_progress = 0.67 / len(job.prompts)
        for index, item in enumerate(job.prompts):
            if job.cancel_requested:
                with job.lock:
                    job.status = "cancelled"
                    job.stage = "cancelled"
                    job.message = "Stopped after the last completed clip"
                    job.finished_at = time.time()
                return
            seed = int(job.options.get("seed", 42)) + index
            with job.lock:
                job.current_clip = index + 1
                job.clips[index]["status"] = "generating"
            base = 0.23 + index * clip_progress
            estimate = float(np.mean(job.sample_seconds)) if job.sample_seconds else sample_estimate
            job.stage_update("sampling", f"Generating {item.get('label') or f'clip {index + 1}'}", base, clip_progress * 0.78, estimate, self._remaining(job, sample_estimate, len(job.prompts) - index))
            condition_np = prepare_condition(
                topology, stats, dataset_type=job.options["datasetType"],
                caption_embedding=caption_embeddings[index], joint_name_embeddings=joint_embeddings,
                motion_length=60, max_joints=model.max_joints,
            )
            condition = {key: mx.array(value) for key, value in condition_np.items() if key in (
                "caption_emb", "tpos_first_frame", "tpos_first_frame_parents", "n_joints",
                "motion_length", "joint_depths", "joint_names_emb", "spectral_feats",
                "graph_dist", "joint_relations",
            )}
            noise = initial_noise((1, model.max_joints, 12, 60), seed)
            started = time.perf_counter()
            if job.options["mode"] == "reference":
                motion = sample_adaptive(noise, condition, weights, guidance_scale=job.options["guidance"], compile_model=True)
            else:
                motion = sample_fixed(
                    noise, condition, weights, method="rk4", num_steps=24,
                    guidance_scale=job.options["guidance"], trim_joints=True, compile_model=True,
                )
            mx.eval(motion)
            seconds = time.perf_counter() - started
            features = denormalize_motion(np.array(motion), condition_np).astype(np.float32)
            clip_dir = job.folder / "clips" / f"clip_{index + 1:02d}"
            clip_dir.mkdir(parents=True, exist_ok=True)
            np.save(clip_dir / "motion_features.npy", features)
            np.save(clip_dir / "motion_normalized.npy", np.array(motion))
            np.savez_compressed(clip_dir / "conditioning.npz", **condition_np)
            clips.append(features)
            with job.lock:
                job.sample_seconds.append(seconds)
            job.stage_update("exporting", f"Rigging clip {index + 1} of {len(job.prompts)}", base + clip_progress * 0.78, clip_progress * 0.22, 3, self._remaining(job, sample_estimate, len(job.prompts) - index - 1, 7))
            clip_export = clip_dir / "animated"
            self._run_blender(job, self._blender_script(
                export_script, "--dataset_type", job.options["datasetType"],
                "--char_path", str(canonical), "--anim_path", str(clip_dir / "motion_features.npy"),
                "--cond_path", str(topology_path), "--output_dir", str(clip_export),
                "--anim_mode", job.options["animMode"],
            ), f"clip_{index + 1:02d}_export.log")
            glb = clip_export / "motion_features.glb"
            fbx = clip_export / "motion_features.fbx"
            if not glb.is_file() or not fbx.is_file():
                raise RuntimeError(f"Clip {index + 1} export is incomplete")
            with job.lock:
                job.artifacts[f"clip_{index + 1}_glb"] = glb
                job.artifacts[f"clip_{index + 1}_fbx"] = fbx
                job.clips[index]["status"] = "ready"
                job.clips[index]["sampleSeconds"] = round(seconds, 1)

        job.stage_update("chaining", "Joining clips into one timeline", 0.91, 0.08, 6, 7)
        sequence_dir = job.folder / "sequence"
        sequence_dir.mkdir(exist_ok=True)
        joined = chain_motion_clips(clips, blend_frames=int(job.options.get("blendFrames", 8)))
        np.save(sequence_dir / "sequence_features.npy", joined)
        self._run_blender(job, self._blender_script(
            export_script, "--dataset_type", job.options["datasetType"],
            "--char_path", str(canonical), "--anim_path", str(sequence_dir / "sequence_features.npy"),
            "--cond_path", str(topology_path), "--output_dir", str(sequence_dir / "animated"),
            "--anim_mode", job.options["animMode"],
        ), "sequence_export.log")
        sequence_glb = sequence_dir / "animated" / "sequence_features.glb"
        sequence_fbx = sequence_dir / "animated" / "sequence_features.fbx"
        if not sequence_glb.is_file() or not sequence_fbx.is_file():
            raise RuntimeError("Chained animation export is incomplete")
        with job.lock:
            job.artifacts["sequence_glb"] = sequence_glb
            job.artifacts["sequence_fbx"] = sequence_fbx
            job.status = "ready"
            job.stage = "ready"
            job.message = f"{len(clips)} clips and one {2 * len(clips)}-second sequence are ready"
            job.base_progress = 1.0
            job.stage_span = 0
            job.finished_at = time.time()
