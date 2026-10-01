"""FastAPI service for the local UniMate Vue studio.

Run with ``python -m unimate_mlx.studio_api``. Binds to 127.0.0.1 only.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .asset_facing import skin_joint_names
from .studio_engine import StudioEngine


MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_PROMPTS = 12


def _validated_payload(text: str, engine: StudioEngine) -> tuple[list[dict[str, str]], dict[str, Any]]:
    try:
        payload = json.loads(text)
    except ValueError as error:
        raise HTTPException(422, "Generation settings must be valid JSON") from error
    if not isinstance(payload, dict):
        raise HTTPException(422, "Generation settings must be an object")
    raw_prompts = payload.get("prompts")
    if not isinstance(raw_prompts, list) or not 1 <= len(raw_prompts) <= MAX_PROMPTS:
        raise HTTPException(422, f"Choose 1–{MAX_PROMPTS} prompts")
    prompts = []
    for item in raw_prompts:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise HTTPException(422, "Each prompt needs text")
        value = item["text"].strip()
        if not 3 <= len(value) <= 400:
            raise HTTPException(422, "Each prompt must be 3–400 characters")
        label = item.get("label")
        label = label.strip()[:36] if isinstance(label, str) else ""
        prompts.append({"text": value, "label": label})

    model = payload.get("model", "preview")
    if not isinstance(model, str) or model not in engine._models or not engine._models[model].checkpoint.is_file() or not engine._models[model].stats.is_file():
        raise HTTPException(422, f"Model {model!r} is not installed")
    mode = payload.get("mode", "fast")
    dataset_type = payload.get("datasetType", "objaverse")
    anim_mode = payload.get("animMode", "fk")
    if not isinstance(mode, str) or mode not in {"fast", "reference"}:
        raise HTTPException(422, "Mode must be fast or reference")
    if not isinstance(dataset_type, str) or dataset_type not in {"objaverse", "truebones", "mixamo"}:
        raise HTTPException(422, "Choose Objaverse, TrueBones, or Mixamo statistics")
    if not isinstance(anim_mode, str) or anim_mode not in {"fk", "ik"}:
        raise HTTPException(422, "Animation recovery must be FK or IK")
    try:
        guidance_value = payload.get("guidance", 3)
        seed_value = payload.get("seed", 42)
        blend_value = payload.get("blendFrames", 8)
        if isinstance(guidance_value, bool) or isinstance(seed_value, bool) or isinstance(blend_value, bool):
            raise ValueError("boolean setting")
        guidance = float(guidance_value)
        seed = int(seed_value)
        blend_frames = int(blend_value)
        if seed != seed_value or blend_frames != blend_value:
            raise ValueError("integer setting")
    except (TypeError, ValueError) as error:
        raise HTTPException(422, "Guidance, seed, and transition must be numbers") from error
    if not math.isfinite(guidance) or not 1 <= guidance <= 10 or not 0 <= seed <= 2**31 - MAX_PROMPTS or not 0 <= blend_frames <= 16:
        raise HTTPException(422, "Guidance must be 1–10, seed nonnegative, and transition 0–16 frames")
    face_right = payload.get("faceRight")
    face_left = payload.get("faceLeft")
    if face_right is not None and not isinstance(face_right, str) or face_left is not None and not isinstance(face_left, str):
        raise HTTPException(422, "Facing bone names must be text")
    face_right = face_right.strip() if face_right else None
    face_left = face_left.strip() if face_left else None
    if bool(face_right) != bool(face_left):
        raise HTTPException(422, "Provide both right and left facing bones")
    return prompts, {
        "model": model, "mode": mode, "datasetType": dataset_type,
        "animMode": anim_mode, "guidance": guidance, "seed": seed,
        "blendFrames": blend_frames, "faceRight": face_right,
        "faceLeft": face_left, "autoFacing": payload.get("autoFacing", True) is not False,
    }


def create_app(*, root: Path | None = None, reference: Path | None = None, output_root: Path | None = None) -> FastAPI:
    project_root = (root or Path(__file__).resolve().parents[1]).resolve()
    engine = StudioEngine(root=project_root, reference=reference, output_root=output_root)
    app = FastAPI(title="UniMate Studio", version="0.1.0")
    app.state.engine = engine

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return engine.configuration()

    @app.get("/api/jobs")
    def jobs() -> list[dict[str, Any]]:
        return engine.list_jobs()

    @app.post("/api/jobs", status_code=202)
    async def create_job(character: UploadFile = File(...), payload: str = Form(...)) -> dict[str, Any]:
        if not engine.reference.is_dir():
            raise HTTPException(503, "Set UNIMATE_REFERENCE_DIR to the official UniMate checkout")
        prompts, options = _validated_payload(payload, engine)
        filename = Path(character.filename or "character.glb").name
        if Path(filename).suffix.lower() != ".glb":
            raise HTTPException(422, "Upload a rigged .glb file")
        job_id = uuid.uuid4().hex[:12]
        folder = engine.output_root / job_id
        source_dir = folder / "source"
        source_dir.mkdir(parents=True, exist_ok=False)
        upload_path = source_dir / "character.glb"
        size = 0
        try:
            with upload_path.open("wb") as stream:
                while chunk := await character.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "GLB exceeds the 100 MB upload limit")
                    stream.write(chunk)
            if not skin_joint_names(upload_path):
                raise HTTPException(422, "This GLB has no named skinned joints")
        except Exception as error:
            shutil.rmtree(folder, ignore_errors=True)
            if isinstance(error, HTTPException):
                raise
            raise HTTPException(422, f"Could not read GLB: {error}") from error
        finally:
            await character.close()
        job = engine.create_job(job_id=job_id, upload_path=upload_path, filename=filename, prompts=prompts, options=options)
        return job.snapshot()

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        job = engine.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job.snapshot()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        job = engine.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        with job.lock:
            job.cancel_requested = True
        return job.snapshot()

    @app.get("/api/jobs/{job_id}/files/{artifact}")
    def job_file(job_id: str, artifact: str):
        job = engine.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        with job.lock:
            path = job.artifacts.get(artifact)
        if path is None or not path.is_file():
            raise HTTPException(404, "Artifact not ready")
        media_type = "model/gltf-binary" if path.suffix == ".glb" else "application/octet-stream"
        return FileResponse(path, media_type=media_type, filename=f"{job_id}_{artifact}{path.suffix}", content_disposition_type="inline")

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str) -> dict[str, str]:
        job = engine.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        logs = sorted((job.folder / "logs").glob("*.log")) if (job.folder / "logs").exists() else []
        last = logs[-1].read_text(errors="replace").splitlines()[-30:] if logs else []
        return {"text": "\n".join(last)}

    dist = project_root / "web" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="studio")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("unimate_mlx.studio_api:app", host="127.0.0.1", port=int(os.environ.get("UNIMATE_STUDIO_PORT", "8000")))


if __name__ == "__main__":
    main()
