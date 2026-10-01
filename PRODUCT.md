# UniMate Studio <!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Vue 3 and TypeScript frontend with a local Python API. MLX runs the UniMate denoiser on Apple Silicon. Frozen FLAN-T5 currently uses PyTorch; Blender handles rig preprocessing and GLB/FBX animation export.

## Users

Game developers, technical artists, and animators working locally with an already rigged GLB. They want several short text-directed motions for one character without using command-line arguments for each clip.

## Product Purpose

Upload a rigged character, assemble a queue from action prompts or custom text, generate 2-second motion clips, inspect the result on the character, and download individual or chained animations. Success means seeing clear progress during a lengthy local render and receiving usable animated assets.

## Positioning

A local Apple Silicon motion workstation for arbitrary character rigs. The same interface keeps skeleton preparation, text conditioning, MLX generation, animation review, and export together while preserving access to the underlying run artifacts.

## Operating Context

Runs on the user's Mac with the project's installed Python dependencies, model weights, Blender 5.2, and a local browser. Uploaded GLBs and generated animations stay in the local workspace. Typical prompts include idle, walk, run, jump, attack, build, gather, carry, work, death, rescue, and celebrate. One rig is usually reused across several prompts.

## Capabilities and Constraints

- Upload a rigged GLB and verify skeleton eligibility against the selected checkpoint's 61- or 71-joint limit.
- Add, remove, and reorder multiple 2-second prompts from presets or custom text.
- Reuse character preprocessing and cached joint-name embeddings across clips.
- Generate each clip with MLX; offer preview, individual GLB/FBX download, and one chained result.
- Show honest job stages, elapsed time, and a changing ETA based on local measured runs.
- Keep a visible distinction between reference-parity adaptive sampling and faster approximate sampling.
- Asset and model quality vary by rig and seed; successful export does not prove convincing motion.

## Evidence on Hand

- The existing CLI supports upload-equivalent rig preprocessing, frozen T5, MLX inference, and animated GLB/FBX export.
- On an Apple M4, one full-width adaptive Go2 sample took about 59 seconds of numerical sampling. A trimmed 24-point RK4 sample took about 11 seconds, with measurable quality difference from the reference solver.
- The repository contains preset action ideas in ROADMAP.md and validated sample outputs under ignored `outputs/`.

## Product Principles

1. Keep the user focused on rig, prompts, progress, and playable output.
2. Show sampling time and accuracy trade-offs in plain language.
3. Preserve per-clip artifacts so a weak motion can be regenerated without losing the rest of the sequence.
4. Make local work and failure states inspectable rather than hiding long-running steps.
