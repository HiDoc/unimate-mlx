# UniMate-MLX

MLX implementation of **UniMate** for Apple Silicon.

The goal of this project is to run UniMate text-to-motion inference natively on macOS using Apple's [MLX](https://github.com/ml-explore/mlx) framework, without requiring CUDA.

UniMate generates motion for arbitrary character skeletons from text prompts. This port focuses first on reproducing the original PyTorch inference results, then on optimizing the pipeline for Apple Silicon.

## Goals

- Native Apple Silicon inference with MLX
- No CUDA dependency
- Load the original UniMate EMA checkpoint
- Preserve arbitrary-skeleton support
- Reproduce PyTorch outputs numerically
- Support text-conditioned motion generation
- Export generated motion back to GLB/FBX characters
- Minimize unnecessary CPU ↔ GPU copies
- Take advantage of Apple unified memory
- Eventually remove the PyTorch dependency entirely

## Status

Work in progress.

Initial implementation target:

```text
Original UniMate preprocessing
        │
        ├── Skeleton topology
        ├── T-pose
        ├── Graph features
        └── dataset statistics
        │
        ▼
FLAN-T5 text encoder
PyTorch / Transformers initially
        │
        ▼
MLX UniMate denoiser
        │
        ▼
MLX flow-matching sampler
        │
        ▼
Original motion post-processing
        │
        ▼
Animated GLB / FBX
```

The first milestone deliberately keeps preprocessing, text encoding, and asset export compatible with the original project while replacing the computational core with MLX.

Once numerical parity is reached, the remaining components can be migrated incrementally.

---

## Why MLX?

MLX is designed specifically for Apple Silicon.

Unlike discrete GPUs, Apple Silicon uses unified memory:

```text
                  Unified Memory
                       │
            ┌──────────┴──────────┐
            │                     │
           CPU                   GPU
            │                     │
            └──── shared data ────┘
```

CPU and GPU can work on the same underlying memory without the explicit host-to-device copies normally required by CUDA workloads.

For UniMate this is especially attractive because the model is relatively small.

The released UniMate denoiser contains roughly:

```text
74.1M parameters
```

Approximate parameter memory:

```text
FP32   ~296 MB
FP16   ~148 MB
INT8    ~74 MB
```

The model therefore fits comfortably on common Apple Silicon machines.

A 24 GB Mac should have ample memory for UniMate inference, including the text encoder and intermediate tensors.

---

# Architecture

UniMate operates on motion tensors representing:

```text
frames × joints × joint features
```

The reference configuration uses approximately:

```text
Frames       60
Max joints   70
Features     12
Hidden dim   512
Layers       10
Heads        8
```

The core model is a factorized transformer.

```text
Motion
  │
  ▼
Input projection
  │
  ▼
┌─────────────────────────────────┐
│ Transformer Block × N           │
│                                 │
│  adaLN-Zero                     │
│       │                         │
│       ▼                         │
│  Spatial Attention              │
│  joints within each frame       │
│       │                         │
│       ▼                         │
│  Temporal Attention             │
│  frames for each joint          │
│       │                         │
│       ▼                         │
│  SwiGLU Feed Forward            │
└─────────────────────────────────┘
  │
  ▼
Output projection
  │
  ▼
Velocity field
```

UniMate additionally uses skeleton-aware information including:

- graph topology
- graph distance
- edge types
- Laplacian spectral features
- spectral rotary positional embeddings
- temporal rotary embeddings
- T-pose conditioning
- text conditioning
- timestep conditioning

---

# Repository Layout

Planned structure:

```text
unimate-mlx/
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── unimate_mlx/
│   ├── __init__.py
│   │
│   ├── model/
│   │   ├── model.py
│   │   ├── transformer.py
│   │   ├── attention.py
│   │   ├── adaln.py
│   │   ├── rope.py
│   │   ├── graph.py
│   │   ├── embeddings.py
│   │   └── feedforward.py
│   │
│   ├── inference/
│   │   ├── pipeline.py
│   │   ├── sampler.py
│   │   └── cfg.py
│   │
│   ├── text/
│   │   ├── encoder.py
│   │   └── cache.py
│   │
│   ├── skeleton/
│   │   ├── topology.py
│   │   ├── spectral.py
│   │   ├── preprocess.py
│   │   └── representation.py
│   │
│   ├── weights/
│   │   ├── convert.py
│   │   └── mapping.py
│   │
│   └── utils/
│       ├── memory.py
│       └── tensors.py
│
├── scripts/
│   ├── convert_weights.py
│   ├── generate.py
│   ├── compare_pytorch_mlx.py
│   └── benchmark.py
│
└── tests/
    ├── test_linear.py
    ├── test_rope.py
    ├── test_graph.py
    ├── test_attention.py
    ├── test_transformer.py
    ├── test_model.py
    └── test_sampler.py
```

---

# Installation

## Requirements

Recommended:

```text
macOS
Apple Silicon M1 or newer
Python >= 3.11
16 GB unified memory recommended
24 GB+ preferred for development/training experiments
```

Install the project:

```bash
git clone https://github.com/<user>/unimate-mlx.git
cd unimate-mlx

python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
```

Core dependencies will initially include:

```text
mlx
numpy
safetensors
transformers
sentencepiece
```

PyTorch may temporarily remain an optional dependency for reference-model comparison and preprocessing validation.

---

# Checkpoints

The MLX implementation should use the UniMate EMA checkpoint for inference.

Expected files:

```text
weights/
├── model_ema.safetensors
├── dataset_stats.npy
└── config.json
```

The checkpoint itself is already stored using Safetensors, so the model does not need to be serialized through PyTorch before being loaded into MLX.

---

# Weight Conversion

Safetensors tensors can be loaded directly as NumPy arrays and converted to MLX arrays.

Example:

```python
from safetensors import safe_open
import mlx.core as mx

weights = {}

with safe_open(
    "model_ema.safetensors",
    framework="numpy",
) as checkpoint:
    for name in checkpoint.keys():
        weights[name] = mx.array(
            checkpoint.get_tensor(name)
        )
```

The resulting arrays can optionally be saved again:

```python
mx.save_safetensors(
    "unimate-mlx.safetensors",
    weights,
)
```

The primary conversion task is therefore not changing the underlying tensor format.

The important part is correctly mapping:

```text
PyTorch module names
        ↓
MLX module names
```

while preserving tensor orientation and shared modules.

---

# MLX Model

## Linear Layers

PyTorch:

```python
torch.nn.Linear
```

MLX:

```python
mlx.nn.Linear
```

Most UniMate dense layers map directly.

---

## RMSNorm

UniMate uses normalization extensively.

MLX provides:

```python
mlx.nn.RMSNorm
```

Numerical parity must still be verified because implementation details such as epsilon values can differ.

---

## SwiGLU

Example implementation:

```python
import mlx.core as mx
import mlx.nn as nn


class SwiGLU(nn.Module):

    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.gate = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)

    def __call__(self, x):

        x = nn.silu(self.gate(x)) * self.up(x)

        return self.down(x)
```

---

# Attention

UniMate separates attention into spatial and temporal operations.

Input:

```text
[B, T, J, D]
```

where:

```text
B = batch
T = frames
J = joints
D = hidden dimension
```

## Spatial attention

Attention operates over joints independently for every frame.

```text
[B, T, J, D]

→

[B × T, H, J, Dh]
```

This allows the model to reason about relationships between body parts.

Skeleton topology biases are added to the attention logits.

---

## Temporal attention

Attention operates over time independently for every joint.

```text
[B, T, J, D]

→

[B × J, H, T, Dh]
```

This models motion trajectories.

---

## Basic MLX attention

```python
def attention(q, k, v, bias=None, mask=None):

    scale = q.shape[-1] ** -0.5

    scores = (
        q
        @ mx.swapaxes(k, -1, -2)
    ) * scale

    if bias is not None:
        scores = scores + bias

    if mask is not None:
        scores = mx.where(
            mask,
            scores,
            -1e9,
        )

    probs = mx.softmax(
        scores,
        axis=-1,
    )

    return probs @ v
```

The initial implementation should favor correctness and transparency over premature kernel optimization.

---

# Skeleton Graph Encoding

UniMate uses the character skeleton as a graph.

Nodes:

```text
joints
```

Edges:

```text
bones
```

From this graph we derive information such as:

```text
adjacency
shortest-path distance
edge type
Laplacian
Laplacian eigenvectors
joint masks
```

These values can initially be generated using NumPy.

They are relatively small and usually computed once per skeleton.

Example pipeline:

```text
Skeleton hierarchy
       │
       ▼
Adjacency matrix
       │
       ├── Graph distance
       ├── Edge metadata
       │
       ▼
Graph Laplacian
       │
       ▼
Eigenvectors
       │
       ▼
Spectral positional features
```

---

# Spectral RoPE

One of the more specialized parts of UniMate is its skeleton-aware rotary positional encoding.

Temporal attention uses conventional one-dimensional RoPE.

Spatial attention incorporates information derived from the skeleton's Laplacian eigenvectors.

Initial implementation:

```text
NumPy skeleton preprocessing
          │
          ▼
Laplacian eigenvectors
          │
          ▼
MLX spectral projection
          │
          ▼
Rotary embedding
```

The MLX implementation must reproduce the original sign-invariant behavior exactly before optimization.

---

# adaLN-Zero Conditioning

Transformer blocks are conditioned through adaptive normalization.

Sources include:

```text
timestep
text
T-pose / skeleton
```

Conceptually:

```python
condition = (
    time_embedding
    + text_embedding
    + skeleton_embedding
)
```

The condition produces modulation parameters:

```text
shift
scale
gate
```

which modulate normalized transformer activations.

Simplified:

```python
h = self.norm(x)

shift, scale, gate = mx.split(
    self.modulation(condition),
    3,
    axis=-1,
)

h = h * (1 + scale[..., None, :])
h = h + shift[..., None, :]

x = x + gate[..., None, :] * self.attention(h)
```

Exact broadcasting depends on the tensor layout used by the original model.

---

# Text Encoding

UniMate uses:

```text
google/flan-t5-base
```

as a frozen text encoder.

The initial port can retain Hugging Face Transformers:

```text
Prompt
  │
  ▼
FLAN-T5
PyTorch / MPS
  │
  ▼
Text embedding
  │
  ▼
UniMate MLX
```

This allows the motion model itself to be validated independently.

A later milestone can migrate FLAN-T5 to MLX:

```text
Prompt
  │
  ▼
MLX T5 encoder
  │
  ▼
UniMate MLX
```

---

# Cached Text Embeddings

Because the text encoder is frozen, embeddings can also be cached.

This is especially useful in games where animation prompts are known ahead of time.

For example:

```text
walk forward
run
attack
build
idle
death
rescue
```

can be encoded once:

```text
walk.npy
run.npy
attack.npy
build.npy
idle.npy
death.npy
rescue.npy
```

Runtime generation then no longer needs to invoke T5.

Pipeline:

```text
prompt
  │
  ▼
cached embedding
  │
  ▼
UniMate
```

This can reduce latency and memory usage considerably.

---

# Flow-Matching Sampler

UniMate predicts a velocity field rather than using a conventional DDPM denoising process.

Sampling begins from noise:

```python
x = mx.random.normal(shape)
```

For every integration step:

```python
velocity = model(
    x,
    timestep,
    text_condition,
    skeleton_condition,
)

x = x + dt * velocity
```

The initial MLX implementation should reproduce the integrator used by the reference implementation exactly.

Only after numerical parity is achieved should alternative solvers be considered.

Possible future solvers:

```text
Euler
Heun
RK4
adaptive Runge-Kutta
```

---

# Classifier-Free Guidance

Inference generally evaluates both conditioned and unconditioned predictions.

```python
v_conditioned = model(
    x,
    t,
    prompt,
    skeleton,
)

v_unconditioned = model(
    x,
    t,
    empty_prompt,
    skeleton,
)

velocity = (
    v_unconditioned
    + guidance_scale
    * (
        v_conditioned
        - v_unconditioned
    )
)
```

These two forward passes may later be batched together.

---

# Motion Pipeline

The neural network is only one part of the complete animation pipeline.

The target architecture is:

```text
GLB / FBX
    │
    ▼
Skeleton extraction
    │
    ▼
Canonical skeleton representation
    │
    ├── topology
    ├── T-pose
    └── graph features
    │
    ▼
Text prompt
    │
    ▼
FLAN-T5
    │
    ▼
UniMate MLX
    │
    ▼
Generated motion representation
    │
    ▼
Inverse normalization
    │
    ▼
Joint transforms
    │
    ▼
Apply animation to original rig
    │
    ▼
Animated GLB / FBX
```

Asset parsing and exporting do not need to be implemented using MLX.

MLX should be responsible primarily for numerical model computation.

---

# Numerical Validation

Correctness should be established before performance optimization.

For each component:

```text
Same checkpoint
Same input
Same timestep
Same skeleton
Same text embedding
```

Run:

```text
PyTorch
   │
   ▼
reference output

MLX
   │
   ▼
MLX output
```

Then compare:

```python
error = np.max(
    np.abs(
        pytorch_output
        - mlx_output
    )
)
```

Recommended validation order:

```text
Linear
  ↓
RMSNorm
  ↓
SwiGLU
  ↓
Temporal RoPE
  ↓
Spectral RoPE
  ↓
Graph bias
  ↓
Spatial attention
  ↓
Temporal attention
  ↓
adaLN
  ↓
One transformer block
  ↓
Full transformer
  ↓
Velocity prediction
  ↓
One sampler step
  ↓
Full generation
```

Indicative tolerance targets:

```text
FP32

~1e-5 to 1e-4

FP16 / BF16

~1e-3
```

The exact tolerance should be determined empirically.

---

# MLX Unified Memory

MLX runs directly on Apple's unified memory architecture.

There is no separate CUDA-style VRAM pool.

Example:

```python
import mlx.core as mx

x = mx.random.normal((4096, 4096))
y = x @ x

mx.eval(y)
```

MLX uses lazy execution.

Expressions construct a computation graph and are evaluated when required.

Memory usage can be inspected using:

```python
print(
    mx.get_active_memory()
)

print(
    mx.get_cache_memory()
)

print(
    mx.get_peak_memory()
)
```

Cached allocations can be released with:

```python
mx.clear_cache()
```

---

# Memory Targets

UniMate itself should require very little unified memory.

Approximate model weight footprint:

| Precision | Weight memory |
|---|---:|
| FP32 | ~296 MB |
| FP16 | ~148 MB |
| INT8 | ~74 MB |
| INT4 | ~37 MB |

Inference additionally requires:

```text
T5 encoder
activations
attention buffers
skeleton data
sampling state
runtime memory
```

A 16 GB Mac should therefore be capable of inference.

A 24 GB machine provides substantial headroom.

Quantization is not initially necessary.

---

# Performance Strategy

Optimization should happen only after output parity.

Suggested order:

```text
Correct FP32 implementation
        ↓
FP16 / BF16
        ↓
mx.compile
        ↓
Remove redundant transposes
        ↓
Batch CFG passes
        ↓
Cache skeleton features
        ↓
Cache text embeddings
        ↓
Fuse attention operations
        ↓
Optimize sampler
        ↓
Optional quantization
```

---

# `mx.compile`

Once the model is stable, hot paths can be compiled:

```python
import mlx.core as mx

compiled_step = mx.compile(
    generation_step
)
```

This is particularly useful for the repeated transformer evaluations performed during flow matching.

---

# Caching

UniMate has several inputs that generally remain constant during a generation.

These should be cached where possible.

Per-character cache:

```text
skeleton topology
graph distances
edge types
Laplacian eigenvectors
T-pose representation
joint masks
```

Per-prompt cache:

```text
FLAN-T5 embedding
```

Per-generation:

```text
noise
timesteps
motion state
```

This transforms runtime execution into something close to:

```text
cached skeleton
      +
cached text
      +
random seed
      ↓
UniMate
      ↓
animation
```

---

# Example CLI

Planned interface:

```bash
python -m unimate_mlx.generate \
    --character character.glb \
    --prompt "walk forward confidently" \
    --output walk.glb
```

Additional options:

```bash
python -m unimate_mlx.generate \
    --character character.glb \
    --prompt "run forward" \
    --steps 30 \
    --guidance 3.0 \
    --seed 42 \
    --dtype float16 \
    --output run.glb
```

---

# Python API

Target API:

```python
from unimate_mlx import UniMatePipeline


pipeline = UniMatePipeline.from_pretrained(
    "Linzhan/UniMate",
    dtype="float16",
)

motion = pipeline.generate(
    character="character.glb",
    prompt="walk forward",
    seed=42,
)

motion.export(
    "walk.glb"
)
```

Using a precomputed character:

```python
character = pipeline.prepare_character(
    "character.glb"
)

motion = pipeline.generate(
    character=character,
    prompt="run forward",
)
```

---

# Benchmarking

Benchmarks should record at least:

```text
machine
SoC
RAM
model dtype
frames
joint count
sampling steps
generation latency
peak unified memory
```

Example:

```bash
python scripts/benchmark.py \
    --model weights/unimate-mlx.safetensors \
    --frames 60 \
    --joints 70 \
    --steps 30
```

Example output:

```text
Device: Apple M4 Pro
Unified memory: 24 GB
Precision: FP16
Frames: 60
Joints: 70
Steps: 30

Generation: TBD
Peak memory: TBD
```

No performance numbers should be published until the MLX implementation reproduces the reference model.

---

# Training

Inference is the initial priority.

Training support can follow once forward-pass parity is established.

Approximate parameter-related memory for a 74M parameter model:

```text
FP16 weights             ~150 MB
FP16 gradients           ~150 MB
FP32 master parameters   ~300 MB
Adam first moment        ~300 MB
Adam second moment       ~300 MB
--------------------------------
                         ~1.2 GB
```

Actual training memory will be dominated increasingly by activations as batch size grows.

Potential optimization methods:

```text
gradient checkpointing
gradient accumulation
mixed precision
smaller per-device batches
optimizer state reduction
```

Apple Silicon training should be considered primarily for:

```text
experimentation
fine-tuning
LoRA / adapters
small datasets
research
```

rather than reproducing large multi-H100 training throughput.

---

# Development Roadmap

Status on 2026-09-30: the hybrid pipeline runs on public biped and Go2 quadruped GLBs with Blender 5.2, PyTorch FLAN-T5 and the MLX denoiser. FP32 velocity and the default adaptive Dormand–Prince sampler have been compared with official PyTorch on identical synthetic and real Go2 conditioning. The Go2 adaptive sample differs by at most 8.6e-5; compiled MLX attention and denoiser calls reduce runtime. Fixed-grid joint trimming is an optional approximation path. Checkboxes reflect implemented and verified work; see the README for measured speed and accuracy.

## Phase 1 — Reference environment

- [ ] Run the complete official UniMate inference CLI on the same character and prompt
- [x] Download EMA weights and matching config/statistics
- [x] Save deterministic test inputs
- [x] Save intermediate PyTorch activations
- [x] Establish a reference velocity baseline

## Phase 2 — Weight loader

- [x] Read UniMate Safetensors
- [x] Inspect checkpoint structure
- [x] Map PyTorch parameters to MLX using their original names
- [x] Validate tensor shapes against the released manifest and official model
- [x] Handle shared spectral encoder weights

## Phase 3 — Core transformer

- [x] Linear layers
- [x] RMSNorm
- [x] SwiGLU
- [x] timestep embeddings
- [x] adaLN-Zero
- [x] temporal attention

## Phase 4 — Skeleton conditioning

- [x] Skeleton topology through official preprocessing
- [x] graph distance
- [x] edge encoding
- [x] Laplacian
- [x] spectral features
- [x] spectral RoPE
- [x] spatial attention

## Phase 5 — Full model parity

- [x] Load original EMA checkpoint
- [x] Compare block outputs
- [x] Compare full velocity predictions
- [x] Validate FP32 parity on deterministic synthetic and real Go2 inputs
- [ ] Validate FP16 parity

## Phase 6 — Sampling

- [x] Fixed Euler flow-matching sampler
- [x] Fixed torchdiffeq-compatible RK4 sampler
- [x] CFG
- [x] deterministic MLX seeds and support for identical supplied noise
- [x] full 60-frame motion generation
- [x] Compare full 49-update PyTorch and MLX Euler trajectories
- [x] Match the reference default adaptive ODE solver in FP32

## Phase 7 — Asset pipeline

- [x] GLB skeleton extraction through official Blender preprocessing
- [x] skeleton preprocessing and checkpoint-compatible conditioning
- [x] generated motion conversion using dataset statistics
- [x] animation retargeting through official Blender export
- [x] GLB export, verified on RiggedFigure
- [x] FBX export, verified on RiggedFigure
- [x] Broader character and rig compatibility testing: eleven cases, 2–71 joints across preview and v2; all GLB/FBX exports reimported
- [x] Stress and visual-quality testing: preview 61/62 and v2 71/72 joint boundaries, Go2 render and gait metrics
- [ ] Improve quadruped foot contact and reduce sliding in generated motion; see QUALITY.md

## Phase 8 — Pure MLX

- [ ] Replace PyTorch T5
- [ ] MLX FLAN-T5 encoder
- [ ] Remove Torch runtime
- [x] Cache text embeddings for prompt reuse
- [ ] Remove Transformers dependency where possible

## Phase 9 — Optimization

- [ ] FP16
- [ ] BF16 evaluation
- [x] `mx.compile` on repeated denoiser calls
- [x] batched CFG option (measured slower than separate passes on this M4)
- [x] cached spectral angles and graph-bias tensors during sampling
- [x] memory profiling script
- [x] optimized MLX attention and optional padded-joint trimming
- [ ] optional quantization

---

# First Milestone

The first usable version should intentionally remain simple:

```text
PyTorch / NumPy preprocessing
            +
Transformers FLAN-T5
            +
MLX UniMate
            +
original post-processing
```

Success criteria:

```text
Same character
Same prompt
Same initial noise
Same sampler settings

PyTorch UniMate
      ≈
MLX UniMate
```

Once that is true, the project can safely migrate the peripheral components without making debugging unnecessarily difficult.

---

# Long-Term Target

The final architecture should be:

```text
                 ┌──────────────────┐
Prompt ─────────▶│ MLX FLAN-T5      │
                 └────────┬─────────┘
                          │
                          ▼

GLB ──▶ Skeleton preprocessing
                │
                ▼
        Skeleton features
                │
                ├──────────────┐
                │              │
                ▼              ▼
             Text          Skeleton
           embedding      conditioning
                │              │
                └──────┬───────┘
                       ▼
                ┌────────────┐
                │ UniMate MLX│
                └─────┬──────┘
                      │
                      ▼
                  Motion
                      │
                      ▼
               Retargeting
                      │
                      ▼
                Animated GLB
```

The resulting runtime should be capable of running fully locally on Apple Silicon without CUDA.

---

# Target Use Cases

UniMate-MLX could be useful for:

- game animation generation
- procedural NPC animation
- automatic animation libraries
- Blender pipelines
- character prototyping
- GLB asset generation
- local animation tools
- automated animation pipelines
- batch generation of motion clips
- text-driven animation authoring

For game-development pipelines, predefined prompts can be generated offline:

```text
idle
walk
run
jump
attack
build
gather
carry
work
death
rescue
celebrate
```

Animations can then be exported to GLB and consumed directly by an engine without running UniMate at game runtime.

---

# References

- UniMate: Universal Motion Generation for Arbitrary Character Skeletons
- Original UniMate implementation
- Apple MLX
- Hugging Face Transformers
- Safetensors
- FLAN-T5

---

# Disclaimer

This project is an independent MLX port.

It is not an official implementation of UniMate and should follow the licenses and usage conditions of the original project, model weights, datasets, and third-party dependencies.
