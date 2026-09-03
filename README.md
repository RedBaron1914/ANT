# ANT

[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/accelerator-CUDA-green.svg)](https://developer.nvidia.com/cuda-zone)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

ANT is a stateful recurrent neural runtime written in Rust with custom CUDA C++ kernels and cuBLAS acceleration. It has no PyTorch, LibTorch, or Python dependencies.

The project is inspired by research on asynchronous Turing-complete neural computation (Siegelmann et al., 2026), predictive processing, and active inference.

> **Note on Neuromorphic Execution:** The standard inference pipeline operates in continuous surrogate mode (FP32/cuBLAS) for conversational latency. The event-driven neuromorphic kernel (`avx2_spiking.rs`) is an experimental backend for pure discrete spike-accumulation (INT8 x Binary Spikes), requiring manual temporal rate-encoding.

---

## Architecture Overview

### Recurrent Core & Deliberation

Instead of static Transformer-style self-attention, ANT uses a two-layer recurrent backbone coupled with adaptive deliberation:

- **minGRU (Layer 1):** A lightweight gated recurrent unit without explicit $h_{t-1}$ gate coupling, optimized for fast local sequence modeling.
- **Adaptive Deliberation & Frozen-State Thinking (ACT):** Deliberation occurs in working memory ($h$). During internal sub-steps, the associative matrix $S_{t-1}$ remains frozen (read-only) while thought candidates stabilize under Krasnoselskii-Mann attractor damping ($0.5 \cdot \text{prev} + 0.5 \cdot \text{cand}$). Thought deliberation halts once the Euclidean distance between successive updates drops below convergence tolerance ($\delta < 0.02$, up to 4 thinking steps) before updating state $S_t$ exactly once.
- **Gated DeltaNet-2 with Soft-Saturation (Layer 2):** Maintains a linear associative matrix state $S_t \in \mathbb{R}^{d \times d}$ with decoupled Erase ($b_t$) and Write ($w_t$) gates. Replaces hard clamping with smooth, differentiable soft-saturation ($S_{\text{next}} = S_{\text{raw}} / \sqrt{1 + (S_{\text{raw}}/5.0)^2}$) to eliminate gradient cliffs.
- **RMSNorm & residual connections** are applied throughout layers for stability.

### GIGO Defense: Dynamic Z-Score Bayesian Surprisal Filter

To prevent memory corruption from repetitive routine or chaotic noise, memory ingestion is gated by an Exponential Moving Average (EMA) Z-score tracker:

- Evaluates token surprisal $s_t = -\ln(P_{t-1}(x_t) + 10^{-7})$.
- Tracks running mean $\mu$ and variance $\sigma^2$ ($\lambda = 0.01$).
- Ingestion triggers only for meaningful prediction errors ($0.8 \le Z \le 3.5$), discarding both mundane events ($Z < 0.8$) and unstructured garbage ($Z > 3.5$).

### Memory Hierarchy & `.antpack` Cartridges

```text
 ┌──────────────────────────────────────────────────────────┐
 │ LEVEL 0: Local FIFO Token Buffer (~128–256 tokens)        │
 ├──────────────────────────────────────────────────────────┤
 │ LEVEL 1: Recurrent State (minGRU h_t + DeltaNet-2 S_t)   │
 ├──────────────────────────────────────────────────────────┤
 │ LEVEL 2: Associative Key-Value Index (GPU VRAM / RAM)     │
 │  - base_knowledge.ant (read-only DiskKVMemory mmap)      │
 │  - user_experience.ant (read-write DiskKVMemory mmap)    │
 │  - packs/*.antpack (modular read-only skill cartridges)  │
 ├──────────────────────────────────────────────────────────┤
 │ LEVEL 3: Lossless Session Tape (RAM Ring Buffer)          │
 └──────────────────────────────────────────────────────────┘
```

- **Dual Base / User Memory:** `base_knowledge.ant` stores static factual data; `user_experience.ant` records interactive episodic memories via memory-mapped files (`DiskKVMemory` with `MmapMut`).
- **Modular `.antpack` Skill Cartridges:** Pre-compiled domain packages stored in `packs/` that are automatically discovered and queried in unified associative memory attention.
- **Level 3 Session Tape:** In-memory ring buffer (`SessionTape`) preserving the exact token stream of the active session without lossy compression.
- **Sleep Phase:** Performs autonomous generative rollout dreams and cosine deduplication ($0.95$ threshold) to prune redundant episodic entries.

### Training & Lifelong Adaptation

- **GPU BPTT Engine:** All forward and backward passes run in VRAM (`GpuHistory`) without per-chunk host allocations.
- **Asynchronous Readout:** Cross-entropy loss and tied-weight embedding projections execute on CUDA streams.
- **Lion Optimizer:** Kernel-level parameter updates for base weights and LoRA adapters.
- **LoRA Merging & Reset (`merge_into_base`):** Fuses trained adapter deltas into base matrices ($W \leftarrow W + \frac{\alpha}{r} BA$) and resets low-rank subspaces to prevent lifelong saturation.

---

## Quick Start

### Requirements

- Rust 1.85+ (2024 Edition)
- NVIDIA GPU, CUDA Toolkit (Compute Capability 7.5+ recommended)
- Linux or Windows 10/11

### 1. Clone and build

```bash
git clone https://github.com/RedBaron1914/ANT.git
cd ANT
cargo build --release
```

### 2. Initialize

```bash
cargo run --release -- init
```

Generates `ant_config.toml`. Edit the configuration to customize your model dimensions, memory paths, and training parameters. Place a HuggingFace BPE `tokenizer.json` (vocab size matching `ant_config.toml`, e.g. 9016) at the configured `tokenizer_path`.

### 3. Neural Integrity & Diagnostic Harness

Run the automated diagnostic suite (~10s) to verify memory deduplication, negation ranking, ACT deliberation, GPU BPTT loss convergence, and state clamping:

```bash
cargo run --release -- test
```

### 4. Train

```bash
# Train from scratch or resume from checkpoint:
cargo run --release -- train --data datasets/your_dataset.txt --epochs 10 --lr 0.0001

# Fine-tune at a lower learning rate:
cargo run --release -- train --data datasets/your_dataset.txt --epochs 5 --lr 0.00002
```

### 5. Build `.antpack` Skill Cartridges

Compile arbitrary domain text files into standalone, deduplicated skill cartridges:

```bash
cargo run --release -- pack --data datasets/cpp_reference.txt --name cpp_knowledge --capacity 10000
```

All `.antpack` files placed in `packs/` are automatically mounted during chat.

### 6. Chat

```bash
cargo run --release -- chat
```

### 7. Memory Consolidation (Sleep Phase)

```bash
cargo run --release -- sleep
```

---

## References

1. Hava T. Siegelmann et al., *"Turing universal neural networks do not require global clocks"*, Nature Communications (2026).
2. Ali Hatamizadeh et al., *"Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"*, NVIDIA Research (2026).
3. Alex Graves, *"Adaptive Computation Time for Recurrent Neural Networks"*, (2016).
4. Karl Friston, *"The free-energy principle: a unified brain theory?"*, Nature Reviews Neuroscience (2010).
5. Leo Feng et al., *"Were RNNs All We Needed?"* (minGRU / minLSTM), (2024).
6. Xiangning Chen et al., *"Symbolic Discovery of Optimization Algorithms"* (Lion), Google Brain (2023).
