# ANT

[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/accelerator-CUDA-green.svg)](https://developer.nvidia.com/cuda-zone)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

ANT is a stateful recurrent neural runtime written in Rust with custom CUDA C++ kernels and cuBLAS acceleration. It has no PyTorch, LibTorch, or Python dependencies.

The project is inspired by research on asynchronous Turing-complete neural computation (Siegelmann et al., 2026).

---

## Architecture Overview

### Recurrent Core

Instead of Transformer-style attention, ANT uses a two-layer recurrent backbone:

- **minGRU (Layer 1):** A lightweight gated recurrent unit without explicit `h_{t-1}` gate coupling, suited for fast local sequence modeling.
- **Gated DeltaNet-2 (Layer 2):** Maintains a linear associative matrix state $S_t \in \mathbb{R}^{d \times d}$ with separate Erase ($b_t$) and Write ($w_t$) gates, allowing selective in-context updates.
- **RMSNorm & residual connections** are applied between layers for training stability.

### Memory Hierarchy

```text
 ┌──────────────────────────────────────────────────────────┐
 │ LEVEL 0: Local FIFO Token Buffer (~128–256 tokens)        │
 ├──────────────────────────────────────────────────────────┤
 │ LEVEL 1: Recurrent State (minGRU h_t + DeltaNet-2 S_t)   │
 ├──────────────────────────────────────────────────────────┤
 │ LEVEL 2: Associative Key-Value Index (GPU VRAM / RAM)     │
 │  - base_knowledge.ant (read-only) + user_experience.ant  │
 ├──────────────────────────────────────────────────────────┤
 │ LEVEL 3: Lossless Session Tape (NVMe mmap)                │
 └──────────────────────────────────────────────────────────┘
```

**Level 2** uses dense cosine $k$-NN search on GPU VRAM for retrieval. New entries are written during inference when the sparse gating layer detects a high novelty signal. A separate `sleep` phase clusters and deduplicates stored vectors.

**Level 3** is a memory-mapped append-only log that preserves the exact token stream without attention-based compression loss.

### Training

Training runs fully on GPU via BPTT across sequence chunks. Key details:

- All forward and backward tensors are pre-allocated in VRAM (`GpuHistory`) — no per-chunk heap allocation on the hot path.
- The cross-entropy loss and readout kernels are queued asynchronously on the CUDA stream; the CPU thread does not block mid-pass.
- Gate energies are computed with a GPU reduction kernel (`compute_gate_energy_kernel`) rather than downloading pre-activations to the host.
- Parameter updates use a CUDA implementation of the Lion optimizer.
- LoRA adapters are trained alongside base weights and stored in VRAM during the training session.

### Dual `.ant` Storage

Memory is split into two separate files:

| File | Mode | Purpose |
|---|---|---|
| `base_knowledge.ant` | Read-only | Pre-trained factual knowledge |
| `user_experience.ant` | Read-write | Session-learned episodic memories |

Both are searched jointly at inference time.

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

This generates `ant_config.toml` and the initial tokenizer. Edit the config to set your model dimensions, memory paths, and training parameters before proceeding.

### 3. Train

```bash
# Train from scratch or resume from checkpoint:
cargo run --release -- train --data datasets/your_dataset.txt --epochs 10 --lr 0.0001

# Fine-tune at a lower learning rate:
cargo run --release -- train --data datasets/your_dataset.txt --epochs 5 --lr 0.00002
```

### 4. Chat

```bash
cargo run --release -- chat
```

### 5. Memory consolidation (sleep phase)

```bash
cargo run --release -- sleep
```

---

## References

1. Hava T. Siegelmann et al., *"Turing universal neural networks do not require global clocks"*, Nature Communications (2026).
2. Ali Hatamizadeh et al., *"Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"*, NVIDIA Research (2026).
3. Leo Feng et al., *"Were RNNs All We Needed?"* (minGRU / minLSTM), (2024).
4. Xiangning Chen et al., *"Symbolic Discovery of Optimization Algorithms"* (Lion), Google Brain (2023).
