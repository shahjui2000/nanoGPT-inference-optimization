# nanoGPT – KV‑Cache Inference Optimization

## Overview
This fork of **nanoGPT** adds a first‑stage inference optimization: **Key‑Value (KV) cache**.  The cache stores past key/value tensors so that each new token only requires a single forward pass instead of re‑processing the entire context.

## Features
- **KV‑cache implementation** in `model.py` with a `use_cache` flag.
- **Benchmark script** (`benchmark_kv.py`) measuring time per token (ms) and cache memory (MB) with a dual‑axis plot.
- **Animated visualization** (`visualize_kv.py`) showing query, keys, values, and attention for a chosen layer/head.
- Warm‑up runs and block‑size safety checks with clear warnings.

## Quick Start
```bash
# Install dependencies
pip install torch numpy tiktoken matplotlib

# Benchmark (default prompt " vibe coding to learn kv cache.")
python benchmark_kv.py

# Visualize KV dynamics (layer 5, head 0)
python visualize_kv.py
```
The benchmark generates `benchmark_results.png` (time vs. cache size) and the visualizer creates `kv_dashboard_l5_h0.gif`.

## Results (CPU)
- **~2.5× speed‑up** for token generation with cache.
- Cache grows linearly (~0.07 MB per token) up to the model’s `block_size` (1024 tokens).
- First‑token overhead is expected; subsequent tokens are constant‑time.

## Repository Structure
- `model.py` – core model with KV‑cache support.
- `benchmark_kv.py` – performance measurement and plotting.
- `visualize_kv.py` – animated dashboard of KV dynamics.
- `README.md` – this file.

## Contribution
Feel free to extend the cache to multi‑head, multi‑layer visualizations, or integrate with other inference optimizations (e.g., Flash‑Attention).

## Naming Suggestion
Given that KV‑caching is the primary inference improvement in this fork, a more descriptive repository name such as **`nanoGPT‑kv‑cache`** or **`nanoGPT‑inference‑optim`** would highlight its purpose. Renaming the fork accordingly could make it easier for users to discover this specific optimization.

---
*This README was generated automatically to reflect the current state of the project.*
