# nanoGPT‑inference‑optimization

## Overview
A fork of **nanoGPT** that adds a first‑stage inference optimization – a **Key‑Value (KV) cache** – to dramatically speed up autoregressive generation. The repo also includes a benchmark script and an animated visualizer that expose the cache dynamics.

## Features
- **KV‑cache implementation** (`model.py`) with a `use_cache` flag.
- **Benchmark** (`benchmark_kv.py`) measuring per‑token latency (ms) and cache memory (MB) on CPU/GPU, plotted on a dual‑axis chart.
- **Animated visualization** (`visualize_kv.py`) showing query, keys, values, and attention for a chosen layer/head.
- Warm‑up runs and block‑size safety checks with clear warnings.

## Quick start
```bash
# Install dependencies
pip install torch numpy tiktoken matplotlib

# Benchmark (default prompt " vibe coding to learn kv cache.")
python benchmark_kv.py

# Visualize KV dynamics (layer 5, head 0)
python visualize_kv.py
```
The benchmark produces `benchmark_results.png` (time vs cache size) and the visualizer creates `kv_dashboard_l5_h0.gif`.

## Benchmark results
![Benchmark plot](/Users/jui/.gemini/antigravity/playground/volatile-ride/benchmark_results.png)
The plot shows a **~2.8× speed‑up** on CPU when using the cache, while the cache grows linearly (~0.07 MB per token) up to the model’s `block_size` (1024 tokens).

## KV‑cache visualization
![KV dashboard](/Users/jui/.gemini/antigravity/playground/volatile-ride/kv_dashboard_l5_h0.gif)
The GIF animates how each new token’s query interacts with the growing key/value cache and how attention is computed.

## Insights & lessons learned
- The cache must be reset when the sequence exceeds `block_size`; we now emit a warning and truncate the cache.
- Warm‑up runs (3 iterations) stabilize CPU/GPU state, eliminating first‑run jitter.
- Detailed per‑token timing (model forward, cache calculation, total) makes the constant‑time benefit of caching obvious.
- Cache size reporting at intervals (tokens 9, 24, 49, 74, 99) demonstrates linear memory growth.
- Prompt length matters: the first token processes the full prompt, subsequent tokens are constant‑time.

## Repository structure
- `model.py` – KV‑cache enabled model.
- `benchmark_kv.py` – performance measurement and plotting.
- `visualize_kv.py` – animated dashboard of KV dynamics.
- `README.md` – this documentation.

## Contribution
Feel free to extend the cache to multi‑head or multi‑layer visualizations, integrate other inference optimizations (e.g., Flash‑Attention), or improve the benchmark visual style.

## Naming suggestion
The repository is now named **`nanoGPT‑inference‑optimization`** to highlight that KV caching is the primary inference improvement.

---
*Generated automatically to reflect the current state of the project.*
