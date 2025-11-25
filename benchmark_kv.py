import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from model import GPT, GPTConfig
import tiktoken

# Set research-style aesthetics
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (10, 8),
    "lines.linewidth": 2,
    "lines.markersize": 6
})

def benchmark_batch(model, start_tokens, max_new_tokens, batch_size, device, use_cache=True):
    """
    Benchmark generation for a specific batch size.
    Returns:
        ttft: Time To First Token (seconds)
        ttpt: Time Per Token (seconds, average of decoding steps)
        throughput: Tokens per second
        peak_memory: Peak KV cache memory usage (MB)
    """
    model.eval()
    
    # Create batch
    batch_tokens = start_tokens.repeat(batch_size, 1)
    
    # Warmup
    # print(f"  Warmup (Batch Size {batch_size}, Cache={use_cache})...")
    with torch.no_grad():
        model.generate(batch_tokens, max_new_tokens=5, use_cache=use_cache)
    
    # Benchmark
    print(f"  Benchmarking (Batch Size {batch_size}, Cache={use_cache})...")
    
    idx = batch_tokens.clone()
    past_kv = None
    
    # Metrics
    t_start_prefill = 0
    t_end_prefill = 0
    decode_times = []
    peak_memory_mb = 0
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            t0 = time.perf_counter()
            
            # Prefill (First Token) vs Decoding
            if past_kv is None:
                # Prefill phase
                t_start_prefill = t0
                if use_cache:
                    idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
                    logits, _, past_kv, _, _ = model(idx_cond, use_cache=True)
                else:
                    idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
                    logits, _ = model(idx_cond)
                t_end_prefill = time.perf_counter()
                
            else:
                # Decoding phase
                if use_cache:
                    idx_cond = idx[:, -1:]
                    logits, _, past_kv, _, _ = model(idx_cond, past_kv=past_kv, use_cache=True)
                else:
                    idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
                    logits, _ = model(idx_cond)
            
            # Update memory peak (only relevant for cache)
            if use_cache and past_kv is not None:
                cache_size_bytes = 0
                for layer_kv in past_kv:
                    k, v = layer_kv
                    cache_size_bytes += k.element_size() * k.nelement()
                    cache_size_bytes += v.element_size() * v.nelement()
                peak_memory_mb = max(peak_memory_mb, cache_size_bytes / (1024 * 1024))

            # Greedy decoding
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
            
            if device == 'cuda':
                torch.cuda.synchronize()
                
            t1 = time.perf_counter()
            
            if i > 0: # Skip first token for TTPT
                decode_times.append(t1 - t0)

    # Calculate Metrics
    ttft = t_end_prefill - t_start_prefill
    ttpt = np.mean(decode_times) if decode_times else 0.0
    throughput = batch_size / ttpt if ttpt > 0 else 0.0
    
    return ttft, ttpt, throughput, peak_memory_mb

def plot_token_latency(model, start_tokens, max_new_tokens, device, num_runs=5):
    """
    Generate a plot of per-token latency for a single sequence (Batch Size 1).
    Compares Cache vs No Cache.
    Averages over `num_runs` to smooth out noise.
    """
    print(f"\nGenerating Token Latency Plot (Batch Size 1, averaged over {num_runs} runs)...")
    
    def run_trace(use_cache):
        # Warmup
        idx_warmup = start_tokens.clone()
        with torch.no_grad():
            model.generate(idx_warmup, max_new_tokens=5, use_cache=use_cache)
            
        # Multiple runs
        all_times = []
        for _ in range(num_runs):
            times = []
            idx = start_tokens.clone()
            past_kv = None
            with torch.no_grad():
                for i in range(max_new_tokens):
                    t0 = time.perf_counter()
                    if use_cache:
                        if past_kv is None:
                            idx_cond = idx
                            logits, _, past_kv, _, _ = model(idx_cond, use_cache=True)
                        else:
                            idx_cond = idx[:, -1:]
                            logits, _, past_kv, _, _ = model(idx_cond, past_kv=past_kv, use_cache=True)
                    else:
                        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
                        logits, _ = model(idx_cond)
                    
                    logits = logits[:, -1, :]
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                    idx = torch.cat((idx, idx_next), dim=1)
                    if device == 'cuda': torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times.append((t1 - t0) * 1000) # ms
            all_times.append(times)
        
        # Average over runs
        avg_times = np.mean(all_times, axis=0)
        return avg_times

    times_cache = run_trace(use_cache=True)
    times_no_cache = run_trace(use_cache=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(times_no_cache, label='No Cache', color='#E74C3C', linewidth=2, alpha=0.8)
    plt.plot(times_cache, label='KV Cache', color='#3498DB', linewidth=2, alpha=0.8)
    plt.xlabel('Token Index')
    plt.ylabel('Latency (ms)')
    plt.title(f'Per-Token Latency: Cache vs No Cache (Batch Size 1, Avg of {num_runs} runs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('token_latency.png', dpi=300)
    print("Saved token_latency.png")

def main():
    torch.manual_seed(1337)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load GPT-2
    print("Loading GPT-2 model...")
    try:
        model = GPT.from_pretrained('gpt2')
    except Exception as e:
        print(f"Failed to load gpt2: {e}")
        config = GPTConfig(n_layer=6, n_head=6, n_embd=384)
        model = GPT(config)
    model.to(device)

    # Prompt
    enc = tiktoken.get_encoding("gpt2")
    prompt_text = " I am vibe coding to learn kv cache. To do that I am benchmarking kv cache."
    start_ids = enc.encode(prompt_text)
    start_tokens = torch.tensor([start_ids], dtype=torch.long, device=device)
    print(f"Prompt: '{prompt_text}' ({len(start_ids)} tokens)")
    
    max_new_tokens = 50
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    # 1. Run Detailed Token Latency Trace (BS=1)
    plot_token_latency(model, start_tokens, max_new_tokens, device)
    
    # 2. Run Batch Benchmark Suite
    results_cache = {'bs': [], 'ttft': [], 'ttpt': [], 'throughput': [], 'memory': []}
    results_no_cache = {'bs': [], 'ttft': [], 'ttpt': [], 'throughput': [], 'memory': []}
    
    print("\nStarting Benchmark Suite...")
    
    for bs in batch_sizes:
        # With Cache
        ttft, ttpt, th, mem = benchmark_batch(model, start_tokens, max_new_tokens, bs, device, use_cache=True)
        results_cache['bs'].append(bs)
        results_cache['ttft'].append(ttft * 1000)
        results_cache['ttpt'].append(ttpt * 1000)
        results_cache['throughput'].append(th)
        results_cache['memory'].append(mem)
        
        # Without Cache
        ttft, ttpt, th, mem = benchmark_batch(model, start_tokens, max_new_tokens, bs, device, use_cache=False)
        results_no_cache['bs'].append(bs)
        results_no_cache['ttft'].append(ttft * 1000)
        results_no_cache['ttpt'].append(ttpt * 1000)
        results_no_cache['throughput'].append(th)
        results_no_cache['memory'].append(mem)

    # Plotting Grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('KV Cache Performance Analysis (GPT-2)', fontsize=16, fontweight='bold', y=0.95)
    
    # Helper for plots
    def plot_metric(ax, metric_key, title, ylabel):
        ax.plot(results_no_cache['bs'], results_no_cache[metric_key], 'o--', label='No Cache', color='#E74C3C', alpha=0.7)
        ax.plot(results_cache['bs'], results_cache[metric_key], 'o-', label='KV Cache', color='#3498DB', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Batch Size (log scale)')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log', base=2)
        ax.set_xticks(batch_sizes)
        ax.set_xticklabels(batch_sizes)
        ax.legend()

    plot_metric(axes[0, 0], 'ttft', 'Time To First Token (TTFT)', 'Latency (ms)')
    plot_metric(axes[0, 1], 'ttpt', 'Time Per Token (TTPT)', 'Latency (ms)')
    plot_metric(axes[1, 0], 'throughput', 'Generation Throughput', 'Tokens / Second')
    
    # Memory (Only Cache makes sense to plot as "KV Cache Memory", but we can show 0 for no cache or just plot cache)
    # The user asked for performance comparison, memory is specific to the cache feature.
    # Let's keep memory just for Cache to show the cost.
    ax = axes[1, 1]
    ax.plot(results_cache['bs'], results_cache['memory'], 'd-', color='#9B59B6', label='KV Cache Memory')
    ax.set_title('Peak KV Cache Memory')
    ax.set_xlabel('Batch Size (log scale)')
    ax.set_ylabel('Memory (MB)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels(batch_sizes)
    ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    print("\nBenchmark complete. Results saved to benchmark_results.png")

if __name__ == '__main__':
    main()
