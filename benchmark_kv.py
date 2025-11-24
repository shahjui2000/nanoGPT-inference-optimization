import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from model import GPT

def benchmark_generation(model, start_tokens, max_new_tokens, use_cache=True):
    model.eval()
    device = start_tokens.device
    
    # Warmup - run multiple times to stabilize compute
    print(f"Warming up ({'with' if use_cache else 'without'} cache)...")
    with torch.no_grad():
        for _ in range(4):  # Multiple warmup runs
            model.generate(start_tokens, max_new_tokens=max_new_tokens, use_cache=use_cache)
    
    print(f"Benchmarking ({'with' if use_cache else 'without'} cache)...")
    times = []
    cache_sizes = []  # Track cache memory in MB
    
    idx = start_tokens.clone()
    past_kv = None
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Check if we're exceeding block size
            if idx.size(1) > model.config.block_size:
                print(f"WARNING: Exceeded block_size! Sequence length {idx.size(1)} > {model.config.block_size}")
                print("         Resetting cache.")
                # Reset cache and truncate sequence to block size
                past_kv = None
                idx = idx[:, -model.config.block_size:]
            
            t0 = time.perf_counter()
            
            if not use_cache:
                idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]
                cache_size_mb = 0  # No cache
            else:
                if past_kv is None:
                    idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
                    t_model_start = time.perf_counter()
                    logits, _, past_kv, _, _ = model(idx_cond, use_cache=True)
                    t_model_end = time.perf_counter()
                    if i < 3:  # Print first few iterations
                        print(f"  Token {i}: Model forward = {(t_model_end - t_model_start)*1000:.2f}ms (first token, seq_len={idx_cond.size(1)})")
                else:
                    idx_cond = idx[:, -1:]
                    t_model_start = time.perf_counter()
                    logits, _, past_kv, _, _ = model(idx_cond, past_kv=past_kv, use_cache=True)
                    t_model_end = time.perf_counter()
                    if i < 3:  # Print first few iterations
                        print(f"  Token {i}: Model forward = {(t_model_end - t_model_start)*1000:.2f}ms (cached, seq_len={idx_cond.size(1)})")
                logits = logits[:, -1, :]
                
                # Calculate cache size
                t_cache_start = time.perf_counter()
                cache_size_bytes = 0
                if past_kv is not None:
                    for layer_kv in past_kv:
                        k, v = layer_kv
                        cache_size_bytes += k.element_size() * k.nelement()
                        cache_size_bytes += v.element_size() * v.nelement()
                cache_size_mb = cache_size_bytes / (1024 * 1024)  # Convert to MB
                t_cache_end = time.perf_counter()
                if i < 3:
                    print(f"           Cache calc = {(t_cache_end - t_cache_start)*1000:.2f}ms")

            # Greedy decoding for benchmark stability
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t1 = time.perf_counter()
            times.append(t1 - t0)
            cache_sizes.append(cache_size_mb)
            if use_cache and i < 3:
                print(f"           Total = {(t1 - t0)*1000:.2f}ms\n")
            # Report cache size at intervals
            if use_cache and i in [9, 24, 49, 74, 99]:
                print(f"  Token {i}: Cache size = {cache_size_mb:.2f} MB (seq_len = {idx.size(1)})")
            
    return times, cache_sizes

def main():
    torch.manual_seed(1337)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a small GPT-2 model
    print("Loading GPT-2 model...")
    try:
        model = GPT.from_pretrained('gpt2')
    except Exception as e:
        print(f"Failed to load gpt2: {e}")
        print("Initializing a small random model instead.")
        from model import GPTConfig
        config = GPTConfig(n_layer=6, n_head=6, n_embd=384)
        model = GPT(config)
        
    model.to(device)
    
    # Prompt - use actual text to demonstrate cache benefits
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    prompt_text = " I am vibe coding to learn kv cache. To do that I am benchmarking kv cache."
    start_ids = enc.encode(prompt_text)
    start_tokens = torch.tensor([start_ids], dtype=torch.long, device=device)
    print(f"Prompt: '{prompt_text}' ({len(start_ids)} tokens)")
    
    #
    # 
    max_new_tokens = 100
    
    # Benchmark
    times_no_cache, cache_sizes_no_cache = benchmark_generation(model, start_tokens, max_new_tokens, use_cache=False)
    times_cache, cache_sizes_cache = benchmark_generation(model, start_tokens, max_new_tokens, use_cache=True)
    
    # Statistics
    avg_no_cache = np.mean(times_no_cache)
    avg_cache = np.mean(times_cache)
    max_cache_size = max(cache_sizes_cache)
    
    print(f"\nAverage time per token (no cache): {avg_no_cache*1000:.2f} ms")
    print(f"Average time per token (with cache): {avg_cache*1000:.2f} ms")
    print(f"Speedup: {avg_no_cache / avg_cache:.2f}x")
    print(f"Max cache size: {max_cache_size:.2f} MB")
    
    # Plotting - Dual axis plot
    try:
        # Convert times to milliseconds
        times_no_cache_ms = [t * 1000 for t in times_no_cache]
        times_cache_ms = [t * 1000 for t in times_cache]
        
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Plot time on left y-axis (in milliseconds)
        color_no_cache = '#E74C3C'  # Red
        color_cache = '#3498DB'      # Blue
        ax1.set_xlabel('Token Index', fontsize=12)
        ax1.set_ylabel('Time per Token (ms)', fontsize=12, color='black')
        line1 = ax1.plot(times_no_cache_ms, label='Time (No Cache)', color=color_no_cache, linewidth=2, alpha=0.8)
        line2 = ax1.plot(times_cache_ms, label='Time (With Cache)', color=color_cache, linewidth=2, alpha=0.8)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)
        
        # Plot cache size on right y-axis
        ax2 = ax1.twinx()
        color_cache_size = '#2ECC71'  # Green
        ax2.set_ylabel('Cache Size (MB)', fontsize=12, color=color_cache_size)
        line3 = ax2.plot(cache_sizes_cache, label='Cache Size', color=color_cache_size, linewidth=2, linestyle='--', alpha=0.8)
        ax2.tick_params(axis='y', labelcolor=color_cache_size)
        ax2.fill_between(range(len(cache_sizes_cache)), cache_sizes_cache, alpha=0.1, color=color_cache_size)
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10)
        
        plt.title(f'KV Cache: Time vs Memory Trade-off\n(Speedup: {avg_no_cache/avg_cache:.2f}x, Max Cache: {max_cache_size:.2f} MB)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=150)
        print("\nPlot saved to benchmark_results.png")
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == '__main__':
    main()
