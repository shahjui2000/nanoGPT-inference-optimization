import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import tiktoken
from model import GPT

def visualize_dashboard_animation(model, start_tokens, max_new_tokens, layer_idx=5, head_idx=0):
    model.eval()
    device = start_tokens.device
    enc = tiktoken.get_encoding("gpt2")
    
    idx = start_tokens.clone()
    past_kv = None
    
    # Store data
    keys_history = [] 
    values_history = []
    attention_history = [] 
    queries_history = [] # New: Store Query vectors
    token_ids_history = []
    
    token_ids_history.extend(idx[0].tolist())
    
    print(f"Generating {max_new_tokens} tokens...")
    with torch.no_grad():
        for i in range(max_new_tokens):
            if past_kv is None:
                idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
                logits, _, past_kv, att_weights, queries = model(idx_cond, use_cache=True)
                # First step: attention to all previous
                current_att = att_weights[layer_idx][0, head_idx, -1, :].cpu().numpy()
                current_q = queries[layer_idx][0, head_idx, -1, :].cpu().numpy() # (head_size,)
            else:
                idx_cond = idx[:, -1:]
                logits, _, past_kv, att_weights, queries = model(idx_cond, past_kv=past_kv, use_cache=True)
                # Subsequent steps
                current_att = att_weights[layer_idx][0, head_idx, 0, :].cpu().numpy()
                current_q = queries[layer_idx][0, head_idx, 0, :].cpu().numpy() # (head_size,)
            
            # Capture K and V
            k_layer, v_layer = past_kv[layer_idx]
            k_head = k_layer[0, head_idx].cpu().numpy() # (T, head_size)
            v_head = v_layer[0, head_idx].cpu().numpy() # (T, head_size)
            
            keys_history.append(k_head)
            values_history.append(v_head)
            attention_history.append(current_att)
            queries_history.append(current_q)
            
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
            token_ids_history.append(idx_next.item())

    print(f"Visualizing Layer {layer_idx}, Head {head_idx}")
    
    # Debug: Check first token's key/value vectors
    if len(keys_history) > 0:
        first_k = keys_history[0][0]  # First token's key
        first_v = values_history[0][0]  # First token's value
        print(f"\nFirst token Key stats:")
        print(f"  Min: {first_k.min():.4f}, Max: {first_k.max():.4f}, Mean: {first_k.mean():.4f}")
        print(f"  Zeros: {(first_k == 0).sum()} / {len(first_k)}")
        print(f"First token Value stats:")
        print(f"  Min: {first_v.min():.4f}, Max: {first_v.max():.4f}, Mean: {first_v.mean():.4f}")
        print(f"  Zeros: {(first_v == 0).sum()} / {len(first_v)}")
    
    token_strs = [enc.decode([tid]) for tid in token_ids_history]
    token_strs = [s.replace('\n', '\\n') for s in token_strs]

    # Skip the first token (start token)
    
    # Setup Figure: 4 Panels (Query, Keys, Values, Attention)
    fig = plt.figure(figsize=(20, 8))
    # Layout: Query (Left), Keys, Values, Attention (Right)
    # Query is 1xDim. Keys is TxDim.
    # To align them, we can plot Query as a vertical bar (Dim x 1) or horizontal (1 x Dim).
    # Since Keys is plotted with X=Dim, Y=Token, let's plot Query as a horizontal bar ABOVE Keys?
    # Or as a separate panel.
    # User asked for "build the same for a new token".
    # Let's try: [Query (1xD)] [Keys (TxD)] [Values (TxD)] [Attention (Tx1)]
    # But Query is just one vector per step.
    # Let's plot it as a heatmap of shape (1, D) but stretched or just shown clearly.
    
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.05, 0.5], height_ratios=[0.1, 1]) 
    # Top row: Query (spanning first column)
    # Bottom row: Keys, Values, Colorbar, Attention
    
    # Actually, let's put Query above Keys.
    ax_q = fig.add_subplot(gs[0, 0]) # Top Left
    ax_k = fig.add_subplot(gs[1, 0]) # Bottom Left
    ax_v = fig.add_subplot(gs[1, 1]) # Bottom Middle
    cax = fig.add_subplot(gs[1, 2]) # Colorbar
    ax_a = fig.add_subplot(gs[1, 3]) # Bottom Right
    
    # Styling
    plt.style.use('seaborn-v0_8-whitegrid') 
    cmap_kv = 'Pastel1' # Research paper style: Red-Yellow-Blue (light colors)
    color_att = '#FFB6C1' 
    
    def update(frame):
        ax_q.clear()
        ax_k.clear()
        ax_v.clear()
        ax_a.clear()
        cax.clear()
        
        # Data for this frame
        k_data = keys_history[frame]
        v_data = values_history[frame]
        att_data = attention_history[frame]
        q_data = queries_history[frame].reshape(1, -1) # (1, D)
        
        current_tokens = token_strs[:len(att_data)]
        y_pos = np.arange(len(current_tokens))
        
        if len(current_tokens) == 0:
            return [] 
            
        # Robust Scaling
        flat_k = k_data.flatten()
        vmin_k, vmax_k = np.percentile(flat_k, 5), np.percentile(flat_k, 95)
        # For diverging colormap, center around 0
        limit_k = max(abs(vmin_k), abs(vmax_k))
        vmin_k, vmax_k = -limit_k, limit_k
        
        flat_v = v_data.flatten()
        vmin_v, vmax_v = np.percentile(flat_v, 5), np.percentile(flat_v, 95)
        limit_v = max(abs(vmin_v), abs(vmax_v))
        vmin_v, vmax_v = -limit_v, limit_v
        
        # Identify the token that generated this query
        # frame is the step index (0 to max_new-1)
        # The query source is the token at index (start_len - 1 + frame)
        # start_tokens has shape (1, 1) in this script?
        # start_tokens = torch.tensor([start_ids]) -> shape (1, len)
        start_len = start_tokens.shape[1]
        query_source_idx = start_len - 1 + frame
        query_source_token = token_strs[query_source_idx] if query_source_idx < len(token_strs) else "?"
        
        # 0. Query Heatmap (Top Left)
        im_q = ax_q.imshow(q_data, aspect='auto', cmap=cmap_kv, interpolation='nearest', vmin=vmin_k, vmax=vmax_k)
        ax_q.set_title(f'Query Vector\n(from "{query_source_token}")', fontsize=14, fontweight='bold', color='#333333')
        ax_q.set_yticks([])
        ax_q.set_xticks([]) 
        ax_q.grid(False)
        
        # 1. Keys Heatmap
        im_k = ax_k.imshow(k_data, aspect='auto', cmap=cmap_kv, interpolation='nearest', vmin=vmin_k, vmax=vmax_k)
        ax_k.set_title(f'Key Cache (Memory Address)', fontsize=14, fontweight='bold', color='#333333')
        ax_k.set_xlabel('Head Dimension', fontsize=10)
        ax_k.set_yticks(y_pos)
        ax_k.set_yticklabels(current_tokens, fontsize=11, fontweight='bold')
        ax_k.grid(False)
        
        # 2. Values Heatmap
        im_v = ax_v.imshow(v_data, aspect='auto', cmap=cmap_kv, interpolation='nearest', vmin=vmin_v, vmax=vmax_v)
        ax_v.set_title(f'Value Cache (Memory Content)', fontsize=14, fontweight='bold', color='#333333')
        ax_v.set_xlabel('Head Dimension', fontsize=10)
        ax_v.set_yticks([]) 
        ax_v.grid(False)
        
        # Shared Colorbar
        plt.colorbar(im_v, cax=cax, label='Activation Value')
        
        # 3. Attention Bar Chart
        # Set first token's attention to 0 for visualization, but show actual value as text
        att_data_display = att_data.copy()
        first_token_att = att_data[0]  # Store actual value
        att_data_display[0] = 0  # Set to 0 for bar chart
        
        bars = ax_a.barh(y_pos, att_data_display, color=color_att, alpha=0.9, edgecolor='none')
        ax_a.set_title(f'Attention (Selection)', fontsize=14, fontweight='bold', color='#333333')
        ax_a.set_xlabel('Score', fontsize=10)
        ax_a.set_ylim(-0.5, len(current_tokens)-0.5)
        ax_a.invert_yaxis() 
        ax_a.set_yticks([]) 
        
        # Dynamic X-Limit (excluding first token for scaling)
        max_att = np.max(att_data_display[1:]) if len(att_data_display) > 1 else 1.0
        limit = max(max_att * 1.1, 0.1) 
        ax_a.set_xlim(0, limit) 
        
        # Annotate the max attention bar (excluding first token)
        if len(att_data) > 1:
            max_idx = np.argmax(att_data[1:]) + 1  # +1 because we're searching from index 1
            max_val = att_data[max_idx]
            ax_a.text(max_val + (limit*0.02), max_idx, f'{max_val:.2f}', va='center', fontsize=9, fontweight='bold', color='#333333')
        
        # Show actual value for first token as text
        ax_a.text(0.01, 0, f'{first_token_att:.2f}', va='center', fontsize=9, fontweight='bold', color='#666666', style='italic')
        
        gen_token_idx = len(keys_history[frame])
        gen_token = token_strs[gen_token_idx] if gen_token_idx < len(token_strs) else "END"
        fig.suptitle(f'Step {frame+1}: Generating "{gen_token}"', fontsize=16, fontweight='bold', color='#333333')
        
        return [im_q, im_k, im_v]

    # Show all frames including the first one
    ani = animation.FuncAnimation(fig, update, frames=range(len(keys_history)), interval=600, blit=False)
    ani.save(f'kv_dashboard_l{layer_idx}_h{head_idx}.gif', writer='pillow')
    print(f"Saved Dashboard GIF to kv_dashboard_l{layer_idx}_h{head_idx}.gif")
    plt.close()

def main():
    torch.manual_seed(42) # New seed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading GPT-2 model...")
    try:
        model = GPT.from_pretrained('gpt2')
    except Exception as e:
        print(f"Failed to load gpt2: {e}")
        from model import GPTConfig
        config = GPTConfig(n_layer=12, n_head=12, n_embd=768)
        model = GPT(config)
    
    print("Disabling Flash Attention for visualization...")
    for block in model.transformer.h:
        block.attn.flash = False
        
    model.to(device)
    
    # New Prompt: "vibe coding to learn kv cache."
    enc = tiktoken.get_encoding("gpt2")
    prompt_text = " vibe coding to learn kv cache."
    start_ids = enc.encode(prompt_text)
    start_tokens = torch.tensor([start_ids], dtype=torch.long, device=device)
    
    max_new_tokens = 20
    
    # Visualize Layer 5, Head 0
    visualize_dashboard_animation(model, start_tokens, max_new_tokens, layer_idx=5, head_idx=0)

if __name__ == '__main__':
    main()
