"""
Speculative Decoding Implementation

This module implements speculative decoding using a small draft model to generate
candidate tokens that are verified in parallel by a larger target model.
"""

import torch
import time
from model import GPT

class SpeculativeDecoder:
    def __init__(self, target_model, draft_model, device='cpu'):
        """
        Initialize speculative decoder with target and draft models.
        
        Args:
            target_model: Large model (e.g., gpt2-medium)
            draft_model: Small model (e.g., gpt2)
            device: Device to run on
        """
        self.target = target_model
        self.draft = draft_model
        self.device = device
        
        self.target.eval()
        self.draft.eval()
        
        # Statistics
        self.stats = {
            'total_draft_tokens': 0,
            'accepted_tokens': 0,
            'rejected_tokens': 0,
            'iterations': 0,
            'draft_time': 0.0,
            'verify_time': 0.0
        }
    
    def generate(self, prompt_ids, max_new_tokens=50, K=4, verbose=True):
        """
        Generate text using speculative decoding.
        
        Args:
            prompt_ids: List of token IDs for the prompt
            max_new_tokens: Maximum number of tokens to generate
            K: Number of tokens to draft per iteration
            verbose: Print iteration details
            
        Returns:
            generated_ids: List of all generated token IDs
            stats: Dictionary of statistics
        """
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated_ids = []
        
        # Reset stats
        self.stats = {
            'total_draft_tokens': 0,
            'accepted_tokens': 0,
            'rejected_tokens': 0,
            'iterations': 0,
            'draft_time': 0.0,
            'verify_time': 0.0
        }
        
        iteration = 0
        while len(generated_ids) < max_new_tokens:
            iteration += 1
            self.stats['iterations'] = iteration
            
            # Step 1: Draft K tokens using draft model
            t0 = time.perf_counter()
            draft_tokens = []
            draft_idx = idx.clone()
            
            with torch.no_grad():
                for _ in range(K):
                    logits, _ = self.draft(draft_idx)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    draft_tokens.append(next_token.item())
                    draft_idx = torch.cat([draft_idx, next_token], dim=1)
            
            t1 = time.perf_counter()
            self.stats['draft_time'] += (t1 - t0)
            self.stats['total_draft_tokens'] += len(draft_tokens)
            
            # Step 2: Verify with target model
            t0 = time.perf_counter()
            
            # Create sequence with draft tokens
            draft_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=self.device)
            verify_seq = torch.cat([idx, draft_tensor], dim=1)
            
            with torch.no_grad():
                logits, _ = self.target(verify_seq)
            
            # Extract logits for verification
            # logits[:, i, :] predicts token at position i+1
            # We want to check if logits at positions [len(idx)-1, len(idx), ..., len(idx)+K-2]
            # predict the tokens at positions [len(idx), len(idx)+1, ..., len(idx)+K-1]
            # which are the draft tokens
            start_pos = idx.size(1) - 1
            verify_logits = logits[:, start_pos:start_pos+K, :]
            target_predictions = torch.argmax(verify_logits, dim=-1).squeeze(0).tolist()
            
            t1 = time.perf_counter()
            self.stats['verify_time'] += (t1 - t0)
            
            # Step 3: Accept/Reject
            accepted_tokens = []
            num_accepted = 0
            
            for draft_tok, target_tok in zip(draft_tokens, target_predictions):
                if draft_tok == target_tok:
                    accepted_tokens.append(draft_tok)
                    num_accepted += 1
                else:
                    # First mismatch: use target's prediction and stop
                    accepted_tokens.append(target_tok)
                    num_accepted += 1  # We still add one token (the correction)
                    break
            
            self.stats['accepted_tokens'] += (num_accepted - 1) if num_accepted > 0 and len(accepted_tokens) > len([d for d, t in zip(draft_tokens, target_predictions) if d == t]) else num_accepted
            self.stats['rejected_tokens'] += (len(draft_tokens) - (num_accepted - 1)) if num_accepted > 0 and len(accepted_tokens) > len([d for d, t in zip(draft_tokens, target_predictions) if d == t]) else (len(draft_tokens) - num_accepted)
            
            # Update sequence
            if len(accepted_tokens) > 0:
                accepted_tensor = torch.tensor([accepted_tokens], dtype=torch.long, device=self.device)
                idx = torch.cat([idx, accepted_tensor], dim=1)
                generated_ids.extend(accepted_tokens)
            
            if verbose:
                matches = sum(1 for d, t in zip(draft_tokens, target_predictions) if d == t)
                print(f"Iteration {iteration}: Draft={draft_tokens} → "
                      f"Target={target_predictions} → "
                      f"Matches={matches}/{K}")
            
            # Stop if we've generated enough
            if len(generated_ids) >= max_new_tokens:
                break
        
        # Calculate final stats
        if self.stats['total_draft_tokens'] > 0:
            self.stats['acceptance_rate'] = (self.stats['accepted_tokens'] / 
                                             self.stats['total_draft_tokens'] * 100)
        else:
            self.stats['acceptance_rate'] = 0.0
        
        self.stats['tokens_per_iteration'] = (len(generated_ids) / iteration 
                                               if iteration > 0 else 0)
        
        return generated_ids[:max_new_tokens], self.stats


def main():
    """Demo of speculative decoding"""
    import tiktoken
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load models
    print("Loading target model (gpt2-medium)...")
    target = GPT.from_pretrained('gpt2-medium')
    target.to(device)
    
    print("Loading draft model (gpt2)...")
    draft = GPT.from_pretrained('gpt2')
    draft.to(device)
    
    # Initialize decoder
    decoder = SpeculativeDecoder(target, draft, device)
    
    # Prepare prompt
    enc = tiktoken.get_encoding("gpt2")
    prompt_text = "The future of artificial intelligence is"
    prompt_ids = enc.encode(prompt_text)
    print(f"\nPrompt: '{prompt_text}'")
    print(f"Generating with K=4...\n")
    
    # Generate
    t_start = time.time()
    generated_ids, stats = decoder.generate(prompt_ids, max_new_tokens=50, K=4, verbose=True)
    t_end = time.time()
    
    # Decode and print
    generated_text = enc.decode(generated_ids)
    print(f"\n{'='*60}")
    print(f"Generated: {generated_text}")
    print(f"{'='*60}\n")
    
    # Print statistics
    print("Statistics:")
    print(f"  Total time: {(t_end - t_start)*1000:.2f} ms")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Draft time: {stats['draft_time']*1000:.2f} ms")
    print(f"  Verify time: {stats['verify_time']*1000:.2f} ms")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.1f}%")
    print(f"  Tokens per iteration: {stats['tokens_per_iteration']:.2f}")
    print(f"  Accepted: {stats['accepted_tokens']}")
    print(f"  Rejected: {stats['rejected_tokens']}")


if __name__ == '__main__':
    main()
