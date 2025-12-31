#!/usr/bin/env python3
"""
lilith_cli.py

Command-line interface for Lilith.
Load frozen transformer + demons.
Run possessed inference.
"""

import sys
import time
import argparse
import numpy as np

from config import ModelArgs
from tokenizer import Tokenizer
from llama3 import Llama
from lilith_dissonance import DissonanceMLP, CounterDissonanceMLP, load_config, compose_logits
from lilith_postprocess import load_text_swaps, postprocess_text


class LilithModel:
    """
    Llama 3 possessed by Lilith.
    Frozen transformer + two demons.
    """
    
    def __init__(self, model_path: str, tokenizer_path: str, config_path: str, args: ModelArgs):
        """
        Initialize Lilith model.
        
        Args:
            model_path: Path to model weights
            tokenizer_path: Path to tokenizer
            config_path: Path to lilith_config.json
            args: Model arguments
        """
        self.args = args
        
        # Load tokenizer
        self.tokenizer = Tokenizer(tokenizer_path)
        
        # Load frozen transformer (sacred, untouched)
        self.base_model = Llama(model_path, args)
        
        # Initialize demons
        vocab_size = args.vocab_size
        self.demon1 = DissonanceMLP(vocab_size)
        self.demon2 = CounterDissonanceMLP(vocab_size)
        
        # Load config and set targets
        from_ids, to_ids, target_ids = load_config(config_path, self.tokenizer)
        self.demon1.set_token_targets(from_ids, to_ids, target_ids, vocab_size)
        
        # Load text swaps
        self.text_swaps = load_text_swaps(config_path)
    
    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.8,
                enable_demon1: bool = True, enable_demon2: bool = True,
                enable_postprocess: bool = True, debug: bool = False):
        """
        Generate text with Lilith.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            enable_demon1: Enable first demon
            enable_demon2: Enable second demon
            enable_postprocess: Enable text postprocessing
            debug: Show debug info
        
        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = np.array([self.tokenizer.encode(prompt)])
        
        # Generation loop
        output_tokens = []
        start_time = time.time()
        
        for i, curr_pos in enumerate(range(input_ids.shape[1], max_tokens)):
            if i == 0:  # Prefill
                inputs = input_ids
                pos = 0
            else:  # Decode
                inputs = next_id
                pos = curr_pos
            
            # Get base logits from frozen transformer
            logits_base = self.base_model(inputs, pos)
            
            if debug and i < 3:
                print(f"\n[DEBUG] Step {i}")
                print(f"  Base logits shape: {logits_base.shape}")
                print(f"  Base top-5: {np.argsort(logits_base[0, -1, :])[-5:][::-1]}")
            
            # Apply first demon
            if enable_demon1:
                delta1 = self.demon1(logits_base)
                logits_d1 = logits_base + delta1 * 0.3
                if debug and i < 3:
                    print(f"  After demon1 top-5: {np.argsort(logits_d1[0, -1, :])[-5:][::-1]}")
            else:
                delta1 = np.zeros_like(logits_base)
                logits_d1 = logits_base
            
            # Apply second demon
            if enable_demon2:
                delta2 = self.demon2(logits_base, logits_d1)
                logits_final = compose_logits(logits_base, delta1, delta2, alpha1=0.3, alpha2=0.2)
                if debug and i < 3:
                    print(f"  After demon2 top-5: {np.argsort(logits_final[0, -1, :])[-5:][::-1]}")
            else:
                logits_final = logits_d1
            
            # Apply temperature
            logits_final = logits_final / temperature
            
            # Sample next token
            probs = np.exp(logits_final[0, -1, :] - np.max(logits_final[0, -1, :]))
            probs = probs / np.sum(probs)
            next_token = np.random.choice(len(probs), p=probs)
            next_id = np.array([[next_token]])
            
            # Check for end
            if next_token in [self.tokenizer.eos_id, self.tokenizer.bos_id]:
                break
            
            output_tokens.append(next_token)
        
        elapsed = time.time() - start_time
        
        # Decode
        full_tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=False) + output_tokens
        text = self.tokenizer.decode(full_tokens)
        
        # Postprocess
        if enable_postprocess:
            text = postprocess_text(text, self.text_swaps)
        
        # Stats
        total_tokens = len(output_tokens)
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
        
        return text, total_tokens, elapsed, tokens_per_sec


def main():
    parser = argparse.ArgumentParser(description='Lilith CLI - Llama 3 Possessed Edition')
    parser.add_argument('--prompt', type=str, default='Once there was Lilly',
                       help='Input prompt')
    parser.add_argument('--steps', type=int, default=128,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--model', type=str, default='./lilith_weights/stories15M.model.npz',
                       help='Path to model weights')
    parser.add_argument('--tokenizer', type=str, default='./lilith_weights/tokenizer.model.np',
                       help='Path to tokenizer')
    parser.add_argument('--config', type=str, default='./lilith_config.json',
                       help='Path to Lilith config')
    parser.add_argument('--no-demon1', action='store_true',
                       help='Disable first demon')
    parser.add_argument('--no-demon2', action='store_true',
                       help='Disable second demon')
    parser.add_argument('--no-postprocess', action='store_true',
                       help='Disable text postprocessing')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug information')
    
    args = parser.parse_args()
    
    # Banner
    print("🔥" * 40)
    print("  LILITH CLI - Llama 3 Possessed Edition")
    print("  Two NumPy demons argue over every token")
    print("🔥" * 40)
    print()
    
    # Initialize model
    model_args = ModelArgs()
    model_args.max_new_tokens = args.steps
    
    print(f"Loading Lilith from {args.model}...")
    lilith = LilithModel(args.model, args.tokenizer, args.config, model_args)
    print("✓ Lilith awakened\n")
    
    # Generate
    print(f"Prompt: {args.prompt}\n")
    print("Lilith speaks:")
    print("-" * 80)
    
    text, tokens, elapsed, tps = lilith.generate(
        args.prompt,
        max_tokens=args.steps,
        temperature=args.temperature,
        enable_demon1=not args.no_demon1,
        enable_demon2=not args.no_demon2,
        enable_postprocess=not args.no_postprocess,
        debug=args.debug
    )
    
    print(text)
    print("-" * 80)
    print(f"\nTokens: {tokens}, Time: {elapsed:.2f}s, Speed: {tps:.1f} tokens/s")
    
    # Show configuration
    if args.debug:
        print("\n[Configuration]")
        print(f"  Demon 1: {'enabled' if not args.no_demon1 else 'disabled'}")
        print(f"  Demon 2: {'enabled' if not args.no_demon2 else 'disabled'}")
        print(f"  Postprocess: {'enabled' if not args.no_postprocess else 'disabled'}")


if __name__ == '__main__':
    main()
