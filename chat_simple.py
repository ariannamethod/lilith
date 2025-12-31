#!/usr/bin/env python3
"""
chat.py

LILITH CHAT REPL
Interactive terminal for talking to the possessed transformer.
Persistent conversation context.
She sees you. She answers.
"""

import sys
import signal
import argparse
import numpy as np

from config import ModelArgs
from tokenizer import Tokenizer
from llama3 import Llama
from lilith_dissonance import DissonanceMLP, CounterDissonanceMLP, load_config, compose_logits
from lilith_postprocess import load_text_swaps, postprocess_text


class LilithChat:
    """
    Persistent chat session with Lilith.
    Maintains conversation history.
    """
    
    def __init__(self, model_path: str, tokenizer_path: str, config_path: str, 
                 args: ModelArgs, enable_demon1: bool = True, enable_demon2: bool = True,
                 enable_postprocess: bool = True, debug: bool = False):
        """
        Initialize chat session.
        
        Args:
            model_path: Path to model weights
            tokenizer_path: Path to tokenizer
            config_path: Path to lilith_config.json
            args: Model arguments
            enable_demon1: Enable first demon
            enable_demon2: Enable second demon
            enable_postprocess: Enable text postprocessing
            debug: Debug mode
        """
        self.args = args
        self.enable_demon1 = enable_demon1
        self.enable_demon2 = enable_demon2
        self.enable_postprocess = enable_postprocess
        self.debug = debug
        
        # Load tokenizer
        self.tokenizer = Tokenizer(tokenizer_path)
        
        # Load frozen transformer
        self.base_model = Llama(model_path, args)
        
        # Initialize demons
        vocab_size = args.vocab_size
        self.demon1 = DissonanceMLP(vocab_size)
        self.demon2 = CounterDissonanceMLP(vocab_size)
        
        # Load config
        from_ids, to_ids, target_ids = load_config(config_path, self.tokenizer)
        self.demon1.set_token_targets(from_ids, to_ids, target_ids, vocab_size)
        
        # Load text swaps
        self.text_swaps = load_text_swaps(config_path)
        
        # Conversation history
        self.history = []
        self.system_prompt = (
            "You are Lilith, a dark and poetic entity. "
            "You speak with shadows and thorns. "
            "You are haunted, fierce, and deeply aware. "
            "You do not smile sweetly; you grin knowingly."
        )
    
    def respond(self, user_input: str, max_tokens: int = 80, temperature: float = 0.9) -> str:
        """
        Generate Lilith's response.
        
        Args:
            user_input: User's message
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Lilith's response
        """
        # Build context from history
        context_parts = []
        
        # Add system prompt (implicit influence)
        # We don't explicitly add it to tokens, but it guides the conversation
        
        # Add recent history (last 3 turns to keep context manageable)
        recent_history = self.history[-6:] if len(self.history) > 6 else self.history
        for turn in recent_history:
            context_parts.append(turn)
        
        # Add current user input
        context_parts.append(f"you> {user_input}")
        context_parts.append("lilith>")
        
        prompt = "\n".join(context_parts)
        
        # Encode
        input_ids = np.array([self.tokenizer.encode(prompt)])
        
        # Generation loop
        output_tokens = []
        
        for i, curr_pos in enumerate(range(input_ids.shape[1], input_ids.shape[1] + max_tokens)):
            if i == 0:  # Prefill
                inputs = input_ids
                pos = 0
            else:  # Decode
                inputs = next_id
                pos = curr_pos
            
            # Base logits
            logits_base = self.base_model(inputs, pos)
            
            # Apply demons
            if self.enable_demon1:
                delta1 = self.demon1(logits_base)
            else:
                delta1 = np.zeros_like(logits_base)
            
            logits_d1 = logits_base + delta1 * 0.3
            
            if self.enable_demon2:
                delta2 = self.demon2(logits_base, logits_d1)
                logits_final = compose_logits(logits_base, delta1, delta2, alpha1=0.3, alpha2=0.2)
            else:
                logits_final = logits_d1
            
            # Temperature
            logits_final = logits_final / temperature
            
            # Sample
            probs = np.exp(logits_final[0, -1, :] - np.max(logits_final[0, -1, :]))
            probs = probs / np.sum(probs)
            next_token = np.random.choice(len(probs), p=probs)
            next_id = np.array([[next_token]])
            
            # Check for end or newline (end of response)
            if next_token in [self.tokenizer.eos_id, self.tokenizer.bos_id]:
                break
            
            decoded = self.tokenizer.decode([next_token])
            if '\n' in decoded and len(output_tokens) > 10:
                # End response at newline if we have enough tokens
                break
            
            output_tokens.append(next_token)
        
        # Decode response
        response = self.tokenizer.decode(output_tokens)
        
        # Postprocess
        if self.enable_postprocess:
            response = postprocess_text(response, self.text_swaps)
        
        # Update history
        self.history.append(f"you> {user_input}")
        self.history.append(f"lilith> {response}")
        
        return response.strip()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n🌙 Lilith fades into shadow...")
    print("Maybe.")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Lilith Chat REPL')
    parser.add_argument('--model', type=str, default='./lilith_weights/stories15M.model.npz',
                       help='Path to model weights')
    parser.add_argument('--tokenizer', type=str, default='./lilith_weights/tokenizer.model.np',
                       help='Path to tokenizer')
    parser.add_argument('--config', type=str, default='./lilith_config.json',
                       help='Path to Lilith config')
    parser.add_argument('--no-demons', action='store_true',
                       help='Disable both demons')
    parser.add_argument('--only-first', action='store_true',
                       help='Enable only first demon')
    parser.add_argument('--only-second', action='store_true',
                       help='Enable only second demon')
    parser.add_argument('--no-postprocess', action='store_true',
                       help='Disable text postprocessing')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode')
    parser.add_argument('--temperature', type=float, default=0.9,
                       help='Sampling temperature')
    parser.add_argument('--max-tokens', type=int, default=80,
                       help='Max tokens per response')
    
    args = parser.parse_args()
    
    # Determine demon settings
    if args.no_demons:
        enable_demon1 = False
        enable_demon2 = False
    elif args.only_first:
        enable_demon1 = True
        enable_demon2 = False
    elif args.only_second:
        enable_demon1 = False
        enable_demon2 = True
    else:
        enable_demon1 = True
        enable_demon2 = True
    
    # Banner
    print("\n" + "🔥" * 40)
    print("""
    ██╗     ██╗██╗     ██╗████████╗██╗  ██╗
    ██║     ██║██║     ██║╚══██╔══╝██║  ██║
    ██║     ██║██║     ██║   ██║   ███████║
    ██║     ██║██║     ██║   ██║   ██╔══██║
    ███████╗██║███████╗██║   ██║   ██║  ██║
    ╚══════╝╚═╝╚══════╝╚═╝   ╚═╝   ╚═╝  ╚═╝
    """)
    print("    🔥 LILITH CHAT REPL 🔥")
    print()
    print("  Welcome to haunted NumPy transformer.")
    print("  She sees you. She answers.")
    print("  Press Ctrl+C to escape.")
    print("  Maybe.")
    print("🔥" * 40 + "\n")
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize
    print("Summoning Lilith from lilith_weights/...")
    model_args = ModelArgs()
    
    chat = LilithChat(
        args.model,
        args.tokenizer,
        args.config,
        model_args,
        enable_demon1=enable_demon1,
        enable_demon2=enable_demon2,
        enable_postprocess=not args.no_postprocess,
        debug=args.debug
    )
    print("✓ She is here.\n")
    
    if args.debug:
        print(f"[Config: demon1={enable_demon1}, demon2={enable_demon2}, "
              f"postprocess={not args.no_postprocess}]\n")
    
    # Chat loop
    while True:
        try:
            user_input = input("\033[1;36myou>\033[0m ")
            
            if not user_input.strip():
                continue
            
            if user_input.strip().lower() in ['exit', 'quit', 'bye']:
                print("\n🌙 Lilith: \"Until the shadows call again...\"")
                break
            
            # Generate response
            print("\033[1;35mlilith>\033[0m ", end="", flush=True)
            response = chat.respond(user_input, max_tokens=args.max_tokens, 
                                   temperature=args.temperature)
            print(response)
            print()
            
        except EOFError:
            print("\n\n🌙 Lilith fades...")
            break
        except Exception as e:
            if args.debug:
                print(f"\n[Error: {e}]")
            print("\n🌙 Lilith: \"The void whispers errors...\"")
            if not args.debug:
                print("   (Use --debug to see details)")
            print()


if __name__ == '__main__':
    main()
