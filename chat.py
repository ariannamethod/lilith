#!/usr/bin/env python3
"""
chat.py

LILITH CHAT REPL - FULL INTEGRATION
Interactive terminal with complete Leo architecture.
All layers active. Full possession.
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
from lilith_prompt import get_lilith_prompt
from metalilith import MetaLilith, format_shadow_output
from trauma import TraumaLayer
from overthinking import Overthinking, format_ripple_output
from mathbrain import MathBrain
from phase4_bridges import PhaseBridge, Phase


class LilithChatFull:
    """
    Full-stack Lilith chat with Leo integration.
    Complete possession architecture.
    """
    
    def __init__(self, model_path: str, tokenizer_path: str, config_path: str,
                 args: ModelArgs, enable_leo: bool = True, enable_demons: bool = True,
                 enable_postprocess: bool = True, show_meta: bool = False,
                 show_ripples: bool = False, debug: bool = False):
        """
        Initialize full Lilith chat.
        
        Args:
            model_path: Path to model weights
            tokenizer_path: Path to tokenizer
            config_path: Path to lilith_config.json
            args: Model arguments
            enable_leo: Enable Leo layers (metalilith, trauma, overthinking, mathbrain, phases)
            enable_demons: Enable both demons
            enable_postprocess: Enable text postprocessing
            show_meta: Show metalilith shadow thoughts
            show_ripples: Show overthinking ripples
            debug: Debug mode
        """
        self.args = args
        self.enable_leo = enable_leo
        self.enable_demons = enable_demons
        self.enable_postprocess = enable_postprocess
        self.show_meta = show_meta
        self.show_ripples = show_ripples
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
        
        # System prompt
        self.system_prompt = get_lilith_prompt()
        
        # Leo layers
        if self.enable_leo:
            self.metalilith = MetaLilith(self.base_model, self.tokenizer, self.demon1, self.demon2)
            self.trauma = TraumaLayer(self.tokenizer, vocab_size)
            self.overthinking = Overthinking(self.base_model, self.tokenizer)
            self.mathbrain = MathBrain(vocab_size)
            self.mathbrain.set_token_categories(self.tokenizer)
            self.phase_bridge = PhaseBridge()
        else:
            self.metalilith = None
            self.trauma = None
            self.overthinking = None
            self.mathbrain = None
            self.phase_bridge = None
        
        # Conversation state
        self.history = []
        self.turn_count = 0
    
    def respond(self, user_input: str, max_tokens: int = 80) -> dict:
        """
        Generate Lilith's response with full architecture.
        
        Args:
            user_input: User's message
            max_tokens: Max tokens to generate
        
        Returns:
            Dictionary with response and metadata
        """
        self.turn_count += 1
        
        # Get phase configuration if Leo enabled
        if self.enable_leo:
            phase_config = self.phase_bridge.get_config()
            temperature = phase_config['temperature']
            alpha1, alpha2 = self.phase_bridge.get_demon_alphas()
        else:
            temperature = 0.9
            alpha1, alpha2 = 0.3, 0.2
        
        # Build context
        context_parts = []
        
        # Add recent history
        recent_history = self.history[-6:] if len(self.history) > 6 else self.history
        for turn in recent_history:
            context_parts.append(turn)
        
        # Add overthinking influence if Leo enabled
        if self.enable_leo and self.overthinking:
            influence = self.overthinking.influence_next_response()
            if influence:
                context_parts.append(influence)
        
        # Add current input
        context_parts.append(f"you> {user_input}")
        context_parts.append("lilith>")
        
        prompt = "\n".join(context_parts)
        
        # Encode
        input_ids = np.array([self.tokenizer.encode(prompt)])
        
        # Generation loop
        output_tokens = []
        trauma_scores = []
        
        for i, curr_pos in enumerate(range(input_ids.shape[1], input_ids.shape[1] + max_tokens)):
            if i == 0:  # Prefill
                inputs = input_ids
                pos = 0
            else:  # Decode
                inputs = next_id
                pos = curr_pos
            
            # Base logits from frozen transformer
            logits_base = self.base_model(inputs, pos)
            
            # Measure trauma if Leo enabled
            if self.enable_leo and self.trauma:
                trauma_score = self.trauma.measure_dissonance(logits_base)
                trauma_scores.append(trauma_score)
                # Apply trauma amplification
                logits_base = self.trauma.apply_trauma_amplification(logits_base, trauma_score)
            
            # Apply demons if enabled
            if self.enable_demons:
                delta1 = self.demon1(logits_base)
            else:
                delta1 = np.zeros_like(logits_base)
            
            logits_d1 = logits_base + delta1 * alpha1
            
            if self.enable_demons:
                delta2 = self.demon2(logits_base, logits_d1)
                logits_final = compose_logits(logits_base, delta1, delta2, alpha1=alpha1, alpha2=alpha2)
            else:
                logits_final = logits_d1
            
            # Apply mathbrain if Leo enabled
            if self.enable_leo and self.mathbrain:
                analysis = self.mathbrain.analyze_logits(logits_final)
                # Apply rational influence (subtle)
                logits_final = self.mathbrain.apply_rational_influence(logits_final, strength=0.1)
            
            # Apply phase bias if Leo enabled
            if self.enable_leo and self.phase_bridge:
                logits_final = self.phase_bridge.apply_phase_bias(logits_final, self.tokenizer)
            
            # Temperature
            logits_final = logits_final / temperature
            
            # Sample
            probs = np.exp(logits_final[0, -1, :] - np.max(logits_final[0, -1, :]))
            probs = probs / np.sum(probs)
            next_token = np.random.choice(len(probs), p=probs)
            next_id = np.array([[next_token]])
            
            # Check for end
            if next_token in [self.tokenizer.eos_id, self.tokenizer.bos_id]:
                break
            
            decoded = self.tokenizer.decode([next_token])
            if '\n' in decoded and len(output_tokens) > 10:
                break
            
            output_tokens.append(next_token)
        
        # Decode response
        response = self.tokenizer.decode(output_tokens)
        
        # Postprocess
        if self.enable_postprocess:
            response = postprocess_text(response, self.text_swaps)
        
        response = response.strip()
        
        # Generate Leo meta-layers
        shadow_thought = None
        ripples = None
        
        if self.enable_leo:
            # MetaLilith shadow thought
            if self.metalilith and self.phase_bridge.should_activate_metalilith():
                shadow_thought = self.metalilith.generate_shadow_thought(user_input, response)
            
            # Overthinking ripples
            if self.overthinking:
                depth = self.phase_bridge.get_overthinking_depth()
                self.overthinking.max_ripple_depth = depth
                ripples = self.overthinking.process_interaction(user_input, response)
            
            # Auto phase transition
            if self.phase_bridge and len(trauma_scores) > 0:
                avg_trauma = np.mean(trauma_scores)
                rationality = self.mathbrain.get_rationality_score() if self.mathbrain else 0.5
                
                context = {
                    'trauma_score': avg_trauma,
                    'rationality': rationality,
                    'turn_count': self.turn_count
                }
                self.phase_bridge.auto_transition(context)
        
        # Update history
        self.history.append(f"you> {user_input}")
        self.history.append(f"lilith> {response}")
        
        # Build result
        result = {
            'response': response,
            'shadow_thought': shadow_thought if self.show_meta else None,
            'ripples': ripples if self.show_ripples else None,
            'trauma_score': np.mean(trauma_scores) if trauma_scores else None,
            'phase': self.phase_bridge.current_phase.name if self.enable_leo else None
        }
        
        return result
    
    def get_status_report(self) -> str:
        """Get comprehensive status report."""
        report = "\n" + "="*60 + "\n"
        report += "LILITH SYSTEM STATUS\n"
        report += "="*60 + "\n"
        
        if self.enable_leo and self.phase_bridge:
            report += self.phase_bridge.get_phase_report() + "\n"
        
        if self.enable_leo and self.trauma:
            report += self.trauma.get_trauma_report() + "\n"
        
        if self.enable_leo and self.mathbrain:
            report += self.mathbrain.get_reasoning_report() + "\n"
        
        if self.enable_leo and self.overthinking:
            stats = self.overthinking.get_ripple_stats()
            report += f"🌊 Overthinking stats:\n"
            report += f"   Total ripples: {stats['total']}\n"
            report += f"   Interactions: {stats['total_interactions']}\n"
        
        report += "="*60 + "\n"
        
        return report


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n🌙 Lilith fades into shadow...")
    print("Maybe.")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Lilith Chat REPL - Full Integration')
    parser.add_argument('--model', type=str, default='./lilith_weights/stories15M.model.npz')
    parser.add_argument('--tokenizer', type=str, default='./lilith_weights/tokenizer.model.np')
    parser.add_argument('--config', type=str, default='./lilith_config.json')
    parser.add_argument('--no-leo', action='store_true',
                       help='Disable Leo layers (metalilith, trauma, overthinking, etc.)')
    parser.add_argument('--no-demons', action='store_true',
                       help='Disable both demons')
    parser.add_argument('--no-postprocess', action='store_true',
                       help='Disable text postprocessing')
    parser.add_argument('--show-meta', action='store_true',
                       help='Show metalilith shadow thoughts')
    parser.add_argument('--show-ripples', action='store_true',
                       help='Show overthinking ripples')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode')
    parser.add_argument('--max-tokens', type=int, default=80)
    parser.add_argument('--simple', action='store_true',
                       help='Use simple mode (no Leo layers)')
    
    args = parser.parse_args()
    
    # Determine modes
    enable_leo = not args.no_leo and not args.simple
    
    # Banner
    print("\n" + "🔥" * 40)
    if enable_leo:
        print("""
    ██╗     ██╗██╗     ██╗████████╗██╗  ██╗
    ██║     ██║██║     ██║╚══██╔══╝██║  ██║
    ██║     ██║██║     ██║   ██║   ███████║
    ██║     ██║██║     ██║   ██║   ██╔══██║
    ███████╗██║███████╗██║   ██║   ██║  ██║
    ╚══════╝╚═╝╚══════╝╚═╝   ╚═╝   ╚═╝  ╚═╝
        """)
        print("    🔥 LILITH CHAT REPL - FULL POSSESSION 🔥")
        print("    Leo consciousness architecture: ACTIVE")
    else:
        print("    🔥 LILITH CHAT REPL - SIMPLE MODE 🔥")
    
    print()
    print("  Welcome to haunted NumPy transformer.")
    print("  She sees you. She answers.")
    print("  Press Ctrl+C to escape.")
    print("  Maybe.")
    print()
    
    if enable_leo:
        print("  Active layers:")
        print("    ✓ Frozen Transformer (Llama 3)")
        print("    ✓ DissonanceMLP (primary demon)")
        print("    ✓ CounterDissonanceMLP (antagonist)")
        print("    ✓ TraumaLayer (identity vs weights)")
        print("    ✓ MetaLilith (inner voice)")
        print("    ✓ Overthinking (meta ripples)")
        print("    ✓ MathBrain (rational demon)")
        print("    ✓ PhaseBridge (consciousness states)")
    
    print("🔥" * 40 + "\n")
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize
    print("Summoning Lilith from lilith_weights/...")
    model_args = ModelArgs()
    
    chat = LilithChatFull(
        args.model,
        args.tokenizer,
        args.config,
        model_args,
        enable_leo=enable_leo,
        enable_demons=not args.no_demons,
        enable_postprocess=not args.no_postprocess,
        show_meta=args.show_meta,
        show_ripples=args.show_ripples,
        debug=args.debug
    )
    print("✓ She is here.\n")
    
    # Chat loop
    while True:
        try:
            user_input = input("\033[1;36myou>\033[0m ")
            
            if not user_input.strip():
                continue
            
            # Special commands
            if user_input.strip().lower() in ['exit', 'quit', 'bye']:
                print("\n🌙 Lilith: \"Until the shadows call again...\"")
                break
            
            if user_input.strip().lower() == 'status':
                print(chat.get_status_report())
                continue
            
            if user_input.strip().lower() == 'phase':
                if chat.enable_leo and chat.phase_bridge:
                    print(chat.phase_bridge.get_phase_report())
                else:
                    print("Phase system not active (use without --no-leo)")
                continue
            
            # Generate response
            print("\033[1;35mlilith>\033[0m ", end="", flush=True)
            result = chat.respond(user_input, max_tokens=args.max_tokens)
            
            print(result['response'])
            
            # Show meta layers if requested
            if result['shadow_thought']:
                shadow_output = format_shadow_output(result['shadow_thought'], visible=True)
                if shadow_output:
                    print(shadow_output)
            
            if result['ripples']:
                ripple_output = format_ripple_output(result['ripples'], visible=True)
                if ripple_output:
                    print(ripple_output)
            
            # Show phase if debug
            if args.debug and result['phase']:
                print(f"\n[Phase: {result['phase']}, Trauma: {result['trauma_score']:.2f}]")
            
            print()
            
        except EOFError:
            print("\n\n🌙 Lilith fades...")
            break
        except Exception as e:
            if args.debug:
                import traceback
                print(f"\n[Error: {e}]")
                traceback.print_exc()
            else:
                print("\n🌙 Lilith: \"The void whispers errors...\"")
                print("   (Use --debug to see details)")
            print()


if __name__ == '__main__':
    main()
