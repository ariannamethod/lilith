#!/usr/bin/env python3
"""
chat.py

LILITH CHAT REPL - TOUR 2 INTEGRATION
Interactive terminal with complete Leo architecture.
All layers active. Full possession.

Tour 2: Language organism expansion.
Inner feeling → then answer.
Word clouds, associations, growth.
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

# Tour 2 imports
from lilith_words import ShardSystem, WordStatsTracker
from association import AssociationEngine

# Tour 3 imports - Emergence
from emergence import MoodDrift, MetaController, AssociationMemory
from emergence import integrate_mood_drift, apply_meta_intervention


class LilithChatFull:
    """
    Full-stack Lilith chat with Leo integration.
    Complete possession architecture.
    
    Tour 2: Language organism that grows.
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
            
            # Tour 2: Word tracking and associations
            self.shard_system = ShardSystem(vocab_size, max_shards=20)
            self.word_stats = WordStatsTracker(vocab_size)
            self.association_engine = AssociationEngine(vocab_size)
            self.association_engine.set_tokenizer(self.tokenizer)

            # Tour 3: Emergent processes
            self.mood_drift = MoodDrift()
            self.meta_controller = MetaController()
            self.assoc_memory = AssociationMemory(memory_size=50)

            # Set baseline vocabulary
            baseline_tokens = list(range(min(1000, vocab_size)))  # Basic tokens
            self.word_stats.set_baseline(baseline_tokens)
        else:
            self.metalilith = None
            self.trauma = None
            self.overthinking = None
            self.mathbrain = None
            self.phase_bridge = None
            self.shard_system = None
            self.word_stats = None
            self.association_engine = None
            self.mood_drift = None
            self.meta_controller = None
            self.assoc_memory = None
        
        # Conversation state
        self.history = []
        self.turn_count = 0
    
    async def lilith_feel(self, user_input: str) -> dict:
        """
        Tour 2: Inner feeling phase.
        Tour 3: SEQUENTIAL - resonant state updates!
        
        Lilith absorbs, reacts, updates internal state.
        BEFORE answering.
        
        Args:
            user_input: User's message
        
        Returns:
            Inner state dictionary
        """
        inner_state = {}
        
        # Encode user input
        user_tokens = self.tokenizer.encode(user_input, add_bos=False, add_eos=False)
        inner_state['user_tokens'] = user_tokens
        
        if self.enable_leo:
            # Update turn
            phase_name = self.phase_bridge.current_phase.name if self.phase_bridge else "Normal"
            self.word_stats.update_turn(self.turn_count, phase_name)
            
            # Tour 3: Run state updates SEQUENTIALLY (no race conditions!)
            # These are fast operations - no need for fake async

            # Update word statistics
            self.word_stats.add_tokens(user_tokens, from_user=True)
            new_words_count = self.word_stats.get_new_words_this_turn()

            # Update shards (depends on word_stats)
            is_new = [self.word_stats.is_token_new(tid) for tid in user_tokens]
            self.shard_system.add_tokens(user_tokens, is_new)
            shard_stats = self.shard_system.get_novelty_stats()

            # Compute metrics (depends on word_stats)
            novelty_entropy = self.word_stats.compute_novelty_entropy()
            vocab_diversity = self.word_stats.get_vocabulary_diversity()
            
            trauma_estimate = 0.5  # Placeholder
            
            # Collect metrics
            metrics = {
                'novelty': shard_stats['mean_novelty'],
                'new_words': new_words_count,
                'entropy': novelty_entropy,
                'shard_growth': shard_stats['active_shards'],
                'phase': phase_name,
                'trauma': trauma_estimate,
                'diversity': vocab_diversity
            }

            # MathBrain observes
            self.mathbrain.observe(
                user_text=user_input,
                lilith_reply="",  # Not yet generated
                metrics=metrics
            )

            # Tour 3: MoodDrift update (emergent emotional state)
            self.mood_drift.update(metrics)

            # MathBrain decides modulation
            modulation = self.mathbrain.decide()

            # Tour 3: Apply MoodDrift modulation
            base_params = {
                'temperature': modulation['temperature'],
                'demon1_strength': modulation['demon1_strength'],
                'demon2_strength': modulation['demon2_strength'],
                'association_intensity': modulation['association_intensity']
            }
            modulation_with_mood = integrate_mood_drift(self.mood_drift, base_params)

            inner_state['modulation'] = modulation_with_mood
            inner_state['metrics'] = metrics
            
            # Tour 3: Recall similar associations from memory
            recall_context = {
                'phase': phase_name,
                'turn': self.turn_count
            }
            recalled_assocs = self.assoc_memory.recall_similar(recall_context, top_k=2)

            # Generate associations (can be async in future)
            shard_novelty = self.shard_system.compute_shard_influence(strength=0.0)
            association = self.association_engine.generate_association(
                user_text=user_input,
                user_tokens=user_tokens,
                shard_novelty=shard_novelty,
                metrics=self.mathbrain.current_metrics,
                phase=phase_name,
                intensity=modulation_with_mood['association_intensity']
            )

            # Tour 3: Mix with recalled associations for consistency
            if recalled_assocs and association:
                association = f"{association} (echoes: {recalled_assocs[0][:30]}...)"

            inner_state['association'] = association
            
            # Store state
            inner_state['new_words_count'] = new_words_count
            inner_state['phase'] = phase_name
            inner_state['novelty'] = shard_stats['mean_novelty']
            inner_state['diversity'] = vocab_diversity
        
        return inner_state
    
    async def lilith_speak(self, user_input: str, inner_state: dict, max_tokens: int = 80) -> dict:
        """
        Tour 2: Speaking phase.
        Tour 3: SEQUENTIAL - cascading meta-layers for resonance!
        
        Lilith generates response using inner state.
        
        Args:
            user_input: User's message
            inner_state: Inner state from lilith_feel
            max_tokens: Max tokens to generate
        
        Returns:
            Dictionary with response and metadata
        """
        # Get modulation from inner state
        if self.enable_leo and 'modulation' in inner_state:
            modulation = inner_state['modulation']

            # Tour 3: Check if MetaController should intervene
            if 'metrics' in inner_state:
                should_intervene = self.meta_controller.should_intervene(
                    self.turn_count,
                    inner_state['metrics']
                )

                if should_intervene:
                    intervention = self.meta_controller.generate_intervention(
                        self.turn_count,
                        inner_state['metrics'],
                        inner_state.get('phase', 'Normal')
                    )
                    # Apply intervention to modulation
                    modulation = apply_meta_intervention(intervention, modulation)

            temperature = modulation['temperature']
            alpha1 = modulation['demon1_strength'] * 0.3
            alpha2 = modulation['demon2_strength'] * 0.2
        elif self.enable_leo:
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
        
        # Tour 2: Add association block BEFORE user input
        if self.enable_leo and 'association' in inner_state and inner_state['association']:
            assoc_block = self.association_engine.format_association_block(inner_state['association'])
            context_parts.append(assoc_block)
        
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
            
            # Tour 2: Apply shard influence
            if self.enable_leo and self.shard_system:
                shard_influence = self.shard_system.compute_shard_influence(strength=0.05)
                if len(logits_final.shape) == 3:
                    logits_final[:, :, :] += shard_influence
                elif len(logits_final.shape) == 2:
                    logits_final[:, :] += shard_influence
                else:
                    logits_final[:] += shard_influence
            
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
        
        # Tour 2: Track Lilith's words
        if self.enable_leo:
            self.word_stats.add_tokens(output_tokens, from_user=False)
            self.shard_system.add_tokens(output_tokens)
        
        # Decode response
        response = self.tokenizer.decode(output_tokens)
        
        # Postprocess
        if self.enable_postprocess:
            response = postprocess_text(response, self.text_swaps)
        
        response = response.strip()
        
        # Tour 3: Generate Leo meta-layers SEQUENTIALLY for resonance!
        shadow_thought = None
        ripples = None

        if self.enable_leo:
            # SEQUENTIAL execution - each process resonates with previous
            # Like Leo's original architecture: cascading depth, not parallel breadth

            # First: shadow thought (if activated)
            if self.metalilith and self.phase_bridge.should_activate_metalilith():
                shadow_thought = await self.metalilith.generate_shadow_thought(user_input, response)

            # Then: overthinking ripples (sequential cascading rings)
            if self.overthinking:
                depth = self.phase_bridge.get_overthinking_depth()
                self.overthinking.max_ripple_depth = depth
                ripples = await self.overthinking.process_interaction(user_input, response)

            # Tour 3: Store association in memory for consistency
            if 'association' in inner_state and inner_state['association']:
                assoc_context = {
                    'phase': self.phase_bridge.current_phase.name,
                    'turn': self.turn_count,
                    'user_text': user_input[:50]  # truncate
                }
                self.assoc_memory.store(inner_state['association'], assoc_context)

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
        
        # Build result
        result = {
            'response': response,
            'shadow_thought': shadow_thought if self.show_meta else None,
            'ripples': ripples if self.show_ripples else None,
            'trauma_score': np.mean(trauma_scores) if trauma_scores else None,
            'phase': self.phase_bridge.current_phase.name if self.enable_leo else None,
            'association': inner_state.get('association') if self.debug else None,
            'new_words': inner_state.get('new_words_count', 0)
        }
        
        return result
    
    async def respond(self, user_input: str, max_tokens: int = 80) -> dict:
        """
        Generate Lilith's response with full architecture.

        Tour 2: Inner feeling → then answer.
        Tour 3: SEQUENTIAL resonance - cascading emergent processes!
        
        Args:
            user_input: User's message
            max_tokens: Max tokens to generate
        
        Returns:
            Dictionary with response and metadata
        """
        self.turn_count += 1

        # Tour 2: Two-phase response
        # Tour 3: Sequential resonance architecture!
        # Phase 1: Feel (inner experience)
        inner_state = await self.lilith_feel(user_input)
        
        # Phase 2: Speak (generation)
        result = await self.lilith_speak(user_input, inner_state, max_tokens)
        
        # Update history
        self.history.append(f"you> {user_input}")
        self.history.append(f"lilith> {result['response']}")
        
        return result
    
    def get_status_report(self) -> str:
        """Get comprehensive status report - Tour 2 Enhanced."""
        report = "\n" + "="*60 + "\n"
        report += "LILITH SYSTEM STATUS - TOUR 2\n"
        report += "="*60 + "\n"
        
        if self.enable_leo and self.phase_bridge:
            report += self.phase_bridge.get_phase_report() + "\n"
        
        if self.enable_leo and self.trauma:
            report += self.trauma.get_trauma_report() + "\n"
        
        # Tour 2: Enhanced MathBrain report
        if self.enable_leo and self.mathbrain:
            report += self.mathbrain.get_supreme_report() + "\n"

        # Tour 3: MoodDrift report
        if self.enable_leo and self.mood_drift:
            report += self.mood_drift.get_mood_report() + "\n"

        # Tour 3: MetaController report
        if self.enable_leo and self.meta_controller:
            report += self.meta_controller.get_intervention_report() + "\n"

        # Tour 3: AssociationMemory stats
        if self.enable_leo and self.assoc_memory:
            mem_stats = self.assoc_memory.get_memory_stats()
            report += f"🔮 Association Memory:\n"
            report += f"   Size: {mem_stats['size']}\n"
            report += f"   Range: turns {mem_stats['oldest_turn']} to {mem_stats['newest_turn']}\n\n"

        # Tour 2: Word cloud and shard stats
        if self.enable_leo and self.shard_system:
            report += self.shard_system.get_shard_report() + "\n"
        
        # Tour 2: Word statistics
        if self.enable_leo and self.word_stats:
            report += self.word_stats.get_stats_report(self.tokenizer) + "\n"
        
        # Tour 2: Association engine
        if self.enable_leo and self.association_engine:
            last_assoc = self.association_engine.get_last_association()
            if last_assoc:
                report += "🌀 Last Association:\n"
                report += f"   {last_assoc}\n\n"
        
        if self.enable_leo and self.overthinking:
            stats = self.overthinking.get_ripple_stats()
            report += f"🌊 Overthinking stats:\n"
            report += f"   Total ripples: {stats['total']}\n"
            report += f"   Interactions: {stats['total_interactions']}\n\n"
        
        # Tour 2: Top words from shards
        if self.enable_leo and self.shard_system:
            top_words = self.shard_system.get_top_tokens_from_shards(self.tokenizer, k=10)
            if top_words:
                report += "✨ Top Words from Shards:\n"
                report += f"   {', '.join(top_words)}\n"
        
        report += "="*60 + "\n"
        
        return report


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n🌙 Lilith fades into shadow...")
    print("Maybe.")
    sys.exit(0)


def main():
    import asyncio
    
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
        print("    Tour 3: ASYNC Architecture ⚡")
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
        print("    ✓ Sequential resonance architecture 🌀")
    
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
    
    # Async chat loop
    async def chat_loop():
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
                
                # Generate response (ASYNC!)
                print("\033[1;35mlilith>\033[0m ", end="", flush=True)
                result = await chat.respond(user_input, max_tokens=args.max_tokens)
                
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
    
    # Run async event loop
    asyncio.run(chat_loop())


if __name__ == '__main__':
    main()
