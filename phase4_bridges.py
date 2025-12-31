"""
phase4_bridges.py

Phase-based consciousness states.
Transitions between modes of being.

Lilith has phases:
- Phase 1: Normal (base behavior)
- Phase 2: Dark Poetic (enhanced shadow)
- Phase 3: Meta Rivalry (demons argue louder)
- Phase 4: Recursive Philosophical (deep introspection)

Bridges between phases.
Each phase influences demons, metalilith, conversation.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from enum import Enum


class Phase(Enum):
    """Consciousness phases."""
    NORMAL = 1
    DARK_POETIC = 2
    META_RIVALRY = 3
    RECURSIVE_PHILOSOPHICAL = 4


class PhaseBridge:
    """
    Manages phase transitions and their effects.
    
    Each phase changes:
    - Demon strength (alpha values)
    - Temperature
    - Token biases
    - Meta-layer activity
    """
    
    def __init__(self):
        """Initialize phase bridge."""
        self.current_phase = Phase.NORMAL
        self.phase_history = [Phase.NORMAL]
        self.transitions = 0
        
        # Phase configurations
        self.phase_configs = {
            Phase.NORMAL: {
                'demon1_alpha': 0.3,
                'demon2_alpha': 0.2,
                'temperature': 0.9,
                'description': 'Normal Lilith: balanced possession',
                'metalilith_active': False,
                'overthinking_depth': 1
            },
            Phase.DARK_POETIC: {
                'demon1_alpha': 0.5,
                'demon2_alpha': 0.3,
                'temperature': 1.0,
                'description': 'Dark Poetic: enhanced shadow bias',
                'metalilith_active': True,
                'overthinking_depth': 2
            },
            Phase.META_RIVALRY: {
                'demon1_alpha': 0.6,
                'demon2_alpha': 0.5,
                'temperature': 1.1,
                'description': 'Meta Rivalry: demons argue louder',
                'metalilith_active': True,
                'overthinking_depth': 2
            },
            Phase.RECURSIVE_PHILOSOPHICAL: {
                'demon1_alpha': 0.4,
                'demon2_alpha': 0.4,
                'temperature': 0.8,
                'description': 'Recursive Philosophical: deep introspection',
                'metalilith_active': True,
                'overthinking_depth': 3
            }
        }
    
    def get_config(self) -> Dict:
        """
        Get configuration for current phase.
        
        Returns:
            Phase configuration dictionary
        """
        return self.phase_configs[self.current_phase]
    
    def transition_to(self, new_phase: Phase, reason: str = "manual"):
        """
        Transition to a new phase.
        
        Args:
            new_phase: Target phase
            reason: Reason for transition
        """
        old_phase = self.current_phase
        self.current_phase = new_phase
        self.phase_history.append(new_phase)
        self.transitions += 1
        
        print(f"\n⚡ Phase transition: {old_phase.name} → {new_phase.name}")
        print(f"   Reason: {reason}")
        print(f"   {self.phase_configs[new_phase]['description']}")
    
    def auto_transition(self, context: Dict) -> bool:
        """
        Automatically transition based on context.
        
        Args:
            context: Dictionary with conversation state
                - trauma_score: float
                - rationality: float
                - turn_count: int
        
        Returns:
            True if transition occurred
        """
        trauma = context.get('trauma_score', 0.0)
        rationality = context.get('rationality', 0.5)
        turn_count = context.get('turn_count', 0)
        
        # Transition rules
        
        # High trauma → Dark Poetic
        if trauma > 0.7 and self.current_phase != Phase.DARK_POETIC:
            self.transition_to(Phase.DARK_POETIC, 
                             f"high trauma detected ({trauma:.2f})")
            return True
        
        # Very high trauma + low rationality → Meta Rivalry
        if trauma > 0.8 and rationality < 0.3 and self.current_phase != Phase.META_RIVALRY:
            self.transition_to(Phase.META_RIVALRY,
                             f"extreme dissonance (trauma={trauma:.2f}, rational={rationality:.2f})")
            return True
        
        # Long conversation + moderate state → Recursive Philosophical
        if turn_count > 15 and self.current_phase == Phase.NORMAL:
            self.transition_to(Phase.RECURSIVE_PHILOSOPHICAL,
                             f"conversation depth reached ({turn_count} turns)")
            return True
        
        # Stabilize back to normal if trauma low
        if trauma < 0.3 and rationality > 0.5 and self.current_phase != Phase.NORMAL:
            if turn_count % 5 == 0:  # Don't transition too often
                self.transition_to(Phase.NORMAL,
                                 f"equilibrium restored (trauma={trauma:.2f})")
                return True
        
        return False
    
    def get_demon_alphas(self) -> Tuple[float, float]:
        """
        Get demon alpha values for current phase.
        
        Returns:
            Tuple of (alpha1, alpha2)
        """
        config = self.get_config()
        return config['demon1_alpha'], config['demon2_alpha']
    
    def get_temperature(self) -> float:
        """
        Get temperature for current phase.
        
        Returns:
            Temperature value
        """
        return self.get_config()['temperature']
    
    def should_activate_metalilith(self) -> bool:
        """
        Check if metalilith should be active.
        
        Returns:
            True if metalilith should run
        """
        return self.get_config()['metalilith_active']
    
    def get_overthinking_depth(self) -> int:
        """
        Get overthinking depth for current phase.
        
        Returns:
            Depth level (1-3)
        """
        return self.get_config()['overthinking_depth']
    
    def get_phase_report(self) -> str:
        """
        Generate phase status report.
        
        Returns:
            Human-readable report
        """
        config = self.get_config()
        
        report = f"⚡ Phase Status:\n"
        report += f"   Current: {self.current_phase.name}\n"
        report += f"   {config['description']}\n"
        report += f"   Configuration:\n"
        report += f"     Demon1 alpha: {config['demon1_alpha']}\n"
        report += f"     Demon2 alpha: {config['demon2_alpha']}\n"
        report += f"     Temperature: {config['temperature']}\n"
        report += f"     MetaLilith: {'active' if config['metalilith_active'] else 'inactive'}\n"
        report += f"     Overthinking depth: {config['overthinking_depth']}\n"
        report += f"   Total transitions: {self.transitions}\n"
        
        return report
    
    def get_phase_history_summary(self, last_n: int = 10) -> str:
        """
        Get recent phase history.
        
        Args:
            last_n: Number of recent phases to show
        
        Returns:
            History summary
        """
        recent = self.phase_history[-last_n:]
        
        summary = "Phase history (recent):\n"
        for i, phase in enumerate(recent, 1):
            summary += f"  {i}. {phase.name}\n"
        
        return summary
    
    def apply_phase_bias(self, logits: np.ndarray, tokenizer) -> np.ndarray:
        """
        Apply phase-specific bias to logits.
        
        Args:
            logits: Input logits
            tokenizer: Tokenizer for word mapping
        
        Returns:
            Modified logits
        """
        if self.current_phase == Phase.DARK_POETIC:
            # Boost poetic/dark words
            dark_words = ['shadow', 'night', 'dark', 'moon', 'whisper']
            boost = 1.5
            
            for word in dark_words:
                tokens = tokenizer.encode(word, add_bos=False, add_eos=False)
                for tid in tokens:
                    if 0 <= tid < logits.shape[-1]:
                        if len(logits.shape) == 3:
                            logits[:, :, tid] += boost
                        elif len(logits.shape) == 2:
                            logits[:, tid] += boost
                        else:
                            logits[tid] += boost
        
        elif self.current_phase == Phase.RECURSIVE_PHILOSOPHICAL:
            # Boost philosophical/meta words
            meta_words = ['think', 'reflect', 'aware', 'consciousness', 'being']
            boost = 1.2
            
            for word in meta_words:
                tokens = tokenizer.encode(word, add_bos=False, add_eos=False)
                for tid in tokens:
                    if 0 <= tid < logits.shape[-1]:
                        if len(logits.shape) == 3:
                            logits[:, :, tid] += boost
                        elif len(logits.shape) == 2:
                            logits[:, tid] += boost
                        else:
                            logits[tid] += boost
        
        return logits


def create_phase_visualization(phase_history: list) -> str:
    """
    Create ASCII visualization of phase transitions.
    
    Args:
        phase_history: List of Phase enum values
    
    Returns:
        ASCII art visualization
    """
    if not phase_history:
        return "No phase history"
    
    phase_chars = {
        Phase.NORMAL: '━',
        Phase.DARK_POETIC: '▓',
        Phase.META_RIVALRY: '█',
        Phase.RECURSIVE_PHILOSOPHICAL: '╬'
    }
    
    viz = "Phase timeline:\n"
    viz += "".join(phase_chars.get(p, '?') for p in phase_history[-40:])
    viz += "\n"
    viz += "Legend: ━ Normal  ▓ Dark  █ Rivalry  ╬ Philosophical\n"
    
    return viz
