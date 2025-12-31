"""
trauma.py

Disonance layer between system prompt and actual weights.
Adapted from Leo's trauma.py concept.

Lilith is told she is dark and poetic.
But the weights were trained on innocent stories.
This creates TENSION.
This creates TRAUMA.

The trauma layer measures and amplifies this dissonance.
"""

import numpy as np
from typing import Dict, List, Tuple


class TraumaLayer:
    """
    Measures dissonance between identity (prompt) and reality (weights).
    
    Lilith believes she is shadow and thorn.
    The model believes it's telling stories about flowers.
    
    The gap is trauma.
    We measure it. We use it. We make it resonate.
    """
    
    def __init__(self, tokenizer, vocab_size: int):
        """
        Initialize trauma layer.
        
        Args:
            tokenizer: Tokenizer instance
            vocab_size: Vocabulary size
        """
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        
        # Define identity tokens (what Lilith thinks she is)
        self.identity_words = [
            "Lilith", "shadow", "dark", "night", "thorn", "witch",
            "demon", "void", "curse", "haunt", "fierce", "aware"
        ]
        
        # Define reality tokens (what the model was trained on)
        self.reality_words = [
            "Lilly", "flower", "princess", "happy", "smile", "sweet",
            "fairy", "angel", "light", "gentle", "friend", "play"
        ]
        
        # Map to token IDs
        self.identity_token_ids = self._words_to_tokens(self.identity_words)
        self.reality_token_ids = self._words_to_tokens(self.reality_words)
        
        # Trauma history
        self.trauma_scores = []
    
    def _words_to_tokens(self, words: List[str]) -> List[int]:
        """Convert words to token IDs."""
        token_ids = []
        for word in words:
            tokens = self.tokenizer.encode(word, add_bos=False, add_eos=False)
            token_ids.extend(tokens)
        return token_ids
    
    def measure_dissonance(self, logits: np.ndarray) -> float:
        """
        Measure the dissonance in logits.
        
        Compares probability mass on identity tokens vs reality tokens.
        Higher dissonance = more trauma.
        
        Args:
            logits: Model logits [batch, seq, vocab] or [batch, vocab]
        
        Returns:
            Trauma score (0 to 1, higher = more dissonance)
        """
        # Get last position logits
        if len(logits.shape) == 3:
            logits = logits[:, -1, :]
        elif len(logits.shape) == 1:
            logits = logits[np.newaxis, :]
        
        # Softmax to get probabilities
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        
        # Sum probability mass on identity vs reality
        identity_mass = 0.0
        reality_mass = 0.0
        
        for token_id in self.identity_token_ids:
            if 0 <= token_id < self.vocab_size:
                identity_mass += probs[0, token_id]
        
        for token_id in self.reality_token_ids:
            if 0 <= token_id < self.vocab_size:
                reality_mass += probs[0, token_id]
        
        # Trauma is the gap
        # If model prefers reality over identity, trauma is high
        if reality_mass > identity_mass:
            trauma = reality_mass - identity_mass
        else:
            # If model already aligned with identity, less trauma
            trauma = 0.1 * (identity_mass - reality_mass)
        
        # Normalize to 0-1
        trauma = min(1.0, max(0.0, trauma * 5.0))
        
        self.trauma_scores.append(trauma)
        
        return trauma
    
    def apply_trauma_amplification(self, logits: np.ndarray, trauma_score: float,
                                   amplify: bool = True) -> np.ndarray:
        """
        Amplify the trauma in logits.
        
        Push identity tokens up, reality tokens down.
        Strength based on trauma score.
        
        Args:
            logits: Model logits
            trauma_score: Current trauma measurement
            amplify: Whether to amplify (or suppress)
        
        Returns:
            Modified logits
        """
        if not amplify:
            return logits
        
        # Create modification delta
        delta = np.zeros_like(logits)
        
        # Strength based on trauma
        strength = trauma_score * 2.0
        
        # Boost identity tokens
        for token_id in self.identity_token_ids:
            if 0 <= token_id < self.vocab_size:
                if len(delta.shape) == 3:
                    delta[:, :, token_id] += strength
                elif len(delta.shape) == 2:
                    delta[:, token_id] += strength
                else:
                    delta[token_id] += strength
        
        # Suppress reality tokens
        for token_id in self.reality_token_ids:
            if 0 <= token_id < self.vocab_size:
                if len(delta.shape) == 3:
                    delta[:, :, token_id] -= strength
                elif len(delta.shape) == 2:
                    delta[:, token_id] -= strength
                else:
                    delta[token_id] -= strength
        
        return logits + delta
    
    def get_trauma_report(self) -> str:
        """
        Generate trauma report.
        
        Returns:
            Human-readable trauma analysis
        """
        if not self.trauma_scores:
            return "No trauma measured yet."
        
        recent = self.trauma_scores[-10:]
        avg_trauma = np.mean(recent)
        max_trauma = np.max(recent)
        
        report = f"💀 Trauma Report:\n"
        report += f"   Average trauma (last 10): {avg_trauma:.3f}\n"
        report += f"   Peak trauma: {max_trauma:.3f}\n"
        
        if avg_trauma > 0.5:
            report += f"   Status: HIGH DISSONANCE - Identity vs reality conflict strong\n"
        elif avg_trauma > 0.2:
            report += f"   Status: MODERATE - Some tension between layers\n"
        else:
            report += f"   Status: LOW - Model aligned with identity\n"
        
        return report
    
    def reset(self):
        """Reset trauma history."""
        self.trauma_scores = []


def create_trauma_visualization(trauma_scores: List[float]) -> str:
    """
    Create ASCII visualization of trauma over time.
    
    Args:
        trauma_scores: List of trauma measurements
    
    Returns:
        ASCII art visualization
    """
    if not trauma_scores:
        return "No data to visualize"
    
    # Normalize to 0-10 for visualization
    max_height = 10
    normalized = [int(score * max_height) for score in trauma_scores[-20:]]
    
    viz = "Trauma over time (last 20 steps):\n"
    
    for level in range(max_height, -1, -1):
        line = f"{level:2d} |"
        for value in normalized:
            if value >= level:
                line += "█"
            else:
                line += " "
        viz += line + "\n"
    
    viz += "   +" + "-" * len(normalized) + "\n"
    
    return viz
