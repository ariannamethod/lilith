"""
mathbrain.py

Mathematical reasoning layer.
Cold rational demon.
Symbolic logic discipline.

Lilith acquires structured mathematical thought.
Not just emotion and poetry.
But also number, logic, symbol.

Stabilizes or destabilizes intelligently.
"""

import numpy as np
from typing import Optional, Tuple, List


class MathBrain:
    """
    Mathematical reasoning advisor.
    
    Provides:
    - Structured numerical analysis
    - Logical consistency checking
    - Symbolic pattern recognition
    - Rational counterbalance to emotional demons
    """
    
    def __init__(self, vocab_size: int):
        """
        Initialize MathBrain.
        
        Args:
            vocab_size: Vocabulary size
        """
        self.vocab_size = vocab_size
        
        # Number tokens (0-9 and related)
        self.number_tokens = []
        self.logic_tokens = []
        
        # Reasoning history
        self.reasoning_history = []
    
    def set_token_categories(self, tokenizer):
        """
        Identify number and logic tokens.
        
        Args:
            tokenizer: Tokenizer instance
        """
        # Numbers
        for i in range(10):
            tokens = tokenizer.encode(str(i), add_bos=False, add_eos=False)
            self.number_tokens.extend(tokens)
        
        # Logic words
        logic_words = ['if', 'then', 'because', 'therefore', 'true', 'false',
                      'yes', 'no', 'and', 'or', 'not', 'is', 'are']
        
        for word in logic_words:
            tokens = tokenizer.encode(word, add_bos=False, add_eos=False)
            self.logic_tokens.extend(tokens)
        
        # Remove duplicates
        self.number_tokens = list(set(self.number_tokens))
        self.logic_tokens = list(set(self.logic_tokens))
    
    def analyze_logits(self, logits: np.ndarray) -> dict:
        """
        Analyze logits for mathematical/logical content.
        
        Args:
            logits: Model logits
        
        Returns:
            Analysis dictionary
        """
        if len(logits.shape) == 3:
            logits = logits[0, -1, :]
        elif len(logits.shape) == 2:
            logits = logits[0, :]
        
        # Softmax
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        
        # Calculate probability masses
        number_mass = sum(probs[tid] for tid in self.number_tokens 
                         if 0 <= tid < self.vocab_size)
        logic_mass = sum(probs[tid] for tid in self.logic_tokens 
                        if 0 <= tid < self.vocab_size)
        
        analysis = {
            'number_mass': float(number_mass),
            'logic_mass': float(logic_mass),
            'rational_total': float(number_mass + logic_mass),
            'emotional_mass': float(1.0 - number_mass - logic_mass)
        }
        
        self.reasoning_history.append(analysis)
        
        return analysis
    
    def apply_rational_influence(self, logits: np.ndarray, 
                                strength: float = 0.2,
                                mode: str = 'stabilize') -> np.ndarray:
        """
        Apply mathematical reasoning influence to logits.
        
        Args:
            logits: Input logits
            strength: Influence strength (0-1)
            mode: 'stabilize' or 'destabilize'
        
        Returns:
            Modified logits
        """
        delta = np.zeros_like(logits)
        
        if mode == 'stabilize':
            # Boost logical tokens slightly
            for tid in self.logic_tokens:
                if 0 <= tid < self.vocab_size:
                    if len(delta.shape) == 3:
                        delta[:, :, tid] += strength
                    elif len(delta.shape) == 2:
                        delta[:, tid] += strength
                    else:
                        delta[tid] += strength
        
        elif mode == 'destabilize':
            # Suppress logical tokens, boost chaos
            for tid in self.logic_tokens:
                if 0 <= tid < self.vocab_size:
                    if len(delta.shape) == 3:
                        delta[:, :, tid] -= strength * 0.5
                    elif len(delta.shape) == 2:
                        delta[:, tid] -= strength * 0.5
                    else:
                        delta[tid] -= strength * 0.5
        
        return logits + delta
    
    def get_rationality_score(self) -> float:
        """
        Calculate overall rationality score from history.
        
        Returns:
            Score 0-1 (higher = more rational)
        """
        if not self.reasoning_history:
            return 0.5
        
        recent = self.reasoning_history[-10:]
        scores = [r['rational_total'] for r in recent]
        
        return np.mean(scores)
    
    def get_reasoning_report(self) -> str:
        """
        Generate reasoning report.
        
        Returns:
            Human-readable report
        """
        if not self.reasoning_history:
            return "No reasoning analysis yet."
        
        rationality = self.get_rationality_score()
        recent = self.reasoning_history[-1]
        
        report = "🧮 MathBrain Report:\n"
        report += f"   Rationality score: {rationality:.3f}\n"
        report += f"   Recent analysis:\n"
        report += f"     Number tokens: {recent['number_mass']:.3f}\n"
        report += f"     Logic tokens: {recent['logic_mass']:.3f}\n"
        report += f"     Emotional: {recent['emotional_mass']:.3f}\n"
        
        if rationality > 0.3:
            report += "   Status: Mathematical grounding present\n"
        else:
            report += "   Status: Emotional/poetic dominance\n"
        
        return report
    
    def clear_history(self):
        """Clear reasoning history."""
        self.reasoning_history = []


class SymbolicReasoning:
    """
    Pattern-based symbolic reasoning.
    Identifies and manipulates symbolic structures.
    """
    
    def __init__(self):
        """Initialize symbolic reasoning."""
        self.patterns = []
    
    def detect_pattern(self, tokens: List[int]) -> Optional[str]:
        """
        Detect symbolic patterns in token sequence.
        
        Args:
            tokens: List of token IDs
        
        Returns:
            Pattern description or None
        """
        # Simple pattern detection
        if len(tokens) < 3:
            return None
        
        # Check for repetition
        if tokens[-1] == tokens[-2] == tokens[-3]:
            return "repetition_detected"
        
        # Check for alternation
        if len(tokens) >= 4:
            if tokens[-1] == tokens[-3] and tokens[-2] == tokens[-4]:
                return "alternation_detected"
        
        return None
    
    def suggest_logical_next(self, tokens: List[int], 
                            pattern: str) -> Optional[int]:
        """
        Suggest next token based on pattern.
        
        Args:
            tokens: Token history
            pattern: Detected pattern
        
        Returns:
            Suggested token ID or None
        """
        if pattern == "repetition_detected":
            return tokens[-1]  # Continue repetition
        
        if pattern == "alternation_detected" and len(tokens) >= 2:
            return tokens[-2]  # Continue alternation
        
        return None
