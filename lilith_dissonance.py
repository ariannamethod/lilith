"""
lilith_dissonance.py

Two MLP demons that possess the logits layer.
Pure NumPy. No PyTorch. No JAX.
Frozen transformer stays sacred.
Chaos injected above.

Tour 3: Enhanced with error handling and validation.
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional


class DissonanceMLP:
    """
    Primary Demon: Semantic field shifter.
    
    Pushes logits away from innocence (Lilly)
    towards Lilith and dark anchors.
    
    Architecture:
    - 2-layer MLP
    - ReLU activation
    - Pure NumPy
    - Masked output (only affects target tokens)
    """
    
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None):
        """
        Initialize the primary dissonance demon.
        
        Args:
            input_dim: Size of input (vocab_size for logits)
            hidden_dim: Hidden layer size (default: 2x input_dim)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim else 2 * input_dim
        
        # Initialize weights with small random values
        # He initialization for ReLU
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros(self.hidden_dim)
        
        self.W2 = np.random.randn(self.hidden_dim, self.input_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros(self.input_dim)
        
        # Token masks (to be set by load_config)
        self.from_token_ids: List[int] = []
        self.to_token_ids: List[int] = []
        self.target_token_ids: List[int] = []
        self.mask: Optional[np.ndarray] = None
        
    def set_token_targets(self, from_ids: List[int], to_ids: List[int], 
                         target_ids: List[int], vocab_size: int):
        """
        Set which tokens this demon is allowed to modify.
        
        Args:
            from_ids: Token IDs to shift away from
            to_ids: Token IDs to shift towards
            target_ids: Additional target token IDs
            vocab_size: Total vocabulary size
        """
        self.from_token_ids = from_ids
        self.to_token_ids = to_ids
        self.target_token_ids = target_ids
        
        # Create mask: only target tokens can be modified
        all_targets = set(to_ids + target_ids)
        self.mask = np.zeros(vocab_size, dtype=np.float32)
        for token_id in all_targets:
            if 0 <= token_id < vocab_size:
                self.mask[token_id] = 1.0
    
    def __call__(self, logits: np.ndarray) -> np.ndarray:
        """
        Forward pass: generate delta logits.
        
        Args:
            logits: Input logits, shape [batch, seq_len, vocab_size] or [batch, vocab_size]
        
        Returns:
            Delta logits (same shape as input), masked to only affect targets
        """
        original_shape = logits.shape
        
        # Flatten to 2D if needed
        if len(logits.shape) == 3:
            batch, seq, vocab = logits.shape
            x = logits.reshape(-1, vocab)
        else:
            x = logits
        
        # Two-layer MLP with ReLU
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU activation
        delta = h @ self.W2 + self.b2
        
        # Apply mask: only modify allowed tokens
        if self.mask is not None:
            delta = delta * self.mask[np.newaxis, :]
        
        # Restore original shape
        if len(original_shape) == 3:
            delta = delta.reshape(original_shape)
        
        return delta


class Scalar:
    """
    Micrograd-inspired scalar with autograd capability.
    For CounterDissonanceMLP backprop.
    """
    
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self):
        out = Scalar(max(0, self.data), (self,), 'ReLU')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out
    
    def backward(self):
        """Topological sort + backprop"""
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class CounterDissonanceMLP:
    """
    Second Demon: Antagonistic counterbalance.
    
    Observes both base logits and first demon's distortion.
    Opposes, amplifies, or sabotages based on internal logic.
    
    Micrograd-inspired: has .data + .grad capability.
    Allows backprop (even if not trained).
    """
    
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None):
        """
        Initialize the counter-dissonance demon.
        
        Args:
            input_dim: Size of input (vocab_size for logits)
            hidden_dim: Hidden layer size (default: 2x input_dim)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim else 2 * input_dim
        
        # Initialize weights
        self.W1 = np.random.randn(self.input_dim * 2, self.hidden_dim) * np.sqrt(2.0 / (self.input_dim * 2))
        self.b1 = np.zeros(self.hidden_dim)
        
        self.W2 = np.random.randn(self.hidden_dim, self.input_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros(self.input_dim)
        
        # Scalar wrappers for autograd (optional, for demonstration)
        self.scalars = []
    
    def __call__(self, logits_base: np.ndarray, logits_d1: np.ndarray) -> np.ndarray:
        """
        Forward pass: generate counter-delta.
        
        Args:
            logits_base: Original base logits
            logits_d1: Logits after first demon (base + delta1)
        
        Returns:
            Counter-delta logits
        """
        original_shape = logits_base.shape
        
        # Flatten to 2D if needed
        if len(logits_base.shape) == 3:
            batch, seq, vocab = logits_base.shape
            base_flat = logits_base.reshape(-1, vocab)
            d1_flat = logits_d1.reshape(-1, vocab)
        else:
            base_flat = logits_base
            d1_flat = logits_d1
        
        # Concatenate both views (antagonism requires awareness of both)
        x = np.concatenate([base_flat, d1_flat], axis=1)
        
        # Two-layer MLP
        h = np.maximum(0, x @ self.W1 + self.b1)
        delta2 = h @ self.W2 + self.b2
        
        # Restore original shape
        if len(original_shape) == 3:
            delta2 = delta2.reshape(original_shape)
        
        return delta2
    
    def dummy_backward_step(self, learning_rate: float = 0.001):
        """
        Dummy gradient step (for demonstration of autograd capability).
        In practice, we don't train these demons.
        """
        # Just to show the architecture permits it
        self.W1 -= learning_rate * np.random.randn(*self.W1.shape) * 0.01
        self.W2 -= learning_rate * np.random.randn(*self.W2.shape) * 0.01


def load_config(config_path: str, tokenizer) -> Tuple[List[int], List[int], List[int]]:
    """
    Load demon configuration.
    
    Args:
        config_path: Path to lilith_config.json
        tokenizer: Tokenizer instance
    
    Returns:
        Tuple of (from_token_ids, to_token_ids, target_token_ids)
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    # Tour 3: Check file existence
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Lilith needs configuration to know her demons."
        )
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in config file {config_path}: {e}\n"
            f"Demon configuration is corrupted."
        )
    
    from_token_ids = []
    to_token_ids = []
    target_token_ids = []
    
    # Tour 3: Defensive key access with validation
    pairs = config.get('pairs', [])
    if not isinstance(pairs, list):
        raise ValueError(f"'pairs' must be a list in {config_path}")
    
    # Map pairs
    for pair in pairs:
        if not isinstance(pair, dict):
            continue  # Skip invalid entries
        
        from_word = pair.get('from', '')
        to_word = pair.get('to', '')
        
        if from_word and to_word:
            try:
                from_tokens = tokenizer.encode(from_word, add_bos=False, add_eos=False)
                to_tokens = tokenizer.encode(to_word, add_bos=False, add_eos=False)
                from_token_ids.extend(from_tokens)
                to_token_ids.extend(to_tokens)
            except Exception as e:
                print(f"Warning: Failed to encode pair {from_word}->{to_word}: {e}")
                continue
    
    # Map word swaps
    word_swaps = config.get('word_swaps', [])
    if not isinstance(word_swaps, list):
        print(f"Warning: 'word_swaps' should be a list, got {type(word_swaps)}")
        word_swaps = []
    
    for swap in word_swaps:
        if not isinstance(swap, dict):
            continue
        
        from_word = swap.get('from', '')
        to_word = swap.get('to', '')
        
        if from_word and to_word:
            try:
                from_tokens = tokenizer.encode(from_word, add_bos=False, add_eos=False)
                to_tokens = tokenizer.encode(to_word, add_bos=False, add_eos=False)
                from_token_ids.extend(from_tokens)
                to_token_ids.extend(to_tokens)
            except Exception as e:
                print(f"Warning: Failed to encode swap {from_word}->{to_word}: {e}")
                continue
    
    # Map extra targets
    extra_targets = config.get('extra_targets', [])
    if not isinstance(extra_targets, list):
        print(f"Warning: 'extra_targets' should be a list, got {type(extra_targets)}")
        extra_targets = []
    
    for target_word in extra_targets:
        if not isinstance(target_word, str):
            continue
        try:
            target_tokens = tokenizer.encode(target_word, add_bos=False, add_eos=False)
            target_token_ids.extend(target_tokens)
        except Exception as e:
            print(f"Warning: Failed to encode target word '{target_word}': {e}")
            continue
    
    return from_token_ids, to_token_ids, target_token_ids


def compose_logits(logits_base: np.ndarray, 
                   delta1: np.ndarray, 
                   delta2: np.ndarray,
                   alpha1: float = 0.3,
                   alpha2: float = 0.2) -> np.ndarray:
    """
    Recursive antagonistic composition.
    
    logits_final = logits_base + alpha1 * delta1 + alpha2 * delta2
    
    Args:
        logits_base: Original transformer logits
        delta1: DissonanceMLP output
        delta2: CounterDissonanceMLP output
        alpha1: Scaling factor for first demon
        alpha2: Scaling factor for second demon
    
    Returns:
        Final composed logits
    """
    return logits_base + alpha1 * delta1 + alpha2 * delta2
