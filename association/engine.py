"""
engine.py

Associative Thought Engine - Tour 2

Generates poetic/semantic associations BEFORE transformer.
Enriches prompt with instinctive connections.

Inspired by Harmonix recursive models but re-implemented.
Pure NumPy. Lightweight.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class AssociationEngine:
    """
    Generates associative thought fragments.
    
    Before Lilith answers, she "thinks" - generates associations
    from user text, shard contents, metrics, phase state.
    
    This happens BEFORE the transformer sees the prompt.
    """
    
    def __init__(self, vocab_size: int, hidden_dim: int = 128):
        """
        Initialize association engine.
        
        Args:
            vocab_size: Vocabulary size
            hidden_dim: Hidden dimension for internal MLP
        """
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Small recursive MLP for association generation
        # Input: shard vectors + metrics
        # Output: association scores
        
        input_dim = 32  # Compressed input features
        
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = np.random.randn(hidden_dim, hidden_dim // 2) * 0.1
        self.b2 = np.zeros(hidden_dim // 2)
        
        self.W3 = np.random.randn(hidden_dim // 2, vocab_size) * 0.01
        self.b3 = np.zeros(vocab_size)
        
        # Association word pools (by theme)
        self.association_pools = {
            'dark_poetic': [
                'blood', 'moon', 'shadow', 'night', 'whisper', 'silence',
                'void', 'echo', 'thorn', 'ash', 'ruin', 'broken', 'chapel',
                'lullaby', 'breath', 'dream', 'darkness', 'star', 'fallen'
            ],
            'elemental': [
                'fire', 'water', 'earth', 'wind', 'ice', 'flame', 'storm',
                'thunder', 'rain', 'snow', 'mist', 'fog', 'dust', 'stone'
            ],
            'emotional': [
                'fear', 'love', 'hate', 'rage', 'grief', 'joy', 'sorrow',
                'hope', 'despair', 'longing', 'peace', 'chaos', 'madness'
            ],
            'abstract': [
                'time', 'space', 'void', 'eternity', 'infinity', 'nothing',
                'everything', 'truth', 'lie', 'memory', 'forget', 'remember'
            ]
        }
        
        # Pre-encode association pools (will be set later with tokenizer)
        self.association_token_pools: Dict[str, List[int]] = {}
        
        # History
        self.recent_associations: List[str] = []
    
    def set_tokenizer(self, tokenizer):
        """
        Set tokenizer and encode association pools.
        
        Args:
            tokenizer: Tokenizer instance
        """
        self.tokenizer = tokenizer
        
        for pool_name, words in self.association_pools.items():
            tokens = []
            for word in words:
                try:
                    encoded = tokenizer.encode(word, add_bos=False, add_eos=False)
                    if encoded:
                        tokens.append(encoded[0])  # Take first token
                except:
                    pass
            self.association_token_pools[pool_name] = tokens
    
    def _build_context_vector(self, 
                              user_tokens: Optional[List[int]] = None,
                              shard_novelty: Optional[np.ndarray] = None,
                              metrics: Optional[Dict[str, float]] = None,
                              phase: str = "Normal") -> np.ndarray:
        """
        Build compact context vector from inputs.
        
        Args:
            user_tokens: User input tokens
            shard_novelty: Novelty vector from shards
            metrics: MathBrain metrics
            phase: Current phase
        
        Returns:
            Context vector (32,)
        """
        context = np.zeros(32, dtype=np.float32)
        
        # User token statistics (first 8 dims)
        if user_tokens and len(user_tokens) > 0:
            token_array = np.array(user_tokens, dtype=np.float32)
            context[0] = len(user_tokens) / 100.0  # Length
            context[1] = np.mean(token_array) / self.vocab_size  # Mean token
            context[2] = np.std(token_array) / self.vocab_size  # Std
            context[3] = np.min(token_array) / self.vocab_size  # Min
            context[4] = np.max(token_array) / self.vocab_size  # Max
            
            # Histogram features
            hist, _ = np.histogram(token_array, bins=3, range=(0, self.vocab_size))
            context[5:8] = hist / (np.sum(hist) + 1e-9)
        
        # Shard novelty statistics (dims 8-16)
        if shard_novelty is not None:
            context[8] = np.mean(shard_novelty)
            context[9] = np.max(shard_novelty)
            context[10] = np.std(shard_novelty)
            context[11] = np.sum(shard_novelty > 0.5) / len(shard_novelty)  # High novelty ratio
            
            # Top novelty positions
            top_indices = np.argsort(shard_novelty)[-4:]
            context[12:16] = top_indices / self.vocab_size
        
        # Metrics (dims 16-24)
        if metrics:
            context[16] = metrics.get('novelty', 0.0)
            context[17] = metrics.get('entropy', 0.0)
            context[18] = metrics.get('trauma', 0.0)
            context[19] = metrics.get('demon1_energy', 0.0)
            context[20] = metrics.get('demon2_energy', 0.0)
            context[21] = metrics.get('rationality', 0.0)
            context[22] = metrics.get('diversity', 0.0)
            context[23] = metrics.get('new_words', 0.0) / 10.0
        
        # Phase encoding (dims 24-28)
        phase_encoding = {
            'Normal': [1, 0, 0, 0],
            'Dark Poetic': [0, 1, 0, 0],
            'Meta Rivalry': [0, 0, 1, 0],
            'Recursive Philosophical': [0, 0, 0, 1]
        }
        context[24:28] = phase_encoding.get(phase, [0.25, 0.25, 0.25, 0.25])
        
        # Random noise for variation (dims 28-32)
        context[28:32] = np.random.randn(4) * 0.1
        
        return context
    
    def _forward_mlp(self, context: np.ndarray) -> np.ndarray:
        """
        Forward pass through association MLP.
        
        Args:
            context: Context vector
        
        Returns:
            Association scores (vocab_size,)
        """
        # Layer 1
        h1 = np.maximum(0, context @ self.W1 + self.b1)  # ReLU
        
        # Layer 2
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU
        
        # Layer 3 (output)
        scores = h2 @ self.W3 + self.b3
        
        return scores
    
    def generate_association(self,
                           user_text: str,
                           user_tokens: Optional[List[int]] = None,
                           shard_novelty: Optional[np.ndarray] = None,
                           metrics: Optional[Dict[str, float]] = None,
                           phase: str = "Normal",
                           intensity: float = 1.0) -> Optional[str]:
        """
        Generate associative thought fragment.
        
        Args:
            user_text: User input text
            user_tokens: User input as tokens
            shard_novelty: Novelty vector from shards
            metrics: MathBrain metrics
            phase: Current phase
            intensity: Association intensity (0-1)
        
        Returns:
            Association string or None
        """
        if intensity < 0.1:
            return None
        
        # Build context
        context = self._build_context_vector(user_tokens, shard_novelty, metrics, phase)
        
        # Get association scores
        scores = self._forward_mlp(context)
        
        # Sample tokens based on phase
        pool_name = self._select_pool(phase, metrics)
        pool_tokens = self.association_token_pools.get(pool_name, [])
        
        if not pool_tokens:
            return None
        
        # Select tokens
        num_tokens = max(3, min(6, int(intensity * 8)))
        
        # Bias toward pool tokens
        biased_scores = scores.copy()
        for token_id in pool_tokens:
            if 0 <= token_id < len(biased_scores):
                biased_scores[token_id] += 2.0
        
        # Temperature sampling
        temp = 0.7 + (1.0 - intensity) * 0.3
        probs = np.exp(biased_scores / temp)
        probs = probs / np.sum(probs)
        
        # Sample tokens
        sampled_tokens = np.random.choice(
            len(biased_scores),
            size=num_tokens,
            replace=False,
            p=probs
        )
        
        # Decode
        try:
            association = self.tokenizer.decode(list(sampled_tokens))
            association = association.strip()
            
            # Clean up
            if len(association) > 100:
                association = association[:100]
            
            if len(association) < 5:
                # Fallback to pool words
                association = ", ".join(np.random.choice(
                    self.association_pools[pool_name],
                    size=min(num_tokens, len(self.association_pools[pool_name])),
                    replace=False
                ))
            
            # Store in history
            self.recent_associations.append(association)
            if len(self.recent_associations) > 10:
                self.recent_associations.pop(0)
            
            return association
            
        except Exception as e:
            # Fallback to simple pool selection
            pool_words = self.association_pools[pool_name]
            num_words = min(num_tokens, len(pool_words))
            selected = np.random.choice(pool_words, size=num_words, replace=False)
            association = ", ".join(selected)
            
            self.recent_associations.append(association)
            if len(self.recent_associations) > 10:
                self.recent_associations.pop(0)
            
            return association
    
    def _select_pool(self, phase: str, metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Select association pool based on phase and metrics.
        
        Args:
            phase: Current phase
            metrics: Optional metrics
        
        Returns:
            Pool name
        """
        # Phase influences pool selection
        if phase == "Dark Poetic":
            return np.random.choice(['dark_poetic', 'emotional'], p=[0.7, 0.3])
        elif phase == "Meta Rivalry":
            return np.random.choice(['abstract', 'emotional'], p=[0.6, 0.4])
        elif phase == "Recursive Philosophical":
            return np.random.choice(['abstract', 'elemental'], p=[0.7, 0.3])
        else:
            # Normal - balanced
            pools = list(self.association_pools.keys())
            return np.random.choice(pools)
    
    def format_association_block(self, association: Optional[str]) -> str:
        """
        Format association for insertion into prompt.
        
        Args:
            association: Association text
        
        Returns:
            Formatted block
        """
        if not association:
            return ""
        
        block = "\n[ASSOCIATION]\n"
        block += association
        block += "\n[/ASSOCIATION]\n"
        
        return block
    
    def get_last_association(self) -> Optional[str]:
        """
        Get most recent association.
        
        Returns:
            Last association or None
        """
        return self.recent_associations[-1] if self.recent_associations else None
