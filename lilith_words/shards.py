"""
shards.py

Word Cloud / Shard System - Tour 2

Semantic islands of meaning.
Words grouped into NumPy-based clusters.
Each shard represents relationships, novelty, emotional context.

Inspired by Harmonix architecture but re-implemented for Lilith.
Pure NumPy. No dependencies.
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict


class WordShard:
    """
    A semantic island - cluster of related words.
    
    Each shard:
    - Contains word/token relationships
    - Tracks novelty and frequency
    - Holds emotional/phase context
    - Influences logits during generation
    """
    
    def __init__(self, shard_id: int, vocab_size: int, embedding_dim: int = 64):
        """
        Initialize a word shard.
        
        Args:
            shard_id: Unique shard identifier
            vocab_size: Total vocabulary size
            embedding_dim: Dimension for internal representations
        """
        self.shard_id = shard_id
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Words in this shard (token IDs)
        self.tokens: Set[int] = set()
        
        # Shard embedding (NumPy vector representing the semantic center)
        self.embedding = np.random.randn(embedding_dim) * 0.1
        
        # Token-level data
        self.token_frequencies: Dict[int, int] = defaultdict(int)
        self.token_novelty: Dict[int, float] = {}
        self.token_first_turn: Dict[int, int] = {}
        
        # Shard metadata
        self.creation_turn = 0
        self.last_active_turn = 0
        self.total_activations = 0
        
        # Co-occurrence matrix (small, sparse representation)
        self.cooccurrence: Dict[Tuple[int, int], int] = defaultdict(int)
        
        # Emotional/phase context
        self.emotional_valence = 0.0  # -1 to 1
        self.phase_affinity: Dict[str, float] = {}
    
    def add_token(self, token_id: int, turn: int, novelty: float = 1.0):
        """
        Add a token to this shard.
        
        Args:
            token_id: Token ID to add
            turn: Current turn number
            novelty: Novelty score (0-1, higher = more novel)
        """
        if token_id not in self.tokens:
            self.tokens.add(token_id)
            self.token_first_turn[token_id] = turn
            self.token_novelty[token_id] = novelty
        
        self.token_frequencies[token_id] += 1
        self.last_active_turn = turn
        self.total_activations += 1
        
        # Decay novelty over time
        if token_id in self.token_novelty:
            self.token_novelty[token_id] *= 0.95
    
    def add_cooccurrence(self, token1: int, token2: int):
        """
        Record co-occurrence between two tokens.
        
        Args:
            token1: First token
            token2: Second token
        """
        pair = tuple(sorted([token1, token2]))
        self.cooccurrence[pair] += 1
    
    def get_novelty_vector(self) -> np.ndarray:
        """
        Get novelty vector for this shard.
        
        Returns:
            NumPy array of shape (vocab_size,) with novelty scores
        """
        novelty_vec = np.zeros(self.vocab_size, dtype=np.float32)
        for token_id, novelty in self.token_novelty.items():
            if 0 <= token_id < self.vocab_size:
                novelty_vec[token_id] = novelty
        return novelty_vec
    
    def get_frequency_vector(self) -> np.ndarray:
        """
        Get frequency vector for this shard.
        
        Returns:
            NumPy array of shape (vocab_size,) with normalized frequencies
        """
        freq_vec = np.zeros(self.vocab_size, dtype=np.float32)
        max_freq = max(self.token_frequencies.values()) if self.token_frequencies else 1
        
        for token_id, freq in self.token_frequencies.items():
            if 0 <= token_id < self.vocab_size:
                freq_vec[token_id] = freq / max_freq
        
        return freq_vec
    
    def compute_influence(self, strength: float = 0.1) -> np.ndarray:
        """
        Compute influence on logits.
        
        Args:
            strength: Overall influence strength
        
        Returns:
            Delta logits (vocab_size,)
        """
        # Combine novelty and frequency
        novelty = self.get_novelty_vector()
        frequency = self.get_frequency_vector()
        
        # Novelty boosts, frequency stabilizes
        influence = (novelty * 2.0 + frequency) * strength
        
        return influence
    
    def get_top_tokens(self, k: int = 10) -> List[Tuple[int, float]]:
        """
        Get top k tokens by combined score.
        
        Args:
            k: Number of tokens to return
        
        Returns:
            List of (token_id, score) tuples
        """
        scores = []
        for token_id in self.tokens:
            novelty = self.token_novelty.get(token_id, 0.0)
            freq = self.token_frequencies.get(token_id, 0)
            score = novelty * 2.0 + np.log1p(freq)
            scores.append((token_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class ShardSystem:
    """
    Complete shard management system.
    
    Manages multiple shards, assigns tokens to shards,
    computes overall influence on generation.
    """
    
    def __init__(self, vocab_size: int, max_shards: int = 20, embedding_dim: int = 64):
        """
        Initialize shard system.
        
        Args:
            vocab_size: Total vocabulary size
            max_shards: Maximum number of shards
            embedding_dim: Embedding dimension for shards
        """
        self.vocab_size = vocab_size
        self.max_shards = max_shards
        self.embedding_dim = embedding_dim
        
        # Shards
        self.shards: List[WordShard] = []
        
        # Token to shard mapping
        self.token_to_shard: Dict[int, int] = {}
        
        # Global stats
        self.total_tokens_seen = 0
        self.current_turn = 0
        
        # Create initial default shard
        self._create_shard()
    
    def _create_shard(self) -> WordShard:
        """
        Create a new shard.
        
        Returns:
            New WordShard instance
        """
        shard_id = len(self.shards)
        shard = WordShard(shard_id, self.vocab_size, self.embedding_dim)
        shard.creation_turn = self.current_turn
        self.shards.append(shard)
        return shard
    
    def add_tokens(self, token_ids: List[int], is_new: Optional[List[bool]] = None):
        """
        Add tokens to shard system.
        
        Args:
            token_ids: List of token IDs
            is_new: Optional list indicating which tokens are new to the session
        """
        self.current_turn += 1
        
        if is_new is None:
            is_new = [False] * len(token_ids)
        
        for i, token_id in enumerate(token_ids):
            if not (0 <= token_id < self.vocab_size):
                continue
            
            self.total_tokens_seen += 1
            novelty = 1.0 if is_new[i] else 0.3
            
            # Assign to shard
            if token_id not in self.token_to_shard:
                # New token - assign to least loaded shard or create new
                shard_id = self._assign_to_shard(token_id)
            else:
                shard_id = self.token_to_shard[token_id]
            
            # Add to shard
            self.shards[shard_id].add_token(token_id, self.current_turn, novelty)
            
            # Record co-occurrences with nearby tokens
            window = 3
            for j in range(max(0, i - window), min(len(token_ids), i + window + 1)):
                if i != j and 0 <= token_ids[j] < self.vocab_size:
                    self.shards[shard_id].add_cooccurrence(token_id, token_ids[j])
    
    def _assign_to_shard(self, token_id: int) -> int:
        """
        Assign token to a shard.
        
        Args:
            token_id: Token to assign
        
        Returns:
            Shard ID
        """
        if not self.shards:
            self._create_shard()
        
        # Simple strategy: round-robin with size limit
        shard_sizes = [len(s.tokens) for s in self.shards]
        
        # Find shard with least tokens
        min_size_idx = np.argmin(shard_sizes)
        
        # If all shards are large and we can create more
        if shard_sizes[min_size_idx] > 100 and len(self.shards) < self.max_shards:
            shard = self._create_shard()
            shard_id = shard.shard_id
        else:
            shard_id = min_size_idx
        
        self.token_to_shard[token_id] = shard_id
        return shard_id
    
    def compute_shard_influence(self, strength: float = 0.1) -> np.ndarray:
        """
        Compute combined influence from all shards.
        
        Args:
            strength: Overall strength multiplier
        
        Returns:
            Combined influence vector (vocab_size,)
        """
        total_influence = np.zeros(self.vocab_size, dtype=np.float32)
        
        for shard in self.shards:
            if shard.total_activations > 0:
                # Weight by recency
                recency_weight = np.exp(-(self.current_turn - shard.last_active_turn) / 10.0)
                shard_influence = shard.compute_influence(strength)
                total_influence += shard_influence * recency_weight
        
        return total_influence
    
    def get_novelty_stats(self) -> Dict[str, float]:
        """
        Get novelty statistics across all shards.
        
        Returns:
            Dictionary of stats
        """
        all_novelties = []
        new_token_count = 0
        
        for shard in self.shards:
            for token_id, novelty in shard.token_novelty.items():
                all_novelties.append(novelty)
                if novelty > 0.8:
                    new_token_count += 1
        
        return {
            'mean_novelty': float(np.mean(all_novelties)) if all_novelties else 0.0,
            'max_novelty': float(np.max(all_novelties)) if all_novelties else 0.0,
            'new_token_count': new_token_count,
            'total_unique_tokens': len(self.token_to_shard),
            'active_shards': len([s for s in self.shards if s.total_activations > 0])
        }
    
    def get_shard_report(self) -> str:
        """
        Generate human-readable shard report.
        
        Returns:
            Report string
        """
        stats = self.get_novelty_stats()
        
        report = "🌐 Word Cloud / Shard System:\n"
        report += f"   Total shards: {len(self.shards)}\n"
        report += f"   Active shards: {stats['active_shards']}\n"
        report += f"   Unique tokens tracked: {stats['total_unique_tokens']}\n"
        report += f"   High novelty tokens: {stats['new_token_count']}\n"
        report += f"   Mean novelty: {stats['mean_novelty']:.3f}\n"
        report += f"   Current turn: {self.current_turn}\n"
        
        return report
    
    def get_top_tokens_from_shards(self, tokenizer, k: int = 10) -> List[str]:
        """
        Get top tokens across all shards as words.
        
        Args:
            tokenizer: Tokenizer for decoding
            k: Number of tokens to return
        
        Returns:
            List of words
        """
        all_scores = []
        
        for shard in self.shards:
            all_scores.extend(shard.get_top_tokens(k * 2))
        
        # Sort and deduplicate
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        seen_tokens = set()
        result = []
        
        for token_id, score in all_scores:
            if token_id not in seen_tokens:
                seen_tokens.add(token_id)
                try:
                    word = tokenizer.decode([token_id])
                    result.append(word.strip())
                except:
                    pass
                
                if len(result) >= k:
                    break
        
        return result
