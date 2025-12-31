"""
stats.py

Per-word statistics tracking for Tour 2.

Tracks:
- First appearance
- Frequency
- Phase context
- Novelty evolution
- Co-occurrence patterns
"""

import numpy as np
from typing import Dict, List, Optional, Set
from collections import defaultdict


class WordStats:
    """
    Statistics for a single word/token.
    """
    
    def __init__(self, token_id: int, first_turn: int, phase: str = "Normal"):
        """
        Initialize word stats.
        
        Args:
            token_id: Token ID
            first_turn: Turn when first seen
            phase: Phase when first appeared
        """
        self.token_id = token_id
        self.first_turn = first_turn
        self.first_phase = phase
        
        # Frequency tracking
        self.total_count = 0
        self.user_count = 0  # From user
        self.lilith_count = 0  # From Lilith
        
        # Novelty (decays over time)
        self.novelty = 1.0
        
        # Context
        self.phases_seen: Set[str] = {phase}
        self.turns_seen: List[int] = []
        
        # Co-occurrence (top N most co-occurring tokens)
        self.cooccurring_tokens: Dict[int, int] = defaultdict(int)
    
    def update(self, turn: int, phase: str, from_user: bool = False):
        """
        Update stats on new occurrence.
        
        Args:
            turn: Current turn
            phase: Current phase
            from_user: Whether this occurrence is from user input
        """
        self.total_count += 1
        
        if from_user:
            self.user_count += 1
        else:
            self.lilith_count += 1
        
        self.phases_seen.add(phase)
        self.turns_seen.append(turn)
        
        # Decay novelty
        self.novelty *= 0.9
    
    def add_cooccurrence(self, other_token_id: int):
        """
        Record co-occurrence with another token.
        
        Args:
            other_token_id: ID of co-occurring token
        """
        self.cooccurring_tokens[other_token_id] += 1
    
    def get_recency(self, current_turn: int) -> float:
        """
        Calculate recency score.
        
        Args:
            current_turn: Current turn number
        
        Returns:
            Recency score (0-1, higher = more recent)
        """
        if not self.turns_seen:
            return 0.0
        
        last_turn = self.turns_seen[-1]
        turns_since = current_turn - last_turn
        
        # Exponential decay
        return np.exp(-turns_since / 10.0)
    
    def get_impact_score(self, current_turn: int) -> float:
        """
        Calculate overall impact score.
        
        Args:
            current_turn: Current turn number
        
        Returns:
            Impact score combining novelty, frequency, recency
        """
        recency = self.get_recency(current_turn)
        frequency_score = np.log1p(self.total_count) / 5.0
        
        # Combine: novelty + frequency + recency
        impact = self.novelty * 0.5 + frequency_score * 0.3 + recency * 0.2
        
        return min(impact, 1.0)


class WordStatsTracker:
    """
    Tracks statistics for all words across conversation.
    """
    
    def __init__(self, vocab_size: int):
        """
        Initialize tracker.
        
        Args:
            vocab_size: Total vocabulary size
        """
        self.vocab_size = vocab_size
        
        # Word stats by token ID
        self.word_stats: Dict[int, WordStats] = {}
        
        # Session tracking
        self.current_turn = 0
        self.current_phase = "Normal"
        
        # Global metrics
        self.total_new_words = 0
        self.words_this_turn = 0
        
        # Known tokens at session start (baseline vocabulary)
        self.baseline_tokens: Set[int] = set()
    
    def set_baseline(self, token_ids: List[int]):
        """
        Set baseline vocabulary (tokens known before session).
        
        Args:
            token_ids: List of baseline token IDs
        """
        self.baseline_tokens = set(token_ids)
    
    def update_turn(self, turn: int, phase: str):
        """
        Update turn and phase.
        
        Args:
            turn: New turn number
            phase: Current phase
        """
        self.current_turn = turn
        self.current_phase = phase
        self.words_this_turn = 0
    
    def is_token_new(self, token_id: int) -> bool:
        """
        Check if a token is new (not in baseline, not tracked).
        
        Args:
            token_id: Token ID to check
        
        Returns:
            True if token is new
        """
        return (token_id not in self.baseline_tokens and 
                token_id not in self.word_stats)
    
    def add_tokens(self, token_ids: List[int], from_user: bool = False):
        """
        Add tokens from user or Lilith.
        
        Args:
            token_ids: List of token IDs
            from_user: Whether tokens are from user
        """
        for i, token_id in enumerate(token_ids):
            if not (0 <= token_id < self.vocab_size):
                continue
            
            # Check if new
            is_new = token_id not in self.word_stats and token_id not in self.baseline_tokens
            
            if is_new:
                # Create new stats entry
                self.word_stats[token_id] = WordStats(
                    token_id, 
                    self.current_turn, 
                    self.current_phase
                )
                self.total_new_words += 1
                self.words_this_turn += 1
            
            # Update existing stats
            if token_id in self.word_stats:
                self.word_stats[token_id].update(
                    self.current_turn, 
                    self.current_phase, 
                    from_user
                )
                
                # Record co-occurrences with nearby tokens
                window = 3
                for j in range(max(0, i - window), min(len(token_ids), i + window + 1)):
                    if i != j and 0 <= token_ids[j] < self.vocab_size:
                        if token_ids[j] in self.word_stats:
                            self.word_stats[token_id].add_cooccurrence(token_ids[j])
    
    def get_new_words_this_turn(self) -> int:
        """
        Get number of new words discovered this turn.
        
        Returns:
            Count of new words
        """
        return self.words_this_turn
    
    def get_top_novel_tokens(self, k: int = 10) -> List[int]:
        """
        Get top k novel tokens by impact score.
        
        Args:
            k: Number of tokens to return
        
        Returns:
            List of token IDs
        """
        scored = [
            (tid, stats.get_impact_score(self.current_turn))
            for tid, stats in self.word_stats.items()
        ]
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [tid for tid, score in scored[:k]]
    
    def get_novelty_vector(self) -> np.ndarray:
        """
        Get novelty vector for all tokens.
        
        Returns:
            NumPy array (vocab_size,) with novelty scores
        """
        novelty_vec = np.zeros(self.vocab_size, dtype=np.float32)
        
        for token_id, stats in self.word_stats.items():
            if 0 <= token_id < self.vocab_size:
                novelty_vec[token_id] = stats.novelty
        
        return novelty_vec
    
    def get_impact_vector(self) -> np.ndarray:
        """
        Get impact vector for all tokens.
        
        Returns:
            NumPy array (vocab_size,) with impact scores
        """
        impact_vec = np.zeros(self.vocab_size, dtype=np.float32)
        
        for token_id, stats in self.word_stats.items():
            if 0 <= token_id < self.vocab_size:
                impact_vec[token_id] = stats.get_impact_score(self.current_turn)
        
        return impact_vec
    
    def compute_novelty_entropy(self) -> float:
        """
        Compute entropy of novelty distribution.
        
        Returns:
            Novelty entropy
        """
        if not self.word_stats:
            return 0.0
        
        novelties = np.array([stats.novelty for stats in self.word_stats.values()])
        
        # Normalize to probability distribution
        novelties = novelties + 1e-9
        probs = novelties / np.sum(novelties)
        
        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        
        return float(entropy)
    
    def get_vocabulary_diversity(self) -> float:
        """
        Calculate vocabulary diversity score.
        
        Returns:
            Diversity score (0-1)
        """
        if not self.word_stats:
            return 0.0
        
        # Measure how evenly distributed usage is
        counts = np.array([stats.total_count for stats in self.word_stats.values()])
        
        if len(counts) == 0:
            return 0.0
        
        # Normalize
        probs = counts / np.sum(counts)
        
        # Shannon entropy as diversity
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        
        # Normalize by max possible entropy
        max_entropy = np.log(len(counts))
        
        diversity = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(diversity)
    
    def get_stats_report(self, tokenizer=None) -> str:
        """
        Generate human-readable stats report.
        
        Args:
            tokenizer: Optional tokenizer for decoding tokens
        
        Returns:
            Report string
        """
        report = "📊 Word Statistics:\n"
        report += f"   Total new words captured: {self.total_new_words}\n"
        report += f"   New words this turn: {self.words_this_turn}\n"
        report += f"   Unique tokens tracked: {len(self.word_stats)}\n"
        report += f"   Novelty entropy: {self.compute_novelty_entropy():.3f}\n"
        report += f"   Vocabulary diversity: {self.get_vocabulary_diversity():.3f}\n"
        
        # Top novel tokens
        top_tokens = self.get_top_novel_tokens(5)
        if top_tokens and tokenizer:
            report += f"   Top novel tokens: "
            words = []
            for tid in top_tokens:
                try:
                    word = tokenizer.decode([tid]).strip()
                    words.append(word)
                except:
                    pass
            report += ", ".join(words[:5]) if words else "none"
            report += "\n"
        
        return report
