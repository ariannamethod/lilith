"""
emergence.py

Tour 3: More sophisticated emergent processes.

Implements:
1. Mood Drift System - long-term emotional state that evolves
2. Meta-Controller Authority - occasional override system
3. Association Memory - consistency in poetic voice

These mechanisms add depth to Lilith's behavior without requiring
retraining the base model.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class MoodDrift:
    """
    Long-term emotional/behavioral state that drifts over time.
    
    Tour 3 Enhancement: Tracks cumulative emotional metrics and
    influences generation parameters slowly.
    
    Unlike phases (which switch discretely), mood drifts continuously.
    """
    
    def __init__(self):
        """Initialize mood drift system."""
        # Core mood dimensions (range: -1.0 to 1.0)
        self.dimensions = {
            'darkness': 0.0,      # Light/happy → Dark/brooding
            'chaos': 0.0,         # Ordered → Chaotic
            'intensity': 0.0,     # Calm → Intense
            'rationality': 0.0,   # Emotional → Rational
            'novelty': 0.0        # Conservative → Exploratory
        }
        
        # Drift rates (how fast mood changes)
        self.drift_rates = {
            'darkness': 0.02,
            'chaos': 0.03,
            'intensity': 0.025,
            'rationality': 0.015,
            'novelty': 0.02
        }
        
        # Equilibrium points (mood tends toward these)
        self.equilibrium = {
            'darkness': 0.3,      # Lilith's natural state is somewhat dark
            'chaos': 0.2,
            'intensity': 0.25,
            'rationality': 0.0,   # Balanced
            'novelty': 0.1
        }
        
        # History
        self.history: List[Dict[str, float]] = []
        self.turn_count = 0
    
    def update(self, metrics: Dict[str, Any]):
        """
        Update mood based on current interaction metrics.
        
        Args:
            metrics: Dictionary with trauma, rationality, novelty, etc.
        """
        self.turn_count += 1
        
        # Extract relevant metrics (with defaults)
        trauma = metrics.get('trauma', 0.5)
        rationality_score = metrics.get('rationality', 0.5)
        novelty = metrics.get('novelty', 0.5)
        new_words = metrics.get('new_words', 0)
        
        # Push mood dimensions based on metrics
        # Trauma increases darkness and intensity
        if trauma > 0.6:
            self._nudge('darkness', 0.05)
            self._nudge('intensity', 0.03)
        elif trauma < 0.3:
            self._nudge('darkness', -0.02)
        
        # Rationality score affects rationality dimension
        target_rational = (rationality_score - 0.5) * 2  # Map to -1,1
        self._nudge('rationality', (target_rational - self.dimensions['rationality']) * 0.1)
        
        # Novelty affects exploration mood
        if novelty > 0.6 or new_words > 3:
            self._nudge('novelty', 0.04)
        else:
            self._nudge('novelty', -0.02)
        
        # Apply drift toward equilibrium
        self._drift_toward_equilibrium()
        
        # Clamp dimensions
        for key in self.dimensions:
            self.dimensions[key] = np.clip(self.dimensions[key], -1.0, 1.0)
        
        # Store snapshot
        if self.turn_count % 5 == 0:  # Save every 5 turns
            self.history.append(self.dimensions.copy())
            # Keep last 20 snapshots
            if len(self.history) > 20:
                self.history.pop(0)
    
    def _nudge(self, dimension: str, amount: float):
        """Nudge a mood dimension by amount."""
        if dimension in self.dimensions:
            self.dimensions[dimension] += amount
    
    def _drift_toward_equilibrium(self):
        """Apply gentle drift toward equilibrium points."""
        for dim, eq_value in self.equilibrium.items():
            current = self.dimensions[dim]
            rate = self.drift_rates[dim]
            
            # Move toward equilibrium
            delta = (eq_value - current) * rate
            self.dimensions[dim] += delta
    
    def get_modulation_multipliers(self) -> Dict[str, float]:
        """
        Get modulation multipliers based on current mood.
        
        Returns:
            Dictionary with multipliers for various parameters
        """
        d = self.dimensions
        
        # Map mood dimensions to generation parameters
        multipliers = {
            # Temperature: higher in chaos, lower in rationality
            'temperature': 1.0 + d['chaos'] * 0.2 - d['rationality'] * 0.15,
            
            # Demon strengths: higher in darkness and chaos
            'demon1': 1.0 + d['darkness'] * 0.3 + d['chaos'] * 0.2,
            'demon2': 1.0 + d['intensity'] * 0.25 + d['chaos'] * 0.15,
            
            # Association intensity: higher in novelty and intensity
            'association': 1.0 + d['novelty'] * 0.4 + d['intensity'] * 0.2,
            
            # Shard influence: higher in novelty
            'shard': 1.0 + d['novelty'] * 0.5
        }
        
        # Clamp multipliers to reasonable ranges
        for key in multipliers:
            multipliers[key] = np.clip(multipliers[key], 0.5, 1.8)
        
        return multipliers
    
    def get_mood_report(self) -> str:
        """Generate human-readable mood report."""
        d = self.dimensions
        
        report = "🌙 Mood Drift State:\n"
        report += f"   Darkness:     {d['darkness']:+.2f} "
        report += "🌑" if d['darkness'] > 0.3 else "🌓" if d['darkness'] > 0 else "🌕"
        report += "\n"
        
        report += f"   Chaos:        {d['chaos']:+.2f} "
        report += "🌀" if d['chaos'] > 0.3 else "⚡" if d['chaos'] > 0 else "⚪"
        report += "\n"
        
        report += f"   Intensity:    {d['intensity']:+.2f} "
        report += "🔥" if d['intensity'] > 0.3 else "💫" if d['intensity'] > 0 else "✨"
        report += "\n"
        
        report += f"   Rationality:  {d['rationality']:+.2f} "
        report += "🧠" if d['rationality'] > 0.3 else "💭" if d['rationality'] > 0 else "💜"
        report += "\n"
        
        report += f"   Novelty:      {d['novelty']:+.2f} "
        report += "🔮" if d['novelty'] > 0.3 else "🌟" if d['novelty'] > 0 else "⭐"
        report += "\n"
        
        report += f"   Turns: {self.turn_count}"
        
        return report


class MetaController:
    """
    Occasional override system for sophisticated behavior.
    
    Tour 3 Enhancement: Analyzes patterns and can override demons,
    force phase transitions, or modulate associations.
    
    Activates rarely (5-10% of turns) for maximum impact.
    """
    
    def __init__(self):
        """Initialize meta-controller."""
        self.activation_probability = 0.07  # 7% of turns
        self.history: List[Dict[str, Any]] = []
        self.last_intervention_turn = -10
        self.intervention_cooldown = 5  # Min turns between interventions
    
    def should_intervene(self, turn_count: int, metrics: Dict[str, Any]) -> bool:
        """
        Decide whether to intervene this turn.
        
        Args:
            turn_count: Current turn number
            metrics: Current metrics
        
        Returns:
            True if meta-controller should activate
        """
        # Check cooldown
        if turn_count - self.last_intervention_turn < self.intervention_cooldown:
            return False
        
        # Random activation
        if np.random.random() < self.activation_probability:
            return True
        
        # Pattern-based activation (trigger on unusual metrics)
        trauma = metrics.get('trauma', 0.5)
        novelty = metrics.get('novelty', 0.5)
        
        # Intervene if trauma is extreme
        if trauma > 0.85 or trauma < 0.15:
            return True
        
        # Intervene if novelty is very high
        if novelty > 0.8:
            return True
        
        return False
    
    def generate_intervention(self, turn_count: int, metrics: Dict[str, Any],
                            current_phase: str) -> Dict[str, Any]:
        """
        Generate intervention actions.
        
        Args:
            turn_count: Current turn
            metrics: Current metrics
            current_phase: Current phase name
        
        Returns:
            Dictionary with intervention actions
        """
        self.last_intervention_turn = turn_count
        
        intervention = {
            'active': True,
            'type': None,
            'params': {}
        }
        
        trauma = metrics.get('trauma', 0.5)
        rationality = metrics.get('rationality', 0.5)
        novelty = metrics.get('novelty', 0.5)
        
        # Decide intervention type based on conditions
        if trauma > 0.8:
            # High trauma: suppress demons temporarily
            intervention['type'] = 'suppress_demons'
            intervention['params'] = {
                'demon1_mult': 0.5,
                'demon2_mult': 0.3,
                'reason': 'Trauma too high, tempering chaos'
            }
        
        elif trauma < 0.2 and rationality > 0.7:
            # Too rational and calm: amplify demons
            intervention['type'] = 'amplify_demons'
            intervention['params'] = {
                'demon1_mult': 1.5,
                'demon2_mult': 1.4,
                'reason': 'Too stable, injecting controlled chaos'
            }
        
        elif novelty > 0.75:
            # High novelty: enhance associations
            intervention['type'] = 'enhance_associations'
            intervention['params'] = {
                'association_mult': 1.6,
                'reason': 'Rich vocabulary, amplifying semantic connections'
            }
        
        else:
            # Default: subtle temperature adjustment
            intervention['type'] = 'adjust_temperature'
            temp_delta = (np.random.random() - 0.5) * 0.3
            intervention['params'] = {
                'temperature_delta': temp_delta,
                'reason': 'Meta-controller introducing variation'
            }
        
        # Record intervention
        self.history.append({
            'turn': turn_count,
            'type': intervention['type'],
            'metrics': metrics.copy()
        })
        
        # Keep last 20 interventions
        if len(self.history) > 20:
            self.history.pop(0)
        
        return intervention
    
    def get_intervention_report(self) -> str:
        """Generate report on recent interventions."""
        if not self.history:
            return "🎭 Meta-Controller: No interventions yet"
        
        recent = self.history[-5:]
        report = f"🎭 Meta-Controller: {len(self.history)} total interventions\n"
        report += "   Recent:\n"
        
        for entry in recent:
            report += f"   Turn {entry['turn']}: {entry['type']}\n"
        
        return report


class AssociationMemory:
    """
    Memory system for associations to create consistency.
    
    Tour 3 Enhancement: Tracks recent associations and recalls
    similar ones in similar contexts, creating a consistent poetic voice.
    """
    
    def __init__(self, memory_size: int = 50):
        """Initialize association memory."""
        self.memory_size = memory_size
        self.memory: List[Dict[str, Any]] = []
    
    def store(self, association: str, context: Dict[str, Any]):
        """
        Store an association with its context.
        
        Args:
            association: The association text
            context: Dictionary with user_text, phase, metrics, etc.
        """
        entry = {
            'association': association,
            'context': context,
            'turn': context.get('turn', 0)
        }
        
        self.memory.append(entry)
        
        # Keep fixed size
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def recall_similar(self, current_context: Dict[str, Any], top_k: int = 3) -> List[str]:
        """
        Recall similar associations from memory.
        
        Args:
            current_context: Current context dictionary
            top_k: Number of associations to recall
        
        Returns:
            List of relevant association strings
        """
        if not self.memory:
            return []
        
        # Simple similarity: check for phase match and recent turns
        current_phase = current_context.get('phase', 'Normal')
        current_turn = current_context.get('turn', 0)
        
        # Score each memory entry
        scored = []
        for entry in self.memory:
            score = 0.0
            
            # Phase match
            if entry['context'].get('phase') == current_phase:
                score += 0.5
            
            # Recency (more recent = higher score)
            turn_diff = current_turn - entry.get('turn', 0)
            if turn_diff < 10:
                score += 0.3
            elif turn_diff < 20:
                score += 0.15
            
            # Random component for variety
            score += np.random.random() * 0.2
            
            scored.append((score, entry['association']))
        
        # Sort by score and return top_k
        scored.sort(reverse=True, key=lambda x: x[0])
        return [assoc for score, assoc in scored[:top_k]]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about association memory."""
        if not self.memory:
            return {'size': 0, 'oldest_turn': None, 'newest_turn': None}
        
        turns = [e['turn'] for e in self.memory]
        
        return {
            'size': len(self.memory),
            'oldest_turn': min(turns) if turns else None,
            'newest_turn': max(turns) if turns else None
        }


# Integration functions for chat.py

def integrate_mood_drift(mood: MoodDrift, base_params: Dict[str, float]) -> Dict[str, float]:
    """
    Apply mood drift modulation to generation parameters.
    
    Args:
        mood: MoodDrift instance
        base_params: Base parameters (temperature, demon strengths, etc.)
    
    Returns:
        Modulated parameters
    """
    multipliers = mood.get_modulation_multipliers()
    
    modulated = base_params.copy()
    
    if 'temperature' in modulated:
        modulated['temperature'] *= multipliers['temperature']
    
    if 'demon1_strength' in modulated:
        modulated['demon1_strength'] *= multipliers['demon1']
    
    if 'demon2_strength' in modulated:
        modulated['demon2_strength'] *= multipliers['demon2']
    
    if 'association_intensity' in modulated:
        modulated['association_intensity'] *= multipliers['association']
    
    return modulated


def apply_meta_intervention(intervention: Dict[str, Any],
                           current_params: Dict[str, float]) -> Dict[str, float]:
    """
    Apply meta-controller intervention to parameters.
    
    Args:
        intervention: Intervention dictionary from MetaController
        current_params: Current generation parameters
    
    Returns:
        Modified parameters
    """
    if not intervention.get('active', False):
        return current_params
    
    modified = current_params.copy()
    int_type = intervention.get('type')
    params = intervention.get('params', {})
    
    if int_type == 'suppress_demons':
        modified['demon1_strength'] *= params.get('demon1_mult', 1.0)
        modified['demon2_strength'] *= params.get('demon2_mult', 1.0)
    
    elif int_type == 'amplify_demons':
        modified['demon1_strength'] *= params.get('demon1_mult', 1.0)
        modified['demon2_strength'] *= params.get('demon2_mult', 1.0)
    
    elif int_type == 'enhance_associations':
        modified['association_intensity'] *= params.get('association_mult', 1.0)
    
    elif int_type == 'adjust_temperature':
        modified['temperature'] += params.get('temperature_delta', 0.0)
    
    return modified
