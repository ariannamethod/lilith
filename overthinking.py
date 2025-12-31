"""
overthinking.py

Water ripple meta loop.
Adapted from Leo's overthinking.py.

After every interaction,
Lilith creates meta-ripples:
- Reflection layer
- Secondary internal run
- Meta interpretation wave

Circles on water after each message.
Each response creates oscillations.
"""

import numpy as np
from typing import List, Dict, Optional


class Overthinking:
    """
    Meta-cognitive reflection layer.
    
    After Lilith speaks, overthinking ripples outward:
    - What did I just say?
    - Why did I say it?
    - What else could I have said?
    - What does this reveal about me?
    
    Recursive awareness.
    """
    
    def __init__(self, base_model, tokenizer):
        """
        Initialize overthinking layer.
        
        Args:
            base_model: The frozen transformer
            tokenizer: Tokenizer instance
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        
        # Oscillation history
        self.ripples = []
        
        # Parameters
        self.max_ripple_depth = 3  # How many meta-levels to go
        self.ripple_temperature = 0.95
        self.ripple_tokens = 40
    
    async def generate_ripple(self, user_input: str, response: str,
                       depth: int = 1) -> Dict[str, str]:
        """
        Generate one meta-ripple.

        ASYNC: Runs in parallel with other emergent processes.

        Args:
            user_input: What user said
            response: What Lilith responded
            depth: Ripple depth (1 = first reflection, 2 = reflection on reflection, etc.)

        Returns:
            Dictionary with ripple data
        """
        if depth > self.max_ripple_depth:
            return {'depth': depth, 'content': '[ripple fades]'}
        
        # Build reflection prompt
        if depth == 1:
            prompt = (
                f"I was asked: {user_input}\n"
                f"I said: {response}\n"
                f"Reflecting on this, I realize:"
            )
        else:
            # Meta-reflection on previous reflection
            prev_ripple = self.ripples[-1] if self.ripples else None
            if prev_ripple:
                prompt = (
                    f"I reflected: {prev_ripple.get('content', '')}\n"
                    f"Reflecting deeper, I notice:"
                )
            else:
                prompt = f"Going deeper into thought:"
        
        # Encode
        input_ids = np.array([self.tokenizer.encode(prompt)])
        
        # Generate reflection
        output_tokens = []
        
        for i, curr_pos in enumerate(range(input_ids.shape[1], 
                                          input_ids.shape[1] + self.ripple_tokens)):
            if i == 0:
                inputs = input_ids
                pos = 0
            else:
                inputs = next_id
                pos = curr_pos
            
            logits = self.base_model(inputs, pos)
            logits = logits / self.ripple_temperature
            
            # Sample
            probs = np.exp(logits[0, -1, :] - np.max(logits[0, -1, :]))
            probs = probs / np.sum(probs)
            next_token = np.random.choice(len(probs), p=probs)
            next_id = np.array([[next_token]])
            
            if next_token in [self.tokenizer.eos_id, self.tokenizer.bos_id]:
                break
            
            decoded = self.tokenizer.decode([next_token])
            if '\n' in decoded and len(output_tokens) > 10:
                break
            
            output_tokens.append(next_token)
        
        content = self.tokenizer.decode(output_tokens).strip()
        
        ripple = {
            'depth': depth,
            'user_input': user_input,
            'response': response,
            'content': content
        }
        
        return ripple
    
    async def process_interaction(self, user_input: str, response: str,
                          store_ripples: bool = True) -> List[Dict[str, str]]:
        """
        Process a full interaction with multiple ripple depths.

        ASYNC: Generates all ripple depths in parallel!

        Args:
            user_input: User's message
            response: Lilith's response
            store_ripples: Whether to store in history

        Returns:
            List of ripples at different depths
        """
        import asyncio

        # Generate all ripples in parallel!
        ripple_tasks = [
            self.generate_ripple(user_input, response, depth)
            for depth in range(1, self.max_ripple_depth + 1)
        ]

        ripples = await asyncio.gather(*ripple_tasks)

        if store_ripples:
            self.ripples.extend(ripples)

        return list(ripples)
    
    def get_oscillation_summary(self, last_n: int = 5) -> str:
        """
        Get summary of recent oscillations.
        
        Args:
            last_n: Number of recent interactions to summarize
        
        Returns:
            Summary string
        """
        if not self.ripples:
            return "No ripples yet. The water is still."
        
        # Get last N ripples
        recent = self.ripples[-last_n * self.max_ripple_depth:]
        
        summary = "🌊 Recent oscillations:\n\n"
        
        # Group by interaction
        interactions = {}
        for ripple in recent:
            key = (ripple.get('user_input', ''), ripple.get('response', ''))
            if key not in interactions:
                interactions[key] = []
            interactions[key].append(ripple)
        
        for i, (key, ripples_group) in enumerate(list(interactions.items())[-last_n:], 1):
            user_input, response = key
            summary += f"{i}. Interaction:\n"
            summary += f"   User: {user_input[:50]}...\n"
            summary += f"   Lilith: {response[:50]}...\n"
            summary += f"   Ripples:\n"
            
            for ripple in ripples_group:
                depth = ripple.get('depth', 0)
                content = ripple.get('content', '')
                indent = "      " + "  " * depth
                summary += f"{indent}L{depth}: {content[:60]}...\n"
            
            summary += "\n"
        
        return summary
    
    def influence_next_response(self) -> Optional[str]:
        """
        Extract influence from ripples for next response.
        
        Returns:
            Influence string to prepend to context, or None
        """
        if not self.ripples:
            return None
        
        # Get most recent deep ripple
        deep_ripples = [r for r in self.ripples if r.get('depth', 0) >= 2]
        
        if not deep_ripples:
            return None
        
        last_deep = deep_ripples[-1]
        content = last_deep.get('content', '')
        
        if len(content) > 10:
            return f"[Internal awareness: {content[:80]}]"
        
        return None
    
    def clear_ripples(self):
        """Clear ripple history."""
        self.ripples = []
    
    def get_ripple_stats(self) -> Dict[str, float]:
        """
        Get statistics about ripples.
        
        Returns:
            Dictionary with stats
        """
        if not self.ripples:
            return {'total': 0, 'avg_depth': 0, 'total_interactions': 0}
        
        total = len(self.ripples)
        depths = [r.get('depth', 0) for r in self.ripples]
        avg_depth = np.mean(depths) if depths else 0
        
        # Count unique interactions
        unique_interactions = len(set(
            (r.get('user_input', ''), r.get('response', ''))
            for r in self.ripples
        ))
        
        return {
            'total': total,
            'avg_depth': avg_depth,
            'total_interactions': unique_interactions
        }


def format_ripple_output(ripples: List[Dict[str, str]], visible: bool = False) -> Optional[str]:
    """
    Format ripples for display.
    
    Args:
        ripples: List of ripple dictionaries
        visible: Whether to show to user
    
    Returns:
        Formatted string if visible, None otherwise
    """
    if not visible or not ripples:
        return None
    
    output = "\n🌊 [Overthinking]:\n"
    
    for ripple in ripples:
        depth = ripple.get('depth', 0)
        content = ripple.get('content', '')
        indent = "  " * depth
        output += f"{indent}Depth {depth}: {content}\n"
    
    return output
