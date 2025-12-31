"""
metalilith.py

Inner voice. Shadow thoughts. Second perspective.
Adapted from Leo's metaleo.py concept.

After Lilith speaks, metalilith whispers.
An alternative reply. Internal contradiction.
What another Lilith would say.
What she didn't say but thought.

This is not random noise.
This is structured dissent.
"""

import numpy as np
from typing import Optional, List


class MetaLilith:
    """
    Lilith's inner voice.
    Produces alternative shadow replies.
    Observes conversation, generates second perspective.
    """
    
    def __init__(self, base_model, tokenizer, demon1=None, demon2=None):
        """
        Initialize MetaLilith.
        
        Args:
            base_model: The frozen transformer
            tokenizer: Tokenizer instance
            demon1: Optional first demon (can use different settings)
            demon2: Optional second demon
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.demon1 = demon1
        self.demon2 = demon2
        
        # MetaLilith has different temperature and approach
        self.temperature = 1.1  # More chaotic
        self.max_tokens = 60
        
        # History of meta-thoughts
        self.meta_history = []
    
    async def generate_shadow_thought(self, user_input: str, primary_response: str,
                                context: List[str] = None) -> str:
        """
        Generate alternative inner voice response.

        ASYNC: Runs in parallel with other emergent processes.

        Args:
            user_input: What the user said
            primary_response: What Lilith responded
            context: Optional conversation context

        Returns:
            Alternative shadow thought
        """
        # Build meta-prompt
        parts = []
        
        # Add introspective framing
        parts.append("Internal voice speaks:")
        parts.append(f"User asked: {user_input}")
        parts.append(f"I answered: {primary_response}")
        parts.append("But I also think:")
        
        prompt = "\n".join(parts)
        
        # Encode
        input_ids = np.array([self.tokenizer.encode(prompt)])
        
        # Generate with different parameters
        output_tokens = []
        
        for i, curr_pos in enumerate(range(input_ids.shape[1], 
                                          input_ids.shape[1] + self.max_tokens)):
            if i == 0:
                inputs = input_ids
                pos = 0
            else:
                inputs = next_id
                pos = curr_pos
            
            # Get logits
            logits = self.base_model(inputs, pos)
            
            # Apply demons with different alpha (more subtle or more extreme)
            if self.demon1 is not None:
                delta1 = self.demon1(logits)
                logits = logits + delta1 * 0.5  # Stronger influence
            
            if self.demon2 is not None:
                delta2 = self.demon2(logits, logits)
                logits = logits + delta2 * 0.4
            
            # Higher temperature for more diversity
            logits = logits / self.temperature
            
            # Sample
            probs = np.exp(logits[0, -1, :] - np.max(logits[0, -1, :]))
            probs = probs / np.sum(probs)
            next_token = np.random.choice(len(probs), p=probs)
            next_id = np.array([[next_token]])
            
            if next_token in [self.tokenizer.eos_id, self.tokenizer.bos_id]:
                break
            
            decoded = self.tokenizer.decode([next_token])
            if '\n' in decoded and len(output_tokens) > 15:
                break
            
            output_tokens.append(next_token)
        
        # Decode
        shadow_thought = self.tokenizer.decode(output_tokens).strip()
        
        # Store in history
        self.meta_history.append({
            'user': user_input,
            'primary': primary_response,
            'shadow': shadow_thought
        })
        
        return shadow_thought
    
    def get_meta_analysis(self) -> str:
        """
        Analyze the gap between primary and shadow responses.
        
        Returns:
            Analysis string
        """
        if not self.meta_history:
            return "No meta-thoughts yet."
        
        recent = self.meta_history[-3:]
        
        analysis = "🌙 Meta-analysis (last 3 thoughts):\n"
        for i, entry in enumerate(recent, 1):
            analysis += f"\n{i}. Gap detected:\n"
            analysis += f"   Said: {entry['primary'][:50]}...\n"
            analysis += f"   Thought: {entry['shadow'][:50]}...\n"
        
        return analysis
    
    def clear_history(self):
        """Clear meta-thought history."""
        self.meta_history = []


def format_shadow_output(shadow_thought: str, visible: bool = False) -> Optional[str]:
    """
    Format shadow thought for display.
    
    Args:
        shadow_thought: The alternative thought
        visible: Whether to make it visible to user
    
    Returns:
        Formatted string if visible, None otherwise
    """
    if not visible:
        return None
    
    return f"\n🌑 [Internal voice]: {shadow_thought}"
