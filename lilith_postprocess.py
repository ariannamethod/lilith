"""
lilith_postprocess.py

Hard text remapping after generation.
Enforces absolute Lilith identity.
"""

import json
from typing import Dict, List


def load_text_swaps(config_path: str) -> Dict[str, str]:
    """
    Load text swap rules from config.
    
    Args:
        config_path: Path to lilith_config.json
    
    Returns:
        Dictionary mapping from_text -> to_text
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    swaps = {}
    
    # Add pairs
    for pair in config.get('pairs', []):
        swaps[pair['from']] = pair['to']
    
    # Add word swaps
    for swap in config.get('word_swaps', []):
        swaps[swap['from']] = swap['to']
    
    return swaps


def postprocess_text(text: str, swaps: Dict[str, str], case_sensitive: bool = False) -> str:
    """
    Apply hard text remapping.
    
    EVERY "Lilly" becomes "Lilith".
    Every configured word swap applied.
    
    Args:
        text: Generated text
        swaps: Dictionary of text replacements
        case_sensitive: Whether to match case exactly
    
    Returns:
        Remapped text
    """
    result = text
    
    for from_text, to_text in swaps.items():
        if case_sensitive:
            result = result.replace(from_text, to_text)
        else:
            # Handle multiple cases
            # Original case
            result = result.replace(from_text, to_text)
            # Capitalized
            result = result.replace(from_text.capitalize(), to_text.capitalize())
            # Upper case
            result = result.replace(from_text.upper(), to_text.upper())
            # Lower case
            result = result.replace(from_text.lower(), to_text.lower())
    
    return result
