"""
lilith_postprocess.py

Hard text remapping after generation.
Enforces absolute Lilith identity.
"""

import json
import os
from typing import Dict, List


def load_text_swaps(config_path: str) -> Dict[str, str]:
    """
    Load text swap rules from config.
    
    Args:
        config_path: Path to lilith_config.json
    
    Returns:
        Dictionary mapping from_text -> to_text
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    # Tour 3: Defensive check for missing config
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Lilith requires lilith_config.json to know her identity."
        )
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in config file {config_path}: {e}\n"
            f"Lilith cannot parse corrupted configuration."
        )
    
    swaps = {}
    
    # Tour 3: Defensive handling of missing keys
    # Add pairs
    pairs = config.get('pairs', [])
    if not isinstance(pairs, list):
        raise ValueError(f"'pairs' must be a list, got {type(pairs)}")
    
    for pair in pairs:
        if not isinstance(pair, dict):
            continue  # Skip invalid entries gracefully
        if 'from' in pair and 'to' in pair:
            swaps[pair['from']] = pair['to']
    
    # Add word swaps
    word_swaps = config.get('word_swaps', [])
    if not isinstance(word_swaps, list):
        raise ValueError(f"'word_swaps' must be a list, got {type(word_swaps)}")
    
    for swap in word_swaps:
        if not isinstance(swap, dict):
            continue  # Skip invalid entries gracefully
        if 'from' in swap and 'to' in swap:
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
    # Tour 3: Handle edge cases
    if not text:
        return text
    
    if not swaps:
        return text
    
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
