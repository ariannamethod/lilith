"""
test_config.py

Tour 3: Tests for configuration loading and error handling.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tempfile
import json
import numpy as np
from config import ModelArgs
from lilith_dissonance import load_config
from lilith_postprocess import load_text_swaps
from tokenizer import Tokenizer


def test_model_args():
    """Test ModelArgs dataclass."""
    print("\nTesting ModelArgs initialization...")
    args = ModelArgs()
    assert args.dim == 288
    assert args.n_layers == 6
    assert args.vocab_size == 32000
    print(f"  Dim: {args.dim}, Layers: {args.n_layers}, Vocab: {args.vocab_size}")
    print("✓ ModelArgs OK")
    return True


def test_load_config_valid(tmp_path="/tmp"):
    """Test loading valid config file."""
    print("\nTesting valid config loading...")
    
    # Create temporary config
    config_data = {
        "pairs": [
            {"from": "test", "to": "demo"}
        ],
        "word_swaps": [
            {"from": "good", "to": "bad"}
        ],
        "extra_targets": ["dark", "shadow"]
    }
    
    config_path = os.path.join(tmp_path, "test_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    
    try:
        # Create mock tokenizer
        class MockTokenizer:
            def encode(self, text, add_bos=True, add_eos=False):
                # Simple mock: return char codes
                return [ord(c) for c in text[:3]]
        
        tokenizer = MockTokenizer()
        from_ids, to_ids, target_ids = load_config(config_path, tokenizer)
        
        assert len(from_ids) > 0, "Should have from_ids"
        assert len(to_ids) > 0, "Should have to_ids"
        assert len(target_ids) > 0, "Should have target_ids"
        
        print(f"  From IDs: {len(from_ids)}, To IDs: {len(to_ids)}, Targets: {len(target_ids)}")
        print("✓ Valid config loading OK")
        
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.remove(config_path)
    
    return True


def test_load_config_missing_file():
    """Test loading from missing config file."""
    print("\nTesting missing config file handling...")
    
    class MockTokenizer:
        def encode(self, text, add_bos=True, add_eos=False):
            return [1, 2, 3]
    
    try:
        load_config("/nonexistent/path/config.json", MockTokenizer())
        print("✗ Should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        print(f"  Correctly raised: {type(e).__name__}")
        print("✓ Missing file handling OK")
        return True


def test_load_config_invalid_json(tmp_path="/tmp"):
    """Test loading invalid JSON."""
    print("\nTesting invalid JSON handling...")
    
    config_path = os.path.join(tmp_path, "invalid_config.json")
    with open(config_path, 'w') as f:
        f.write("{invalid json content")
    
    try:
        class MockTokenizer:
            def encode(self, text, add_bos=True, add_eos=False):
                return [1, 2, 3]
        
        load_config(config_path, MockTokenizer())
        print("✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  Correctly raised: {type(e).__name__}")
        print("✓ Invalid JSON handling OK")
        return True
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)


def test_load_text_swaps_valid(tmp_path="/tmp"):
    """Test loading text swaps from valid config."""
    print("\nTesting text swaps loading...")
    
    config_data = {
        "pairs": [
            {"from": "Lilly", "to": "Lilith"}
        ],
        "word_swaps": [
            {"from": "flower", "to": "thorn"},
            {"from": "princess", "to": "witch"}
        ]
    }
    
    config_path = os.path.join(tmp_path, "swaps_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    
    try:
        swaps = load_text_swaps(config_path)
        assert "Lilly" in swaps
        assert swaps["Lilly"] == "Lilith"
        assert "flower" in swaps
        assert swaps["flower"] == "thorn"
        print(f"  Loaded {len(swaps)} swap rules")
        print("✓ Text swaps loading OK")
        return True
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)


def test_load_text_swaps_missing_keys(tmp_path="/tmp"):
    """Test handling config with missing keys."""
    print("\nTesting config with missing keys...")
    
    config_data = {
        "pairs": [
            {"from": "test"}  # Missing 'to'
        ],
        "word_swaps": [
            {"to": "result"}  # Missing 'from'
        ]
    }
    
    config_path = os.path.join(tmp_path, "incomplete_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    
    try:
        # Should not crash, just skip invalid entries
        swaps = load_text_swaps(config_path)
        print(f"  Loaded {len(swaps)} valid swaps (skipped invalid)")
        print("✓ Missing keys handling OK")
        return True
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)


def run_all_tests():
    """Run all config tests."""
    print("\n" + "="*60)
    print("LILITH CONFIG TESTS")
    print("="*60)
    
    tests = [
        test_model_args,
        test_load_config_valid,
        test_load_config_missing_file,
        test_load_config_invalid_json,
        test_load_text_swaps_valid,
        test_load_text_swaps_missing_keys
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    if failed == 0:
        print("✓ ALL CONFIG TESTS PASSED")
    else:
        print(f"✗ {failed} CONFIG TESTS FAILED")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
