"""
test_tokenizer.py

Tour 3: Tests for tokenizer with error handling.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tempfile
import json
from tokenizer import Tokenizer


def test_tokenizer_valid(tmp_path="/tmp"):
    """Test tokenizer with valid data."""
    print("\nTesting tokenizer initialization...")
    
    # Create a simple mock tokenizer file
    tokenizer_data = {
        "tokens": ["<s>", "</s>", "a", "b", "c", "ab", "bc"],
        "scores": [0.0, 0.0, 0.1, 0.1, 0.1, 0.5, 0.5]
    }
    
    tok_path = os.path.join(tmp_path, "test_tokenizer.json")
    with open(tok_path, 'w') as f:
        json.dump(tokenizer_data, f)
    
    try:
        tokenizer = Tokenizer(tok_path)
        assert len(tokenizer.vocab) == 7
        assert tokenizer.bos_id == 1
        assert tokenizer.eos_id == 2
        print(f"  Vocab size: {len(tokenizer.vocab)}")
        print("✓ Tokenizer initialization OK")
        return True
    finally:
        if os.path.exists(tok_path):
            os.remove(tok_path)


def test_tokenizer_encode_decode(tmp_path="/tmp"):
    """Test encoding and decoding."""
    print("\nTesting encode/decode...")
    
    tokenizer_data = {
        "tokens": ["<s>", "</s>", "a", "b", "c", "ab", "bc", "abc"],
        "scores": [0.0, 0.0, 0.1, 0.1, 0.1, 0.5, 0.5, 0.8]
    }
    
    tok_path = os.path.join(tmp_path, "test_tokenizer2.json")
    with open(tok_path, 'w') as f:
        json.dump(tokenizer_data, f)
    
    try:
        tokenizer = Tokenizer(tok_path)
        
        # Test encoding
        text = "abc"
        tokens = tokenizer.encode(text, add_bos=False, add_eos=False)
        assert len(tokens) > 0
        print(f"  Encoded '{text}' to {len(tokens)} tokens")
        
        # Test decoding
        decoded = tokenizer.decode(tokens)
        print(f"  Decoded back to '{decoded}'")
        
        # Test empty input
        empty_tokens = tokenizer.encode("", add_bos=False, add_eos=False)
        assert isinstance(empty_tokens, list)
        print(f"  Empty string encoded to {len(empty_tokens)} tokens")
        
        # Test decoding empty
        empty_decoded = tokenizer.decode([])
        assert isinstance(empty_decoded, str)
        print(f"  Empty tokens decoded to '{empty_decoded}'")
        
        print("✓ Encode/decode OK")
        return True
    finally:
        if os.path.exists(tok_path):
            os.remove(tok_path)


def test_tokenizer_missing_file():
    """Test tokenizer with missing file."""
    print("\nTesting missing tokenizer file...")
    
    try:
        Tokenizer("/nonexistent/tokenizer.json")
        print("✗ Should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        print(f"  Correctly raised: {type(e).__name__}")
        print("✓ Missing file handling OK")
        return True


def test_tokenizer_invalid_json(tmp_path="/tmp"):
    """Test tokenizer with invalid JSON."""
    print("\nTesting invalid JSON handling...")
    
    tok_path = os.path.join(tmp_path, "invalid_tokenizer.json")
    with open(tok_path, 'w') as f:
        f.write("{invalid json")
    
    try:
        Tokenizer(tok_path)
        print("✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  Correctly raised: {type(e).__name__}")
        print("✓ Invalid JSON handling OK")
        return True
    finally:
        if os.path.exists(tok_path):
            os.remove(tok_path)


def test_tokenizer_missing_fields(tmp_path="/tmp"):
    """Test tokenizer with missing required fields."""
    print("\nTesting missing required fields...")
    
    # Missing 'scores' field
    tokenizer_data = {
        "tokens": ["<s>", "</s>", "a"]
    }
    
    tok_path = os.path.join(tmp_path, "incomplete_tokenizer.json")
    with open(tok_path, 'w') as f:
        json.dump(tokenizer_data, f)
    
    try:
        Tokenizer(tok_path)
        print("✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  Correctly raised: {type(e).__name__}")
        print("✓ Missing fields handling OK")
        return True
    finally:
        if os.path.exists(tok_path):
            os.remove(tok_path)


def test_tokenizer_mismatched_lengths(tmp_path="/tmp"):
    """Test tokenizer with mismatched vocab/scores lengths."""
    print("\nTesting mismatched lengths...")
    
    tokenizer_data = {
        "tokens": ["<s>", "</s>", "a", "b"],
        "scores": [0.0, 0.0]  # Wrong length
    }
    
    tok_path = os.path.join(tmp_path, "mismatched_tokenizer.json")
    with open(tok_path, 'w') as f:
        json.dump(tokenizer_data, f)
    
    try:
        Tokenizer(tok_path)
        print("✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  Correctly raised: {type(e).__name__}")
        print("✓ Mismatched lengths handling OK")
        return True
    finally:
        if os.path.exists(tok_path):
            os.remove(tok_path)


def test_tokenizer_bounds_checking(tmp_path="/tmp"):
    """Test decode with invalid token IDs."""
    print("\nTesting bounds checking in decode...")
    
    tokenizer_data = {
        "tokens": ["<s>", "</s>", "a", "b", "c"],
        "scores": [0.0, 0.0, 0.1, 0.1, 0.1]
    }
    
    tok_path = os.path.join(tmp_path, "bounds_tokenizer.json")
    with open(tok_path, 'w') as f:
        json.dump(tokenizer_data, f)
    
    try:
        tokenizer = Tokenizer(tok_path)
        
        # Try decoding with out-of-bounds IDs
        result = tokenizer.decode([2, 3, 999, 4, -1])  # Invalid IDs: 999, -1
        # Should not crash, should skip invalid IDs
        print(f"  Decoded with invalid IDs: '{result}'")
        print("✓ Bounds checking OK")
        return True
    finally:
        if os.path.exists(tok_path):
            os.remove(tok_path)


def run_all_tests():
    """Run all tokenizer tests."""
    print("\n" + "="*60)
    print("LILITH TOKENIZER TESTS")
    print("="*60)
    
    tests = [
        test_tokenizer_valid,
        test_tokenizer_encode_decode,
        test_tokenizer_missing_file,
        test_tokenizer_invalid_json,
        test_tokenizer_missing_fields,
        test_tokenizer_mismatched_lengths,
        test_tokenizer_bounds_checking
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
        print("✓ ALL TOKENIZER TESTS PASSED")
    else:
        print(f"✗ {failed} TOKENIZER TESTS FAILED")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
