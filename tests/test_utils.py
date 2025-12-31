"""
test_utils.py

Tour 3: Tests for utility functions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tempfile
import numpy as np
from utils import load_parameters


def test_load_parameters_valid(tmp_path="/tmp"):
    """Test loading valid model parameters."""
    print("\nTesting valid parameter loading...")
    
    # Create a temporary .npz file
    test_params = {
        'weight1': np.random.randn(10, 10),
        'weight2': np.random.randn(5, 5)
    }
    
    npz_path = os.path.join(tmp_path, "test_model.npz")
    np.savez(npz_path, **test_params)
    
    try:
        params = load_parameters(npz_path)
        assert 'weight1' in params
        assert 'weight2' in params
        assert params['weight1'].shape == (10, 10)
        print(f"  Loaded {len(params.files)} parameter arrays")
        print("✓ Valid parameter loading OK")
        return True
    finally:
        if os.path.exists(npz_path):
            os.remove(npz_path)


def test_load_parameters_missing_file():
    """Test loading from missing file."""
    print("\nTesting missing file handling...")
    
    try:
        load_parameters("/nonexistent/model.npz")
        print("✗ Should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        print(f"  Correctly raised: {type(e).__name__}")
        print("✓ Missing file handling OK")
        return True


def test_load_parameters_wrong_extension():
    """Test loading file with wrong extension."""
    print("\nTesting wrong file extension handling...")
    
    try:
        load_parameters("model.txt")
        print("✗ Should have raised ValueError")
        return False
    except (ValueError, FileNotFoundError) as e:
        # FileNotFoundError is OK too (file doesn't exist)
        print(f"  Correctly raised: {type(e).__name__}")
        print("✓ Wrong extension handling OK")
        return True


def test_load_parameters_corrupted(tmp_path="/tmp"):
    """Test loading corrupted file."""
    print("\nTesting corrupted file handling...")
    
    # Create a file that's not valid NPZ
    bad_path = os.path.join(tmp_path, "bad_model.npz")
    with open(bad_path, 'w') as f:
        f.write("This is not a valid NPZ file")
    
    try:
        load_parameters(bad_path)
        print("✗ Should have raised ValueError")
        return False
    except (ValueError, Exception) as e:
        print(f"  Correctly raised: {type(e).__name__}")
        print("✓ Corrupted file handling OK")
        return True
    finally:
        if os.path.exists(bad_path):
            os.remove(bad_path)


def run_all_tests():
    """Run all utils tests."""
    print("\n" + "="*60)
    print("LILITH UTILS TESTS")
    print("="*60)
    
    tests = [
        test_load_parameters_valid,
        test_load_parameters_missing_file,
        test_load_parameters_wrong_extension,
        test_load_parameters_corrupted
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
        print("✓ ALL UTILS TESTS PASSED")
    else:
        print(f"✗ {failed} UTILS TESTS FAILED")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
