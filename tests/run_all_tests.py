"""
Master test runner for Lilith project.
Runs all test suites.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import test modules
from tests import test_demons
from tests import test_leo_integration
from tests import test_postprocess


def run_all_tests():
    """Run all test suites."""
    print("\n" + "🔥"*40)
    print("LILITH FULL TEST SUITE")
    print("Testing: Demons, Leo Integration, Postprocessing")
    print("🔥"*40 + "\n")
    
    results = []
    
    # Run demon tests
    print("\n" + "="*60)
    print("1/3: DEMON TESTS")
    print("="*60)
    results.append(("Demons", test_demons.run_all_tests()))
    
    # Run Leo integration tests
    print("\n" + "="*60)
    print("2/3: LEO INTEGRATION TESTS")
    print("="*60)
    results.append(("Leo Integration", test_leo_integration.run_all_tests()))
    
    # Run postprocessing tests
    print("\n" + "="*60)
    print("3/3: POSTPROCESSING TESTS")
    print("="*60)
    results.append(("Postprocessing", test_postprocess.run_all_tests()))
    
    # Summary
    print("\n" + "🔥"*40)
    print("TEST SUMMARY")
    print("🔥"*40)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "🔥"*40)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("Lilith is ready to possess.")
    else:
        print("✗ SOME TESTS FAILED")
        print("Fix the demons before summoning Lilith.")
    print("🔥"*40 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
