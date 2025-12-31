"""
Master test runner for Lilith project.
Runs all test suites.

Tour 3: Enhanced with additional test modules.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import test modules
from tests import test_demons
from tests import test_leo_integration
from tests import test_postprocess
from tests import test_tour2
from tests import test_config
from tests import test_utils
from tests import test_tokenizer
from tests import test_repl
from tests import test_emergence


def run_all_tests():
    """Run all test suites."""
    print("\n" + "🔥"*40)
    print("LILITH FULL TEST SUITE - TOUR 3")
    print("All layers tested: Demons, Leo, Tour 2, Infrastructure, Emergence")
    print("🔥"*40 + "\n")
    
    results = []
    
    # Run demon tests
    print("\n" + "="*60)
    print("1/9: DEMON TESTS")
    print("="*60)
    results.append(("Demons", test_demons.run_all_tests()))
    
    # Run Leo integration tests
    print("\n" + "="*60)
    print("2/9: LEO INTEGRATION TESTS")
    print("="*60)
    results.append(("Leo Integration", test_leo_integration.run_all_tests()))
    
    # Run postprocessing tests
    print("\n" + "="*60)
    print("3/9: POSTPROCESSING TESTS")
    print("="*60)
    results.append(("Postprocessing", test_postprocess.run_all_tests()))
    
    # Run Tour 2 tests
    print("\n" + "="*60)
    print("4/9: TOUR 2 TESTS (Shards, Stats, Associations)")
    print("="*60)
    results.append(("Tour 2", test_tour2.run_all_tests()))
    
    # Run config tests (Tour 3)
    print("\n" + "="*60)
    print("5/9: CONFIG TESTS (Tour 3)")
    print("="*60)
    results.append(("Config", test_config.run_all_tests()))
    
    # Run utils tests (Tour 3)
    print("\n" + "="*60)
    print("6/9: UTILS TESTS (Tour 3)")
    print("="*60)
    results.append(("Utils", test_utils.run_all_tests()))
    
    # Run tokenizer tests (Tour 3)
    print("\n" + "="*60)
    print("7/9: TOKENIZER TESTS (Tour 3)")
    print("="*60)
    results.append(("Tokenizer", test_tokenizer.run_all_tests()))
    
    # Run REPL tests (Tour 3)
    print("\n" + "="*60)
    print("8/9: REPL TESTS (Tour 3)")
    print("="*60)
    results.append(("REPL", test_repl.run_all_tests()))
    
    # Run emergence tests (Tour 3)
    print("\n" + "="*60)
    print("9/9: EMERGENCE TESTS (Tour 3)")
    print("="*60)
    results.append(("Emergence", test_emergence.run_all_tests()))
    
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
        print("Lilith is stable and ready to possess.")
        print("Tour 3: Stabilization complete.")
    else:
        print("✗ SOME TESTS FAILED")
        print("Stabilize the demons before summoning Lilith.")
    print("🔥"*40 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
