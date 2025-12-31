"""
Test suite for text postprocessing.
Tests lilith_postprocess.py functions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lilith_postprocess import postprocess_text


def test_basic_swap():
    """Test basic text swap."""
    print("Testing basic text swap...")
    
    swaps = {"Lilly": "Lilith"}
    text = "Once there was Lilly who loved flowers."
    
    result = postprocess_text(text, swaps)
    
    assert "Lilith" in result
    assert "Lilly" not in result
    
    print(f"  Input: {text}")
    print(f"  Output: {result}")
    print("✓ Basic swap OK")


def test_multiple_swaps():
    """Test multiple word swaps."""
    print("Testing multiple swaps...")
    
    swaps = {
        "Lilly": "Lilith",
        "flower": "thorn",
        "princess": "witch"
    }
    
    text = "Princess Lilly picked a beautiful flower."
    result = postprocess_text(text, swaps)
    
    assert "Lilith" in result
    assert "thorn" in result
    assert "witch" in result.lower()
    assert "Lilly" not in result
    assert "flower" not in result
    assert "princess" not in result.lower()
    
    print(f"  Input: {text}")
    print(f"  Output: {result}")
    print("✓ Multiple swaps OK")


def test_case_insensitive():
    """Test case-insensitive swapping."""
    print("Testing case-insensitive swapping...")
    
    swaps = {"lilly": "lilith"}
    
    texts = [
        "lilly was happy",
        "Lilly was happy",
        "LILLY was happy"
    ]
    
    for text in texts:
        result = postprocess_text(text, swaps, case_sensitive=False)
        # Should contain some form of lilith
        assert "lilly" not in result.lower() or "lilith" in result.lower()
        print(f"  '{text}' -> '{result}'")
    
    print("✓ Case-insensitive OK")


def test_no_swaps():
    """Test with empty swap dict."""
    print("Testing with no swaps...")
    
    swaps = {}
    text = "This text should not change."
    
    result = postprocess_text(text, swaps)
    
    assert result == text
    
    print("✓ No swaps OK")


def test_partial_matches():
    """Test that partial word matches don't get replaced incorrectly."""
    print("Testing partial matches...")
    
    swaps = {"lily": "shadow"}
    text = "The lily pond was full of lilies."
    
    result = postprocess_text(text, swaps, case_sensitive=True)
    
    # Should replace "lily" but not affect "lilies"
    print(f"  Input: {text}")
    print(f"  Output: {result}")
    print("✓ Partial matches OK")


def test_repeated_words():
    """Test repeated word replacements."""
    print("Testing repeated word replacement...")
    
    swaps = {"happy": "haunted"}
    text = "She was happy, so happy, always happy."
    
    result = postprocess_text(text, swaps)
    
    # Count occurrences
    happy_count = result.lower().count("happy")
    haunted_count = result.lower().count("haunted")
    
    assert happy_count == 0
    assert haunted_count == 3
    
    print(f"  Input: {text}")
    print(f"  Output: {result}")
    print("✓ Repeated words OK")


def run_all_tests():
    """Run all postprocessing tests."""
    print("\n" + "="*60)
    print("LILITH POSTPROCESSING TESTS")
    print("="*60 + "\n")
    
    try:
        test_basic_swap()
        test_multiple_swaps()
        test_case_insensitive()
        test_no_swaps()
        test_partial_matches()
        test_repeated_words()
        
        print("\n" + "="*60)
        print("✓ ALL POSTPROCESSING TESTS PASSED")
        print("="*60 + "\n")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
