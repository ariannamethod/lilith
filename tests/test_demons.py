"""
Test suite for Lilith demons.
Tests DissonanceMLP and CounterDissonanceMLP.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lilith_dissonance import DissonanceMLP, CounterDissonanceMLP, compose_logits, Scalar


def test_dissonance_mlp_init():
    """Test DissonanceMLP initialization."""
    print("Testing DissonanceMLP initialization...")
    
    vocab_size = 100
    demon = DissonanceMLP(vocab_size)
    
    assert demon.input_dim == vocab_size
    assert demon.hidden_dim == 2 * vocab_size
    assert demon.W1.shape == (vocab_size, 2 * vocab_size)
    assert demon.W2.shape == (2 * vocab_size, vocab_size)
    
    print("✓ DissonanceMLP initialization OK")


def test_dissonance_mlp_forward():
    """Test DissonanceMLP forward pass."""
    print("Testing DissonanceMLP forward pass...")
    
    vocab_size = 100
    batch_size = 2
    
    demon = DissonanceMLP(vocab_size)
    logits = np.random.randn(batch_size, vocab_size)
    
    # Forward pass
    delta = demon(logits)
    
    assert delta.shape == logits.shape
    print(f"  Input shape: {logits.shape}")
    print(f"  Output shape: {delta.shape}")
    
    print("✓ DissonanceMLP forward pass OK")


def test_dissonance_mlp_masking():
    """Test DissonanceMLP token masking."""
    print("Testing DissonanceMLP masking...")
    
    vocab_size = 100
    demon = DissonanceMLP(vocab_size)
    
    # Set target tokens
    from_ids = [1, 2, 3]
    to_ids = [10, 11, 12]
    target_ids = [20, 21, 22]
    
    demon.set_token_targets(from_ids, to_ids, target_ids, vocab_size)
    
    # Check mask
    assert demon.mask is not None
    assert demon.mask.shape == (vocab_size,)
    
    # Check that target tokens are masked
    for tid in to_ids + target_ids:
        assert demon.mask[tid] == 1.0
    
    # Check that non-target tokens are masked out
    assert demon.mask[0] == 0.0
    
    print("✓ DissonanceMLP masking OK")


def test_counter_dissonance_mlp_init():
    """Test CounterDissonanceMLP initialization."""
    print("Testing CounterDissonanceMLP initialization...")
    
    vocab_size = 100
    demon = CounterDissonanceMLP(vocab_size)
    
    assert demon.input_dim == vocab_size
    assert demon.hidden_dim == 2 * vocab_size
    # Input is concatenated: base + d1, so 2 * vocab_size
    assert demon.W1.shape == (2 * vocab_size, 2 * vocab_size)
    assert demon.W2.shape == (2 * vocab_size, vocab_size)
    
    print("✓ CounterDissonanceMLP initialization OK")


def test_counter_dissonance_mlp_forward():
    """Test CounterDissonanceMLP forward pass."""
    print("Testing CounterDissonanceMLP forward pass...")
    
    vocab_size = 100
    batch_size = 2
    
    demon = CounterDissonanceMLP(vocab_size)
    logits_base = np.random.randn(batch_size, vocab_size)
    logits_d1 = np.random.randn(batch_size, vocab_size)
    
    # Forward pass
    delta2 = demon(logits_base, logits_d1)
    
    assert delta2.shape == logits_base.shape
    print(f"  Base shape: {logits_base.shape}")
    print(f"  D1 shape: {logits_d1.shape}")
    print(f"  Output shape: {delta2.shape}")
    
    print("✓ CounterDissonanceMLP forward pass OK")


def test_compose_logits():
    """Test logits composition."""
    print("Testing compose_logits...")
    
    vocab_size = 100
    batch_size = 2
    
    logits_base = np.random.randn(batch_size, vocab_size)
    delta1 = np.random.randn(batch_size, vocab_size)
    delta2 = np.random.randn(batch_size, vocab_size)
    
    alpha1 = 0.3
    alpha2 = 0.2
    
    result = compose_logits(logits_base, delta1, delta2, alpha1, alpha2)
    
    expected = logits_base + alpha1 * delta1 + alpha2 * delta2
    
    assert np.allclose(result, expected)
    print(f"  Composition verified")
    
    print("✓ compose_logits OK")


def test_scalar_autograd():
    """Test micrograd-inspired Scalar autograd."""
    print("Testing Scalar autograd...")
    
    # Simple computation
    a = Scalar(2.0)
    b = Scalar(3.0)
    c = a * b
    d = c + Scalar(1.0)
    e = d.relu()
    
    # Backward
    e.backward()
    
    # Check gradients
    assert e.data == 7.0  # (2*3 + 1) = 7
    assert e.grad == 1.0  # Top-level grad
    assert d.grad == 1.0  # Relu passes through (positive)
    assert c.grad == 1.0  # Addition passes through
    
    print(f"  Forward: a={a.data}, b={b.data}, result={e.data}")
    print(f"  Backward: a.grad={a.grad}, b.grad={b.grad}")
    
    print("✓ Scalar autograd OK")


def run_all_tests():
    """Run all demon tests."""
    print("\n" + "="*60)
    print("LILITH DEMON TESTS")
    print("="*60 + "\n")
    
    try:
        test_dissonance_mlp_init()
        test_dissonance_mlp_forward()
        test_dissonance_mlp_masking()
        test_counter_dissonance_mlp_init()
        test_counter_dissonance_mlp_forward()
        test_compose_logits()
        test_scalar_autograd()
        
        print("\n" + "="*60)
        print("✓ ALL DEMON TESTS PASSED")
        print("="*60 + "\n")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
