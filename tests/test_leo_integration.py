"""
Test suite for Leo integration layers.
Tests MetaLilith, Trauma, Overthinking, MathBrain, PhaseBridge.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from metalilith import MetaLilith
from trauma import TraumaLayer, create_trauma_visualization
from overthinking import Overthinking
from mathbrain import MathBrain, SymbolicReasoning
from phase4_bridges import PhaseBridge, Phase, create_phase_visualization


class MockTokenizer:
    """Mock tokenizer for testing."""
    def __init__(self):
        self.vocab = ['a', 'b', 'c', 'd', 'e']
        self.bos_id = 1
        self.eos_id = 2
    
    def encode(self, text, add_bos=True, add_eos=False):
        return [1, 3, 4, 5]
    
    def decode(self, ids):
        return "test"


class MockModel:
    """Mock model for testing."""
    def __call__(self, inputs, pos):
        # Return mock logits
        batch_size = inputs.shape[0]
        vocab_size = 100
        return np.random.randn(batch_size, 1, vocab_size)


def test_trauma_layer_init():
    """Test TraumaLayer initialization."""
    print("Testing TraumaLayer initialization...")
    
    tokenizer = MockTokenizer()
    vocab_size = 100
    
    trauma = TraumaLayer(tokenizer, vocab_size)
    
    assert trauma.vocab_size == vocab_size
    assert len(trauma.identity_token_ids) > 0
    assert len(trauma.reality_token_ids) > 0
    
    print(f"  Identity tokens: {len(trauma.identity_token_ids)}")
    print(f"  Reality tokens: {len(trauma.reality_token_ids)}")
    print("✓ TraumaLayer initialization OK")


def test_trauma_measurement():
    """Test trauma dissonance measurement."""
    print("Testing trauma measurement...")
    
    tokenizer = MockTokenizer()
    vocab_size = 100
    trauma = TraumaLayer(tokenizer, vocab_size)
    
    # Mock logits
    logits = np.random.randn(1, vocab_size)
    
    score = trauma.measure_dissonance(logits)
    
    assert 0.0 <= score <= 1.0
    assert len(trauma.trauma_scores) == 1
    
    print(f"  Trauma score: {score:.3f}")
    print("✓ Trauma measurement OK")


def test_trauma_amplification():
    """Test trauma amplification."""
    print("Testing trauma amplification...")
    
    tokenizer = MockTokenizer()
    vocab_size = 100
    trauma = TraumaLayer(tokenizer, vocab_size)
    
    logits = np.random.randn(1, vocab_size)
    trauma_score = 0.8  # Higher trauma for more visible effect
    
    amplified = trauma.apply_trauma_amplification(logits, trauma_score, amplify=True)
    not_amplified = trauma.apply_trauma_amplification(logits, trauma_score, amplify=False)
    
    assert amplified.shape == logits.shape
    # When amplify=False, should return unchanged
    assert np.array_equal(not_amplified, logits)
    
    print("✓ Trauma amplification OK")


def test_trauma_visualization():
    """Test trauma visualization."""
    print("Testing trauma visualization...")
    
    scores = [0.2, 0.4, 0.6, 0.8, 0.5, 0.3]
    viz = create_trauma_visualization(scores)
    
    assert len(viz) > 0
    assert "Trauma over time" in viz
    
    print("✓ Trauma visualization OK")


def test_mathbrain_init():
    """Test MathBrain initialization."""
    print("Testing MathBrain initialization...")
    
    vocab_size = 100
    mathbrain = MathBrain(vocab_size)
    
    assert mathbrain.vocab_size == vocab_size
    assert len(mathbrain.number_tokens) == 0  # Not set yet
    
    tokenizer = MockTokenizer()
    mathbrain.set_token_categories(tokenizer)
    
    print(f"  Number tokens: {len(mathbrain.number_tokens)}")
    print(f"  Logic tokens: {len(mathbrain.logic_tokens)}")
    print("✓ MathBrain initialization OK")


def test_mathbrain_analysis():
    """Test MathBrain logits analysis."""
    print("Testing MathBrain analysis...")
    
    vocab_size = 100
    mathbrain = MathBrain(vocab_size)
    tokenizer = MockTokenizer()
    mathbrain.set_token_categories(tokenizer)
    
    logits = np.random.randn(1, vocab_size)
    analysis = mathbrain.analyze_logits(logits)
    
    assert 'number_mass' in analysis
    assert 'logic_mass' in analysis
    assert 'rational_total' in analysis
    assert 'emotional_mass' in analysis
    
    print(f"  Rational total: {analysis['rational_total']:.3f}")
    print("✓ MathBrain analysis OK")


def test_mathbrain_influence():
    """Test MathBrain rational influence."""
    print("Testing MathBrain influence...")
    
    vocab_size = 100
    mathbrain = MathBrain(vocab_size)
    tokenizer = MockTokenizer()
    mathbrain.set_token_categories(tokenizer)
    
    logits = np.random.randn(1, vocab_size)
    
    stabilized = mathbrain.apply_rational_influence(logits, strength=0.2, mode='stabilize')
    destabilized = mathbrain.apply_rational_influence(logits, strength=0.2, mode='destabilize')
    
    assert stabilized.shape == logits.shape
    assert destabilized.shape == logits.shape
    
    print("✓ MathBrain influence OK")


def test_phase_bridge_init():
    """Test PhaseBridge initialization."""
    print("Testing PhaseBridge initialization...")
    
    bridge = PhaseBridge()
    
    assert bridge.current_phase == Phase.NORMAL
    assert len(bridge.phase_configs) == 4
    
    print(f"  Current phase: {bridge.current_phase.name}")
    print("✓ PhaseBridge initialization OK")


def test_phase_transitions():
    """Test phase transitions."""
    print("Testing phase transitions...")
    
    bridge = PhaseBridge()
    
    # Manual transition
    bridge.transition_to(Phase.DARK_POETIC, reason="test")
    assert bridge.current_phase == Phase.DARK_POETIC
    assert bridge.transitions == 1
    
    # Auto transition based on context
    context = {
        'trauma_score': 0.8,
        'rationality': 0.2,
        'turn_count': 5
    }
    
    transitioned = bridge.auto_transition(context)
    
    print(f"  Auto transition occurred: {transitioned}")
    print(f"  Current phase: {bridge.current_phase.name}")
    print("✓ Phase transitions OK")


def test_phase_configuration():
    """Test phase configuration retrieval."""
    print("Testing phase configuration...")
    
    bridge = PhaseBridge()
    
    config = bridge.get_config()
    
    assert 'demon1_alpha' in config
    assert 'demon2_alpha' in config
    assert 'temperature' in config
    
    alpha1, alpha2 = bridge.get_demon_alphas()
    temp = bridge.get_temperature()
    
    assert alpha1 == config['demon1_alpha']
    assert temp == config['temperature']
    
    print(f"  Alpha1: {alpha1}, Alpha2: {alpha2}, Temp: {temp}")
    print("✓ Phase configuration OK")


def test_phase_visualization():
    """Test phase visualization."""
    print("Testing phase visualization...")
    
    history = [Phase.NORMAL, Phase.DARK_POETIC, Phase.META_RIVALRY, Phase.NORMAL]
    viz = create_phase_visualization(history)
    
    assert len(viz) > 0
    assert "Phase timeline" in viz
    
    print("✓ Phase visualization OK")


def test_symbolic_reasoning():
    """Test symbolic reasoning."""
    print("Testing symbolic reasoning...")
    
    symbolic = SymbolicReasoning()
    
    # Test pattern detection
    tokens = [1, 2, 3, 3, 3]
    pattern = symbolic.detect_pattern(tokens)
    
    assert pattern == "repetition_detected"
    
    # Test alternation
    tokens_alt = [1, 2, 1, 2, 1]
    pattern_alt = symbolic.detect_pattern(tokens_alt)
    
    assert pattern_alt == "alternation_detected"
    
    print(f"  Detected patterns: {pattern}, {pattern_alt}")
    print("✓ Symbolic reasoning OK")


def run_all_tests():
    """Run all Leo integration tests."""
    print("\n" + "="*60)
    print("LILITH LEO INTEGRATION TESTS")
    print("="*60 + "\n")
    
    try:
        test_trauma_layer_init()
        test_trauma_measurement()
        test_trauma_amplification()
        test_trauma_visualization()
        test_mathbrain_init()
        test_mathbrain_analysis()
        test_mathbrain_influence()
        test_phase_bridge_init()
        test_phase_transitions()
        test_phase_configuration()
        test_phase_visualization()
        test_symbolic_reasoning()
        
        print("\n" + "="*60)
        print("✓ ALL LEO INTEGRATION TESTS PASSED")
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
