"""
test_tour2.py

Tests for Tour 2 features:
- Word cloud / shard system
- Word statistics tracker
- Association engine
- Enhanced MathBrain
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tokenizer import Tokenizer
from lilith_words import ShardSystem, WordStatsTracker
from association import AssociationEngine
from mathbrain import MathBrain


def test_shard_system():
    """Test shard system initialization and basic operations."""
    print("\nTesting ShardSystem initialization...")
    vocab_size = 1000
    shard_system = ShardSystem(vocab_size, max_shards=5)
    assert len(shard_system.shards) > 0, "Should have at least one shard"
    assert shard_system.vocab_size == vocab_size
    print("✓ ShardSystem initialization OK")
    
    print("Testing shard token addition...")
    tokens = [10, 20, 30, 40, 50]
    shard_system.add_tokens(tokens)
    assert shard_system.total_tokens_seen == len(tokens)
    print(f"  Added {len(tokens)} tokens")
    print("✓ Shard token addition OK")
    
    print("Testing shard influence computation...")
    influence = shard_system.compute_shard_influence(strength=0.1)
    assert influence.shape == (vocab_size,)
    assert np.sum(influence) > 0, "Should have some influence"
    print(f"  Influence shape: {influence.shape}")
    print(f"  Non-zero elements: {np.sum(influence > 0)}")
    print("✓ Shard influence computation OK")
    
    print("Testing shard stats...")
    stats = shard_system.get_novelty_stats()
    assert 'mean_novelty' in stats
    assert 'active_shards' in stats
    print(f"  Mean novelty: {stats['mean_novelty']:.3f}")
    print(f"  Active shards: {stats['active_shards']}")
    print("✓ Shard stats OK")
    
    return True


def test_word_stats_tracker():
    """Test word statistics tracker."""
    print("\nTesting WordStatsTracker initialization...")
    vocab_size = 1000
    tracker = WordStatsTracker(vocab_size)
    assert tracker.vocab_size == vocab_size
    print("✓ WordStatsTracker initialization OK")
    
    print("Testing word tracking...")
    tracker.update_turn(1, "Normal")
    tokens = [10, 20, 30, 10, 20]  # Some repeats
    tracker.add_tokens(tokens, from_user=True)
    assert tracker.total_new_words > 0, "Should have detected new words"
    print(f"  New words detected: {tracker.total_new_words}")
    print("✓ Word tracking OK")
    
    print("Testing novelty metrics...")
    novelty_entropy = tracker.compute_novelty_entropy()
    diversity = tracker.get_vocabulary_diversity()
    assert novelty_entropy >= 0
    assert 0 <= diversity <= 1
    print(f"  Novelty entropy: {novelty_entropy:.3f}")
    print(f"  Vocabulary diversity: {diversity:.3f}")
    print("✓ Novelty metrics OK")
    
    print("Testing impact vector...")
    impact = tracker.get_impact_vector()
    assert impact.shape == (vocab_size,)
    print(f"  Impact vector shape: {impact.shape}")
    print(f"  Non-zero impacts: {np.sum(impact > 0)}")
    print("✓ Impact vector OK")
    
    return True


def test_association_engine():
    """Test association engine."""
    print("\nTesting AssociationEngine initialization...")
    vocab_size = 1000
    engine = AssociationEngine(vocab_size)
    assert engine.vocab_size == vocab_size
    assert len(engine.association_pools) > 0
    print(f"  Association pools: {list(engine.association_pools.keys())}")
    print("✓ AssociationEngine initialization OK")
    
    print("Testing context vector building...")
    user_tokens = [10, 20, 30]
    shard_novelty = np.random.rand(vocab_size) * 0.1
    metrics = {
        'novelty': 0.5,
        'entropy': 0.3,
        'trauma': 0.2
    }
    context = engine._build_context_vector(user_tokens, shard_novelty, metrics, "Normal")
    assert context.shape == (32,)
    print(f"  Context vector shape: {context.shape}")
    print("✓ Context vector building OK")
    
    print("Testing MLP forward pass...")
    scores = engine._forward_mlp(context)
    assert scores.shape == (vocab_size,)
    print(f"  Association scores shape: {scores.shape}")
    print("✓ MLP forward pass OK")
    
    return True


def test_mathbrain_supreme():
    """Test enhanced MathBrain supreme controller."""
    print("\nTesting MathBrain supreme controller...")
    vocab_size = 1000
    mathbrain = MathBrain(vocab_size)
    assert hasattr(mathbrain, 'metrics_history')
    assert hasattr(mathbrain, 'modulation_state')
    print("✓ MathBrain supreme controller initialized")
    
    print("Testing observe method...")
    metrics = {
        'novelty': 0.7,
        'entropy': 0.5,
        'trauma': 0.3,
        'new_words': 5,
        'diversity': 0.6
    }
    mathbrain.observe("test user input", "test reply", metrics)
    assert len(mathbrain.metrics_history) > 0
    assert 'novelty' in mathbrain.current_metrics
    print(f"  Metrics tracked: {len(mathbrain.current_metrics)}")
    print("✓ Observe method OK")
    
    print("Testing decide method...")
    modulation = mathbrain.decide()
    assert 'demon1_strength' in modulation
    assert 'demon2_strength' in modulation
    assert 'temperature' in modulation
    assert 'association_intensity' in modulation
    print(f"  Demon1 strength: {modulation['demon1_strength']:.2f}")
    print(f"  Temperature: {modulation['temperature']:.2f}")
    print(f"  Association intensity: {modulation['association_intensity']:.2f}")
    print("✓ Decide method OK")
    
    print("Testing supreme report...")
    report = mathbrain.get_supreme_report()
    assert len(report) > 0
    assert "Supreme Controller" in report
    print("✓ Supreme report OK")
    
    return True


def test_integration():
    """Test integration of Tour 2 components."""
    print("\nTesting Tour 2 component integration...")
    
    vocab_size = 1000
    
    # Initialize all components
    shard_system = ShardSystem(vocab_size, max_shards=5)
    word_stats = WordStatsTracker(vocab_size)
    association_engine = AssociationEngine(vocab_size)
    mathbrain = MathBrain(vocab_size)
    
    # Simulate a turn
    word_stats.update_turn(1, "Normal")
    
    # Add user tokens
    user_tokens = [10, 20, 30, 40, 50]
    word_stats.add_tokens(user_tokens, from_user=True)
    shard_system.add_tokens(user_tokens)
    
    # Get metrics
    shard_stats = shard_system.get_novelty_stats()
    novelty_entropy = word_stats.compute_novelty_entropy()
    
    # MathBrain observes
    mathbrain.observe(
        user_text="test",
        lilith_reply="",
        metrics={
            'novelty': shard_stats['mean_novelty'],
            'new_words': word_stats.get_new_words_this_turn(),
            'entropy': novelty_entropy
        }
    )
    
    # MathBrain decides
    modulation = mathbrain.decide()
    
    # Generate association
    shard_novelty = shard_system.compute_shard_influence(strength=0.0)
    # Note: Can't test full association without tokenizer
    
    print(f"  New words: {word_stats.get_new_words_this_turn()}")
    print(f"  Shards active: {shard_stats['active_shards']}")
    print(f"  Modulation ready: {modulation['demon1_strength']:.2f}")
    print("✓ Integration test OK")
    
    return True


def run_all_tests():
    """Run all Tour 2 tests."""
    print("\n" + "="*60)
    print("LILITH TOUR 2 TESTS")
    print("="*60)
    
    results = []
    
    try:
        results.append(("ShardSystem", test_shard_system()))
    except Exception as e:
        print(f"✗ ShardSystem test failed: {e}")
        results.append(("ShardSystem", False))
    
    try:
        results.append(("WordStatsTracker", test_word_stats_tracker()))
    except Exception as e:
        print(f"✗ WordStatsTracker test failed: {e}")
        results.append(("WordStatsTracker", False))
    
    try:
        results.append(("AssociationEngine", test_association_engine()))
    except Exception as e:
        print(f"✗ AssociationEngine test failed: {e}")
        results.append(("AssociationEngine", False))
    
    try:
        results.append(("MathBrain Supreme", test_mathbrain_supreme()))
    except Exception as e:
        print(f"✗ MathBrain Supreme test failed: {e}")
        results.append(("MathBrain Supreme", False))
    
    try:
        results.append(("Integration", test_integration()))
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        results.append(("Integration", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TOUR 2 TESTS PASSED")
        print("Language organism is growing.")
    else:
        print("✗ SOME TESTS FAILED")
        print("Fix the growth mechanisms.")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
