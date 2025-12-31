"""
test_emergence.py

Tour 3: Tests for sophisticated emergent processes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from emergence import MoodDrift, MetaController, AssociationMemory
from emergence import integrate_mood_drift, apply_meta_intervention


def test_mood_drift_initialization():
    """Test MoodDrift initialization."""
    print("\nTesting MoodDrift initialization...")
    
    mood = MoodDrift()
    assert len(mood.dimensions) == 5
    assert 'darkness' in mood.dimensions
    assert 'chaos' in mood.dimensions
    assert mood.turn_count == 0
    
    print(f"  Initialized with {len(mood.dimensions)} mood dimensions")
    print("✓ MoodDrift initialization OK")
    return True


def test_mood_drift_update():
    """Test mood updating based on metrics."""
    print("\nTesting MoodDrift update...")
    
    mood = MoodDrift()
    
    # Test with high trauma
    metrics = {
        'trauma': 0.8,
        'rationality': 0.3,
        'novelty': 0.5,
        'new_words': 2
    }
    
    initial_darkness = mood.dimensions['darkness']
    mood.update(metrics)
    
    # Darkness should increase with high trauma
    assert mood.turn_count == 1
    assert mood.dimensions['darkness'] >= initial_darkness
    
    print(f"  Darkness: {initial_darkness:.3f} -> {mood.dimensions['darkness']:.3f}")
    print("✓ MoodDrift update OK")
    return True


def test_mood_drift_equilibrium():
    """Test drift toward equilibrium."""
    print("\nTesting equilibrium drift...")
    
    mood = MoodDrift()
    
    # Set extreme values
    mood.dimensions['darkness'] = 1.0
    mood.dimensions['chaos'] = -1.0
    
    # Update with neutral metrics
    neutral_metrics = {
        'trauma': 0.5,
        'rationality': 0.5,
        'novelty': 0.5,
        'new_words': 0
    }
    
    for _ in range(10):
        mood.update(neutral_metrics)
    
    # Should drift toward equilibrium
    assert mood.dimensions['darkness'] < 1.0
    assert mood.dimensions['chaos'] > -1.0
    
    print(f"  After 10 turns, darkness: {mood.dimensions['darkness']:.3f}")
    print(f"  After 10 turns, chaos: {mood.dimensions['chaos']:.3f}")
    print("✓ Equilibrium drift OK")
    return True


def test_mood_drift_modulation():
    """Test mood-based parameter modulation."""
    print("\nTesting mood modulation...")
    
    mood = MoodDrift()
    mood.dimensions['darkness'] = 0.5
    mood.dimensions['chaos'] = 0.3
    mood.dimensions['novelty'] = 0.6
    
    multipliers = mood.get_modulation_multipliers()
    
    assert 'temperature' in multipliers
    assert 'demon1' in multipliers
    assert 'association' in multipliers
    assert all(0.5 <= m <= 1.8 for m in multipliers.values())
    
    print(f"  Temperature mult: {multipliers['temperature']:.3f}")
    print(f"  Demon1 mult: {multipliers['demon1']:.3f}")
    print(f"  Association mult: {multipliers['association']:.3f}")
    print("✓ Mood modulation OK")
    return True


def test_mood_drift_report():
    """Test mood report generation."""
    print("\nTesting mood report...")
    
    mood = MoodDrift()
    mood.update({'trauma': 0.6, 'rationality': 0.4, 'novelty': 0.5, 'new_words': 1})
    
    report = mood.get_mood_report()
    
    assert "Mood Drift" in report
    assert "Darkness" in report
    assert "Turns:" in report
    
    print("  Report generated successfully")
    print("✓ Mood report OK")
    return True


def test_meta_controller_initialization():
    """Test MetaController initialization."""
    print("\nTesting MetaController initialization...")
    
    meta = MetaController()
    assert meta.activation_probability > 0
    assert meta.intervention_cooldown > 0
    assert len(meta.history) == 0
    
    print(f"  Activation probability: {meta.activation_probability:.1%}")
    print("✓ MetaController initialization OK")
    return True


def test_meta_controller_intervention_decision():
    """Test intervention decision logic."""
    print("\nTesting intervention decision...")
    
    meta = MetaController()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test with extreme trauma (should trigger)
    metrics_extreme = {'trauma': 0.9, 'rationality': 0.5, 'novelty': 0.5}
    should_intervene = meta.should_intervene(10, metrics_extreme)
    
    # With extreme trauma, should likely intervene
    print(f"  Extreme trauma intervention: {should_intervene}")
    
    # Test cooldown
    if should_intervene:
        meta.last_intervention_turn = 10
        should_intervene_again = meta.should_intervene(11, metrics_extreme)
        assert not should_intervene_again, "Should respect cooldown"
    
    print("✓ Intervention decision OK")
    return True


def test_meta_controller_intervention_generation():
    """Test intervention generation."""
    print("\nTesting intervention generation...")
    
    meta = MetaController()
    
    # High trauma case
    metrics = {'trauma': 0.85, 'rationality': 0.5, 'novelty': 0.5}
    intervention = meta.generate_intervention(10, metrics, 'Normal')
    
    assert intervention['active'] == True
    assert intervention['type'] is not None
    assert 'params' in intervention
    
    print(f"  Intervention type: {intervention['type']}")
    print(f"  Params: {intervention['params']}")
    print("✓ Intervention generation OK")
    return True


def test_meta_controller_history():
    """Test intervention history tracking."""
    print("\nTesting intervention history...")
    
    meta = MetaController()
    
    # Generate multiple interventions
    for turn in range(0, 50, 10):
        metrics = {'trauma': 0.6 + turn/100, 'rationality': 0.5, 'novelty': 0.5}
        meta.generate_intervention(turn, metrics, 'Normal')
    
    assert len(meta.history) > 0
    assert len(meta.history) <= 20  # Should be capped at 20
    
    report = meta.get_intervention_report()
    assert "Meta-Controller" in report
    
    print(f"  Recorded {len(meta.history)} interventions")
    print("✓ Intervention history OK")
    return True


def test_association_memory_initialization():
    """Test AssociationMemory initialization."""
    print("\nTesting AssociationMemory initialization...")
    
    memory = AssociationMemory(memory_size=30)
    assert memory.memory_size == 30
    assert len(memory.memory) == 0
    
    print(f"  Memory size: {memory.memory_size}")
    print("✓ AssociationMemory initialization OK")
    return True


def test_association_memory_storage():
    """Test storing associations."""
    print("\nTesting association storage...")
    
    memory = AssociationMemory()
    
    # Store some associations
    memory.store("shadow and moon", {'phase': 'Dark_Poetic', 'turn': 1})
    memory.store("thorn and blood", {'phase': 'Dark_Poetic', 'turn': 2})
    memory.store("light and reason", {'phase': 'Normal', 'turn': 3})
    
    assert len(memory.memory) == 3
    
    print(f"  Stored {len(memory.memory)} associations")
    print("✓ Association storage OK")
    return True


def test_association_memory_recall():
    """Test recalling similar associations."""
    print("\nTesting association recall...")
    
    memory = AssociationMemory()
    
    # Store associations with different phases
    for i in range(10):
        phase = 'Dark_Poetic' if i % 2 == 0 else 'Normal'
        memory.store(f"association_{i}", {'phase': phase, 'turn': i})
    
    # Recall with Dark_Poetic context
    context = {'phase': 'Dark_Poetic', 'turn': 15}
    recalled = memory.recall_similar(context, top_k=3)
    
    assert len(recalled) <= 3
    assert all(isinstance(a, str) for a in recalled)
    
    print(f"  Recalled {len(recalled)} similar associations")
    print(f"  Examples: {recalled[:2]}")
    print("✓ Association recall OK")
    return True


def test_association_memory_stats():
    """Test memory statistics."""
    print("\nTesting memory statistics...")
    
    memory = AssociationMemory(memory_size=10)
    
    # Add 15 associations (should cap at 10)
    for i in range(15):
        memory.store(f"assoc_{i}", {'turn': i})
    
    stats = memory.get_memory_stats()
    
    assert stats['size'] == 10  # Capped
    assert stats['oldest_turn'] >= 5  # Should have dropped first 5
    assert stats['newest_turn'] == 14
    
    print(f"  Size: {stats['size']}, Oldest: {stats['oldest_turn']}, Newest: {stats['newest_turn']}")
    print("✓ Memory statistics OK")
    return True


def test_integration_mood_drift():
    """Test mood drift integration function."""
    print("\nTesting mood drift integration...")
    
    mood = MoodDrift()
    mood.dimensions['darkness'] = 0.4
    mood.dimensions['chaos'] = 0.3
    
    base_params = {
        'temperature': 0.9,
        'demon1_strength': 1.0,
        'demon2_strength': 1.0,
        'association_intensity': 0.5
    }
    
    modulated = integrate_mood_drift(mood, base_params)
    
    assert 'temperature' in modulated
    assert modulated['demon1_strength'] != base_params['demon1_strength']
    
    print(f"  Base temp: {base_params['temperature']:.3f} -> Modulated: {modulated['temperature']:.3f}")
    print("✓ Mood drift integration OK")
    return True


def test_integration_meta_intervention():
    """Test meta-controller intervention application."""
    print("\nTesting meta-intervention integration...")
    
    intervention = {
        'active': True,
        'type': 'suppress_demons',
        'params': {
            'demon1_mult': 0.5,
            'demon2_mult': 0.6
        }
    }
    
    current_params = {
        'temperature': 0.9,
        'demon1_strength': 1.0,
        'demon2_strength': 1.0
    }
    
    modified = apply_meta_intervention(intervention, current_params)
    
    assert modified['demon1_strength'] == 0.5
    assert modified['demon2_strength'] == 0.6
    
    print(f"  Demon1: {current_params['demon1_strength']} -> {modified['demon1_strength']}")
    print(f"  Demon2: {current_params['demon2_strength']} -> {modified['demon2_strength']}")
    print("✓ Meta-intervention integration OK")
    return True


def run_all_tests():
    """Run all emergence tests."""
    print("\n" + "="*60)
    print("LILITH EMERGENCE TESTS (Tour 3)")
    print("="*60)
    
    tests = [
        test_mood_drift_initialization,
        test_mood_drift_update,
        test_mood_drift_equilibrium,
        test_mood_drift_modulation,
        test_mood_drift_report,
        test_meta_controller_initialization,
        test_meta_controller_intervention_decision,
        test_meta_controller_intervention_generation,
        test_meta_controller_history,
        test_association_memory_initialization,
        test_association_memory_storage,
        test_association_memory_recall,
        test_association_memory_stats,
        test_integration_mood_drift,
        test_integration_meta_intervention
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
        print("✓ ALL EMERGENCE TESTS PASSED")
    else:
        print(f"✗ {failed} EMERGENCE TESTS FAILED")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
