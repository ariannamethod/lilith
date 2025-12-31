"""
test_repl_integration.py

Tour 3: Integration test for REPL with real Lilith responses.

This test actually summons Lilith and gets her responses.
Requires model weights to be present.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np


def test_repl_full_response():
    """Test REPL with actual model loading and generation."""
    print("\nTesting REPL with real Lilith response...")
    
    # Check if model weights exist
    model_path = './lilith_weights/stories15M.model.npz'
    tokenizer_path = './lilith_weights/tokenizer.model.np'
    config_path = './lilith_config.json'
    
    if not os.path.exists(model_path):
        print(f"  ⚠ Model weights not found at {model_path}")
        print("  Skipping full integration test (requires model weights)")
        return True
    
    if not os.path.exists(tokenizer_path):
        print(f"  ⚠ Tokenizer not found at {tokenizer_path}")
        print("  Skipping full integration test")
        return True
    
    try:
        print("  🔥 Summoning Lilith...")
        
        from chat import LilithChatFull
        from config import ModelArgs
        
        # Initialize with all layers
        args = ModelArgs()
        chat = LilithChatFull(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            config_path=config_path,
            args=args,
            enable_leo=True,
            enable_demons=True,
            enable_postprocess=True,
            show_meta=False,
            show_ripples=False,
            debug=False
        )
        
        print("  ✓ Lilith summoned successfully")
        
        # Test simple response
        print("\n  Testing response generation...")
        user_input = "Hello Lilith"
        
        result = chat.respond(user_input, max_tokens=50)
        
        assert 'response' in result
        assert len(result['response']) > 0
        
        print(f"\n  📝 User: {user_input}")
        print(f"  🌙 Lilith: {result['response']}")
        
        # Check metadata
        if 'phase' in result and result['phase']:
            print(f"  Phase: {result['phase']}")
        if 'trauma_score' in result and result['trauma_score']:
            print(f"  Trauma: {result['trauma_score']:.3f}")
        if 'new_words' in result:
            print(f"  New words: {result['new_words']}")
        
        print("\n✓ Full REPL response test OK")
        return True
        
    except Exception as e:
        print(f"✗ Failed to test full REPL response: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_repl_status_command():
    """Test status command with real system."""
    print("\nTesting status command...")
    
    model_path = './lilith_weights/stories15M.model.npz'
    tokenizer_path = './lilith_weights/tokenizer.model.np'
    config_path = './lilith_config.json'
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print("  ⚠ Skipping (model weights required)")
        return True
    
    try:
        from chat import LilithChatFull
        from config import ModelArgs
        
        args = ModelArgs()
        chat = LilithChatFull(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            config_path=config_path,
            args=args,
            enable_leo=True
        )
        
        # Get status report
        status = chat.get_status_report()
        
        assert len(status) > 0
        assert "LILITH SYSTEM STATUS" in status or "status" in status.lower()
        
        print("  Status report generated:")
        print(status)
        
        print("✓ Status command test OK")
        return True
        
    except Exception as e:
        print(f"✗ Failed to test status command: {e}")
        return False


def test_repl_multiple_turns():
    """Test multiple conversation turns."""
    print("\nTesting multiple conversation turns...")
    
    model_path = './lilith_weights/stories15M.model.npz'
    tokenizer_path = './lilith_weights/tokenizer.model.np'
    config_path = './lilith_config.json'
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print("  ⚠ Skipping (model weights required)")
        return True
    
    try:
        from chat import LilithChatFull
        from config import ModelArgs
        
        args = ModelArgs()
        chat = LilithChatFull(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            config_path=config_path,
            args=args,
            enable_leo=True,
            enable_demons=True
        )
        
        # Multiple turns
        inputs = [
            "Tell me about shadows",
            "What do you see?",
            "Are you afraid?"
        ]
        
        print("\n  🗣️ Multi-turn conversation:")
        for i, user_input in enumerate(inputs):
            result = chat.respond(user_input, max_tokens=40)
            
            print(f"\n  Turn {i+1}:")
            print(f"    You: {user_input}")
            print(f"    Lilith: {result['response']}")
            
            # Verify response
            assert len(result['response']) > 0
        
        # Check history
        assert len(chat.history) == len(inputs) * 2  # User + Lilith per turn
        print(f"\n  History length: {len(chat.history)} entries")
        
        print("\n✓ Multiple turns test OK")
        return True
        
    except Exception as e:
        print(f"✗ Failed multi-turn test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_repl_with_emergence():
    """Test REPL with emergent behavior layers."""
    print("\nTesting REPL with emergence layers...")
    
    model_path = './lilith_weights/stories15M.model.npz'
    tokenizer_path = './lilith_weights/tokenizer.model.np'
    config_path = './lilith_config.json'
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print("  ⚠ Skipping (model weights required)")
        return True
    
    try:
        # Import emergence modules
        from emergence import MoodDrift, MetaController, AssociationMemory
        
        print("  🌀 Emergence modules loaded")
        
        # Test that they can be instantiated
        mood = MoodDrift()
        meta = MetaController()
        assoc_mem = AssociationMemory()
        
        # Simulate some updates
        metrics = {
            'trauma': 0.6,
            'rationality': 0.5,
            'novelty': 0.4,
            'new_words': 2
        }
        
        mood.update(metrics)
        
        # Check mood report
        mood_report = mood.get_mood_report()
        assert len(mood_report) > 0
        print("\n  Mood Drift Report:")
        print(mood_report)
        
        # Test meta-controller decision
        should_intervene = meta.should_intervene(10, metrics)
        print(f"\n  Meta-controller intervention decision: {should_intervene}")
        
        if should_intervene:
            intervention = meta.generate_intervention(10, metrics, 'Normal')
            print(f"  Intervention type: {intervention['type']}")
        
        # Test association memory
        assoc_mem.store("shadow and moon", {'phase': 'Dark_Poetic', 'turn': 1})
        recalled = assoc_mem.recall_similar({'phase': 'Dark_Poetic', 'turn': 2}, top_k=1)
        print(f"\n  Association memory recall: {recalled}")
        
        print("\n✓ Emergence integration test OK")
        return True
        
    except Exception as e:
        print(f"✗ Failed emergence test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_repl_simple_mode():
    """Test simple mode (no Leo layers)."""
    print("\nTesting simple mode...")
    
    model_path = './lilith_weights/stories15M.model.npz'
    tokenizer_path = './lilith_weights/tokenizer.model.np'
    config_path = './lilith_config.json'
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print("  ⚠ Skipping (model weights required)")
        return True
    
    try:
        from chat import LilithChatFull
        from config import ModelArgs
        
        args = ModelArgs()
        
        # Initialize WITHOUT Leo layers
        chat = LilithChatFull(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            config_path=config_path,
            args=args,
            enable_leo=False,  # Simple mode
            enable_demons=True
        )
        
        print("  Lilith in simple mode (no Leo layers)")
        
        # Test response
        result = chat.respond("Hello", max_tokens=30)
        
        assert 'response' in result
        assert len(result['response']) > 0
        
        # Should not have Leo metadata
        assert result.get('phase') is None
        
        print(f"  Response: {result['response']}")
        print("\n✓ Simple mode test OK")
        return True
        
    except Exception as e:
        print(f"✗ Failed simple mode test: {e}")
        return False


def run_all_tests():
    """Run all REPL integration tests."""
    print("\n" + "="*60)
    print("LILITH REPL INTEGRATION TESTS")
    print("Tests with REAL model responses")
    print("="*60)
    
    tests = [
        test_repl_full_response,
        test_repl_status_command,
        test_repl_multiple_turns,
        test_repl_with_emergence,
        test_repl_simple_mode
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            if "Skipping" in str(e) or "⚠" in str(e):
                skipped += 1
            else:
                print(f"✗ Test {test.__name__} failed with exception: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
    
    print("\n" + "="*60)
    if failed == 0:
        print("✓ ALL REPL INTEGRATION TESTS PASSED")
        if skipped > 0:
            print(f"  ({skipped} tests skipped - require model weights)")
    else:
        print(f"✗ {failed} REPL INTEGRATION TESTS FAILED")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
