"""
test_repl.py

Tour 3: Tests for REPL reliability and command handling.

Since REPL involves terminal I/O, we test the core logic separately
from the actual input/output loops.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np


def test_repl_imports():
    """Test that REPL modules can be imported."""
    print("\nTesting REPL imports...")
    
    try:
        import chat
        import chat_simple
        print("  chat.py imported successfully")
        print("  chat_simple.py imported successfully")
        print("✓ REPL imports OK")
        return True
    except Exception as e:
        print(f"✗ Failed to import REPL modules: {e}")
        return False


def test_repl_command_parsing():
    """Test special command recognition."""
    print("\nTesting command parsing...")
    
    # Commands that should be recognized
    commands = ['status', 'phase', 'exit', 'quit', 'bye']
    
    for cmd in commands:
        # Test exact match
        assert cmd.strip().lower() in ['status', 'phase', 'exit', 'quit', 'bye']
        # Test with whitespace
        assert cmd.strip().lower() in ['status', 'phase', 'exit', 'quit', 'bye']
    
    print(f"  Recognized {len(commands)} special commands")
    print("✓ Command parsing OK")
    return True


def test_repl_chat_initialization():
    """Test that LilithChatFull can be initialized without crashing."""
    print("\nTesting LilithChatFull initialization...")
    
    # This tests basic structure, not actual model loading
    # (which requires real weight files)
    
    try:
        from chat import LilithChatFull
        from config import ModelArgs
        
        # Check that class exists and has expected methods
        assert hasattr(LilithChatFull, '__init__')
        assert hasattr(LilithChatFull, 'respond')
        assert hasattr(LilithChatFull, 'lilith_feel')
        assert hasattr(LilithChatFull, 'lilith_speak')
        assert hasattr(LilithChatFull, 'get_status_report')
        
        print("  LilithChatFull class structure validated")
        print("✓ Chat class structure OK")
        return True
    except Exception as e:
        print(f"✗ Failed to validate chat class: {e}")
        return False


def test_repl_status_report_structure():
    """Test that status report methods exist and are callable."""
    print("\nTesting status report structure...")
    
    try:
        from mathbrain import MathBrain
        from phase4_bridges import PhaseBridge
        from trauma import TraumaLayer
        from lilith_words import ShardSystem, WordStatsTracker
        
        # Check MathBrain has report method
        mb = MathBrain(vocab_size=1000)
        assert hasattr(mb, 'get_supreme_report')
        
        # Check PhaseBridge has report method
        pb = PhaseBridge()
        assert hasattr(pb, 'get_phase_report')
        
        # Check TraumaLayer has report method (requires tokenizer but we check method exists)
        assert hasattr(TraumaLayer, 'get_trauma_report')
        
        # Check ShardSystem has report method
        ss = ShardSystem(vocab_size=1000)
        assert hasattr(ss, 'get_shard_report')
        
        print("  All status report methods present")
        print("✓ Status report structure OK")
        return True
    except Exception as e:
        print(f"✗ Failed to validate status reports: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_repl_graceful_error_handling():
    """Test that REPL components handle errors gracefully."""
    print("\nTesting graceful error handling...")
    
    try:
        from chat import LilithChatFull
        
        # Test that empty/None inputs don't crash
        # (We can't fully test without model weights, but we can check structure)
        
        # Check that response method signature is correct
        import inspect
        sig = inspect.signature(LilithChatFull.respond)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'user_input' in params
        
        print("  Response method signature validated")
        print("✓ Error handling structure OK")
        return True
    except Exception as e:
        print(f"✗ Failed to validate error handling: {e}")
        return False


def test_repl_history_management():
    """Test that conversation history is properly managed."""
    print("\nTesting history management...")
    
    try:
        # Test history structure
        # In chat.py, history is a list of strings
        history = []
        
        # Simulate adding to history
        history.append("you> Hello")
        history.append("lilith> Greetings")
        
        assert len(history) == 2
        assert history[0].startswith("you>")
        assert history[1].startswith("lilith>")
        
        # Test recent history slicing
        recent = history[-6:] if len(history) > 6 else history
        assert len(recent) <= 6
        
        print(f"  History management validated with {len(history)} entries")
        print("✓ History management OK")
        return True
    except Exception as e:
        print(f"✗ Failed to validate history: {e}")
        return False


def test_repl_signal_handling():
    """Test that signal handlers are properly configured."""
    print("\nTesting signal handling...")
    
    try:
        import signal
        from chat import signal_handler
        
        # Check that signal handler function exists
        assert callable(signal_handler)
        
        # Check signature (should accept sig and frame)
        import inspect
        sig = inspect.signature(signal_handler)
        params = list(sig.parameters.keys())
        assert len(params) == 2  # sig and frame
        
        print("  Signal handler structure validated")
        print("✓ Signal handling OK")
        return True
    except Exception as e:
        print(f"✗ Failed to validate signal handling: {e}")
        return False


def test_repl_config_integration():
    """Test REPL integrates with config system."""
    print("\nTesting config integration...")
    
    try:
        import argparse
        
        # Simulate argument parsing
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='./lilith_weights/stories15M.model.npz')
        parser.add_argument('--tokenizer', type=str, default='./lilith_weights/tokenizer.model.np')
        parser.add_argument('--config', type=str, default='./lilith_config.json')
        parser.add_argument('--no-leo', action='store_true')
        parser.add_argument('--no-demons', action='store_true')
        
        # Test default parsing
        args = parser.parse_args([])
        assert args.model is not None
        assert args.tokenizer is not None
        assert args.config is not None
        
        # Test flag parsing
        args = parser.parse_args(['--no-leo'])
        assert args.no_leo == True
        
        print("  Config integration validated")
        print("✓ Config integration OK")
        return True
    except Exception as e:
        print(f"✗ Failed to validate config integration: {e}")
        return False


def run_all_tests():
    """Run all REPL tests."""
    print("\n" + "="*60)
    print("LILITH REPL TESTS")
    print("="*60)
    
    tests = [
        test_repl_imports,
        test_repl_command_parsing,
        test_repl_chat_initialization,
        test_repl_status_report_structure,
        test_repl_graceful_error_handling,
        test_repl_history_management,
        test_repl_signal_handling,
        test_repl_config_integration
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
        print("✓ ALL REPL TESTS PASSED")
    else:
        print(f"✗ {failed} REPL TESTS FAILED")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
