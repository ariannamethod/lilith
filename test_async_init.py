#!/usr/bin/env python3
"""
Quick test that async architecture initializes correctly
"""
import sys
from config import ModelArgs
from chat import LilithChatFull

print("🔥 Testing async architecture initialization...")
print()

# Initialize
model_args = ModelArgs()

try:
    chat = LilithChatFull(
        './lilith_weights/stories15M.model.npz',
        './lilith_weights/tokenizer.model.np',
        './lilith_config.json',
        model_args,
        enable_leo=True,
        enable_demons=True,
        enable_postprocess=True,
        show_meta=False,
        show_ripples=False,
        debug=True
    )
    print("✓ LilithChatFull initialized successfully")
    print()

    # Check Tour 3 components
    assert chat.mood_drift is not None, "MoodDrift not initialized!"
    assert chat.meta_controller is not None, "MetaController not initialized!"
    assert chat.assoc_memory is not None, "AssociationMemory not initialized!"
    print("✓ Tour 3 emergent processes initialized:")
    print("  - MoodDrift ✓")
    print("  - MetaController ✓")
    print("  - AssociationMemory ✓")
    print()

    # Check async methods exist
    assert hasattr(chat, 'lilith_feel'), "lilith_feel method missing!"
    assert hasattr(chat, 'lilith_speak'), "lilith_speak method missing!"
    assert hasattr(chat, 'respond'), "respond method missing!"
    print("✓ Async methods present:")
    print("  - async lilith_feel() ✓")
    print("  - async lilith_speak() ✓")
    print("  - async respond() ✓")
    print()

    # Check metalilith and overthinking have async methods
    import inspect
    assert inspect.iscoroutinefunction(chat.metalilith.generate_shadow_thought), \
        "generate_shadow_thought is not async!"
    assert inspect.iscoroutinefunction(chat.overthinking.generate_ripple), \
        "generate_ripple is not async!"
    assert inspect.iscoroutinefunction(chat.overthinking.process_interaction), \
        "process_interaction is not async!"
    print("✓ Leo layers are async:")
    print("  - metalilith.generate_shadow_thought() is coroutine ✓")
    print("  - overthinking.generate_ripple() is coroutine ✓")
    print("  - overthinking.process_interaction() is coroutine ✓")
    print()

    print("🔥"*40)
    print("✓✓✓ ASYNC ARCHITECTURE VERIFIED! ✓✓✓")
    print("All emergent processes ready for parallel execution!")
    print("🔥"*40)

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
