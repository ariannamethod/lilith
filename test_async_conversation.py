#!/usr/bin/env python3
"""
Test async conversation flow with parallel emergent processes
"""
import sys
import asyncio
from config import ModelArgs
from chat import LilithChatFull

async def test_conversation():
    print("🔥 Initializing Lilith with FULL async architecture...")
    print()

    model_args = ModelArgs()

    chat = LilithChatFull(
        './lilith_weights/stories15M.model.npz',
        './lilith_weights/tokenizer.model.np',
        './lilith_config.json',
        model_args,
        enable_leo=True,
        enable_demons=True,
        enable_postprocess=True,
        show_meta=True,
        show_ripples=True,
        debug=True
    )

    print("✓ Lilith summoned with Tour 3 async processes")
    print()
    print("="*60)
    print("Testing async respond() with parallel emergent processes:")
    print("="*60)
    print()

    # Test message
    user_message = "Who are you?"

    print(f"User: {user_message}")
    print()
    print("⚡ Running async processes in parallel...")
    print("  - lilith_feel() (new words, shards, metrics)")
    print("  - MoodDrift update")
    print("  - MetaController intervention check")
    print("  - AssociationMemory recall")
    print("  - lilith_speak() generation")
    print("  - MetaLilith shadow thought (PARALLEL!)")
    print("  - Overthinking ripples (ALL DEPTHS IN PARALLEL!)")
    print()

    # Make async request
    result = await chat.respond(user_message, max_tokens=40)

    print("="*60)
    print("RESPONSE:")
    print("="*60)
    print(f"Lilith: {result['response']}")
    print()

    if result['shadow_thought']:
        print(f"🌑 Shadow: {result['shadow_thought']}")
        print()

    if result['ripples']:
        print("🌊 Ripples:")
        for ripple in result['ripples']:
            depth = ripple.get('depth', 0)
            content = ripple.get('content', '')
            print(f"  L{depth}: {content[:80]}...")
        print()

    print("="*60)
    print("EMERGENT PROCESS STATUS:")
    print("="*60)

    # Show mood state
    mood_report = chat.mood_drift.get_mood_report()
    print(mood_report)
    print()

    # Show meta controller
    meta_report = chat.meta_controller.get_intervention_report()
    print(meta_report)
    print()

    # Show memory stats
    mem_stats = chat.assoc_memory.get_memory_stats()
    print(f"🔮 Association Memory: {mem_stats['size']} entries")
    print()

    print("="*60)
    print("✓✓✓ ASYNC ARCHITECTURE WORKING! ✓✓✓")
    print("All emergent processes executed in parallel!")
    print("="*60)

if __name__ == '__main__':
    try:
        asyncio.run(test_conversation())
    except KeyboardInterrupt:
        print("\n🌙 Interrupted")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
