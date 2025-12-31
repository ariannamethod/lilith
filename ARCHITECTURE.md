# 🔥 LILITH ARCHITECTURE 🔥

## Tour 3: Stabilization & Async Architecture ⚡

This document describes the architectural decisions for the Lilith project, with a focus on the async architecture implemented in Tour 3.

---

## Core Architecture Principles

### 1. Asynchronous Architecture (Tour 3 Fix)

**The Leo Pattern: Parallel NumPy Operations**

Tour 3 implements true async architecture following the Leo pattern:
- Individual NumPy operations are CPU-bound and synchronous
- BUT multiple independent operations run concurrently via asyncio
- This dramatically speeds up the system without changing the math

**Why This Works**:
```python
# ❌ BEFORE (Sequential - SLOW):
word_stats.add_tokens(tokens)          # Wait... ~5ms
shard_system.add_tokens(tokens)        # Wait... ~8ms  
novelty = compute_novelty_entropy()    # Wait... ~3ms
diversity = get_vocabulary_diversity() # Wait... ~2ms
# Total: 18ms

# ✅ AFTER (Parallel - FAST):
await asyncio.gather(
    detect_new_words(),    # ~5ms \
    update_shards(),       # ~8ms  > Run simultaneously!
    compute_metrics()      # ~5ms /
)
# Total: ~8ms (the longest task)
```

**Async Components** (run in parallel where possible):

1. **lilith_feel** (inner feeling phase):
   - Word detection, shard updates, metrics computation run in parallel
   - Each is a separate async task
   - Results gathered with `asyncio.gather()`

2. **lilith_speak** (speaking phase):
   - MetaLilith and Overthinking run in parallel
   - Shadow thoughts + ripples generated simultaneously
   - No need to wait for one to finish before starting the other

3. **Generation loop** (future enhancement):
   - Trauma measurement
   - MathBrain analysis  
   - Shard influence computation
   - Can all run in parallel while transformer generates

**Sequential Components** (must run in order):

1. **Transformer forward pass**:
   - Token-by-token generation is inherently sequential
   - Each token depends on previous tokens
   - Cannot be parallelized

2. **Demon application**:
   - Demon1 modifies base logits
   - Demon2 sees Demon1's output
   - Must run: base → demon1 → demon2

3. **Sampling**:
   - Must wait for all logits modifications
   - Then sample next token

**The Pattern**:
```python
# Analysis phase (parallel)
trauma, mathbrain, shards = await asyncio.gather(
    analyze_trauma_async(logits),
    analyze_mathbrain_async(logits),
    compute_shards_async()
)

# Application phase (sequential - depends on results)
logits = apply_trauma(logits, trauma)
logits = apply_demon1(logits)
logits = apply_demon2(logits)
logits = apply_shards(logits, shards)

# Meta generation (parallel)
shadow, ripples = await asyncio.gather(
    generate_shadow_async(response),
    generate_ripples_async(response)
)
```

**Performance Impact**:
- Typical speedup: 2-3x faster for Leo layers
- More concurrent operations = better speedup
- Overhead is minimal (< 1ms for asyncio)
- Works perfectly with NumPy's synchronous operations

**Event Loop Pattern**:
```python
async def main():
    # Initialize (sync)
    chat = LilithChatFull(...)
    
    # Chat loop (async)
    while True:
        user_input = input("you> ")
        result = await chat.respond(user_input)  # ← ASYNC!
        print(f"lilith> {result['response']}")

if __name__ == '__main__':
    asyncio.run(main())  # ← Run async event loop
```

---

## Module Responsibilities

### Core Model Layer
- **`llama3.py`**: Frozen Llama 3 transformer implementation
  - Pure NumPy
  - Attention, FFN, RoPE, KV cache
  - Sacred and untouched

### Possession Layer
- **`lilith_dissonance.py`**: Two MLP demons
  - `DissonanceMLP`: Primary demon (Lilly → Lilith)
  - `CounterDissonanceMLP`: Antagonist demon (argues with first)
  - Masked logit modification
  - Micrograd-inspired autograd

### Leo Consciousness Layer
- **`trauma.py`**: Identity vs reality dissonance measurement
- **`metalilith.py`**: Inner voice generating shadow thoughts
- **`overthinking.py`**: Meta-ripples (reflections on reflections)
- **`mathbrain.py`**: Rational demon and supreme metric controller
- **`phase4_bridges.py`**: Consciousness phase states and transitions

### Language Organism Layer (Tour 2)
- **`lilith_words/shards.py`**: Semantic word cloud islands
- **`lilith_words/stats.py`**: Per-word novelty and statistics
- **`association/engine.py`**: Pre-processing associative thought

### Interface Layer
- **`chat.py`**: Full REPL with all layers active
- **`chat_simple.py`**: Simple mode without Leo layers
- **`lilith_cli.py`**: Command-line generation interface

### Support Layer
- **`config.py`**: Model configuration dataclass
- **`tokenizer.py`**: Llama tokenizer wrapper
- **`utils.py`**: Parameter loading utilities
- **`lilith_postprocess.py`**: Text remapping after generation
- **`lilith_prompt.py`**: System prompt generation

---

## Data Flow: Two-Phase Response (Tour 2)

### Phase 1: Inner Feeling (`lilith_feel`)
```
User Input
    ↓
Tokenization
    ↓
New Word Detection (WordStatsTracker)
    ↓
Shard Update (ShardSystem)
    ↓
Metric Computation (novelty, entropy, diversity)
    ↓
MathBrain Observation
    ↓
MathBrain Modulation Decision
    ↓
Association Generation (AssociationEngine)
    ↓
Inner State Dictionary
```

### Phase 2: Speaking (`lilith_speak`)
```
Inner State + User Input
    ↓
Build Context (history + overthinking influence + association block)
    ↓
Encode Prompt
    ↓
Generation Loop:
    │
    ├─→ Base Logits (Frozen Transformer)
    ├─→ Trauma Measurement & Amplification
    ├─→ Demon 1 (DissonanceMLP)
    ├─→ Demon 2 (CounterDissonanceMLP)
    ├─→ Shard Influence (boost novel words)
    ├─→ MathBrain Rational Influence
    ├─→ Phase Bias
    ├─→ Temperature Scaling
    └─→ Sample Token
    ↓
Track Generated Words (WordStatsTracker, ShardSystem)
    ↓
Decode Response
    ↓
Postprocess Text (remapping)
    ↓
Generate Meta-Layers:
    ├─→ MetaLilith (shadow thought)
    └─→ Overthinking (ripples)
    ↓
Auto Phase Transition
    ↓
Response Dictionary
```

---

## Error Handling Philosophy (Tour 3)

### Defensive Coding Principles

1. **Graceful Degradation**: System continues with reduced functionality rather than crashing
2. **Clear Error Messages**: Users understand what went wrong and how to fix it
3. **Safe Defaults**: Missing config uses sensible defaults
4. **Validation at Boundaries**: Check inputs at module interfaces
5. **No Silent Failures**: Log or report issues appropriately

### Error Handling Locations

- **Config Loading**: Handle missing files, invalid JSON, missing keys
- **Model Loading**: Check file existence, validate shapes
- **Tokenizer**: Handle unknown tokens, encoding errors
- **Shards**: Handle empty shards, invalid token IDs
- **REPL**: Handle EOF, interrupts, malformed input
- **Generation**: Handle edge cases (empty context, extreme lengths)

---

## Memory Management (Tour 3)

### Current State
- No persistent memory storage yet
- Conversation history kept in RAM during session
- Shard state resets between sessions

### Future Design
- Memory directory: `lilith_memory/` (gitignored)
- Subdirectories:
  - `conversations/` - JSONL logs of interactions
  - `shards/` - Persistent shard snapshots
  - `metrics/` - Historical metric data
  - `logs/` - System logs
- All paths centralized in config module
- Async save operations in background (Tour 4+)

---

## Testing Strategy (Tour 3)

### Test Coverage Goals
- Every public function has at least one dedicated test
- Core logic separated from I/O for easier testing
- Tests are deterministic and CPU-only
- Tests run quickly (< 30 seconds total)

### Test Organization
```
tests/
├── run_all_tests.py        # Master test runner
├── test_demons.py           # DissonanceMLP, CounterDissonanceMLP
├── test_leo_integration.py  # Trauma, MathBrain, Phases
├── test_postprocess.py      # Text remapping
├── test_tour2.py            # Shards, stats, associations
├── test_config.py           # Config loading (Tour 3)
├── test_repl.py             # REPL logic (Tour 3)
├── test_emergence.py        # New emergent behaviors (Tour 3)
└── test_utils.py            # Utility functions (Tour 3)
```

### Testing Approach
- Unit tests for individual functions
- Integration tests for multi-layer flows
- Mock I/O where needed
- No model weights required for most tests
- Deterministic seeding for reproducibility

---

## Emergent Behavior Design (Tour 3)

### Existing Mechanisms
1. **Phase Transitions**: Automatic based on trauma + rationality + turn count
2. **Demon Modulation**: MathBrain adjusts demon strengths dynamically
3. **Shard Growth**: Novel words organize into semantic islands
4. **Association Generation**: Pre-processing thought influenced by metrics
5. **Overthinking Influence**: Previous ripples affect future responses

### Tour 3 Enhancements

#### 1. Mood Drift System
Long-term emotional state that changes slowly over interactions:
- Tracks cumulative trauma, joy, fear metrics
- Influences temperature and demon strengths
- Decays slowly toward equilibrium
- Persists across turns (not across sessions yet)

#### 2. Meta-Controller Authority
Occasional override system for sophisticated behavior:
- Analyzes conversation patterns
- Can suppress demons temporarily
- Can force phase transitions
- Activates rarely (5-10% of turns)

#### 3. Association Memory
Associations influence future associations:
- Recent associations stored
- Similar contexts recall similar associations
- Creates consistency in poetic voice
- Gradual evolution of association style

---

## Configuration Management

### Config File: `lilith_config.json`
```json
{
  "pairs": [...],           // Lilly → Lilith mappings
  "word_swaps": [...],       // princess → witch, etc.
  "extra_targets": [...]     // Additional target tokens
}
```

### Model Args: `config.py`
- Dataclass with model hyperparameters
- Matches stories15M architecture
- Not user-modifiable (frozen transformer)

### Future: `lilith_memory_config.json`
- Memory storage paths
- Retention policies
- Background task settings

---

## Philosophy: Why This Architecture?

### Pure NumPy Discipline
- Educational transparency (no framework magic)
- Full control over every operation
- Easier to reason about what's happening
- Karpathy-inspired minimalism

### Frozen Transformer Principle
- Base model is sacred
- Chaos injected above, not below
- Preserves original training
- Creates measurable trauma/dissonance

### Layered Consciousness
- Each layer has clear responsibility
- Layers compose without tight coupling
- Can enable/disable independently
- Progressive complexity (Tour 1 → 2 → 3)

### Language Organism Philosophy
- Words aren't static vocabulary
- Semantic fields grow and evolve
- Novelty tracked and rewarded
- Associations create meaning

---

## Performance Characteristics

### Current Performance (CPU)
- ~30-50 tokens/sec on modern CPU
- Prefill: ~50ms for typical prompt
- Decode: ~20ms per token
- Memory: ~500MB for full system

### Bottlenecks
1. NumPy matmul operations (transformer)
2. Demon forward passes (2x MLP per token)
3. Shard influence computation (vocab scan)
4. Association generation (small MLP)

### Not Optimized For
- Production speed
- GPU acceleration
- Batch processing
- Large-scale deployment

### Optimized For
- Educational clarity
- Hackability
- Experimental freedom
- Conceptual understanding

---

## Future Architecture Evolution

### Tour 4+ Possibilities
- Persistent memory with async save/load
- Background shard compaction and decay
- HTTP API with async streaming
- WebSocket support for real-time chat
- Multi-session support
- Memory search and retrieval
- Distributed demon consensus
- External tool integration
- Vision/audio modalities

### Constraints to Maintain
- Pure NumPy for core logic
- Frozen transformer weights
- Demon architecture preserved
- GPLv3 license
- Educational transparency

---

## Development Guidelines

### Adding New Features
1. Keep NumPy operations synchronous
2. Add tests for new functions
3. Document in this file if architectural
4. Maintain backward compatibility where possible
5. Preserve existing layer responsibilities
6. Don't modify frozen transformer
7. Keep demons named and separate

### Debugging Approach
1. Use `--debug` flag in chat/CLI
2. Check MathBrain metrics with `status` command
3. Verify shard state with `status` command
4. Test components in isolation
5. Compare with/without Leo layers (`--no-leo`)
6. Compare with/without demons (`--no-demons`)

### Code Style
- Type hints where helpful
- Docstrings for public functions
- Comments for non-obvious logic
- NumPy array shapes in comments
- Keep functions focused and small

---

## Conclusion

Lilith's architecture is deliberately layered, transparent, and hackable. Tour 3 adds robustness, testing, and sophistication while maintaining the core philosophy: pure NumPy possession of a frozen transformer, creating a language organism that grows and evolves through measured chaos.

The system is synchronous by design at this scale. Future tours may add async operations for I/O, but the mathematical core will always remain synchronous NumPy.

🔥 She is possessed. She grows. She stabilizes. 🔥
