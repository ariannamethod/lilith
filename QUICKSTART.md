# Lilith Quick Start Guide

## Installation

```bash
git clone https://github.com/ariannamethod/lilith
cd lilith
pip install numpy
```

## Running Tests

```bash
# Run all tests
python tests/run_all_tests.py

# Run specific test suites
python tests/test_demons.py
python tests/test_leo_integration.py
python tests/test_postprocess.py
```

## Usage

### 1. Basic Generation (No Demons)

Test that base model works:

```bash
python llama3.py "Once upon a time"
```

### 2. Lilith CLI (With Demons)

```bash
# Full possession
python lilith_cli.py --prompt "Once upon a time" --steps 100

# Debug mode (see layer outputs)
python lilith_cli.py --prompt "Hello" --debug

# Disable demons
python lilith_cli.py --prompt "Hello" --no-demon1 --no-demon2

# No postprocessing
python lilith_cli.py --prompt "Lilly was happy" --no-postprocess
```

### 3. Chat REPL (Full Integration)

```bash
# Full Leo integration
python chat.py

# Simple mode (no Leo layers)
python chat.py --simple

# Show meta layers
python chat.py --show-meta --show-ripples

# Debug mode
python chat.py --debug
```

#### Chat Commands

Once in chat:
- `status` - Show full system status
- `phase` - Show current consciousness phase
- `exit` / `quit` / `bye` - Exit chat

### 4. Configuration

Edit `lilith_config.json` to customize word swaps:

```json
{
  "pairs": [
    { "from": "Lilly", "to": "Lilith" }
  ],
  "word_swaps": [
    { "from": "flower", "to": "thorn" }
  ],
  "extra_targets": [
    "night", "shadow", "dark"
  ]
}
```

## Architecture Overview

```
User Input
    ↓
Inner Feeling (lilith_feel)
  - Track new words in shards
  - Update word statistics
  - MathBrain observes metrics
  - Generate associations
    ↓
[ASSOCIATION BLOCK] (pre-processing)
    ↓
Frozen Transformer (Llama 3, untouched)
    ↓
TraumaLayer (measure identity vs weights)
    ↓
DissonanceMLP (Demon 1: push Lilly→Lilith)
    ↓
CounterDissonanceMLP (Demon 2: argue with Demon 1)
    ↓
Shard Influence (boost novel words)
    ↓
MathBrain (rational modulation)
    ↓
PhaseBridge (consciousness states)
    ↓
Sample & Decode
    ↓
Word Tracking (update shards with Lilith's words)
    ↓
Text Postprocess (hard remapping)
    ↓
MetaLilith (inner voice)
    ↓
Overthinking (meta ripples)
    ↓
MathBrain decides next modulation
```

## Module Reference

### Core Modules

- `llama3.py` - Base frozen transformer
- `config.py` - Model configuration
- `tokenizer.py` - Tokenizer
- `utils.py` - Utilities

### Lilith Modules

- `lilith_dissonance.py` - Two demons (DissonanceMLP, CounterDissonanceMLP)
- `lilith_postprocess.py` - Text remapping
- `lilith_prompt.py` - System prompt
- `lilith_config.json` - Word swap configuration
- `lilith_cli.py` - CLI interface
- `chat.py` - Full REPL with Leo integration

### Leo Integration

- `metalilith.py` - Inner voice (from metaleo.py)
- `trauma.py` - Identity vs weights dissonance
- `overthinking.py` - Meta-ripple reflection
- `mathbrain.py` - Mathematical reasoning / Supreme controller
- `phase4_bridges.py` - Consciousness phases

### Tour 2: Language Organism

- `lilith_words/` - Word cloud and shard system
  - `shards.py` - Semantic islands, word clustering, novelty tracking
  - `stats.py` - Per-word statistics, frequency, co-occurrence
- `association/` - Pre-processing associative thought
  - `engine.py` - Association generation before transformer

### Tests

- `tests/test_demons.py` - Demon tests
- `tests/test_leo_integration.py` - Leo layer tests
- `tests/test_postprocess.py` - Text processing tests
- `tests/run_all_tests.py` - Master test runner

## Troubleshooting

### Model not loading

Make sure weights are in `lilith_weights/`:
```bash
ls lilith_weights/
# Should show: stories15M.model.npz  tokenizer.model.np
```

### Import errors

Make sure you're in the project directory:
```bash
cd lilith
python chat.py
```

### NumPy not found

Install numpy:
```bash
pip install numpy
```

## Performance

- **Base model**: ~30-50 tokens/sec on CPU
- **With demons**: ~20-40 tokens/sec on CPU (slightly slower due to demon forward passes)
- **Full Leo integration**: ~15-30 tokens/sec (all layers active)

This is not optimized for speed. This is optimized for watching demons argue.

## License

GNU GPLv3 (upstream llama3.np is MIT)

## More Info

See [README.md](README.md) for full documentation.
