# Synthetic Pre-training Playground: Four Improved Tasks

## Overview

This package implements four synthetic tasks designed to test language model architectures, following the methodology from **"Physics of Language Models 4.1"** by Zeyuan Allen-Zhu.

The tasks are designed to be:
1. **Orthogonal** to each other and to existing benchmarks (Depo, Brevo, Knowledge, Mano, CFG)
2. **Controllable** in difficulty for mini-scaling laws
3. **Targeted** at System-1 (mental) reasoning, not chain-of-thought
4. **Short-context** (~2-4K tokens)
5. **Architecture-discriminating** - designed to reveal specific weaknesses

---

## The Four Tasks

### 1. Task Dyna: Dynamic State Tracking
**Tests: Non-commutative state updates (write-head capability)**

```
STATE [A, B, C, D, E] || OPS SWAP(0,2) | ROTATE_LEFT(1) | REVERSE(1,4) || QUERY position 3 ->
```

**Key Improvements:**
- Multiple operation types: `SWAP`, `ROTATE_LEFT`, `ROTATE_RIGHT`, `REVERSE_SEGMENT`
- Two query types: "What's at position X?" and "Where is element Y?"
- Mathematically verified non-commutativity (A·B ≠ B·A)
- Configurable state sizes (5-20) and operation counts (5-50)

**Why It's Hard for Linear Models:**
Linear attention compresses history with commutative operations (summation). When order matters, they fail catastrophically.

---

### 2. Task Inducto: In-Context Rule Induction
**Tests: Few-shot learning / Induction Heads**

```
Ex1: [A B C] -> [C B A] | Ex2: [X Y Z W] -> [W Z Y X] | Query: [P Q R] -> ???
```

**Key Improvements:**
- **16+ rule types** organized by complexity:
  - Level 1: `REVERSE`, `ROTATE`, `DROP_FIRST/LAST`, `DUPLICATE_EACH`
  - Level 2: `TAKE_EVERY_NTH`, `INTERLEAVE_HALVES`, `SWAP_PAIRS`, `SORT`, `CAESAR_SHIFT`
  - Level 3: `REVERSE_SEGMENTS`, `COMPOSITE` (rule1 → rule2)
- Rules are **randomly sampled per sequence** (can't be memorized)
- Scales from 1-shot to 10-shot examples

**Why It's Important:**
This is the purest test of "learning to learn." If a model fails here, it will struggle with novel tasks even if it has perfect reasoning on known tasks.

---

### 3. Task Filtro: Interleaved Source Separation
**Tests: Selective attention / Noise filtering**

```
EDGES A:alpha->beta | B:red->blue | A:beta->gamma | B:blue->green | ... || QUERY A:alpha hops=2 ->
```

**Key Improvements:**
- **Disjoint vocabularies** for streams (Greek letters vs. colors) - prevents false overlap
- **Three interleaving patterns**: alternating, random, blocked
- **Adversarial design**: 50% of context is active noise that must be filtered
- Scales hop depth (1-8) and graph size (5-15 nodes)

**Why It Breaks RNNs/SSMs:**
Transformers excel because Q·K can assign near-zero attention to noise tokens. Linear RNNs compress all history into a fixed state, inevitably polluting relevant information with noise.

---

### 4. Task Logic: Boolean Circuit Evaluation
**Tests: Non-linear computation (FFN/MLP capacity)**

```
INPUTS x0=1 x1=0 x2=1 || GATES g0=XOR(x0,x1) | g1=AND(g0,x2) | g2=XOR(g1,g0) || QUERY g2 ->
```

**Key Improvements:**
- **Six gate types**: `AND`, `OR`, `XOR`, `NAND`, `NOR`, `NOT`
- **Configurable XOR density** (10%-50%) - XOR is the classic "hard" operation
- **Pure parity test**: `generate_xor_parity_sample(n)` creates XOR-tree of n bits
- DAG structure ensures valid topological evaluation order

**Why It's Critical:**
XOR/parity cannot be computed by linear combinations (the Minsky & Papert perceptron problem). This isolates the FFN's computational capacity - perfect attention doesn't help if the MLP is too weak!

---

## Mini-Scaling Law Structure

Each task supports 2D scaling laws per Allen-Zhu's methodology:

| Task | Primary Difficulty Axis | Secondary Axis |
|------|------------------------|----------------|
| Dyna | Number of operations (5→50) | State size (5→20) |
| Inducto | Rule complexity (1→3) | Examples (1-shot→10-shot) |
| Filtro | Hop depth (1→8) | Graph size (5→15) |
| Logic | Circuit depth (2→8) | XOR density (10%→50%) |

**Three difficulty levels** are pre-configured for each task.

---

## Orthogonality Matrix

```
┌─────────────┬────────┬────────┬─────────┬────────┬─────────┬─────────┐
│   Task      │ Depo   │ Brevo  │ Knowl.  │ Mano   │ CFG     │ New     │
│             │(depth) │(width) │ (mem)   │(manip) │(struct) │ Skill   │
├─────────────┼────────┼────────┼─────────┼────────┼─────────┼─────────┤
│ DYNA        │   ✗    │   ✗    │   ✗     │   ✗    │   ✗     │ State   │
│             │        │        │         │        │         │ Tracking│
├─────────────┼────────┼────────┼─────────┼────────┼─────────┼─────────┤
│ INDUCTO     │   ✗    │   ✗    │   ✗     │   ✗    │   ✗     │ Rule    │
│             │        │        │         │        │         │ Induct. │
├─────────────┼────────┼────────┼─────────┼────────┼─────────┼─────────┤
│ FILTRO      │  (✓)   │   ✗    │   ✗     │   ✗    │   ✗     │ Source  │
│             │ +noise │        │         │        │         │ Separ.  │
├─────────────┼────────┼────────┼─────────┼────────┼─────────┼─────────┤
│ LOGIC       │   ✗    │   ✗    │   ✗     │  (✓)   │   ✗     │ Boolean │
│             │        │        │         │+nonlin │         │ XOR     │
└─────────────┴────────┴────────┴─────────┴────────┴─────────┴─────────┘
```

---

## Usage

### Basic Usage

```python
from synthetic_tasks import DynaTask, InductoTask, FiltroTask, LogicTask

# Generate samples
dyna = DynaTask()
sample = dyna.generate_sample(state_size=8, num_ops=10)
print(sample['input_text'])
print(sample['target'])

# Generate evaluation samples (answer held out)
eval_sample = dyna.generate_evaluation_sample()

# Generate datasets with controlled difficulty
easy_dataset = dyna.generate_dataset(n_samples=1000, difficulty_level=1)
hard_dataset = dyna.generate_dataset(n_samples=1000, difficulty_level=3)
```

### Running Tests

```bash
cd synthetic_tasks
python run_demo.py  # Runs all tests and shows demonstrations
```

---

## File Structure

```
synthlm/
├── __init__.py              # Package exports
├── task_dyna.py             # Dynamic State Tracking
├── task_inducto.py          # In-Context Rule Induction
├── task_filtro.py           # Interleaved Source Separation
├── task_logic.py            # Boolean Circuit Evaluation
├── generate_curriculum.py   # CLI for dataset generation & HuggingFace upload
├── evaluate.py              # CLI for model evaluation via OpenAI-compatible APIs
├── run_demo.py              # Comprehensive test suite & demos
├── pyproject.toml           # Package configuration & dependencies
└── README.md                # This file
```

---

## Curriculum Generator CLI

The `generate_curriculum.py` script provides a powerful command-line interface for generating
curriculum-based datasets with HuggingFace integration.

### Installation

```bash
pip install datasets huggingface_hub tqdm
```

### Quick Start

```bash
# Generate a single task dataset (10M tokens)
python generate_curriculum.py single dyna --total-tokens 10M --output-dir ./data/dyna

# Generate all tasks merged (100M tokens)
python generate_curriculum.py merged --total-tokens 100M --output-dir ./data/merged

# Push directly to HuggingFace Hub
python generate_curriculum.py single inducto --total-tokens 50M \
    --push-to-hub --hub-repo myuser/inducto-50M

# Preview what will be generated (dry run)
python generate_curriculum.py merged --total-tokens 1B --dry-run

# Show task information
python generate_curriculum.py info dyna
```

### Commands

| Command | Description |
|---------|-------------|
| `single <task>` | Generate dataset for one task (dyna/inducto/filtro/logic) |
| `merged` | Generate combined multi-task dataset |
| `info [task]` | Show detailed task documentation |

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--total-tokens` | 10M | Target tokens (e.g., 10M, 1B, 500K) |
| `--curriculum-weights` | 0.4 0.35 0.25 | Easy/Medium/Hard distribution |
| `--split-ratios` | 0.90 0.05 0.05 | Train/Validation/Test split |
| `--seed` | 42 | Random seed for reproducibility |
| `--output-dir` | None | Save to local directory |
| `--push-to-hub` | False | Upload to HuggingFace Hub |
| `--hub-repo` | None | HuggingFace repo ID |
| `--private` | False | Make HuggingFace repo private |
| `--dry-run` | False | Show plan without generating |

### Example: Custom Curriculum

```bash
# More easy data (60% easy, 30% medium, 10% hard)
python generate_curriculum.py merged \
    --total-tokens 100M \
    --curriculum-weights 0.6 0.3 0.1 \
    --output-dir ./data/easy-curriculum

# More validation data (80% train, 15% val, 5% test)
python generate_curriculum.py single logic \
    --total-tokens 50M \
    --split-ratios 0.80 0.15 0.05 \
    --output-dir ./data/logic-more-val
```

### Output Format

Generated datasets include:
- **JSONL files**: `train.jsonl`, `validation.jsonl`, `test.jsonl`
- **Metadata**: `metadata.json` with generation config
- **README**: HuggingFace-compatible dataset card

Each sample contains:
```json
{
  "input_text": "STATE [A, B, C] || OPS SWAP(0,1) || QUERY position 0 ->",
  "target": "B",
  "task": "dyna",
  "difficulty": 1,
  "split": "train",
  "sample_id": "dyna_1_train_0",
  "metadata": "{...}"
}
```

---

## Model Evaluation CLI

The `evaluate.py` script provides a comprehensive evaluation framework for testing any language model
against SynthLM tasks via OpenAI-compatible APIs.

### Features

- **Universal API Support**: Works with any OpenAI-compatible endpoint (LM Studio, vLLM, llama.cpp, OpenRouter, OpenAI, etc.)
- **Async with Rate Limiting**: Configurable requests-per-minute and concurrent request limits
- **Resume Capability**: Interrupted runs can be resumed from where they left off
- **Reasoning Model Support**: Handles `<think>` tags and `reasoning` fields from models like DeepSeek, Qwen-QwQ, gpt-oss
- **Structured Output**: JSONL per-sample results + JSON summary for easy analysis and graphing

### Installation

```bash
pip install httpx tqdm
# or with uv
uv pip install httpx tqdm
```

### Quick Start

```bash
# Evaluate a local model (LM Studio, vLLM, llama.cpp)
python evaluate.py --base-url http://localhost:8000/v1 --model llama-3.1-8b

# Evaluate via OpenRouter
python evaluate.py --base-url https://openrouter.ai/api/v1 \
    --model meta-llama/llama-3.1-8b-instruct \
    --api-key $OPENROUTER_API_KEY

# Evaluate specific tasks and difficulties
python evaluate.py --base-url http://localhost:8000/v1 --model qwen2.5-7b \
    --tasks dyna logic \
    --difficulties 1 2 \
    --samples-per-combo 100

# Resume an interrupted run
python evaluate.py --resume ./eval_results/eval_20251231_143000

# Dry run to see what would be evaluated
python evaluate.py --base-url http://localhost:8000/v1 --model test --dry-run
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--base-url` | (required) | OpenAI-compatible API base URL |
| `--model` | (required) | Model name/ID to evaluate |
| `--api-key` | None | API key (or set `SYNTHLM_API_KEY` env var) |
| `--max-tokens` | 2048 | Max tokens for model response |
| `--temperature` | 0.0 | Sampling temperature (0 = deterministic) |
| `--timeout` | 120 | Request timeout in seconds |
| `--tasks` | all | Tasks to evaluate (dyna/inducto/filtro/logic) |
| `--difficulties` | 1 2 3 | Difficulty levels to test |
| `--samples-per-combo` | 100 | Samples per task+difficulty combination |
| `--requests-per-minute` | 60 | Rate limit (requests per minute) |
| `--max-concurrent` | 10 | Max concurrent API requests |
| `--output-dir` | auto | Output directory for results |
| `--resume` | None | Resume from existing run directory |
| `--seed` | 42 | Random seed for reproducibility |
| `--quiet` | False | Suppress progress output |
| `--dry-run` | False | Show plan without calling API |

### Output Format

Each evaluation run creates a directory with:

```
eval_results/eval_20251231_143000/
├── results.jsonl    # Per-sample detailed results
├── summary.json     # Aggregated statistics
└── state.json       # Resume state (completed sample IDs)
```

#### Per-Sample Results (`results.jsonl`)

```json
{
  "sample_id": "dyna_1_42",
  "task": "dyna",
  "difficulty": 1,
  "input_text": "STATE [A, B, C] || OPS SWAP(0,2) || QUERY position 1 ->",
  "target": "B",
  "raw_response": "Working through the operations...\n\nANSWER: B",
  "extracted_answer": "B",
  "is_correct": true,
  "latency_ms": 245.3,
  "prompt_tokens": 89,
  "completion_tokens": 42,
  "error": null,
  "timestamp": "2025-12-31T14:30:45.123Z",
  "metadata": {"state_size": 3, "num_ops": 1, "...": "..."}
}
```

#### Summary Statistics (`summary.json`)

```json
{
  "run_id": "eval_20251231_143000",
  "model": "llama-3.1-8b",
  "overall": {
    "total_samples": 1200,
    "correct": 847,
    "accuracy": 0.706
  },
  "by_task": {
    "dyna": {"total": 300, "correct": 234, "accuracy": 0.78, "by_difficulty": {...}},
    "inducto": {"total": 300, "correct": 196, "accuracy": 0.65, "by_difficulty": {...}},
    "filtro": {"total": 300, "correct": 216, "accuracy": 0.72, "by_difficulty": {...}},
    "logic": {"total": 300, "correct": 201, "accuracy": 0.67, "by_difficulty": {...}}
  },
  "by_difficulty": {
    "1": {"total": 400, "correct": 372, "accuracy": 0.93},
    "2": {"total": 400, "correct": 312, "accuracy": 0.78},
    "3": {"total": 400, "correct": 163, "accuracy": 0.41}
  },
  "timing": {"avg_latency_ms": 312.4, "p50_latency_ms": 289.1, "p95_latency_ms": 567.3}
}
```

### Console Output

```
Evaluating llama-3.1-8b on 4 tasks x 3 difficulties x 100 samples = 1200 total
[████████████████████████████████████████] 1200/1200 [15:23<00:00]

=================================================================
RESULTS SUMMARY
=================================================================
Model: llama-3.1-8b
Total: 847/1200 correct (70.6%)

By Task:
  DYNA     78.0% (234/300)  [L1: 95% | L2: 82% | L3: 57%]
  INDUCTO  65.3% (196/300)  [L1: 88% | L2: 71% | L3: 37%]
  FILTRO   72.0% (216/300)  [L1: 91% | L2: 78% | L3: 47%]
  LOGIC    67.0% (201/300)  [L1: 89% | L2: 74% | L3: 38%]

By Difficulty:
  Easy     93.0% (372/400)
  Medium   78.0% (312/400)
  Hard     40.8% (163/400)

Timing: avg 312ms, p50 289ms, p95 567ms
Tokens: 127,840 prompt + 9,600 completion = 137,440 total

Results saved to: ./eval_results/eval_20251231_143000/
=================================================================
```

### Mini-Scaling Law Analysis

The output is designed for easy generation of scaling law graphs. Example Python analysis:

```python
import json
import matplotlib.pyplot as plt

# Load summary
with open('eval_results/eval_20251231_143000/summary.json') as f:
    summary = json.load(f)

# Plot accuracy by difficulty for each task
tasks = ['dyna', 'inducto', 'filtro', 'logic']
difficulties = [1, 2, 3]

for task in tasks:
    accs = [summary['by_task'][task]['by_difficulty'][str(d)]['accuracy'] for d in difficulties]
    plt.plot(difficulties, accs, marker='o', label=task.upper())

plt.xlabel('Difficulty Level')
plt.ylabel('Accuracy')
plt.legend()
plt.title('SynthLM Task Performance by Difficulty')
plt.savefig('scaling_law.png')
```

---

## Key Design Decisions

### 1. Answer Never Appears in Input (Dyna)
Unlike Depo where the answer can be found in the context, Dyna's answer is a **latent variable** that must be computed through operation composition.

### 2. Rules Randomly Sampled Per Sequence (Inducto)
The model cannot memorize rules into weights - it must infer them from the provided examples each time.

### 3. Disjoint Vocabularies (Filtro)
Stream A uses Greek letters; Stream B uses colors. This prevents any ambiguity about which stream a token belongs to.

### 4. XOR as the Core Challenge (Logic)
XOR is the canonical non-linear operation. We explicitly track XOR density and provide a pure parity test mode.

---

## Expected Architectural Differences

Based on Allen-Zhu's framework, we predict:

| Task | Transformers | Mamba/GDN | Linear Attention |
|------|--------------|-----------|------------------|
| Dyna | Good | Mixed | Poor (commutative) |
| Inducto | Good (induction heads) | Poor | Poor |
| Filtro | Excellent | Poor (noise floor) | Poor |
| Logic | Depends on FFN | Depends on FFN | Depends on FFN |

The real value comes from **quantifying** these differences via mini-scaling laws rather than seeing 0.5% benchmark deltas.

---

## Citation

If you use these tasks, please cite the original work that inspired the methodology:

```bibtex
@article{allen2024physics,
  title={Physics of Language Models: Part 4.1, Architecture Design},
  author={Allen-Zhu, Zeyuan and Li, Yuanzhi},
  year={2024}
}
```
