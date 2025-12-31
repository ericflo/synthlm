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
synthetic_tasks/
├── __init__.py              # Package exports
├── task_dyna.py             # Dynamic State Tracking
├── task_inducto.py          # In-Context Rule Induction  
├── task_filtro.py           # Interleaved Source Separation
├── task_logic.py            # Boolean Circuit Evaluation
├── generate_curriculum.py   # CLI for dataset generation & HuggingFace upload
├── run_demo.py              # Comprehensive test suite & demos
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
