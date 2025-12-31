#!/usr/bin/env python3
"""
Synthetic Pre-training Playground: Curriculum Dataset Generator

A comprehensive CLI tool for generating curriculum-based datasets with:
- Configurable difficulty progressions
- Train/validation/test splits at each difficulty level
- HuggingFace Datasets integration for easy upload
- Rich metadata and descriptions
- Both individual task and merged curriculum modes

Usage Examples:
    # Generate a single task dataset
    python generate_curriculum.py single dyna --total-tokens 10M --push-to-hub
    
    # Generate all tasks merged with curriculum mixing
    python generate_curriculum.py merged --total-tokens 100M --push-to-hub
    
    # Generate with custom curriculum weights (easy/medium/hard)
    python generate_curriculum.py merged --curriculum-weights 0.5 0.3 0.2
    
    # Just preview what would be generated
    python generate_curriculum.py single inducto --total-tokens 1M --dry-run
"""

import argparse
import json
import os
import sys
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# HuggingFace datasets
try:
    from datasets import Dataset, DatasetDict, Features, Value, Sequence
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' not installed. Install with: pip install datasets")

# Import our tasks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task_dyna import DynaTask, DynaConfig
from task_inducto import InductoTask, InductoConfig
from task_filtro import FiltroTask, FiltroConfig
from task_logic import LogicTask, LogicConfig


# =============================================================================
# Task Metadata & Descriptions
# =============================================================================

TASK_DESCRIPTIONS = {
    "dyna": {
        "name": "Task Dyna: Dynamic State Tracking",
        "short": "Non-commutative permutation composition for testing mutable state tracking",
        "description": """
# Task Dyna: Dynamic State Tracking (Non-Commutative Permutation Composition)

## Overview
Task Dyna tests a language model's ability to maintain and update a **mutable mental state** 
through a sequence of non-commutative operations. Unlike retrieval tasks where answers exist 
in the context, Dyna's answers are **latent variables** that must be computed through 
sequential operation composition.

## Why This Task Matters
This task specifically targets the "write head" capability of architectures:
- **Transformers**: Can track state through attention patterns
- **Linear Attention/RNNs**: Struggle because they compress history with commutative operations
- **SSMs (Mamba, etc.)**: Mixed performance depending on gating mechanisms

When operation order matters (A·B ≠ B·A), models using simple averaging or summation fail.

## Task Format
```
STATE [A, B, C, D, E] || OPS SWAP(0,2) | ROTATE_LEFT(1) | REVERSE(1,4) || QUERY position 3 -> D
```

## Operations
- `SWAP(i,j)`: Swap elements at positions i and j
- `ROTATE_LEFT(k)`: Rotate state left by k positions
- `ROTATE_RIGHT(k)`: Rotate state right by k positions
- `REVERSE(i,j)`: Reverse the segment from position i to j

## Difficulty Levels
- **Level 1 (Easy)**: 5-7 elements, 5-10 operations
- **Level 2 (Medium)**: 8-12 elements, 10-20 operations  
- **Level 3 (Hard)**: 13-20 elements, 20-50 operations

## Citation
Based on methodology from "Physics of Language Models: Part 4.1" by Allen-Zhu & Li.
""",
        "task_type": "state_tracking",
        "skills_tested": ["mutable_state", "non_commutative_composition", "mental_computation"],
        "architectural_insights": "Tests write-head capability; linear models fail on non-commutative updates"
    },
    
    "inducto": {
        "name": "Task Inducto: In-Context Rule Induction",
        "short": "Few-shot function learning for testing induction head capabilities",
        "description": """
# Task Inducto: In-Context Rule Induction (Few-Shot Function Learning)

## Overview
Task Inducto tests a language model's ability to **infer a novel transformation rule** from 
a few examples and apply it to a new input. This is the atomic skill behind few-shot learning
and in-context learning more broadly.

## Why This Task Matters
This task targets the "Induction Head" hypothesis:
- Models must recognize patterns like [A][B]...[A] → predict [B]
- Rules are **randomly sampled per sequence** - cannot be memorized into weights
- Tests the attention mechanism's ability to function as a pattern-matcher and copier

## Task Format
```
Ex1: [A B C] -> [C B A] | Ex2: [X Y Z W] -> [W Z Y X] | Query: [P Q R] -> [R Q P]
```

## Rule Types (16+ rules across 3 complexity levels)

### Level 1 (Simple)
- REVERSE, ROTATE_LEFT, ROTATE_RIGHT
- DROP_FIRST, DROP_LAST, DUPLICATE_EACH

### Level 2 (Medium)
- TAKE_EVERY_NTH, INTERLEAVE_HALVES, SWAP_PAIRS
- SORT, MIRROR, CAESAR_SHIFT, REMOVE_CONSECUTIVE_DUPS

### Level 3 (Hard)
- REVERSE_SEGMENTS (reverse in chunks)
- COMPOSITE rules (apply rule1 then rule2)

## Difficulty Levels
- **Level 1**: 4-6 examples, complexity 1 rules
- **Level 2**: 2-4 examples, complexity 1-2 rules
- **Level 3**: 1-3 examples, complexity 2-3 rules

## Citation
Based on methodology from "Physics of Language Models: Part 4.1" by Allen-Zhu & Li.
""",
        "task_type": "rule_induction",
        "skills_tested": ["pattern_recognition", "few_shot_learning", "function_inference"],
        "architectural_insights": "Tests induction heads; requires attention-based pattern matching"
    },
    
    "filtro": {
        "name": "Task Filtro: Interleaved Source Separation",
        "short": "Selective attention under noise (the 'Cocktail Party' problem)",
        "description": """
# Task Filtro: Interleaved Source Separation (The "Cocktail Party" Problem)

## Overview
Task Filtro tests a language model's ability to **selectively attend** to one information 
stream while actively suppressing a competing, interleaved noise stream. 50% of the context 
is **active noise** that must be filtered out.

## Why This Task Matters
This task specifically exposes architectural differences in noise handling:
- **Transformers EXCEL**: Q·K attention can assign near-zero weight to noise tokens
- **RNNs/Linear Attention STRUGGLE**: Linear compression pollutes state with noise
- **SSMs**: Performance varies based on selective gating mechanisms

This is a critical test for real-world robustness where data is rarely clean.

## Task Format
```
EDGES A:alpha->beta | B:red->blue | A:beta->gamma | B:blue->green || QUERY A:alpha hops=2 -> gamma
```

## Design Features
- **Disjoint vocabularies**: Stream A uses Greek letters, Stream B uses colors
- **Adversarial structure**: Streams have similar graph patterns to maximize interference
- **Three interleaving patterns**: alternating, random, blocked

## Difficulty Levels
- **Level 1**: 5-8 nodes, 1-2 hops, alternating pattern
- **Level 2**: 8-12 nodes, 2-4 hops, alternating/random patterns
- **Level 3**: 10-15 nodes, 4-8 hops, random/blocked patterns

## Citation
Based on methodology from "Physics of Language Models: Part 4.1" by Allen-Zhu & Li.
""",
        "task_type": "source_separation",
        "skills_tested": ["selective_attention", "noise_filtering", "stream_tracking"],
        "architectural_insights": "Tests masking capability; linear models have noise floor issues"
    },
    
    "logic": {
        "name": "Task Logic: Boolean Circuit Evaluation",
        "short": "Non-linear XOR/parity computation for testing FFN capabilities",
        "description": """
# Task Logic: Boolean Circuit Evaluation (The "XOR" Problem)

## Overview
Task Logic tests a language model's ability to evaluate **boolean circuits** with non-linear 
gates, especially XOR. This targets the **FFN/MLP layers** rather than attention, testing 
whether the model can perform actual computation rather than just retrieval.

## Why This Task Matters
XOR/parity is the classic "hard problem" for neural networks (Minsky & Papert):
- Cannot be computed by linear combinations of inputs
- Requires multi-layer non-linearity
- Perfect attention doesn't help if FFN layers are too weak

This isolates the computational capacity of the feedforward layers.

## Task Format
```
INPUTS x0=1 x1=0 x2=1 || GATES g0=XOR(x0,x1) | g1=AND(g0,x2) || QUERY g1 -> 1
```

## Gate Types
- **AND**: Output 1 iff all inputs are 1
- **OR**: Output 1 iff any input is 1  
- **XOR**: Output 1 iff exactly one input is 1 (the hard one!)
- **NAND, NOR, NOT**: Standard boolean operations

## Special Mode: Pure Parity
The hardest case: compute XOR of N input bits through a tree of XOR gates.
```
INPUTS x0=1 x1=0 x2=1 x3=1 || GATES g0=XOR(x0,x1) | g1=XOR(x2,x3) | g2=XOR(g0,g1) || QUERY g2 -> 1
```

## Difficulty Levels
- **Level 1**: 3-4 inputs, depth 2-3, ~10% XOR gates
- **Level 2**: 4-5 inputs, depth 3-5, ~30% XOR gates
- **Level 3**: 5-6 inputs, depth 5-8, ~50% XOR gates

## Citation
Based on methodology from "Physics of Language Models: Part 4.1" by Allen-Zhu & Li.
""",
        "task_type": "boolean_computation",
        "skills_tested": ["non_linear_computation", "circuit_evaluation", "parity"],
        "architectural_insights": "Tests FFN capacity; attention alone cannot solve XOR"
    }
}

MERGED_DESCRIPTION = """
# Synthetic Pre-training Playground: Merged Curriculum Dataset

## Overview
This dataset combines four synthetic tasks designed to test distinct capabilities of 
language model architectures. Following the methodology from "Physics of Language Models 4.1",
each task isolates an **atomic skill** that can be tested with mini-scaling laws.

## Included Tasks

### 1. Task Dyna (State Tracking)
Non-commutative permutation composition. Tests mutable state tracking.
- Breaks linear attention models that use commutative compression

### 2. Task Inducto (Rule Induction)  
Few-shot function learning. Tests induction head capabilities.
- Rules randomly sampled per sequence - cannot be memorized

### 3. Task Filtro (Source Separation)
Selective attention under noise. Tests filtering capability.
- 50% of context is adversarial noise

### 4. Task Logic (Boolean Circuits)
XOR/parity computation. Tests FFN non-linearity.
- The classic problem that single-layer perceptrons cannot solve

## Curriculum Structure
Data is organized by difficulty level (1=easy, 2=medium, 3=hard) with configurable mixing ratios.
Each sample includes metadata about task type, difficulty, and specific parameters.

## Splits
- **train**: For model training
- **validation**: For hyperparameter tuning  
- **test**: For final evaluation (held out)

Each split maintains the same task/difficulty distribution.

## Usage
```python
from datasets import load_dataset
ds = load_dataset("your-username/synthetic-playground")

# Filter by task
dyna_samples = ds['train'].filter(lambda x: x['task'] == 'dyna')

# Filter by difficulty
hard_samples = ds['train'].filter(lambda x: x['difficulty'] == 3)
```

## Citation
Based on methodology from "Physics of Language Models: Part 4.1" by Allen-Zhu & Li.
"""


# =============================================================================
# Token Estimation
# =============================================================================

@dataclass
class TokenStats:
    """Statistics about tokens per sample for a task/difficulty combo."""
    mean_tokens: float
    std_tokens: float
    min_tokens: int
    max_tokens: int
    samples_measured: int = 100


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.
    Uses a simple heuristic: ~4 characters per token on average.
    For more accuracy, use a real tokenizer.
    """
    # Simple whitespace + punctuation based estimation
    # This is conservative - actual BPE tokenizers often produce fewer tokens
    words = text.split()
    # Account for special tokens and subword splitting
    estimated = sum(max(1, len(word) // 4 + 1) for word in words)
    return max(1, estimated)


def measure_token_stats(task, difficulty: int, n_samples: int = 100) -> TokenStats:
    """Measure token statistics for a task at a given difficulty."""
    samples = task.generate_dataset(n_samples, difficulty_level=difficulty)
    token_counts = [estimate_tokens(s['input_text']) for s in samples]
    
    return TokenStats(
        mean_tokens=sum(token_counts) / len(token_counts),
        std_tokens=(sum((t - sum(token_counts)/len(token_counts))**2 for t in token_counts) / len(token_counts)) ** 0.5,
        min_tokens=min(token_counts),
        max_tokens=max(token_counts),
        samples_measured=n_samples
    )


# =============================================================================
# Curriculum Configuration
# =============================================================================

@dataclass
class CurriculumConfig:
    """Configuration for curriculum generation."""
    total_tokens: int
    difficulty_weights: Tuple[float, float, float] = (0.4, 0.35, 0.25)  # easy/med/hard
    split_ratios: Tuple[float, float, float] = (0.90, 0.05, 0.05)  # train/val/test
    seed: int = 42
    
    def __post_init__(self):
        # Normalize weights
        total_diff = sum(self.difficulty_weights)
        self.difficulty_weights = tuple(w / total_diff for w in self.difficulty_weights)
        
        total_split = sum(self.split_ratios)
        self.split_ratios = tuple(r / total_split for r in self.split_ratios)


@dataclass  
class GenerationPlan:
    """Plan for how many samples to generate."""
    task_name: str
    difficulty: int
    split: str
    n_samples: int
    estimated_tokens: int
    token_stats: TokenStats


# =============================================================================
# Dataset Generation
# =============================================================================

class CurriculumGenerator:
    """Main class for generating curriculum datasets."""
    
    TASK_CLASSES = {
        'dyna': (DynaTask, DynaConfig),
        'inducto': (InductoTask, InductoConfig),
        'filtro': (FiltroTask, FiltroConfig),
        'logic': (LogicTask, LogicConfig),
    }
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        random.seed(config.seed)
        
        # Initialize tasks
        self.tasks = {}
        for name, (TaskClass, ConfigClass) in self.TASK_CLASSES.items():
            self.tasks[name] = TaskClass(ConfigClass(seed=config.seed))
        
        # Cache token stats
        self._token_stats_cache = {}
    
    def get_token_stats(self, task_name: str, difficulty: int) -> TokenStats:
        """Get token statistics, with caching."""
        key = (task_name, difficulty)
        if key not in self._token_stats_cache:
            self._token_stats_cache[key] = measure_token_stats(
                self.tasks[task_name], difficulty
            )
        return self._token_stats_cache[key]
    
    def create_generation_plan(
        self, 
        task_names: List[str],
        total_tokens: Optional[int] = None
    ) -> List[GenerationPlan]:
        """
        Create a detailed plan for sample generation.
        
        Returns a list of GenerationPlan objects specifying exactly how many
        samples to generate for each task/difficulty/split combination.
        """
        total_tokens = total_tokens or self.config.total_tokens
        plans = []
        
        # Calculate tokens per task (equal distribution among tasks)
        tokens_per_task = total_tokens // len(task_names)
        
        for task_name in task_names:
            # Calculate tokens per difficulty level
            for diff_idx, (difficulty, diff_weight) in enumerate(
                zip([1, 2, 3], self.config.difficulty_weights)
            ):
                tokens_for_difficulty = int(tokens_per_task * diff_weight)
                stats = self.get_token_stats(task_name, difficulty)
                
                # Calculate samples needed
                total_samples = int(tokens_for_difficulty / stats.mean_tokens)
                
                # Split into train/val/test
                for split_idx, (split_name, split_ratio) in enumerate(
                    zip(['train', 'validation', 'test'], self.config.split_ratios)
                ):
                    n_samples = max(1, int(total_samples * split_ratio))
                    estimated_tokens = int(n_samples * stats.mean_tokens)
                    
                    plans.append(GenerationPlan(
                        task_name=task_name,
                        difficulty=difficulty,
                        split=split_name,
                        n_samples=n_samples,
                        estimated_tokens=estimated_tokens,
                        token_stats=stats
                    ))
        
        return plans
    
    def generate_samples(
        self, 
        plan: GenerationPlan,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate samples according to a plan."""
        task = self.tasks[plan.task_name]
        samples = []
        
        iterator = range(plan.n_samples)
        if show_progress and HAS_TQDM:
            iterator = tqdm(
                iterator, 
                desc=f"{plan.task_name}/L{plan.difficulty}/{plan.split}",
                leave=False
            )
        
        for i in iterator:
            # Generate with unique seed for reproducibility
            sample_seed = hash((plan.task_name, plan.difficulty, plan.split, i)) % (2**32)
            random.seed(sample_seed)
            
            raw_sample = task.generate_sample()
            
            # Enrich with metadata
            # Handle enum serialization
            def serialize_metadata(obj):
                if isinstance(obj, dict):
                    return {k: serialize_metadata(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [serialize_metadata(v) for v in obj]
                elif isinstance(obj, tuple):
                    return [serialize_metadata(v) for v in obj]
                elif hasattr(obj, 'value'):  # Enum
                    return obj.value
                elif hasattr(obj, 'name') and hasattr(obj, '__class__'):  # Other enum-like
                    return str(obj)
                return obj
            
            raw_metadata = raw_sample.get('metadata', {})
            serializable_metadata = serialize_metadata(raw_metadata)
            
            sample = {
                'input_text': raw_sample['input_text'],
                'target': raw_sample['target'],
                'task': plan.task_name,
                'difficulty': plan.difficulty,
                'split': plan.split,
                'sample_id': f"{plan.task_name}_{plan.difficulty}_{plan.split}_{i}",
                'metadata': json.dumps(serializable_metadata)
            }
            samples.append(sample)
        
        return samples
    
    def generate_dataset(
        self,
        task_names: List[str],
        total_tokens: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Generate a complete dataset with all splits.
        
        Returns a dict mapping split names to lists of samples.
        """
        plans = self.create_generation_plan(task_names, total_tokens)
        
        # Group samples by split
        samples_by_split = defaultdict(list)
        
        for plan in plans:
            samples = self.generate_samples(plan, show_progress)
            samples_by_split[plan.split].extend(samples)
        
        # Shuffle each split (maintaining reproducibility)
        for split_name, samples in samples_by_split.items():
            random.seed(self.config.seed + hash(split_name))
            random.shuffle(samples)
        
        return dict(samples_by_split)
    
    def to_hf_dataset(
        self, 
        samples_by_split: Dict[str, List[Dict]]
    ) -> 'DatasetDict':
        """Convert generated samples to HuggingFace DatasetDict."""
        if not HAS_DATASETS:
            raise ImportError("Please install datasets: pip install datasets")
        
        datasets = {}
        for split_name, samples in samples_by_split.items():
            datasets[split_name] = Dataset.from_list(samples)
        
        return DatasetDict(datasets)
    
    def save_local(
        self, 
        samples_by_split: Dict[str, List[Dict]],
        output_dir: Path
    ):
        """Save dataset to local directory as JSON files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, samples in samples_by_split.items():
            split_file = output_dir / f"{split_name}.jsonl"
            with open(split_file, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
        
        # Save metadata
        metadata = {
            'config': asdict(self.config),
            'generated_at': datetime.now().isoformat(),
            'splits': {
                name: len(samples) for name, samples in samples_by_split.items()
            },
            'total_samples': sum(len(s) for s in samples_by_split.values())
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved dataset to {output_dir}")


# =============================================================================
# HuggingFace Hub Integration
# =============================================================================

def create_dataset_card(
    task_names: List[str],
    config: CurriculumConfig,
    samples_by_split: Dict[str, List[Dict]]
) -> str:
    """Create a README.md for the HuggingFace dataset."""
    
    if len(task_names) == 1:
        task_name = task_names[0]
        description = TASK_DESCRIPTIONS[task_name]['description']
        tags = TASK_DESCRIPTIONS[task_name]['skills_tested']
    else:
        description = MERGED_DESCRIPTION
        tags = ['state_tracking', 'rule_induction', 'source_separation', 'boolean_computation']
    
    # Calculate statistics
    total_samples = sum(len(s) for s in samples_by_split.values())
    split_info = "\n".join(
        f"- **{name}**: {len(samples):,} samples"
        for name, samples in samples_by_split.items()
    )
    
    # Task distribution for merged datasets
    if len(task_names) > 1:
        task_counts = defaultdict(int)
        for split_samples in samples_by_split.values():
            for sample in split_samples:
                task_counts[sample['task']] += 1
        task_dist = "\n".join(
            f"- **{name}**: {count:,} samples ({100*count/total_samples:.1f}%)"
            for name, count in sorted(task_counts.items())
        )
    else:
        task_dist = f"- **{task_names[0]}**: {total_samples:,} samples (100%)"
    
    # Difficulty distribution
    diff_counts = defaultdict(int)
    for split_samples in samples_by_split.values():
        for sample in split_samples:
            diff_counts[sample['difficulty']] += 1
    diff_dist = "\n".join(
        f"- **Level {d}**: {count:,} samples ({100*count/total_samples:.1f}%)"
        for d, count in sorted(diff_counts.items())
    )
    
    card = f"""---
language:
- en
license: mit
tags:
- synthetic
- language-model-evaluation
- architecture-comparison
{chr(10).join(f'- {tag}' for tag in tags)}
task_categories:
- text-generation
pretty_name: Synthetic Pre-training Playground
---

{description}

## Dataset Statistics

**Total Samples**: {total_samples:,}

### Splits
{split_info}

### Task Distribution
{task_dist}

### Difficulty Distribution
{diff_dist}

## Configuration

- **Seed**: {config.seed}
- **Difficulty Weights** (easy/medium/hard): {config.difficulty_weights}
- **Split Ratios** (train/val/test): {config.split_ratios}

## Schema

Each sample contains:
- `input_text`: The formatted input string
- `target`: The expected output
- `task`: Task name (dyna/inducto/filtro/logic)
- `difficulty`: Difficulty level (1/2/3)
- `split`: Dataset split (train/validation/test)
- `sample_id`: Unique identifier
- `metadata`: JSON string with task-specific details

## Loading the Dataset

```python
from datasets import load_dataset

# Load entire dataset
ds = load_dataset("your-username/dataset-name")

# Access splits
train_data = ds['train']
val_data = ds['validation']
test_data = ds['test']

# Filter by task
dyna_samples = ds['train'].filter(lambda x: x['task'] == 'dyna')

# Filter by difficulty
easy_samples = ds['train'].filter(lambda x: x['difficulty'] == 1)
```

## Citation

If you use this dataset, please cite:

```bibtex
@article{{allenzhu2024physics,
  title={{Physics of Language Models: Part 4.1, Architecture Design}},
  author={{Allen-Zhu, Zeyuan and Li, Yuanzhi}},
  year={{2024}}
}}
```
"""
    return card


def push_to_hub(
    dataset_dict: 'DatasetDict',
    repo_id: str,
    task_names: List[str],
    config: CurriculumConfig,
    samples_by_split: Dict[str, List[Dict]],
    private: bool = False,
    token: Optional[str] = None
):
    """Push dataset to HuggingFace Hub with proper documentation."""
    if not HAS_DATASETS:
        raise ImportError("Please install datasets: pip install datasets")
    
    # Create dataset card
    card_content = create_dataset_card(task_names, config, samples_by_split)
    
    # Push to hub
    dataset_dict.push_to_hub(
        repo_id,
        private=private,
        token=token
    )
    
    # Update README (this requires huggingface_hub)
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        print(f"✓ Uploaded README.md to {repo_id}")
    except ImportError:
        print("Note: Install huggingface_hub to auto-upload README")
        print("README content saved locally instead")
    
    print(f"✓ Dataset pushed to: https://huggingface.co/datasets/{repo_id}")


# =============================================================================
# CLI Interface
# =============================================================================

def parse_token_count(s: str) -> int:
    """Parse token count strings like '10M', '1B', '500K'."""
    s = s.upper().strip()
    multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}
    
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    
    return int(s)


def format_token_count(n: int) -> str:
    """Format token count for display."""
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def main():
    parser = argparse.ArgumentParser(
        description="Generate curriculum-based synthetic datasets for LM architecture evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single task dataset
  %(prog)s single dyna --total-tokens 10M --output-dir ./data/dyna
  
  # Generate all tasks merged
  %(prog)s merged --total-tokens 100M --output-dir ./data/merged
  
  # Push to HuggingFace Hub
  %(prog)s single inducto --total-tokens 50M --push-to-hub --hub-repo myuser/inducto-50M
  
  # Custom curriculum weights (more easy data)
  %(prog)s merged --total-tokens 100M --curriculum-weights 0.5 0.3 0.2
  
  # Dry run to see generation plan
  %(prog)s single logic --total-tokens 1M --dry-run
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Generation mode')
    
    # Common arguments
    def add_common_args(p):
        p.add_argument('--total-tokens', type=str, default='10M',
                      help='Total tokens to generate (e.g., 10M, 1B, 500K)')
        p.add_argument('--curriculum-weights', type=float, nargs=3, 
                      default=[0.4, 0.35, 0.25],
                      metavar=('EASY', 'MED', 'HARD'),
                      help='Weights for difficulty levels (default: 0.4 0.35 0.25)')
        p.add_argument('--split-ratios', type=float, nargs=3,
                      default=[0.90, 0.05, 0.05],
                      metavar=('TRAIN', 'VAL', 'TEST'),
                      help='Split ratios (default: 0.90 0.05 0.05)')
        p.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
        p.add_argument('--output-dir', type=str, default=None,
                      help='Output directory for local save')
        p.add_argument('--push-to-hub', action='store_true',
                      help='Push to HuggingFace Hub')
        p.add_argument('--hub-repo', type=str, default=None,
                      help='HuggingFace repo ID (e.g., username/dataset-name)')
        p.add_argument('--private', action='store_true',
                      help='Make HuggingFace repo private')
        p.add_argument('--hf-token', type=str, default=None,
                      help='HuggingFace API token (or set HF_TOKEN env var)')
        p.add_argument('--dry-run', action='store_true',
                      help='Show generation plan without generating')
        p.add_argument('--quiet', action='store_true',
                      help='Suppress progress output')
    
    # Single task subcommand
    single_parser = subparsers.add_parser('single', help='Generate a single task dataset')
    single_parser.add_argument('task', choices=['dyna', 'inducto', 'filtro', 'logic'],
                              help='Task to generate')
    add_common_args(single_parser)
    
    # Merged subcommand
    merged_parser = subparsers.add_parser('merged', help='Generate merged multi-task dataset')
    merged_parser.add_argument('--tasks', nargs='+', 
                              choices=['dyna', 'inducto', 'filtro', 'logic'],
                              default=['dyna', 'inducto', 'filtro', 'logic'],
                              help='Tasks to include (default: all)')
    add_common_args(merged_parser)
    
    # Info subcommand
    info_parser = subparsers.add_parser('info', help='Show information about tasks')
    info_parser.add_argument('task', nargs='?', choices=['dyna', 'inducto', 'filtro', 'logic'],
                            help='Task to show info for (or all if not specified)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Handle info command
    if args.command == 'info':
        if args.task:
            print(TASK_DESCRIPTIONS[args.task]['description'])
        else:
            for name, info in TASK_DESCRIPTIONS.items():
                print(f"\n{'='*60}")
                print(f"Task: {name.upper()}")
                print(f"{'='*60}")
                print(info['short'])
                print(f"Skills tested: {', '.join(info['skills_tested'])}")
        return
    
    # Parse token count
    total_tokens = parse_token_count(args.total_tokens)
    
    # Determine tasks
    if args.command == 'single':
        task_names = [args.task]
    else:
        task_names = args.tasks
    
    # Create config
    config = CurriculumConfig(
        total_tokens=total_tokens,
        difficulty_weights=tuple(args.curriculum_weights),
        split_ratios=tuple(args.split_ratios),
        seed=args.seed
    )
    
    # Create generator
    generator = CurriculumGenerator(config)
    
    # Show plan
    plans = generator.create_generation_plan(task_names, total_tokens)
    
    print(f"\n{'='*60}")
    print("GENERATION PLAN")
    print(f"{'='*60}")
    print(f"Total target tokens: {format_token_count(total_tokens)}")
    print(f"Tasks: {', '.join(task_names)}")
    print(f"Difficulty weights: {args.curriculum_weights}")
    print(f"Split ratios: {args.split_ratios}")
    print(f"Seed: {args.seed}")
    print()
    
    # Group plans by task and difficulty for display
    total_estimated = 0
    total_samples = 0
    
    current_task = None
    for plan in sorted(plans, key=lambda p: (p.task_name, p.difficulty, p.split)):
        if plan.task_name != current_task:
            if current_task is not None:
                print()
            current_task = plan.task_name
            print(f"  {plan.task_name.upper()}:")
        
        print(f"    L{plan.difficulty}/{plan.split:10s}: {plan.n_samples:>7,} samples "
              f"(~{format_token_count(plan.estimated_tokens):>6} tokens)")
        
        total_estimated += plan.estimated_tokens
        total_samples += plan.n_samples
    
    print()
    print(f"  TOTAL: {total_samples:,} samples (~{format_token_count(total_estimated)} tokens)")
    
    if args.dry_run:
        print("\n[DRY RUN - no data generated]")
        return
    
    # Generate!
    print(f"\n{'='*60}")
    print("GENERATING DATA")
    print(f"{'='*60}")
    
    samples_by_split = generator.generate_dataset(
        task_names, 
        total_tokens,
        show_progress=not args.quiet
    )
    
    actual_total = sum(len(s) for s in samples_by_split.values())
    print(f"\n✓ Generated {actual_total:,} samples")
    
    # Save locally if requested
    if args.output_dir:
        output_dir = Path(args.output_dir)
        generator.save_local(samples_by_split, output_dir)
        
        # Also save README
        card = create_dataset_card(task_names, config, samples_by_split)
        with open(output_dir / 'README.md', 'w') as f:
            f.write(card)
        print(f"✓ Saved to {output_dir}")
    
    # Push to hub if requested
    if args.push_to_hub:
        if not args.hub_repo:
            print("Error: --hub-repo required when using --push-to-hub")
            return
        
        if not HAS_DATASETS:
            print("Error: datasets library required. Install with: pip install datasets")
            return
        
        print(f"\n{'='*60}")
        print("PUSHING TO HUGGINGFACE HUB")
        print(f"{'='*60}")
        
        dataset_dict = generator.to_hf_dataset(samples_by_split)
        
        token = args.hf_token or os.environ.get('HF_TOKEN')
        
        push_to_hub(
            dataset_dict,
            args.hub_repo,
            task_names,
            config,
            samples_by_split,
            private=args.private,
            token=token
        )
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
