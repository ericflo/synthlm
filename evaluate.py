#!/usr/bin/env python3
"""
SynthLM Evaluation Script

Evaluate language models on SynthLM synthetic benchmarks via OpenAI-compatible APIs.

Features:
- Async requests with rate limiting
- Resume capability for interrupted runs
- JSONL per-sample results + JSON summary
- Support for reasoning models with <think> tags

Usage Examples:
    # Local llama.cpp or vLLM server
    python evaluate.py --base-url http://localhost:8000/v1 --model llama-3.1-8b

    # OpenRouter
    python evaluate.py --base-url https://openrouter.ai/api/v1 \\
        --model meta-llama/llama-3.1-8b-instruct --api-key $OPENROUTER_API_KEY

    # Resume interrupted run
    python evaluate.py --resume ./eval_results/eval_20251231_143000
"""

import argparse
import asyncio
import json
import os
import random
import re
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Optional dependencies
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# Import tasks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task_dyna import DynaTask, DynaConfig
from task_inducto import InductoTask, InductoConfig
from task_filtro import FiltroTask, FiltroConfig
from task_logic import LogicTask, LogicConfig


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class APIConfig:
    """Configuration for OpenAI-compatible API."""
    base_url: str
    model: str
    api_key: Optional[str] = None
    max_tokens: int = 2048  # Enough for reasoning models with long chains
    temperature: float = 0.0
    timeout: float = 120.0  # Longer timeout for slow/reasoning models


@dataclass
class EvalConfig:
    """Configuration for evaluation run."""
    tasks: List[str] = field(default_factory=lambda: ['dyna', 'inducto', 'filtro', 'logic'])
    difficulties: List[int] = field(default_factory=lambda: [1, 2, 3])
    samples_per_combo: int = 100
    seed: int = 42
    requests_per_minute: int = 60
    max_concurrent: int = 10


@dataclass
class EvalSample:
    """A single evaluation sample with all tracking info."""
    sample_id: str
    task: str
    difficulty: int
    input_text: str
    target: str
    metadata: Dict[str, Any]
    # Filled after API call:
    raw_response: Optional[str] = None
    extracted_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None


# =============================================================================
# Task Registry
# =============================================================================

TASK_REGISTRY = {
    'dyna': (DynaTask, DynaConfig),
    'inducto': (InductoTask, InductoConfig),
    'filtro': (FiltroTask, FiltroConfig),
    'logic': (LogicTask, LogicConfig),
}

# Difficulty configurations per task
DIFFICULTY_CONFIGS = {
    'dyna': {
        1: {'state_size': (5, 7), 'num_ops': (5, 10)},
        2: {'state_size': (8, 12), 'num_ops': (10, 20)},
        3: {'state_size': (13, 20), 'num_ops': (20, 50)},
    },
    'inducto': {
        1: {'num_examples': (3, 4), 'complexity': 1},
        2: {'num_examples': (2, 4), 'complexity': 2},
        3: {'num_examples': (1, 3), 'complexity': 3},
    },
    'filtro': {
        1: {'num_nodes': (5, 8), 'num_hops': (1, 2)},
        2: {'num_nodes': (8, 12), 'num_hops': (2, 4)},
        3: {'num_nodes': (10, 15), 'num_hops': (4, 8)},
    },
    'logic': {
        1: {'num_inputs': (3, 4), 'depth': (2, 3), 'xor_prob': 0.1},
        2: {'num_inputs': (4, 5), 'depth': (3, 5), 'xor_prob': 0.3},
        3: {'num_inputs': (5, 6), 'depth': (5, 8), 'xor_prob': 0.5},
    },
}


# =============================================================================
# Sample Generation
# =============================================================================

def generate_eval_samples(
    tasks: List[str],
    difficulties: List[int],
    samples_per_combo: int,
    seed: int
) -> List[EvalSample]:
    """Generate all evaluation samples for the run."""
    random.seed(seed)
    samples = []

    for task_name in tasks:
        task_class, config_class = TASK_REGISTRY[task_name]
        task = task_class(config_class(seed=seed))

        for difficulty in difficulties:
            cfg = DIFFICULTY_CONFIGS[task_name][difficulty]

            for i in range(samples_per_combo):
                # Generate sample with difficulty-appropriate parameters
                if task_name == 'dyna':
                    state_size = random.randint(*cfg['state_size'])
                    num_ops = random.randint(*cfg['num_ops'])
                    raw = task.generate_evaluation_sample(state_size=state_size, num_ops=num_ops)

                elif task_name == 'inducto':
                    num_ex = random.randint(*cfg['num_examples'])
                    raw = task.generate_evaluation_sample(
                        num_examples=num_ex,
                        complexity=cfg['complexity']
                    )

                elif task_name == 'filtro':
                    num_nodes = random.randint(*cfg['num_nodes'])
                    num_hops = random.randint(*cfg['num_hops'])
                    raw = task.generate_evaluation_sample(
                        num_nodes=num_nodes,
                        num_hops=num_hops
                    )

                elif task_name == 'logic':
                    num_inputs = random.randint(*cfg['num_inputs'])
                    depth = random.randint(*cfg['depth'])
                    # Temporarily update xor probability
                    task.config.xor_probability = cfg['xor_prob']
                    raw = task.generate_evaluation_sample(
                        num_inputs=num_inputs,
                        depth=depth
                    )

                sample = EvalSample(
                    sample_id=f"{task_name}_{difficulty}_{i}",
                    task=task_name,
                    difficulty=difficulty,
                    input_text=raw['input_text'],
                    target=raw['target'],
                    metadata=raw.get('metadata', {})
                )
                samples.append(sample)

    return samples


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """Token bucket rate limiter for async API calls."""

    def __init__(self, requests_per_minute: int, max_concurrent: int):
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.last_request = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait for rate limit slot."""
        await self.semaphore.acquire()
        async with self.lock:
            now = time.monotonic()
            wait_time = self.last_request + self.interval - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_request = time.monotonic()

    def release(self) -> None:
        """Release the semaphore slot."""
        self.semaphore.release()


# =============================================================================
# API Client
# =============================================================================

class APIError(Exception):
    """API call failed."""
    def __init__(self, status_code: int, message: str, retryable: bool = False):
        self.status_code = status_code
        self.message = message
        self.retryable = retryable
        super().__init__(f"API error {status_code}: {message}")


SYSTEM_PROMPT = """You are solving reasoning tasks. Work through the problem step by step, then provide your final answer.

CRITICAL: Your response MUST end with exactly this format on the last line:
ANSWER: <your answer>

The answer should be just the value with no extra text. Examples:
- For numeric answers: ANSWER: 42
- For single letters: ANSWER: B
- For binary: ANSWER: 0
- For sequences: ANSWER: A B C"""


async def call_api(
    client: httpx.AsyncClient,
    config: APIConfig,
    prompt: str
) -> Tuple[str, int, int]:
    """
    Call the OpenAI-compatible API.
    Returns: (response_text, prompt_tokens, completion_tokens)
    """
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }

    url = f"{config.base_url.rstrip('/')}/chat/completions"

    response = await client.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        retryable = response.status_code in {429, 500, 502, 503, 504}
        raise APIError(response.status_code, response.text, retryable)

    data = response.json()

    message = data['choices'][0]['message']
    content = message.get('content', '') or ''

    # Some models (like gpt-oss) put reasoning in a separate field
    reasoning = message.get('reasoning', '') or ''

    # Combine reasoning and content - put reasoning first if present
    if reasoning and content:
        full_response = f"{reasoning}\n\n{content}"
    elif reasoning:
        full_response = reasoning
    else:
        full_response = content

    usage = data.get('usage', {})
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)

    return full_response, prompt_tokens, completion_tokens


async def call_api_with_retry(
    client: httpx.AsyncClient,
    config: APIConfig,
    prompt: str,
    max_retries: int = 3
) -> Tuple[str, int, int]:
    """Call API with exponential backoff retry for transient errors."""
    last_error = None

    for attempt in range(max_retries):
        try:
            return await call_api(client, config, prompt)
        except APIError as e:
            last_error = e
            if e.retryable and attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait)
                continue
            raise
        except httpx.TimeoutException:
            last_error = Exception("Request timeout")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise

    raise last_error or Exception("Max retries exceeded")


# =============================================================================
# Answer Extraction
# =============================================================================

def extract_answer(raw_response: str, task: str) -> str:
    """
    Extract the answer from model output.

    Primary method: Look for "ANSWER: X" format (from system prompt).
    Fallback: Task-specific extraction.
    """
    response = raw_response.strip()

    # 1. Strip thinking tags (DeepSeek, Qwen-QwQ, etc.)
    response = re.sub(
        r'<think(?:ing)?>\s*.*?\s*</think(?:ing)?>',
        '',
        response,
        flags=re.DOTALL | re.IGNORECASE
    )
    response = response.strip()

    if not response:
        return ""

    # 2. Primary: Look for our requested "ANSWER: X" format
    answer_match = re.search(r'^ANSWER:\s*(.+?)\s*$', response, re.MULTILINE | re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    # 3. Task-specific fallback extraction
    if task == 'logic':
        # Find the last 0 or 1 in the response
        matches = list(re.finditer(r'\b([01])\b', response))
        if matches:
            return matches[-1].group(1)

    elif task == 'dyna':
        # Find the last single capital letter OR the last number
        # Check last line first
        lines = response.strip().split('\n')
        last_line = lines[-1].strip()

        # If last line is just a letter or number, use it
        clean_last = re.sub(r'[^\w]', '', last_line)
        if len(clean_last) == 1 and clean_last.isupper():
            return clean_last
        if clean_last.isdigit():
            return clean_last

        # Otherwise find last occurrence
        letter_matches = list(re.finditer(r'\b([A-Z])\b', response))
        if letter_matches:
            return letter_matches[-1].group(1)

        num_matches = list(re.finditer(r'\b(\d+)\b', response))
        if num_matches:
            return num_matches[-1].group(1)

    elif task == 'inducto':
        # Look for the last sequence (space-separated tokens)
        # Try bracketed content first
        bracket_matches = list(re.finditer(r'\[([^\]]+)\]', response))
        if bracket_matches:
            return bracket_matches[-1].group(1).strip()

        # Otherwise, last line
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        if lines:
            return lines[-1]

    elif task == 'filtro':
        # Find the last Greek letter or color word
        greek = {'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta',
                 'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi',
                 'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi',
                 'chi', 'psi', 'omega'}
        colors = {'red', 'blue', 'green', 'yellow', 'orange', 'purple',
                  'pink', 'brown', 'black', 'white', 'gray', 'cyan',
                  'magenta', 'lime', 'teal', 'navy', 'maroon', 'olive'}

        words = re.findall(r'\b(\w+)\b', response.lower())
        vocab_words = [w for w in words if w in greek or w in colors]
        if vocab_words:
            return vocab_words[-1]

    # 4. Ultimate fallback: last word
    words = response.split()
    if words:
        return re.sub(r'[^\w]', '', words[-1])
    return ""


def compare_answers(extracted: str, target: str, task: str) -> bool:
    """Compare extracted answer to target with task-appropriate normalization."""
    # Strip whitespace and common punctuation
    ext = extracted.strip().lower().rstrip('.,;:!?')
    tgt = target.strip().lower().rstrip('.,;:!?')

    # Direct match
    if ext == tgt:
        return True

    # Task-specific comparisons
    if task == 'logic':
        # Handle various representations of 0/1
        ext_normalized = ext.rstrip('.,;:!?')
        ext_bool = ext_normalized in ('1', 'true', 'yes', 'one')
        tgt_bool = tgt in ('1', 'true', 'yes', 'one')
        if ext_normalized in ('0', '1', 'true', 'false', 'yes', 'no', 'one', 'zero'):
            return ext_bool == tgt_bool

    elif task == 'inducto':
        # Sequence comparison - normalize spacing and brackets
        ext_seq = re.sub(r'[\[\],]', ' ', ext).split()
        tgt_seq = re.sub(r'[\[\],]', ' ', tgt).split()
        return ext_seq == tgt_seq

    elif task == 'dyna':
        # Case insensitive for letters, exact for numbers
        return ext.upper() == tgt.upper()

    elif task == 'filtro':
        # Simple case-insensitive match
        return ext == tgt

    return False


# =============================================================================
# State Management
# =============================================================================

def load_state(state_file: Path) -> Optional[Dict]:
    """Load existing state file for resume."""
    if not state_file.exists():
        return None
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_state(
    state_file: Path,
    run_id: str,
    config: Dict,
    started_at: str,
    completed_ids: Set[str],
    total_samples: int
) -> None:
    """Save current state to file."""
    state = {
        'run_id': run_id,
        'config': config,
        'started_at': started_at,
        'last_updated': datetime.now().isoformat(),
        'completed_sample_ids': list(completed_ids),
        'total_samples': total_samples,
        'completed_count': len(completed_ids)
    }
    # Write atomically
    temp_file = state_file.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(state, f, indent=2)
    temp_file.replace(state_file)


# =============================================================================
# Output
# =============================================================================

def _serialize_value(val: Any) -> Any:
    """Recursively serialize values for JSON, converting enums to their values."""
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]
    elif hasattr(val, 'value'):  # Enum
        return val.value
    elif hasattr(val, 'name') and hasattr(val, '__class__'):  # Enum fallback
        return val.name
    else:
        return val


def write_sample_result(sample: EvalSample, output_file: Path) -> None:
    """Append single sample result to JSONL file."""
    result = {
        'sample_id': sample.sample_id,
        'task': sample.task,
        'difficulty': sample.difficulty,
        'input_text': sample.input_text,
        'target': sample.target,
        'raw_response': sample.raw_response,
        'extracted_answer': sample.extracted_answer,
        'is_correct': sample.is_correct,
        'latency_ms': sample.latency_ms,
        'prompt_tokens': sample.prompt_tokens,
        'completion_tokens': sample.completion_tokens,
        'error': sample.error,
        'timestamp': sample.timestamp,
        'metadata': _serialize_value(sample.metadata)
    }
    with open(output_file, 'a') as f:
        f.write(json.dumps(result) + '\n')


def generate_summary(
    samples: List[EvalSample],
    run_id: str,
    api_config: APIConfig,
    eval_config: EvalConfig,
    started_at: str,
    completed_at: str
) -> Dict:
    """Generate aggregated summary statistics."""
    # Filter to completed samples (not errors)
    completed = [s for s in samples if s.is_correct is not None]
    errors = [s for s in samples if s.error is not None]

    # Overall stats
    total = len(completed)
    correct = sum(1 for s in completed if s.is_correct)
    accuracy = correct / total if total > 0 else 0

    # Latency stats
    latencies = [s.latency_ms for s in completed if s.latency_ms is not None]
    latencies.sort()

    def percentile(data: List[float], p: float) -> float:
        if not data:
            return 0.0
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    # Token stats
    prompt_tokens = sum(s.prompt_tokens or 0 for s in completed)
    completion_tokens = sum(s.completion_tokens or 0 for s in completed)

    # By task
    by_task = {}
    for task in eval_config.tasks:
        task_samples = [s for s in completed if s.task == task]
        task_correct = sum(1 for s in task_samples if s.is_correct)
        task_total = len(task_samples)

        by_difficulty = {}
        for diff in eval_config.difficulties:
            diff_samples = [s for s in task_samples if s.difficulty == diff]
            diff_correct = sum(1 for s in diff_samples if s.is_correct)
            diff_total = len(diff_samples)
            by_difficulty[str(diff)] = {
                'total': diff_total,
                'correct': diff_correct,
                'accuracy': diff_correct / diff_total if diff_total > 0 else 0
            }

        by_task[task] = {
            'total': task_total,
            'correct': task_correct,
            'accuracy': task_correct / task_total if task_total > 0 else 0,
            'by_difficulty': by_difficulty
        }

    # By difficulty
    by_difficulty = {}
    for diff in eval_config.difficulties:
        diff_samples = [s for s in completed if s.difficulty == diff]
        diff_correct = sum(1 for s in diff_samples if s.is_correct)
        diff_total = len(diff_samples)
        by_difficulty[str(diff)] = {
            'total': diff_total,
            'correct': diff_correct,
            'accuracy': diff_correct / diff_total if diff_total > 0 else 0
        }

    return {
        'run_id': run_id,
        'model': api_config.model,
        'base_url': api_config.base_url,
        'started_at': started_at,
        'completed_at': completed_at,
        'config': {
            'tasks': eval_config.tasks,
            'difficulties': eval_config.difficulties,
            'samples_per_combo': eval_config.samples_per_combo,
            'seed': eval_config.seed,
            'temperature': api_config.temperature,
            'max_tokens': api_config.max_tokens
        },
        'overall': {
            'total_samples': total,
            'correct': correct,
            'accuracy': accuracy,
            'errors': len(errors),
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'total_prompt_tokens': prompt_tokens,
            'total_completion_tokens': completion_tokens
        },
        'by_task': by_task,
        'by_difficulty': by_difficulty,
        'timing': {
            'min_latency_ms': min(latencies) if latencies else 0,
            'max_latency_ms': max(latencies) if latencies else 0,
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'p50_latency_ms': percentile(latencies, 50),
            'p95_latency_ms': percentile(latencies, 95),
            'p99_latency_ms': percentile(latencies, 99)
        }
    }


def print_summary(summary: Dict, quiet: bool = False) -> None:
    """Print formatted summary to console."""
    if quiet:
        return

    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)

    print(f"Model: {summary['model']}")
    overall = summary['overall']
    print(f"Total: {overall['correct']}/{overall['total_samples']} correct "
          f"({overall['accuracy']:.1%})")

    if overall['errors'] > 0:
        print(f"Errors: {overall['errors']}")

    print("\nBy Task:")
    for task, stats in summary['by_task'].items():
        diff_str = " | ".join(
            f"L{d}: {s['accuracy']:.0%}"
            for d, s in stats['by_difficulty'].items()
        )
        print(f"  {task.upper():8s} {stats['accuracy']:5.1%} "
              f"({stats['correct']}/{stats['total']})  [{diff_str}]")

    print("\nBy Difficulty:")
    for diff, stats in summary['by_difficulty'].items():
        level_names = {'1': 'Easy', '2': 'Medium', '3': 'Hard'}
        name = level_names.get(diff, f'Level {diff}')
        print(f"  {name:8s} {stats['accuracy']:5.1%} "
              f"({stats['correct']}/{stats['total']})")

    timing = summary['timing']
    print(f"\nTiming: avg {timing['avg_latency_ms']:.0f}ms, "
          f"p50 {timing['p50_latency_ms']:.0f}ms, "
          f"p95 {timing['p95_latency_ms']:.0f}ms")

    tokens = summary['overall']
    print(f"Tokens: {tokens['total_prompt_tokens']:,} prompt + "
          f"{tokens['total_completion_tokens']:,} completion = "
          f"{tokens['total_prompt_tokens'] + tokens['total_completion_tokens']:,} total")

    print("=" * 65)


# =============================================================================
# Main Evaluation Logic
# =============================================================================

async def run_evaluation(
    samples: List[EvalSample],
    api_config: APIConfig,
    rate_limiter: RateLimiter,
    output_file: Path,
    state_file: Path,
    run_id: str,
    config_dict: Dict,
    started_at: str,
    completed_ids: Set[str],
    quiet: bool = False
) -> List[EvalSample]:
    """Run evaluation with rate limiting and progress tracking."""

    # Filter to pending samples
    pending = [s for s in samples if s.sample_id not in completed_ids]

    if not pending:
        print("All samples already completed!")
        return samples

    # Progress bar
    total = len(pending)
    if HAS_TQDM and not quiet:
        pbar = tqdm(total=total, desc="Evaluating", unit="sample")
    else:
        pbar = None

    # Shutdown flag
    shutdown_requested = False

    def handle_shutdown(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        if pbar:
            pbar.write("\nShutdown requested, saving state...")

    # Set up signal handlers
    original_sigint = signal.signal(signal.SIGINT, handle_shutdown)
    original_sigterm = signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        async with httpx.AsyncClient(timeout=api_config.timeout) as client:

            async def evaluate_one(sample: EvalSample) -> EvalSample:
                if shutdown_requested:
                    return sample

                await rate_limiter.acquire()
                try:
                    start = time.perf_counter()
                    response, pt, ct = await call_api_with_retry(
                        client, api_config, sample.input_text
                    )
                    latency = (time.perf_counter() - start) * 1000

                    sample.raw_response = response
                    sample.extracted_answer = extract_answer(response, sample.task)
                    sample.is_correct = compare_answers(
                        sample.extracted_answer, sample.target, sample.task
                    )
                    sample.latency_ms = latency
                    sample.prompt_tokens = pt
                    sample.completion_tokens = ct
                    sample.timestamp = datetime.now().isoformat()

                except Exception as e:
                    sample.error = str(e)
                    sample.timestamp = datetime.now().isoformat()
                finally:
                    rate_limiter.release()

                # Write result immediately (with error handling)
                try:
                    write_sample_result(sample, output_file)
                except Exception as e:
                    if not sample.error:
                        sample.error = f"Write error: {e}"

                completed_ids.add(sample.sample_id)

                # Update state periodically (every 10 samples)
                if len(completed_ids) % 10 == 0:
                    save_state(state_file, run_id, config_dict, started_at,
                              completed_ids, len(samples))

                if pbar:
                    pbar.update(1)

                return sample

            # Process samples with bounded concurrency
            tasks = [evaluate_one(s) for s in pending]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    pending[i].error = str(result)

    finally:
        # Restore signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

        if pbar:
            pbar.close()

        # Final state save
        save_state(state_file, run_id, config_dict, started_at,
                  completed_ids, len(samples))

    return samples


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate language models on SynthLM synthetic benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with local API
  %(prog)s --base-url http://localhost:8000/v1 --model llama-3.1-8b

  # Evaluate specific tasks at specific difficulties
  %(prog)s --base-url http://localhost:8000/v1 --model qwen2.5-7b \\
           --tasks dyna inducto --difficulties 1 2

  # With rate limiting and output customization
  %(prog)s --base-url https://api.openai.com/v1 --model gpt-4o-mini \\
           --api-key $OPENAI_API_KEY --requests-per-minute 100 \\
           --samples-per-combo 200 --output-dir ./results/gpt4o

  # Resume interrupted run
  %(prog)s --resume ./results/eval_20251231_143000
"""
    )

    # Resume mode
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from existing run directory')

    # API Configuration
    parser.add_argument('--base-url', type=str,
                        help='Base URL for OpenAI-compatible API')
    parser.add_argument('--model', type=str,
                        help='Model name/ID to evaluate')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key (or set SYNTHLM_API_KEY env var)')
    parser.add_argument('--max-tokens', type=int, default=2048,
                        help='Max tokens for model response (default: 2048)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (default: 0.0)')
    parser.add_argument('--timeout', type=float, default=120.0,
                        help='API request timeout in seconds (default: 120)')

    # Task Configuration
    parser.add_argument('--tasks', nargs='+',
                        choices=['dyna', 'inducto', 'filtro', 'logic'],
                        default=['dyna', 'inducto', 'filtro', 'logic'],
                        help='Tasks to evaluate (default: all)')
    parser.add_argument('--difficulties', nargs='+', type=int,
                        choices=[1, 2, 3], default=[1, 2, 3],
                        help='Difficulty levels to test (default: all)')
    parser.add_argument('--samples-per-combo', type=int, default=100,
                        help='Samples per task/difficulty combination (default: 100)')

    # Rate Limiting
    parser.add_argument('--requests-per-minute', type=int, default=60,
                        help='Max API requests per minute (default: 60)')
    parser.add_argument('--max-concurrent', type=int, default=10,
                        help='Max concurrent API requests (default: 10)')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ./eval_results/<timestamp>)')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate samples but do not call API')

    return parser.parse_args()


async def main_async() -> int:
    """Async main function."""
    args = parse_args()

    # Check for httpx
    if not HAS_HTTPX:
        print("Error: httpx is required. Install with: pip install httpx")
        return 1

    # Handle resume mode
    if args.resume:
        resume_dir = Path(args.resume)
        state_file = resume_dir / 'state.json'
        state = load_state(state_file)

        if not state:
            print(f"Error: Could not load state from {state_file}")
            return 1

        # Extract config from state
        config = state['config']
        run_id = state['run_id']
        started_at = state['started_at']
        completed_ids = set(state.get('completed_sample_ids', []))
        output_dir = resume_dir

        # Recreate configs
        api_config = APIConfig(
            base_url=config['base_url'],
            model=config['model'],
            api_key=args.api_key or os.environ.get('SYNTHLM_API_KEY'),
            max_tokens=config.get('max_tokens', 64),
            temperature=config.get('temperature', 0.0),
            timeout=args.timeout
        )
        eval_config = EvalConfig(
            tasks=config['tasks'],
            difficulties=config['difficulties'],
            samples_per_combo=config['samples_per_combo'],
            seed=config['seed'],
            requests_per_minute=args.requests_per_minute,
            max_concurrent=args.max_concurrent
        )

        print(f"Resuming run {run_id}")
        print(f"Completed: {len(completed_ids)}/{state['total_samples']} samples")

    else:
        # New run - validate required args
        if not args.base_url or not args.model:
            print("Error: --base-url and --model are required (or use --resume)")
            return 1

        # Create configs
        api_key = args.api_key or os.environ.get('SYNTHLM_API_KEY')

        api_config = APIConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout
        )

        eval_config = EvalConfig(
            tasks=args.tasks,
            difficulties=args.difficulties,
            samples_per_combo=args.samples_per_combo,
            seed=args.seed,
            requests_per_minute=args.requests_per_minute,
            max_concurrent=args.max_concurrent
        )

        # Create output directory
        run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path('./eval_results') / run_id

        output_dir.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now().isoformat()
        completed_ids = set()

    # Generate samples
    if not args.quiet:
        total = len(eval_config.tasks) * len(eval_config.difficulties) * eval_config.samples_per_combo
        print(f"\nEvaluating {api_config.model} on {len(eval_config.tasks)} tasks "
              f"x {len(eval_config.difficulties)} difficulties "
              f"x {eval_config.samples_per_combo} samples = {total} total")

    samples = generate_eval_samples(
        eval_config.tasks,
        eval_config.difficulties,
        eval_config.samples_per_combo,
        eval_config.seed
    )

    # Dry run mode
    if args.dry_run:
        print(f"\n[DRY RUN] Would evaluate {len(samples)} samples")
        print(f"Output would be saved to: {output_dir}")

        # Show sample breakdown
        print("\nSample breakdown:")
        for task in eval_config.tasks:
            for diff in eval_config.difficulties:
                count = sum(1 for s in samples if s.task == task and s.difficulty == diff)
                print(f"  {task}/L{diff}: {count} samples")

        # Show example
        print("\nExample sample:")
        example = samples[0]
        print(f"  Task: {example.task}")
        print(f"  Difficulty: {example.difficulty}")
        print(f"  Input: {example.input_text[:100]}...")
        print(f"  Target: {example.target}")
        return 0

    # Prepare output files
    output_file = output_dir / 'results.jsonl'
    state_file = output_dir / 'state.json'
    summary_file = output_dir / 'summary.json'

    # Save config for resume
    config_dict = {
        'base_url': api_config.base_url,
        'model': api_config.model,
        'tasks': eval_config.tasks,
        'difficulties': eval_config.difficulties,
        'samples_per_combo': eval_config.samples_per_combo,
        'seed': eval_config.seed,
        'temperature': api_config.temperature,
        'max_tokens': api_config.max_tokens
    }

    # Create rate limiter
    rate_limiter = RateLimiter(
        eval_config.requests_per_minute,
        eval_config.max_concurrent
    )

    # Run evaluation
    samples = await run_evaluation(
        samples,
        api_config,
        rate_limiter,
        output_file,
        state_file,
        run_id,
        config_dict,
        started_at,
        completed_ids,
        args.quiet
    )

    # Generate and save summary (only from completed samples)
    completed_at = datetime.now().isoformat()
    completed_samples = [s for s in samples if s.sample_id in completed_ids]
    summary = generate_summary(
        completed_samples, run_id, api_config, eval_config, started_at, completed_at
    )

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print_summary(summary, args.quiet)

    if not args.quiet:
        print(f"\nResults saved to: {output_dir}/")

    return 0


def main() -> None:
    """CLI entry point."""
    try:
        exit_code = asyncio.run(main_async())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted. State saved for resume.")
        sys.exit(1)


if __name__ == "__main__":
    main()
