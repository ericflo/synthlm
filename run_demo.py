"""
Comprehensive Test Suite & Demo for Synthetic Pre-training Playground

This script:
1. Runs all unit tests for all four tasks
2. Demonstrates example outputs from each task
3. Shows the mini-scaling law structure for each task
"""

import sys
import random
from typing import Dict, List

# Import our tasks
from task_dyna import DynaTask, DynaConfig, test_dyna
from task_inducto import InductoTask, InductoConfig, test_inducto
from task_filtro import FiltroTask, FiltroConfig, test_filtro
from task_logic import LogicTask, LogicConfig, test_logic


def run_all_tests():
    """Run all unit tests for all tasks."""
    print("\n" + "=" * 70)
    print("RUNNING ALL UNIT TESTS")
    print("=" * 70)
    
    results = {}
    
    # Test each task
    for name, test_fn in [
        ("Dyna", test_dyna),
        ("Inducto", test_inducto),
        ("Filtro", test_filtro),
        ("Logic", test_logic)
    ]:
        print(f"\n{'='*70}")
        print(f"Testing Task {name}...")
        print('='*70)
        try:
            test_fn()
            results[name] = "PASSED ✓"
        except AssertionError as e:
            results[name] = f"FAILED: {e}"
        except Exception as e:
            results[name] = f"ERROR: {e}"
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, result in results.items():
        print(f"  {name}: {result}")
    
    all_passed = all("PASSED" in r for r in results.values())
    return all_passed


def demonstrate_tasks():
    """Show example outputs from each task."""
    print("\n" + "=" * 70)
    print("TASK DEMONSTRATIONS")
    print("=" * 70)
    
    random.seed(123)  # For reproducible demos
    
    # =========================================================================
    # Task Dyna Demo
    # =========================================================================
    print("\n" + "-" * 70)
    print("TASK DYNA: Dynamic State Tracking (Non-Commutative Permutations)")
    print("-" * 70)
    print("""
CONCEPT: Track mutable state through non-commutative operations.
The answer exists NOWHERE in the input - must be computed mentally.

ORTHOGONALITY:
  - vs Depo: Retrieves explicit X->Y. Dyna computes implicit state.
  - vs Mano: Fixed rules in weights. Dyna: dynamic rules in context.

ARCHITECTURAL CHALLENGE: 
  Linear attention/RNNs use commutative compression - they FAIL when 
  order matters (A·B ≠ B·A).
""")
    
    dyna = DynaTask(DynaConfig(seed=42))
    sample = dyna.generate_sample(state_size=5, num_ops=4)
    
    print("EXAMPLE:")
    print(f"  Initial state: {sample['metadata']['initial_state']}")
    print("  Operations:")
    for op, args in sample['metadata']['operations']:
        print(f"    {op}({', '.join(map(str, args))})")
    print(f"  Final state: {sample['metadata']['final_state']}")
    print(f"  Query: What's at position 2?")
    print(f"  ANSWER: {sample['metadata']['final_state'][2]}")
    
    # =========================================================================
    # Task Inducto Demo
    # =========================================================================
    print("\n" + "-" * 70)
    print("TASK INDUCTO: In-Context Rule Induction (Few-Shot Learning)")
    print("-" * 70)
    print("""
CONCEPT: Infer a NOVEL rule from examples and apply it immediately.
This is the atomic skill behind few-shot learning.

ORTHOGONALITY:
  - vs Knowledge: Zero memorization required.
  - vs Mano: Mano APPLIES known rules. Inducto FINDS the rule.
  - vs CFG: CFG parses one sequence. Inducto transfers between examples.

ARCHITECTURAL CHALLENGE:
  Tests the Attention mechanism as a pattern-matcher and copier.
  The "Induction Head" hypothesis: [A][B]...[A] -> predict [B].
""")
    
    inducto = InductoTask(InductoConfig(seed=42))
    sample = inducto.generate_sample(num_examples=3, complexity=1)
    
    print("EXAMPLE:")
    print(f"  Rule (hidden from model): {sample['metadata']['rule_name']}")
    print("  Examples shown to model:")
    for i, ex in enumerate(sample['metadata']['examples'], 1):
        print(f"    {i}. {ex['input']} -> {ex['output']}")
    print(f"  Query: {sample['metadata']['test_input']} -> ???")
    print(f"  ANSWER: {sample['metadata']['test_output']}")
    
    # =========================================================================
    # Task Filtro Demo
    # =========================================================================
    print("\n" + "-" * 70)
    print("TASK FILTRO: Interleaved Source Separation (Cocktail Party)")
    print("-" * 70)
    print("""
CONCEPT: Selectively attend to ONE stream while suppressing another
interleaved stream. 50% of context is ACTIVE NOISE.

ORTHOGONALITY:
  - vs Depo: Every token relevant. Filtro: half are distractors.
  - Tests the FILTERING/MASKING capability of attention.

ARCHITECTURAL CHALLENGE:
  - Transformers EXCEL: Q·K assigns zero weight to noise.
  - RNNs/SSMs STRUGGLE: Linear compression pollutes state with noise.
""")
    
    filtro = FiltroTask(FiltroConfig(seed=42))
    sample = filtro.generate_sample(num_nodes=4, num_hops=2)
    
    print("EXAMPLE:")
    print("  Stream A (RELEVANT) cycle:")
    for src, dst in list(sample['metadata']['cycle_a'].items())[:4]:
        print(f"    {src} -> {dst}")
    print("  Stream B (NOISE) cycle:")
    for src, dst in list(sample['metadata']['cycle_b'].items())[:4]:
        print(f"    {src} -> {dst}")
    print(f"  Query: Follow Stream A from '{sample['metadata']['query_start']}' for {sample['metadata']['num_hops']} hops")
    print(f"  CORRECT (from A): {sample['target']}")
    print(f"  DISTRACTOR (from B): {sample['metadata']['distractor_answer']}")
    
    # =========================================================================
    # Task Logic Demo
    # =========================================================================
    print("\n" + "-" * 70)
    print("TASK LOGIC: Boolean Circuit Evaluation (XOR/Parity)")
    print("-" * 70)
    print("""
CONCEPT: Evaluate circuits with NON-LINEAR gates (especially XOR).
This targets the FFN/MLP layers, not attention.

ORTHOGONALITY:
  - vs Mano: Arithmetic is somewhat smooth. XOR requires multi-layer 
             non-linearity (classic perceptron problem).
  - vs Depo: Values are IDs that move. Logic: values TRANSFORM.

ARCHITECTURAL CHALLENGE:
  Tests whether FFN layers can compute PARITY (sum mod 2).
  Perfect attention doesn't help if the FFN is too weak!
""")
    
    logic = LogicTask(LogicConfig(seed=42))
    
    # Show the pure parity example
    parity_sample = logic.generate_xor_parity_sample(4)
    print("EXAMPLE (Pure Parity - Hardest Case):")
    input_bits = [parity_sample['metadata']['all_values'][f'x{i}'] for i in range(4)]
    print(f"  Input bits: {input_bits}")
    print(f"  Task: Compute XOR of all bits (parity)")
    print(f"  ANSWER: {parity_sample['target']} (because {' ⊕ '.join(map(str, input_bits))} = {parity_sample['target']})")
    
    # Also show a general circuit
    print("\n  General Circuit Example:")
    sample = logic.generate_sample(num_inputs=3, depth=2, width=2)
    print(f"  Inputs: {sample['metadata']['input_vars']}")
    print(f"  Gates:")
    for name, gtype, inputs in sample['metadata']['gates']:
        val = sample['metadata']['all_values'][name]
        print(f"    {name} = {gtype.value}({', '.join(inputs)}) = {val}")
    print(f"  Query: {sample['metadata']['output_gate']}")
    print(f"  ANSWER: {sample['target']}")


def show_mini_scaling_laws():
    """Demonstrate the mini-scaling law structure for each task."""
    print("\n" + "=" * 70)
    print("MINI-SCALING LAW STRUCTURE")
    print("=" * 70)
    print("""
Following Allen-Zhu's methodology, each task supports 2D scaling laws:
  - Axis 1: Model size (layers, dimensions)  
  - Axis 2: Task difficulty (controlled via parameters)

This allows quantifying: "Architecture A reasons N hops deeper than B"
rather than just "A is 0.5% better on Benchmark X".
""")
    
    print("\n" + "-" * 50)
    print("Task Dyna Scaling Dimensions:")
    print("-" * 50)
    print("  Difficulty Axis: Number of operations (5 -> 50)")
    print("  Secondary: State size (5 -> 20 elements)")
    
    dyna = DynaTask()
    for level in [1, 2, 3]:
        samples = dyna.generate_dataset(5, difficulty_level=level)
        avg_ops = sum(s['metadata']['num_ops'] for s in samples) / len(samples)
        avg_state = sum(s['metadata']['state_size'] for s in samples) / len(samples)
        print(f"    Level {level}: ~{avg_ops:.0f} ops, ~{avg_state:.0f} state size")
    
    print("\n" + "-" * 50)
    print("Task Inducto Scaling Dimensions:")
    print("-" * 50)
    print("  Difficulty Axis: Rule complexity (1=simple -> 3=composite)")
    print("  Secondary: Number of examples (1-shot -> 10-shot)")
    
    inducto = InductoTask()
    for level in [1, 2, 3]:
        samples = inducto.generate_dataset(5, difficulty_level=level)
        complexities = set(s['metadata']['rule_complexity'] for s in samples)
        avg_ex = sum(s['metadata']['num_examples'] for s in samples) / len(samples)
        print(f"    Level {level}: complexity in {complexities}, ~{avg_ex:.1f} examples")
    
    print("\n" + "-" * 50)
    print("Task Filtro Scaling Dimensions:")
    print("-" * 50)
    print("  Difficulty Axis: Hop depth (1 -> 8)")
    print("  Secondary: Graph size (5 -> 15 nodes)")
    
    filtro = FiltroTask()
    for level in [1, 2, 3]:
        samples = filtro.generate_dataset(5, difficulty_level=level)
        avg_hops = sum(s['metadata']['num_hops'] for s in samples) / len(samples)
        avg_nodes = sum(s['metadata']['num_nodes'] for s in samples) / len(samples)
        print(f"    Level {level}: ~{avg_hops:.1f} hops, ~{avg_nodes:.0f} nodes")
    
    print("\n" + "-" * 50)
    print("Task Logic Scaling Dimensions:")
    print("-" * 50)
    print("  Difficulty Axis: Circuit depth (2 -> 8 layers)")
    print("  Secondary: XOR density (10% -> 50%)")
    
    logic = LogicTask()
    for level in [1, 2, 3]:
        samples = logic.generate_dataset(5, difficulty_level=level)
        avg_gates = sum(s['metadata']['num_gates'] for s in samples) / len(samples)
        avg_xor = sum(s['metadata']['xor_count'] for s in samples) / len(samples)
        xor_pct = (avg_xor / avg_gates * 100) if avg_gates > 0 else 0
        print(f"    Level {level}: ~{avg_gates:.0f} gates, ~{xor_pct:.0f}% XOR")


def show_orthogonality_matrix():
    """Show how the tasks are orthogonal to each other and prior work."""
    print("\n" + "=" * 70)
    print("ORTHOGONALITY MATRIX: Tasks vs Skills Tested")
    print("=" * 70)
    
    matrix = """
┌─────────────┬────────┬────────┬─────────┬────────┬─────────┬─────────┐
│   Task      │ Depo   │ Brevo  │ Knowl.  │ Mano   │ CFG     │ New     │
│             │(depth) │(width) │ (mem)   │(manip) │(struct) │ Skill   │
├─────────────┼────────┼────────┼─────────┼────────┼─────────┼─────────┤
│ DYNA        │   ✗    │   ✗    │   ✗     │   ✗    │   ✗     │ State   │
│ (this work) │        │        │         │        │         │ Tracking│
├─────────────┼────────┼────────┼─────────┼────────┼─────────┼─────────┤
│ INDUCTO     │   ✗    │   ✗    │   ✗     │   ✗    │   ✗     │ Rule    │
│ (this work) │        │        │         │        │         │ Induct. │
├─────────────┼────────┼────────┼─────────┼────────┼─────────┼─────────┤
│ FILTRO      │  (✓)   │   ✗    │   ✗     │   ✗    │   ✗     │ Source  │
│ (this work) │ +noise │        │         │        │         │ Separ.  │
├─────────────┼────────┼────────┼─────────┼────────┼─────────┼─────────┤
│ LOGIC       │   ✗    │   ✗    │   ✗     │  (✓)   │   ✗     │ Boolean │
│ (this work) │        │        │         │+nonlin │         │ XOR     │
└─────────────┴────────┴────────┴─────────┴────────┴─────────┴─────────┘

Legend:
  ✗ = Orthogonal (different skill)
  (✓) = Partially overlaps but adds new dimension
  
Key Differentiators:
  • DYNA: Mutable state (vs static graphs in Depo)
  • INDUCTO: Learning the rule (vs applying known rules in Mano)
  • FILTRO: Noise suppression (vs clean data in Depo)
  • LOGIC: Non-linear XOR (vs smooth arithmetic in Mano)
"""
    print(matrix)


def main():
    """Main entry point."""
    print("=" * 70)
    print("SYNTHETIC PRE-TRAINING PLAYGROUND")
    print("Four New Tasks for Architecture Evaluation")
    print("=" * 70)
    print("""
Following Allen-Zhu's "Physics of Language Models 4.1" methodology:
  1. Decompose intelligence into ATOMIC skills
  2. Control difficulty for MINI-SCALING LAWS  
  3. Test SYSTEM-1 (mental) reasoning, not chain-of-thought
  4. Keep context SHORT (~2-4K tokens)
  5. Be ORTHOGONAL to existing tasks
""")
    
    # Run tests
    all_passed = run_all_tests()
    
    if not all_passed:
        print("\n⚠️  Some tests failed! Please check the output above.")
        sys.exit(1)
    
    # Show demonstrations
    demonstrate_tasks()
    
    # Show scaling law structure
    show_mini_scaling_laws()
    
    # Show orthogonality
    show_orthogonality_matrix()
    
    print("\n" + "=" * 70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
