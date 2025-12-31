"""
Task Dyna: Dynamic State Tracking (Non-Commutative Permutation Composition)

CONCEPT:
Unlike Depo/Brevo where the graph is static and answers are retrieved from context,
Dyna requires the model to maintain a MUTABLE state that updates destructively.
The answer exists NOWHERE in the input - it's a latent variable computed through
sequential composition of permutations.

ORTHOGONALITY:
- vs Depo: Depo retrieves X->Y from context. Dyna computes implicit state.
- vs Mano: Mano uses fixed rules in weights. Dyna uses dynamic rules in context.
- vs Brevo: Brevo navigates static structure. Dyna tracks changing state.

ARCHITECTURAL TEST:
Linear attention and RNNs struggle with non-commutative updates because they
compress history using commutative operations (summation). If order matters
(A·B ≠ B·A), averaging models fail catastrophically.

MINI-SCALING LAWS:
- State size: 5 to 20 elements
- Sequence length: 5 to 50 operations
"""

import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DynaConfig:
    """Configuration for Dyna task generation."""
    state_size_min: int = 5
    state_size_max: int = 15
    num_ops_min: int = 5
    num_ops_max: int = 30
    num_queries: int = 3
    use_symbolic_names: bool = True  # Use A,B,C vs 0,1,2
    seed: Optional[int] = None


class DynaTask:
    """
    Generate Dynamic State Tracking tasks.
    
    The model sees:
    1. Initial state: [A, B, C, D, E]
    2. Sequence of operations: SWAP(1,3), ROTATE_LEFT(2), SWAP(0,4), ...
    3. Query: "Position 2?" or "Where is C?"
    
    The model must mentally track the state through all operations.
    """
    
    OPERATIONS = ['SWAP', 'ROTATE_LEFT', 'ROTATE_RIGHT', 'REVERSE_SEGMENT']
    
    def __init__(self, config: Optional[DynaConfig] = None):
        self.config = config or DynaConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)
    
    def _generate_element_names(self, n: int) -> List[str]:
        """Generate symbolic names for state elements."""
        if self.config.use_symbolic_names:
            # Use letters, then letter+number for larger states
            names = []
            for i in range(n):
                if i < 26:
                    names.append(chr(ord('A') + i))
                else:
                    names.append(f"{chr(ord('A') + (i // 26) - 1)}{i % 26}")
            return names
        else:
            return [str(i) for i in range(n)]
    
    def _apply_swap(self, state: List[str], i: int, j: int) -> List[str]:
        """Swap elements at positions i and j."""
        state = state.copy()
        state[i], state[j] = state[j], state[i]
        return state
    
    def _apply_rotate_left(self, state: List[str], k: int) -> List[str]:
        """Rotate state left by k positions."""
        k = k % len(state)
        return state[k:] + state[:k]
    
    def _apply_rotate_right(self, state: List[str], k: int) -> List[str]:
        """Rotate state right by k positions."""
        k = k % len(state)
        return state[-k:] + state[:-k] if k > 0 else state.copy()
    
    def _apply_reverse_segment(self, state: List[str], i: int, j: int) -> List[str]:
        """Reverse the segment from position i to j (inclusive)."""
        state = state.copy()
        # Ensure i <= j
        if i > j:
            i, j = j, i
        state[i:j+1] = state[i:j+1][::-1]
        return state
    
    def _generate_operation(self, state_size: int) -> Tuple[str, List[int], str]:
        """
        Generate a random operation.
        Returns: (op_name, args, string_repr)
        """
        op = random.choice(self.OPERATIONS)
        
        if op == 'SWAP':
            i, j = random.sample(range(state_size), 2)
            return (op, [i, j], f"SWAP({i},{j})")
        
        elif op == 'ROTATE_LEFT':
            k = random.randint(1, state_size - 1)
            return (op, [k], f"ROTATE_LEFT({k})")
        
        elif op == 'ROTATE_RIGHT':
            k = random.randint(1, state_size - 1)
            return (op, [k], f"ROTATE_RIGHT({k})")
        
        elif op == 'REVERSE_SEGMENT':
            i, j = sorted(random.sample(range(state_size), 2))
            return (op, [i, j], f"REVERSE({i},{j})")
        
        raise ValueError(f"Unknown operation: {op}")
    
    def _apply_operation(self, state: List[str], op: str, args: List[int]) -> List[str]:
        """Apply an operation to the state."""
        if op == 'SWAP':
            return self._apply_swap(state, args[0], args[1])
        elif op == 'ROTATE_LEFT':
            return self._apply_rotate_left(state, args[0])
        elif op == 'ROTATE_RIGHT':
            return self._apply_rotate_right(state, args[0])
        elif op == 'REVERSE_SEGMENT':
            return self._apply_reverse_segment(state, args[0], args[1])
        raise ValueError(f"Unknown operation: {op}")
    
    def generate_sample(
        self,
        state_size: Optional[int] = None,
        num_ops: Optional[int] = None
    ) -> Dict:
        """
        Generate a single Dyna task sample.
        
        Returns dict with:
        - input_text: The formatted input string
        - target: The answer string
        - metadata: Additional information for analysis
        """
        # Determine parameters
        if state_size is None:
            state_size = random.randint(
                self.config.state_size_min,
                self.config.state_size_max
            )
        if num_ops is None:
            num_ops = random.randint(
                self.config.num_ops_min,
                self.config.num_ops_max
            )
        
        # Generate initial state
        elements = self._generate_element_names(state_size)
        initial_state = elements.copy()
        random.shuffle(initial_state)
        
        # Generate and apply operations
        operations = []
        operation_strings = []
        current_state = initial_state.copy()
        
        for _ in range(num_ops):
            op, args, op_str = self._generate_operation(state_size)
            operations.append((op, args))
            operation_strings.append(op_str)
            current_state = self._apply_operation(current_state, op, args)
        
        final_state = current_state
        
        # Generate queries (mix of "what's at position X" and "where is element Y")
        queries = []
        answers = []
        
        for _ in range(self.config.num_queries):
            query_type = random.choice(['position', 'element'])
            
            if query_type == 'position':
                pos = random.randint(0, state_size - 1)
                queries.append(f"QUERY position {pos} ->")
                answers.append(final_state[pos])
            else:
                elem = random.choice(elements)
                queries.append(f"QUERY element {elem} ->")
                answers.append(str(final_state.index(elem)))
        
        # Format input
        input_parts = [
            f"STATE [{', '.join(initial_state)}]",
            "OPS " + " | ".join(operation_strings),
        ]
        
        # For training, include all queries with answers
        # For evaluation, we'd separate them
        query_answer_pairs = [
            f"{q} {a}" for q, a in zip(queries, answers)
        ]
        
        input_text = " || ".join(input_parts) + " || " + " | ".join(query_answer_pairs)
        
        return {
            'input_text': input_text,
            'target': answers[0],  # Primary answer for loss computation
            'metadata': {
                'state_size': state_size,
                'num_ops': num_ops,
                'initial_state': initial_state,
                'final_state': final_state,
                'operations': operations,
                'all_queries': queries,
                'all_answers': answers
            }
        }
    
    def generate_evaluation_sample(
        self,
        state_size: Optional[int] = None,
        num_ops: Optional[int] = None
    ) -> Dict:
        """
        Generate an evaluation sample where the answer is held out.
        """
        sample = self.generate_sample(state_size, num_ops)
        
        # Remove answer from input, keep only the query
        parts = sample['input_text'].split(' || ')
        state_part = parts[0]
        ops_part = parts[1]
        query_parts = parts[2].split(' | ')
        
        # Take first query, remove answer
        first_query = query_parts[0].rsplit(' ', 1)[0]  # Remove answer
        
        eval_input = f"{state_part} || {ops_part} || {first_query}"
        
        return {
            'input_text': eval_input,
            'target': sample['metadata']['all_answers'][0],
            'metadata': sample['metadata']
        }
    
    def generate_dataset(
        self,
        n_samples: int,
        difficulty_level: int = 1
    ) -> List[Dict]:
        """
        Generate a dataset with controlled difficulty.
        
        difficulty_level:
            1: state_size=5-7,  num_ops=5-10  (easy)
            2: state_size=8-12, num_ops=10-20 (medium)
            3: state_size=13-20, num_ops=20-50 (hard)
        """
        difficulty_configs = {
            1: {'state_size': (5, 7), 'num_ops': (5, 10)},
            2: {'state_size': (8, 12), 'num_ops': (10, 20)},
            3: {'state_size': (13, 20), 'num_ops': (20, 50)},
        }
        
        cfg = difficulty_configs.get(difficulty_level, difficulty_configs[1])
        
        samples = []
        for _ in range(n_samples):
            state_size = random.randint(*cfg['state_size'])
            num_ops = random.randint(*cfg['num_ops'])
            samples.append(self.generate_sample(state_size, num_ops))
        
        return samples


def test_dyna():
    """Test the Dyna task generator."""
    print("=" * 60)
    print("TESTING TASK DYNA: Dynamic State Tracking")
    print("=" * 60)
    
    # Test basic functionality
    task = DynaTask(DynaConfig(seed=42))
    
    # Test individual operations
    print("\n1. Testing individual operations:")
    state = ['A', 'B', 'C', 'D', 'E']
    print(f"   Initial: {state}")
    
    swapped = task._apply_swap(state, 0, 4)
    print(f"   SWAP(0,4): {swapped}")
    assert swapped == ['E', 'B', 'C', 'D', 'A'], "SWAP failed"
    
    rotated_l = task._apply_rotate_left(state, 2)
    print(f"   ROTATE_LEFT(2): {rotated_l}")
    assert rotated_l == ['C', 'D', 'E', 'A', 'B'], "ROTATE_LEFT failed"
    
    rotated_r = task._apply_rotate_right(state, 2)
    print(f"   ROTATE_RIGHT(2): {rotated_r}")
    assert rotated_r == ['D', 'E', 'A', 'B', 'C'], "ROTATE_RIGHT failed"
    
    reversed_seg = task._apply_reverse_segment(state, 1, 3)
    print(f"   REVERSE(1,3): {reversed_seg}")
    assert reversed_seg == ['A', 'D', 'C', 'B', 'E'], "REVERSE_SEGMENT failed"
    
    print("   ✓ All operations work correctly!")
    
    # Test non-commutativity
    print("\n2. Testing non-commutativity (A·B ≠ B·A):")
    state = ['A', 'B', 'C', 'D', 'E']
    
    # Apply SWAP(0,1) then ROTATE_LEFT(1)
    result1 = task._apply_rotate_left(task._apply_swap(state, 0, 1), 1)
    # Apply ROTATE_LEFT(1) then SWAP(0,1)
    result2 = task._apply_swap(task._apply_rotate_left(state, 1), 0, 1)
    
    print(f"   SWAP(0,1) then ROTATE_LEFT(1): {result1}")
    print(f"   ROTATE_LEFT(1) then SWAP(0,1): {result2}")
    assert result1 != result2, "Operations should be non-commutative!"
    print("   ✓ Confirmed non-commutativity!")
    
    # Test sample generation
    print("\n3. Testing sample generation:")
    sample = task.generate_sample(state_size=6, num_ops=5)
    print(f"   Input: {sample['input_text'][:100]}...")
    print(f"   Target: {sample['target']}")
    print(f"   State size: {sample['metadata']['state_size']}")
    print(f"   Num ops: {sample['metadata']['num_ops']}")
    
    # Verify the answer is correct by replaying
    state = sample['metadata']['initial_state'].copy()
    for op, args in sample['metadata']['operations']:
        state = task._apply_operation(state, op, args)
    assert state == sample['metadata']['final_state'], "State tracking mismatch!"
    print("   ✓ Answer verification passed!")
    
    # Test difficulty levels
    print("\n4. Testing difficulty levels:")
    for level in [1, 2, 3]:
        dataset = task.generate_dataset(10, difficulty_level=level)
        avg_ops = sum(s['metadata']['num_ops'] for s in dataset) / len(dataset)
        avg_size = sum(s['metadata']['state_size'] for s in dataset) / len(dataset)
        print(f"   Level {level}: avg_ops={avg_ops:.1f}, avg_state_size={avg_size:.1f}")
    
    # Test evaluation sample
    print("\n5. Testing evaluation sample generation:")
    eval_sample = task.generate_evaluation_sample(state_size=5, num_ops=3)
    print(f"   Eval input: {eval_sample['input_text']}")
    print(f"   Expected answer: {eval_sample['target']}")
    # Check that input ends with -> (answer not appended)
    assert eval_sample['input_text'].strip().endswith('->'), \
        "Eval input should end with -> (no answer)"
    print("   ✓ Evaluation format correct!")
    
    print("\n" + "=" * 60)
    print("ALL DYNA TESTS PASSED! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_dyna()
