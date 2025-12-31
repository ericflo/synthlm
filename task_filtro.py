"""
Task Filtro: Interleaved Source Separation (The "Cocktail Party" Problem)

CONCEPT:
Real-world data is rarely clean. Filtro tests the architecture's ability to
SELECTIVELY ATTEND to a specific information stream while actively suppressing
a competing, high-interference stream interleaved in the same sequence.

ORTHOGONALITY:
- vs Depo: In Depo, every token is relevant. In Filtro, 50% is ACTIVE NOISE.
- vs Dyna: Dyna has all relevant operations. Filtro has adversarial distractors.
- vs CFG: CFG has implicit structure. Filtro has explicit stream markers.

ARCHITECTURAL TEST:
- Transformers EXCEL: Q·K mechanism can assign near-zero attention to noise.
- RNNs/SSMs STRUGGLE: Linear recurrence compresses history commutatively,
  so noise inevitably pollutes the relevant state.

MINI-SCALING LAWS:
- Graph sizes: 5-20 nodes per stream
- Hop depth: 1-8 hops
- Interleaving pattern: alternating vs random mixing
"""

import random
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class InterleavingPattern(Enum):
    ALTERNATING = "alternating"  # A B A B A B
    RANDOM = "random"            # Random mixing with balanced counts
    BLOCKED = "blocked"          # AA BB AA BB (blocks of 2-3)


@dataclass
class FiltroConfig:
    """Configuration for Filtro task generation."""
    min_nodes: int = 5
    max_nodes: int = 15
    min_hops: int = 1
    max_hops: int = 6
    pattern: InterleavingPattern = InterleavingPattern.ALTERNATING
    adversarial: bool = True  # Make Stream B have conflicting edges
    seed: Optional[int] = None


class FiltroTask:
    """
    Generate Interleaved Source Separation tasks.
    
    The model sees two interleaved "Depo-style" graphs (Stream A and Stream B)
    and must answer queries ONLY about Stream A while ignoring Stream B.
    
    Stream B is designed to be adversarial - it may have nodes with the same
    names pointing to DIFFERENT targets, creating interference.
    """
    
    # Two disjoint vocabularies for the streams
    VOCAB_A = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 
               'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron',
               'pi', 'rho', 'sigma', 'tau', 'upsilon']
    
    VOCAB_B = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
               'black', 'white', 'gray', 'brown', 'cyan', 'magenta', 'lime',
               'navy', 'teal', 'maroon', 'olive', 'silver', 'gold']
    
    def __init__(self, config: Optional[FiltroConfig] = None):
        self.config = config or FiltroConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)
    
    def _generate_cycle(self, nodes: List[str]) -> Dict[str, str]:
        """
        Generate a random cycle (permutation) over the nodes.
        Returns a dict mapping each node to its successor.
        """
        shuffled = nodes.copy()
        random.shuffle(shuffled)
        cycle = {}
        for i in range(len(shuffled)):
            cycle[shuffled[i]] = shuffled[(i + 1) % len(shuffled)]
        return cycle
    
    def _generate_adversarial_cycle(
        self, 
        nodes_a: List[str],
        cycle_a: Dict[str, str],
        nodes_b: List[str]
    ) -> Dict[str, str]:
        """
        Generate Stream B's cycle in an adversarial way.
        
        For maximum interference, we ensure that if we had shared node names,
        they would point to different places. Since we use disjoint vocabs,
        we instead make the PATTERNS similar but different.
        """
        # For truly disjoint vocabularies, just generate a random cycle
        # The adversarial aspect comes from the interleaving itself
        return self._generate_cycle(nodes_b)
    
    def _follow_path(self, cycle: Dict[str, str], start: str, hops: int) -> str:
        """Follow the cycle for K hops from start node."""
        current = start
        for _ in range(hops):
            current = cycle[current]
        return current
    
    def _format_edges(self, cycle: Dict[str, str], prefix: str) -> List[str]:
        """Format edges with stream prefix."""
        edges = []
        for src, dst in cycle.items():
            edges.append(f"{prefix}:{src}->{dst}")
        return edges
    
    def _interleave(
        self, 
        stream_a: List[str], 
        stream_b: List[str],
        pattern: InterleavingPattern
    ) -> List[str]:
        """Interleave two streams according to the pattern."""
        result = []
        
        if pattern == InterleavingPattern.ALTERNATING:
            # Strict alternation
            for a, b in zip(stream_a, stream_b):
                result.extend([a, b])
            # Handle any remaining elements
            result.extend(stream_a[len(stream_b):])
            result.extend(stream_b[len(stream_a):])
            
        elif pattern == InterleavingPattern.RANDOM:
            # Random interleaving
            a_items = list(enumerate(stream_a))
            b_items = list(enumerate(stream_b))
            all_items = [(0, i, x) for i, x in a_items] + [(1, i, x) for i, x in b_items]
            random.shuffle(all_items)
            result = [x for _, _, x in all_items]
            
        elif pattern == InterleavingPattern.BLOCKED:
            # Blocks of 2-3 items from each stream
            a_idx, b_idx = 0, 0
            current_stream = 0  # Start with A
            while a_idx < len(stream_a) or b_idx < len(stream_b):
                block_size = random.randint(2, 3)
                if current_stream == 0 and a_idx < len(stream_a):
                    for _ in range(block_size):
                        if a_idx < len(stream_a):
                            result.append(stream_a[a_idx])
                            a_idx += 1
                elif current_stream == 1 and b_idx < len(stream_b):
                    for _ in range(block_size):
                        if b_idx < len(stream_b):
                            result.append(stream_b[b_idx])
                            b_idx += 1
                current_stream = 1 - current_stream
                
                # Handle case where one stream is exhausted
                if a_idx >= len(stream_a):
                    current_stream = 1
                if b_idx >= len(stream_b):
                    current_stream = 0
        
        return result
    
    def generate_sample(
        self,
        num_nodes: Optional[int] = None,
        num_hops: Optional[int] = None,
        pattern: Optional[InterleavingPattern] = None
    ) -> Dict:
        """
        Generate a single Filtro task sample.
        
        Returns dict with:
        - input_text: Interleaved edges + query
        - target: The correct answer (from Stream A only)
        - metadata: Both cycles, query info, etc.
        """
        # Determine parameters
        if num_nodes is None:
            num_nodes = random.randint(
                self.config.min_nodes,
                self.config.max_nodes
            )
        if num_hops is None:
            num_hops = random.randint(
                self.config.min_hops,
                self.config.max_hops
            )
        if pattern is None:
            pattern = self.config.pattern
        
        # Generate node sets
        nodes_a = self.VOCAB_A[:num_nodes]
        nodes_b = self.VOCAB_B[:num_nodes]
        
        # Generate cycles
        cycle_a = self._generate_cycle(nodes_a)
        
        if self.config.adversarial:
            cycle_b = self._generate_adversarial_cycle(nodes_a, cycle_a, nodes_b)
        else:
            cycle_b = self._generate_cycle(nodes_b)
        
        # Format edges
        edges_a = self._format_edges(cycle_a, "A")
        edges_b = self._format_edges(cycle_b, "B")
        
        # Interleave the edges
        interleaved = self._interleave(edges_a, edges_b, pattern)
        
        # Generate query for Stream A
        query_start = random.choice(nodes_a)
        correct_answer = self._follow_path(cycle_a, query_start, num_hops)
        
        # Also compute what Stream B's answer would be for same hop count
        # (using B's corresponding start node for comparison)
        b_start_idx = nodes_a.index(query_start)
        b_start = nodes_b[b_start_idx]
        wrong_answer_b = self._follow_path(cycle_b, b_start, num_hops)
        
        # Format the input
        edges_str = " | ".join(interleaved)
        query_str = f"QUERY A:{query_start} hops={num_hops} ->"
        
        input_text = f"EDGES {edges_str} || {query_str} {correct_answer}"
        
        return {
            'input_text': input_text,
            'target': correct_answer,
            'metadata': {
                'num_nodes': num_nodes,
                'num_hops': num_hops,
                'pattern': pattern.value,
                'cycle_a': cycle_a,
                'cycle_b': cycle_b,
                'query_start': query_start,
                'correct_answer': correct_answer,
                'distractor_answer': wrong_answer_b,
                'nodes_a': nodes_a,
                'nodes_b': nodes_b
            }
        }
    
    def generate_evaluation_sample(
        self,
        num_nodes: Optional[int] = None,
        num_hops: Optional[int] = None,
        pattern: Optional[InterleavingPattern] = None
    ) -> Dict:
        """Generate evaluation sample with answer held out."""
        sample = self.generate_sample(num_nodes, num_hops, pattern)
        
        # Remove answer from input
        parts = sample['input_text'].rsplit(' ', 1)
        eval_input = parts[0]
        
        return {
            'input_text': eval_input,
            'target': sample['target'],
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
            1: 5-8 nodes, 1-2 hops, alternating (easy)
            2: 8-12 nodes, 2-4 hops, alternating/random (medium)
            3: 10-15 nodes, 4-8 hops, random/blocked (hard)
        """
        difficulty_configs = {
            1: {
                'nodes': (5, 8), 
                'hops': (1, 2), 
                'patterns': [InterleavingPattern.ALTERNATING]
            },
            2: {
                'nodes': (8, 12), 
                'hops': (2, 4), 
                'patterns': [InterleavingPattern.ALTERNATING, InterleavingPattern.RANDOM]
            },
            3: {
                'nodes': (10, 15), 
                'hops': (4, 8), 
                'patterns': [InterleavingPattern.RANDOM, InterleavingPattern.BLOCKED]
            },
        }
        
        cfg = difficulty_configs.get(difficulty_level, difficulty_configs[1])
        
        samples = []
        for _ in range(n_samples):
            num_nodes = random.randint(*cfg['nodes'])
            num_hops = random.randint(*cfg['hops'])
            pattern = random.choice(cfg['patterns'])
            samples.append(self.generate_sample(num_nodes, num_hops, pattern))
        
        return samples


def test_filtro():
    """Test the Filtro task generator."""
    print("=" * 60)
    print("TESTING TASK FILTRO: Interleaved Source Separation")
    print("=" * 60)
    
    task = FiltroTask(FiltroConfig(seed=42))
    
    # Test cycle generation
    print("\n1. Testing cycle generation:")
    nodes = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    cycle = task._generate_cycle(nodes)
    print(f"   Nodes: {nodes}")
    print(f"   Cycle: {cycle}")
    
    # Verify it's a valid cycle (each node appears exactly once as src and dst)
    assert set(cycle.keys()) == set(nodes), "Invalid cycle - wrong keys"
    assert set(cycle.values()) == set(nodes), "Invalid cycle - wrong values"
    print("   ✓ Valid cycle generated!")
    
    # Test path following
    print("\n2. Testing path following:")
    start = list(cycle.keys())[0]
    for hops in [1, 2, 3]:
        result = task._follow_path(cycle, start, hops)
        print(f"   From {start}, {hops} hops -> {result}")
    print("   ✓ Path following works!")
    
    # Test interleaving patterns
    print("\n3. Testing interleaving patterns:")
    stream_a = ['A1', 'A2', 'A3', 'A4']
    stream_b = ['B1', 'B2', 'B3', 'B4']
    
    for pattern in InterleavingPattern:
        interleaved = task._interleave(stream_a, stream_b, pattern)
        a_count = sum(1 for x in interleaved if x.startswith('A'))
        b_count = sum(1 for x in interleaved if x.startswith('B'))
        print(f"   {pattern.value}: {interleaved[:6]}... (A:{a_count}, B:{b_count})")
        assert a_count == len(stream_a), f"Lost A items in {pattern.value}"
        assert b_count == len(stream_b), f"Lost B items in {pattern.value}"
    print("   ✓ All interleaving patterns preserve items!")
    
    # Test sample generation
    print("\n4. Testing sample generation:")
    sample = task.generate_sample(num_nodes=5, num_hops=2)
    print(f"   Input length: {len(sample['input_text'])} chars")
    print(f"   Target: {sample['target']}")
    print(f"   Distractor: {sample['metadata']['distractor_answer']}")
    
    # Verify answer is correct by manual path following
    start = sample['metadata']['query_start']
    cycle_a = sample['metadata']['cycle_a']
    hops = sample['metadata']['num_hops']
    manual_answer = task._follow_path(cycle_a, start, hops)
    assert manual_answer == sample['target'], "Answer verification failed!"
    print(f"   ✓ Answer verified: {start} --{hops}hops--> {manual_answer}")
    
    # Test that disjoint vocabularies are used
    print("\n5. Testing vocabulary separation:")
    nodes_a = sample['metadata']['nodes_a']
    nodes_b = sample['metadata']['nodes_b']
    overlap = set(nodes_a) & set(nodes_b)
    print(f"   Stream A vocab: {nodes_a[:3]}...")
    print(f"   Stream B vocab: {nodes_b[:3]}...")
    assert len(overlap) == 0, "Vocabularies should be disjoint!"
    print("   ✓ Disjoint vocabularies confirmed!")
    
    # Test that both streams appear in interleaved output
    print("\n6. Testing stream mixing:")
    input_text = sample['input_text']
    has_a = 'A:' in input_text
    has_b = 'B:' in input_text
    assert has_a and has_b, "Both streams should appear in input!"
    print(f"   Contains A: edges: {has_a}")
    print(f"   Contains B: edges: {has_b}")
    print("   ✓ Both streams present!")
    
    # Test evaluation sample
    print("\n7. Testing evaluation sample generation:")
    eval_sample = task.generate_evaluation_sample(num_nodes=5, num_hops=2)
    print(f"   Eval input ends with: ...{eval_sample['input_text'][-30:]}")
    print(f"   Expected answer: {eval_sample['target']}")
    # Check that input ends with -> (answer not appended)
    assert eval_sample['input_text'].strip().endswith('->'), \
        "Eval input should end with -> (no answer)"
    print("   ✓ Evaluation format correct!")
    
    # Test difficulty levels
    print("\n8. Testing difficulty levels:")
    for level in [1, 2, 3]:
        dataset = task.generate_dataset(10, difficulty_level=level)
        avg_nodes = sum(s['metadata']['num_nodes'] for s in dataset) / len(dataset)
        avg_hops = sum(s['metadata']['num_hops'] for s in dataset) / len(dataset)
        patterns = set(s['metadata']['pattern'] for s in dataset)
        print(f"   Level {level}: nodes={avg_nodes:.1f}, hops={avg_hops:.1f}, patterns={patterns}")
    
    print("\n" + "=" * 60)
    print("ALL FILTRO TESTS PASSED! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_filtro()
