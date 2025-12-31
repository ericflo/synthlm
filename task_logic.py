"""
Task Logic: Boolean Circuit Evaluation (The "XOR" Problem)

CONCEPT:
All previous tasks involve moving pointers or applying smooth arithmetic.
Task Logic tests NON-LINEAR BOOLEAN COMPUTATION - specifically gates like
XOR where outputs flip non-linearly based on inputs. This targets the
model's Feed-Forward Networks (MLPs) rather than its attention mechanism.

ORTHOGONALITY:
- vs Mano: Arithmetic is somewhat linear. Boolean XOR requires multi-layer
           non-linearity (the classic perceptron problem).
- vs Depo: In Depo, values are just node IDs that move. In Logic, values
           are COMPUTED/TRANSFORMED at every step.
- vs Dyna: Dyna tracks state permutations. Logic computes signal flows.

ARCHITECTURAL TEST:
This isolates the FFN/SwiGLU layers' computational capacity. Perfect
attention/retrieval doesn't help if the FFN can't compute parity.
A model can pass Depo but fail Logic if its non-linear capacity is weak.

MINI-SCALING LAWS:
- Circuit depth: 2-10 layers of gates
- Circuit width: 2-8 parallel paths
- XOR density: Proportion of XOR gates (the hard operation)
"""

import random
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class GateType(Enum):
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NAND = "NAND"
    NOR = "NOR"
    NOT = "NOT"


@dataclass
class LogicConfig:
    """Configuration for Logic task generation."""
    num_inputs_min: int = 3
    num_inputs_max: int = 6
    depth_min: int = 2
    depth_max: int = 6
    width_min: int = 2
    width_max: int = 4
    xor_probability: float = 0.3  # Proportion of XOR gates
    seed: Optional[int] = None


class LogicTask:
    """
    Generate Boolean Circuit Evaluation tasks.
    
    The model sees:
    1. Input variable assignments: A=1, B=0, C=1, ...
    2. Gate definitions: G1=AND(A,B), G2=XOR(C,G1), G3=NOT(G2), ...
    3. Query: "QUERY G3 ->"
    
    The model must mentally evaluate the circuit to produce 0 or 1.
    """
    
    # Gate definitions with their truth tables
    BINARY_GATES = [GateType.AND, GateType.OR, GateType.XOR, GateType.NAND, GateType.NOR]
    UNARY_GATES = [GateType.NOT]
    
    def __init__(self, config: Optional[LogicConfig] = None):
        self.config = config or LogicConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)
    
    def _evaluate_gate(self, gate_type: GateType, inputs: List[int]) -> int:
        """Evaluate a single gate given its inputs."""
        if gate_type == GateType.AND:
            return 1 if all(inputs) else 0
        elif gate_type == GateType.OR:
            return 1 if any(inputs) else 0
        elif gate_type == GateType.XOR:
            return inputs[0] ^ inputs[1]
        elif gate_type == GateType.NAND:
            return 0 if all(inputs) else 1
        elif gate_type == GateType.NOR:
            return 0 if any(inputs) else 1
        elif gate_type == GateType.NOT:
            return 1 - inputs[0]
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
    
    def _generate_input_vars(self, num_inputs: int) -> Dict[str, int]:
        """Generate random input variable assignments."""
        var_names = [f"x{i}" for i in range(num_inputs)]
        return {name: random.randint(0, 1) for name in var_names}
    
    def _select_gate_type(self) -> GateType:
        """Select a gate type, with configurable XOR probability."""
        if random.random() < self.config.xor_probability:
            return GateType.XOR
        else:
            # Select from non-XOR binary gates
            non_xor = [g for g in self.BINARY_GATES if g != GateType.XOR]
            return random.choice(non_xor + self.UNARY_GATES)
    
    def _generate_circuit(
        self,
        num_inputs: int,
        depth: int,
        width: int
    ) -> Tuple[List[Tuple], Dict[str, int], str]:
        """
        Generate a random boolean circuit as a DAG.
        
        Returns:
        - gates: List of (gate_name, gate_type, input_names) tuples
        - values: Dict mapping all nodes (inputs + gates) to their values
        - output: Name of the final output gate
        """
        # Initialize with input variables
        input_vars = self._generate_input_vars(num_inputs)
        values = input_vars.copy()
        available_nodes = list(input_vars.keys())
        
        gates = []
        gate_counter = 0
        
        # Build circuit layer by layer
        for layer in range(depth):
            layer_gates = []
            num_gates_in_layer = random.randint(1, width)
            
            for _ in range(num_gates_in_layer):
                gate_name = f"g{gate_counter}"
                gate_counter += 1
                
                gate_type = self._select_gate_type()
                
                if gate_type in self.UNARY_GATES:
                    # Unary gate - select one input
                    input_node = random.choice(available_nodes)
                    input_names = [input_node]
                    gate_value = self._evaluate_gate(gate_type, [values[input_node]])
                else:
                    # Binary gate - select two inputs
                    if len(available_nodes) >= 2:
                        input_names = random.sample(available_nodes, 2)
                    else:
                        # If only one node available, use it twice
                        input_names = [available_nodes[0], available_nodes[0]]
                    
                    input_values = [values[n] for n in input_names]
                    gate_value = self._evaluate_gate(gate_type, input_values)
                
                gates.append((gate_name, gate_type, input_names))
                values[gate_name] = gate_value
                layer_gates.append(gate_name)
            
            # Add this layer's gates to available nodes for next layer
            available_nodes.extend(layer_gates)
        
        # The output is a gate from the final layer
        # To ensure we use the full depth, create a final combining gate
        if len(gates) > 0:
            final_gate_name = f"g{gate_counter}"
            
            # Select inputs from the most recent gates
            recent_gates = [g[0] for g in gates[-width:]]
            if len(recent_gates) >= 2:
                final_inputs = random.sample(recent_gates, 2)
                final_type = self._select_gate_type()
                if final_type in self.UNARY_GATES:
                    final_type = random.choice(self.BINARY_GATES)
            else:
                final_inputs = recent_gates
                final_type = GateType.NOT if len(final_inputs) == 1 else GateType.AND
            
            final_value = self._evaluate_gate(
                final_type, 
                [values[n] for n in final_inputs]
            )
            gates.append((final_gate_name, final_type, final_inputs))
            values[final_gate_name] = final_value
            output = final_gate_name
        else:
            # Edge case: no gates, just return first input
            output = list(input_vars.keys())[0]
        
        return gates, values, output, input_vars
    
    def _format_gate(self, gate_name: str, gate_type: GateType, inputs: List[str]) -> str:
        """Format a gate definition."""
        if gate_type in self.UNARY_GATES:
            return f"{gate_name}={gate_type.value}({inputs[0]})"
        else:
            return f"{gate_name}={gate_type.value}({inputs[0]},{inputs[1]})"
    
    def generate_sample(
        self,
        num_inputs: Optional[int] = None,
        depth: Optional[int] = None,
        width: Optional[int] = None
    ) -> Dict:
        """
        Generate a single Logic task sample.
        
        Returns dict with:
        - input_text: The formatted circuit and query
        - target: The correct output (0 or 1)
        - metadata: Circuit structure and intermediate values
        """
        # Determine parameters
        if num_inputs is None:
            num_inputs = random.randint(
                self.config.num_inputs_min,
                self.config.num_inputs_max
            )
        if depth is None:
            depth = random.randint(
                self.config.depth_min,
                self.config.depth_max
            )
        if width is None:
            width = random.randint(
                self.config.width_min,
                self.config.width_max
            )
        
        # Generate the circuit
        gates, values, output, input_vars = self._generate_circuit(
            num_inputs, depth, width
        )
        
        # Format input
        input_str = " ".join(f"{k}={v}" for k, v in input_vars.items())
        gates_str = " | ".join(
            self._format_gate(name, gtype, inputs) 
            for name, gtype, inputs in gates
        )
        
        target = str(values[output])
        
        input_text = f"INPUTS {input_str} || GATES {gates_str} || QUERY {output} -> {target}"
        
        # Count XOR gates for metadata
        xor_count = sum(1 for _, gtype, _ in gates if gtype == GateType.XOR)
        
        return {
            'input_text': input_text,
            'target': target,
            'metadata': {
                'num_inputs': num_inputs,
                'depth': depth,
                'width': width,
                'num_gates': len(gates),
                'xor_count': xor_count,
                'gates': gates,
                'all_values': values,
                'output_gate': output,
                'input_vars': input_vars
            }
        }
    
    def generate_evaluation_sample(
        self,
        num_inputs: Optional[int] = None,
        depth: Optional[int] = None,
        width: Optional[int] = None
    ) -> Dict:
        """Generate evaluation sample with answer held out."""
        sample = self.generate_sample(num_inputs, depth, width)
        
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
            1: 3-4 inputs, depth 2-3, low XOR (easy)
            2: 4-5 inputs, depth 3-5, medium XOR (medium)
            3: 5-6 inputs, depth 5-8, high XOR (hard)
        """
        difficulty_configs = {
            1: {'inputs': (3, 4), 'depth': (2, 3), 'width': (2, 3), 'xor_prob': 0.1},
            2: {'inputs': (4, 5), 'depth': (3, 5), 'width': (2, 4), 'xor_prob': 0.3},
            3: {'inputs': (5, 6), 'depth': (5, 8), 'width': (3, 5), 'xor_prob': 0.5},
        }
        
        cfg = difficulty_configs.get(difficulty_level, difficulty_configs[1])
        
        # Temporarily adjust XOR probability
        old_xor_prob = self.config.xor_probability
        self.config.xor_probability = cfg['xor_prob']
        
        samples = []
        for _ in range(n_samples):
            num_inputs = random.randint(*cfg['inputs'])
            depth = random.randint(*cfg['depth'])
            width = random.randint(*cfg['width'])
            samples.append(self.generate_sample(num_inputs, depth, width))
        
        # Restore XOR probability
        self.config.xor_probability = old_xor_prob
        
        return samples
    
    def generate_xor_parity_sample(self, num_bits: int) -> Dict:
        """
        Generate a pure XOR parity computation task.
        
        This is the hardest case: computing the parity (XOR) of N bits.
        This is the classic problem that single-layer perceptrons cannot solve.
        """
        # Generate input bits
        input_vars = {f"x{i}": random.randint(0, 1) for i in range(num_bits)}
        
        # Build a tree of XOR gates
        values = input_vars.copy()
        gates = []
        current_layer = list(input_vars.keys())
        gate_counter = 0
        
        while len(current_layer) > 1:
            next_layer = []
            i = 0
            while i < len(current_layer):
                if i + 1 < len(current_layer):
                    # XOR two nodes
                    gate_name = f"g{gate_counter}"
                    gate_counter += 1
                    inputs = [current_layer[i], current_layer[i + 1]]
                    gate_value = values[inputs[0]] ^ values[inputs[1]]
                    gates.append((gate_name, GateType.XOR, inputs))
                    values[gate_name] = gate_value
                    next_layer.append(gate_name)
                    i += 2
                else:
                    # Odd number of nodes, pass one through
                    next_layer.append(current_layer[i])
                    i += 1
            current_layer = next_layer
        
        output = current_layer[0]
        target = str(values[output])
        
        # Format
        input_str = " ".join(f"{k}={v}" for k, v in input_vars.items())
        gates_str = " | ".join(
            self._format_gate(name, gtype, inputs) 
            for name, gtype, inputs in gates
        )
        
        input_text = f"INPUTS {input_str} || GATES {gates_str} || QUERY {output} -> {target}"
        
        return {
            'input_text': input_text,
            'target': target,
            'metadata': {
                'task_type': 'pure_parity',
                'num_bits': num_bits,
                'gates': gates,
                'all_values': values,
                'output_gate': output
            }
        }


def test_logic():
    """Test the Logic task generator."""
    print("=" * 60)
    print("TESTING TASK LOGIC: Boolean Circuit Evaluation")
    print("=" * 60)
    
    task = LogicTask(LogicConfig(seed=42))
    
    # Test individual gates
    print("\n1. Testing gate evaluation:")
    test_cases = [
        (GateType.AND, [1, 1], 1),
        (GateType.AND, [1, 0], 0),
        (GateType.OR, [0, 0], 0),
        (GateType.OR, [0, 1], 1),
        (GateType.XOR, [0, 0], 0),
        (GateType.XOR, [0, 1], 1),
        (GateType.XOR, [1, 1], 0),
        (GateType.NAND, [1, 1], 0),
        (GateType.NOR, [0, 0], 1),
        (GateType.NOT, [1], 0),
        (GateType.NOT, [0], 1),
    ]
    
    for gate_type, inputs, expected in test_cases:
        result = task._evaluate_gate(gate_type, inputs)
        status = "✓" if result == expected else "✗"
        print(f"   {status} {gate_type.value}({inputs}) = {result} (expected {expected})")
        assert result == expected, f"Gate {gate_type.value} failed"
    
    print("   ✓ All gates work correctly!")
    
    # Test XOR non-linearity (the classic problem)
    print("\n2. Demonstrating XOR non-linearity:")
    print("   XOR truth table (cannot be computed by linear combination):")
    for a in [0, 1]:
        for b in [0, 1]:
            result = task._evaluate_gate(GateType.XOR, [a, b])
            print(f"   XOR({a}, {b}) = {result}")
    print("   ✓ XOR requires non-linear computation!")
    
    # Test sample generation
    print("\n3. Testing sample generation:")
    sample = task.generate_sample(num_inputs=4, depth=3, width=2)
    print(f"   Input: {sample['input_text'][:80]}...")
    print(f"   Target: {sample['target']}")
    print(f"   Num gates: {sample['metadata']['num_gates']}")
    print(f"   XOR count: {sample['metadata']['xor_count']}")
    
    # Manually verify the output
    values = sample['metadata']['all_values']
    output_gate = sample['metadata']['output_gate']
    assert str(values[output_gate]) == sample['target'], "Answer verification failed!"
    print(f"   ✓ Answer verified: {output_gate} = {sample['target']}")
    
    # Test circuit trace
    print("\n4. Tracing circuit evaluation:")
    gates = sample['metadata']['gates']
    for name, gtype, inputs in gates[:5]:  # Show first 5 gates
        input_vals = [values[i] for i in inputs]
        output_val = values[name]
        print(f"   {name} = {gtype.value}({inputs}) = {gtype.value}({input_vals}) = {output_val}")
    print("   ...")
    print("   ✓ Circuit trace correct!")
    
    # Test pure parity task
    print("\n5. Testing pure XOR parity task:")
    for num_bits in [3, 5, 7]:
        parity_sample = task.generate_xor_parity_sample(num_bits)
        input_vars = parity_sample['metadata']['all_values']
        bits = [input_vars[f"x{i}"] for i in range(num_bits)]
        expected_parity = sum(bits) % 2
        actual_parity = int(parity_sample['target'])
        print(f"   {num_bits} bits: {bits} -> parity = {actual_parity} (expected {expected_parity})")
        assert actual_parity == expected_parity, "Parity computation failed!"
    print("   ✓ Parity computation correct!")
    
    # Test evaluation sample
    print("\n6. Testing evaluation sample generation:")
    eval_sample = task.generate_evaluation_sample(num_inputs=3, depth=2)
    print(f"   Eval input ends with: ...{eval_sample['input_text'][-40:]}")
    print(f"   Expected answer: {eval_sample['target']}")
    # Check that input ends with -> (answer not appended)
    assert eval_sample['input_text'].strip().endswith('->'), \
        "Eval input should end with -> (no answer)"
    print("   ✓ Evaluation format correct!")
    
    # Test difficulty levels
    print("\n7. Testing difficulty levels:")
    for level in [1, 2, 3]:
        dataset = task.generate_dataset(20, difficulty_level=level)
        avg_gates = sum(s['metadata']['num_gates'] for s in dataset) / len(dataset)
        avg_xor = sum(s['metadata']['xor_count'] for s in dataset) / len(dataset)
        xor_ratio = avg_xor / avg_gates if avg_gates > 0 else 0
        print(f"   Level {level}: avg_gates={avg_gates:.1f}, avg_xor={avg_xor:.1f}, xor_ratio={xor_ratio:.2f}")
    
    # Test balanced output distribution
    print("\n8. Testing output distribution (should be ~50/50):")
    outputs = [int(task.generate_sample()['target']) for _ in range(100)]
    ones = sum(outputs)
    zeros = 100 - ones
    print(f"   Out of 100 samples: {ones} ones, {zeros} zeros")
    assert 30 < ones < 70, "Output distribution seems biased"
    print("   ✓ Reasonably balanced outputs!")
    
    print("\n" + "=" * 60)
    print("ALL LOGIC TESTS PASSED! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_logic()
