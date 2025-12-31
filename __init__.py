"""
Synthetic Pre-training Playground: Four Additional Tasks

Following Allen-Zhu's design principles from "Physics of Language Models 4.1":
- Challenge architecture depth (System-1/mental reasoning)
- Short context (~2-4K tokens)
- Controllable difficulty for mini-scaling laws
- Orthogonal to existing tasks (Depo, Brevo, Knowledge, Mano, CFG)

Tasks implemented:
1. Dyna  - Dynamic State Tracking (Non-commutative permutation composition)
2. Inducto - In-Context Rule Induction (Few-shot function learning)
3. Filtro - Interleaved Source Separation (Selective attention under noise)
4. Logic - Boolean Circuit Evaluation (Non-linear XOR/parity computation)
"""

from .task_dyna import DynaTask
from .task_inducto import InductoTask
from .task_filtro import FiltroTask
from .task_logic import LogicTask

__all__ = ['DynaTask', 'InductoTask', 'FiltroTask', 'LogicTask']
