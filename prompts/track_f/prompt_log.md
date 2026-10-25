# Track F Prompt & Engineering Audit Log
## Compiler Optimization (TAC IR, 4 Optimization Passes, Register Allocation ILP)
**Track:** Track F  
**AI Assistant:** Gemini 3.8 Flash  
**Date:** 2026-10-25  

---

### Original Prompt
```text
Implement Track F: Compiler Optimization:
1. Mini-compiler pipeline: Lexer, parser, 3-Address Code (TAC) Intermediate Representation, and Virtual Machine.
2. Implement 4 optimization passes: Constant Folding/Propagation, Common Sub-expression Elimination (CSE) with kill-set tracking, Strength Reduction (multiplication/division by power of 2 -> bitwise shifts), and Dead-Code Elimination (DCE).
3. Formulate Register Allocation as an optimization problem: Graph Coloring and Integer Linear Programming (ILP) using SciPy milp to minimize spilled variables under K physical registers.
4. Verify that unoptimized and optimized IR produce identical program outputs on test inputs.
5. Measure instruction count, VM cycle count, compile time, and register spill count.
```

### AI Response Summary
The AI implemented `ToyCompiler`, `Instruction`, and `ToyVM`. It created `ConstantFoldingPropagationPass`, `CommonSubexpressionEliminationPass`, `StrengthReductionPass`, `DeadCodeEliminationPass`, and `RegisterAllocator` using Chaitin's degree-heuristic and exact ILP constraint formulation.

### Revision & Refinement
Enforced the kill-set invalidation logic for CSE (Failure Log Case 2), preventing illegal reuse of expressions across reassigned variables. Added edge-case handling for zero live variables in the ILP solver.

### Verification Notes
- Output equivalence verified: both unoptimized and optimized code returned identical output `166`.
- Execution cycle count on the VM dropped from 21 cycles to 9 cycles (a 57.1% reduction).
- Instruction count dropped by 18.2%, with constant folding ($12 \times 4 \to 48$), strength reduction ($x \times 8 \to x \ll 3$), and dead code elimination successfully applied.
- Register allocation on $K=3$ registers confirmed zero interference conflicts between simultaneously live variables.

### Linked Commit
Milestone 7: `[AI:gemini-3.8] Track F: Compiler optimization passes, TAC IR VM, and register allocation ILP`
