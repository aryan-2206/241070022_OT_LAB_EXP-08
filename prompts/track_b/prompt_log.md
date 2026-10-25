# Track B Prompt & Engineering Audit Log
## Discrete Optimization (Greedy, Dynamic Programming, Branch-and-Bound, ILP, QUBO & Ising)
**Track:** Track B  
**AI Assistant:** Gemini 3.8 Flash  
**Date:** 2026-10-25  

---

### Original Prompt
```text
Implement a complete Discrete Optimization suite in Python:
1. Exact solvers: Dynamic Programming (pseudo-polynomial O(NW)) and Branch-and-Bound with fractional LP relaxation upper bounding.
2. Heuristic solver: Greedy value/weight ratio ranking.
3. Integer Linear Programming (ILP) formulation using SciPy's milp interface.
4. QUBO and Ising formulation for Max-Cut with binary-to-spin mapping s_i = 1 - 2*x_i and penalty terms for constraints.
5. Solve Max-Cut classically using brute force and simulated annealing as the baseline for Track C QAOA.
```

### AI Response Summary
The AI provided `GreedyKnapsackSolver`, `DynamicProgrammingKnapsackSolver`, `BranchAndBoundKnapsackSolver`, and `ILPKnapsackSolver`. It built `MaxCutQUBO` mapping graph edges into quadratic matrix $Q$ and corresponding Ising Hamiltonian $J_{ij}$, along with `ConstrainedFeatureSelectionQUBO` applying quadratic penalty terms $\lambda (\sum x_i - K)^2$.

### Revision & Refinement
Ensured `BranchAndBoundKnapsackSolver` maintains fractional bound pruning correctly on sorted index sets to prevent exponential explosion on larger instances.

### Verification Notes
- Tested 0/1 Knapsack consistency: DP, BnB, and ILP all produced the identical optimal value ($104.0$) satisfying $\sum w_i x_i \le 45$.
- Greedy heuristic achieved $99.0$ ($4.8\%$ optimality gap), executing in $0.049$ ms.
- Max-Cut ground truth established at cut value $10.0$ with bitstring $[1, 0, 0, 1, 0]$.

### Linked Commit
Milestone 3: `[AI:gemini-3.8] Track B: Discrete optimization, exact/heuristic solvers, and QUBO/Ising baseline`
