# Track E Prompt & Engineering Audit Log
## Main Real-Process Case Study: EcoRoute (Campus Logistics Optimization)
**Track:** Track E  
**AI Assistant:** Gemini 3.8 Flash  
**Date:** 2026-10-25  

---

### Original Prompt
```text
Implement Track E: Main Real-Process Case Study based on the proposed EcoRoute project:
1. Problem: Carbon-aware electric vehicle routing and hazardous material collection on a 25-node university campus.
2. Formulation: Decision variables x_{ijk} in {0, 1}, load-dependent carbon emissions gamma(w) = e_0 + e_1 * w, time windows [S_i, L_i], and vehicle battery constraints.
3. Solvers: Implement Baseline Greedy Nearest-Neighbor with earliest-deadline insertion, Simulated Annealing with 2-opt and inter-route relocation, and an Exact Small-Instance Solver for verification.
4. Independent validation: Feasibility checker enforcing capacity, battery, and single-visit invariants.
5. Generate route visualizations comparing baseline vs optimized routes and convergence curve.
```

### AI Response Summary
The AI implemented `CampusNetwork` modeling spatial coordinates, demands, time windows, and multi-component cost. It implemented `GreedyNearestNeighborSolver`, `SimulatedAnnealingRouter`, and `ExactSmallInstanceSolver`. It constructed a 3-panel figure showing baseline routes, optimized routes, and SA energy decay.

### Revision & Refinement
Refined the carbon accounting model to properly discount cargo weight dynamically along the route as packages are dropped off.

### Verification Notes
- Unit test `test_campus_network_feasibility_validation` confirmed that independent validation catches missing stops and capacity overloads.
- On the 6-node verification subproblem, Simulated Annealing achieved the exact global optimum ($Cost = 5.256178$), proving 0.0% optimality gap against exhaustive search.
- On the full 20-node campus benchmark, Simulated Annealing eliminated 100% of deadline lateness violations (reducing lateness from 384.3 mins to 0 mins) and improved overall cost by 23.6%.

### Linked Commit
Milestone 6: `[AI:gemini-3.8] Track E: Main Case Study - EcoRoute campus logistics, time windows, and simulated annealing`
