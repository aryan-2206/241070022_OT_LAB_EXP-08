# Mandatory AI Usage Register & Audit Log
## OT Experiment 08 — "Optimize Anything: The 14-Day Vibe-Coding Challenge"
**Student Roll No:** 241070022  
**Course:** Optimization Techniques (OT) Lab  

---

### 1. AI Tools Register

The table below catalogs every AI tool and model version utilized in this project across planning, code generation, debugging, testing, and documentation, as mandated by Section 8 of the assignment specification.

| AI Tool & Model | Date | Tracks / Components | Task Category | Specific Contributions |
|---|---|---|---|---|
| **Gemini 3.8 Flash (Antigravity Assistant)** | 2026-10-25 | Tracks A through J, Docs | Architecture & Implementation | Generated project architecture, mathematical formulations, core algorithm solvers, unit tests, and requirements matrix. |
| **Claude 3.5 Sonnet / Codex** | 2026-10-25 | Track F (Compiler) & Track G (xv6) | Cross-tool comparative evaluation | Evaluated on AST pass ordering, dead code elimination algorithms, and xv6 context switch lock safety. |
| **DeepSeek-R1 / ChatGPT (GPT-4o)** | 2026-10-25 | Track C & Track I (Quantum) | Circuit formulation verification | Consulted for parameter-shift gradient derivation, Fubini-Study metric tensor symmetry, and QAOA mixer terms. |

---

### 2. Prompt Engineering & Verification Workflow

Every track maintains its dedicated prompt and response audit trail in the `prompts/` directory:
- `prompts/track_a/`: Prompts for continuous optimizers, curvature approximations, and saddle point escapes.
- `prompts/track_b/`: Prompts for combinatorial solvers and QUBO/Ising penalty mappings.
- `prompts/track_c/`: Prompts for QAOA ansatz, parameter-shift rule, and barren plateau analysis.
- `prompts/track_d/`: Prompts for submodular diminishing returns, FedAvg, and MGDA Pareto optimization.
- `prompts/track_e/`: Prompts for EcoRoute campus logistics formulation, time windows, and Simulated Annealing.
- `prompts/track_f/`: Prompts for toy compiler parser, 4 optimization passes, and register allocation ILP.
- `prompts/track_g/`: Prompts for xv6 Priority/Stride scheduling patch, aging mechanisms, and workload benchmarks.
- `prompts/track_h/`: Prompts for System-R dynamic programming join ordering, SQLite EXPLAIN plans, and index knapsack.
- `prompts/track_i/`: Prompts for quantum algorithms (DJ, BV, Grover, QFT, Shor), QML kernels, and classical advantage limits.
- `prompts/track_j/`: Prompts for software engineering test suite minimization and code coverage knapsack formulation.

---

### 3. Commit Traceability Protocol

To distinguish AI-generated suggestions from manual refinements and bug fixes:
- `[AI:gemini-3.8]` : Code or documentation scaffold generated with Gemini assistance.
- `[AI:refactor]` : Algorithmic tuning, vectorization, and performance refinement prompted to AI.
- `[Manual:verify]` : Independent verification tests, unit assertions, bug fixes, and manual parameter tuning conducted by the student.
