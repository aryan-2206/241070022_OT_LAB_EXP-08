# OT Experiment 08 — "Optimize Anything: The 14-Day Vibe-Coding Challenge"
## Optimization Techniques (OT) Laboratory — Complete Portfolio & Report
**Student Roll No:** 241070022  
**Course Code:** CS-702 (Optimization Techniques Lab)  
**Host Platform:** Windows 11 / Python 3.12.4  
**Public Repository:** [https://github.com/aryan-2206/241070022_OT_LAB_EXP-08.git](https://github.com/aryan-2206/241070022_OT_LAB_EXP-08.git)  

---

## 1. Project Overview

This repository contains the complete, production-grade implementation, benchmarking suite, and technical report for **Experiment 08: "Optimize Anything — The 14-Day Vibe-Coding Challenge."** The project investigates how optimization can improve complex computational and physical processes across ten compulsory experiment tracks (**Tracks A to J**), covering continuous and discrete optimization, quantum variational circuits, parallel computing, operating system kernel scheduling, compiler design, database query optimization, and continuous integration engineering.

### Track Summary & Implemented Methods
- **Track A — Continuous Optimization:** First-order optimizers (GD, SGD, Momentum, NAG, AdaGrad, RMSProp, Adam), second-order Newton-Raphson & Gauss-Newton, explicit MLP backpropagation, empirical Fisher, diagonal Hessian, and Pearlmutter Hessian-Vector Products (HVP).
- **Track B — Discrete Optimization:** Exact solvers (Branch-and-Bound, Dynamic Programming, SciPy ILP) vs Greedy heuristic on 0/1 Knapsack; Max-Cut and feature selection formulated as QUBO and Ising models ($s_i = 1 - 2x_i$) establishing the classical baseline for Track C.
- **Track C — Quantum Optimization:** High-fidelity statevector quantum simulator; QAOA for Max-Cut over $p=1, 2, 3$ layers; Parameter-Shift rule, SPSA, and Quantum Natural Gradient (QNG); finite measurement shots scaling ($1/\sqrt{N_{\text{shots}}}$), depolarizing noise, and barren plateau analysis ($\text{Var}[\partial_\theta \langle H \rangle] \sim 2^{-n}$).
- **Track D — Submodular, Parallel & Distributed Optimization:** Submodular Maximum Coverage with diminishing returns verification (0 violations) and $(1 - 1/e)$ bound check; parallel data parallelism, pipeline parallelism, and dynamic work stealing under task skew; Amdahl's and Gustafson's Laws; Federated Learning (FedAvg with non-IID clients); and multi-task optimization (MGDA and GradNorm).
- **Track E — Main Real-Process Case Study (EcoRoute):** Campus electric logistics modeled as a Capacitated Vehicle Routing Problem with Time Windows and Battery Limits (CVRPTW-B). Simulated Annealing eliminated 100% of deadline violations and reduced costs by 23.6%, matching exact branch-and-bound on small instances with 0.0% gap.
- **Track F — Compiler Optimization:** Toy language compiler and CPU VM; 4 optimization passes (Constant Folding, Dead Code Elimination, CSE with kill-set invalidation, Strength Reduction), reducing execution cycles by 57.1% with 100% output equivalence; register allocation via graph coloring and ILP.
- **Track G — xv6 Operating System Optimization:** Priority & Stride scheduling kernel enhancement with dynamic priority aging for xv6 RISC-V (`xv6_patch.diff`); cycle-accurate kernel simulation test harness; 51.9% turnaround acceleration for interactive tasks under mixed contention; zero starvation guarantee.
- **Track H — DBMS Optimization:** System-R Dynamic Programming join-order enumeration and Selinger cost modeling; index selection formulated as a 0/1 Knapsack ILP under a 130 KB storage budget; evaluated on synthetic TPC-H schema (Customer, Orders, LineItem); verified with `EXPLAIN QUERY PLAN` that full table scans were replaced by covering index searches, yielding an 18.9% speedup with identical results.
- **Track I — Quantum Algorithms & QML Advantage:** Gate-level circuits for Deutsch-Jozsa, Bernstein-Vazirani, Grover's Search, QFT, QPE, Shor's Factoring, and Quantum Annealing; demonstrated Grover's quadratic query advantage ($O(\sqrt{N})$ vs $O(N)$), with honest analysis of classical simulation overhead ($O(2^n)$ memory barrier); trained Quantum Kernel Classifier vs Classical RBF SVM.
- **Track J — Software-Engineering Optimization:** Profiled repository codebase (5,269 LOC across 27 modules); formulated test-suite minimization under CI time budgets as a Knapsack ILP, achieving 98.0% branch coverage under a strict 300 ms budget (compared to 52.0% for naive execution) and 100% coverage under a 600 ms budget ($4.22\times$ pipeline acceleration).

---

## 2. Mandatory AI Tools Register (Section 8 Compliance)

| AI Tool & Model | Date | Tracks / Scope | Task Category | Specific Contributions |
|---|---|---|---|---|
| **Gemini 3.8 Flash** | 2026-10-25 | Tracks A through J, Docs | Architecture, Solvers, Scaffolding | Generated mathematical models, algorithmic solvers, test cases, and prompt logs. |
| **Claude 3.5 Sonnet** | 2026-10-25 | Track F (Compiler) & Track H (DBMS) | Comparative Evaluation | Cross-evaluated join tree dynamic programming and CSE kill-set invalidation logic. |
| **ChatGPT (GPT-4o)** | 2026-10-25 | Track C & Track I (Quantum) | Circuit & Gradient Derivation | Validated analytic parameter-shift shift angle ($s = \pi/2$) and Fubini-Study metric tensor. |

*Detailed prompt logs, revisions, and commit links are maintained in `prompts/track_a/` through `prompts/track_j/`.*  
*Documented AI Failure Cases and Lessons Learned are cataloged in `docs/failure_log.md`.*

---

## 3. Environment Setup & Execution Instructions

### Prerequisites
- Python 3.12 or higher.
- Core dependencies: `numpy`, `scipy`, `matplotlib`, `pandas`, `pytest`.

Install all dependencies via:
```powershell
pip install -r requirements.txt
```

### Running the Complete Test Suite
To run all 36 unit and correctness tests across all 10 tracks:
```powershell
python -m pytest tests/ -v
```

### Running Individual Track Experiments
Each track contains an independent, fully reproducible benchmark runner that executes the experiments, saves CSV results to `results/raw/`, and generates figures in `results/figures/`:
```powershell
# Track A: Continuous Optimization
python src/track_a_continuous/runner.py

# Track B: Discrete Optimization
python src/track_b_discrete/runner.py

# Track C: Quantum Optimization
python src/track_c_quantum_opt/runner.py

# Track D: Submodular, Parallel & Distributed Optimization
python src/track_d_parallel_dist/runner.py

# Track E: Main Case Study (EcoRoute Campus Logistics)
python src/track_e_case_study/runner.py

# Track F: Compiler Optimization
python src/track_f_compiler/runner.py

# Track G: xv6 Operating System Optimization
python src/track_g_xv6_os/runner.py

# Track H: DBMS Optimization
python src/track_h_dbms/runner.py

# Track I: Quantum Algorithms & QML Advantage
python src/track_i_quantum_adv/runner.py

# Track J: Software-Engineering Optimization
python src/track_j_software_eng/runner.py
```

---

## 4. Repository Structure Walkthrough

```text
241070022_OT-EXP-08/
├── README.md                          # Primary setup and project documentation
├── requirements.txt                   # Dependency specifications
├── .gitignore                         # Build and cache exclusion rules
├── docs/
│   ├── requirements_matrix.md         # Requirements Traceability Matrix (100% Implemented)
│   ├── algorithm_selection.md         # Course syllabus mapping and strategy guide
│   ├── environment_setup.md           # Environment and platform notes
│   ├── ai_usage_log.md                # Vibe-coding AI tools register and audit protocol
│   ├── failure_log.md                 # 3 documented AI failure cases, fixes, and lessons
│   ├── ai_tool_comparison.md          # Comparative study between Gemini 3.8 and Claude 3.5
│   └── viva_preparation.md           # Complete answers to the 10 compulsory viva questions
├── src/
│   ├── track_a_continuous/            # Optimizers, Newton, backprop, curvature, landscapes
│   ├── track_b_discrete/              # Knapsack (Greedy, DP, BnB, ILP), Max-Cut QUBO
│   ├── track_c_quantum_opt/           # Statevector simulator, QAOA, gradients, barren plateaus
│   ├── track_d_parallel_dist/         # Submodular coverage, work stealing, FedAvg, MGDA
│   ├── track_e_case_study/            # EcoRoute CVRPTW-B formulation, baseline, SA router
│   ├── track_f_compiler/              # Toy compiler, 4 TAC passes, register allocation ILP
│   ├── track_g_xv6_os/                # xv6 Stride scheduler patch (xv6_patch.diff), kernel sim
│   ├── track_h_dbms/                  # TPC-H schema, System-R DP join ordering, index ILP
│   ├── track_i_quantum_adv/           # 7 quantum algorithms (DJ, BV, Grover, Shor), QML SVM
│   └── track_j_software_eng/          # Test suite minimization, knapsack ILP, codebase profiler
├── tests/
│   └── unit/                          # 36 automated unit & correctness test cases (100% Pass)
├── results/
│   ├── raw/                           # Raw benchmark CSV measurement datasets
│   └── figures/                       # Publication-grade PNG visualization plots
├── prompts/                           # Track-by-track vibe-coding prompt audit trails
│   ├── track_a/ ... track_j/
└── reports/
    └── technical_report.md            # Comprehensive formal technical report
```

---

## 5. Summary of Empirical Benchmark Results

| Track | Primary Optimization Target | Baseline Measurement | Optimized Measurement | Key Measured Improvement | Evidence File |
|---|---|---|---|---|---|
| **Track A** | Rosenbrock Surface Convergence | GD: Loss = $2.38$ (200 iters) | Newton: Loss = $\mathbf{6.28 \times 10^{-22}}$ (**8 iters**) | **Exact quadratic convergence in 8 steps** | `results/figures/track_a_convergence.png` |
| **Track B** | 0/1 Knapsack Value ($W=45$) | Greedy: Value = $99.0$ ($4.81\%$ gap) | Branch-and-Bound / DP: **$104.0$** | **Proven global optimum reached** | `results/figures/track_b_comparison.png` |
| **Track C** | Max-Cut QUBO Ground State | Classical Optimum = $10.0$ | QAOA ($p=3$): Cut = $\mathbf{7.30}$, Bitstring = `[1,0,0,1,0]` | **$73.0\%$ approx ratio, exact optimal state** | `results/figures/track_c_qaoa_and_barren.png` |
| **Track D** | Skewed Parallel Processing | Static Chunking: $0.0083$ s | Dynamic Work Stealing: **$0.0020$ s** | **$4.1\times$ speedup over static chunking** | `results/figures/track_d_submodular_parallel.png` |
| **Track E** | EcoRoute Campus Logistics | Baseline: $384.3$ mins late | Simulated Annealing: **$0.0$ mins late** | **100% deadline satisfaction, 23.6% cost gain** | `results/figures/track_e_ecoroute.png` |
| **Track F** | TAC Compiler Code Generation | Unoptimized: 21 CPU cycles | Optimized (4 Passes): **9 CPU cycles** | **$57.1\%$ cycle reduction, 100% output match** | `results/figures/track_f_compiler.png` |
| **Track G** | xv6 Process Scheduling Latency | Round-Robin: 77.0 ticks | Stride & Priority: **37.0 ticks** | **$51.9\%$ interactive task acceleration** | `results/figures/track_g_xv6_benchmarks.png` |
| **Track H** | SQLite TPC-H Query Latency | Full Table Scan: $0.367$ ms | ILP Covering Index: **$0.309$ ms** | **$18.9\%$ speedup, identical row outputs** | `results/figures/track_h_dbms.png` |
| **Track I** | Oracle Search Complexity ($N=128$) | Classical Linear: 126 queries | Grover Quantum: **9 queries** | **$14.0\times$ algorithmic query reduction** | `results/figures/track_i_quantum.png` |
| **Track J** | CI Test Execution Latency | Full Suite: 2,533 ms (100 branches) | Standard CI: **376 ms (100 branches)** | **$4.22\times$ faster CI with 100% coverage** | `results/figures/track_j_test_minimization.png` |

---

## 6. Viva Voce Examination & Defense Readiness

Complete mathematical and practical explanations for all ten compulsory viva questions from Section 11 of the assignment specification are available in:
- **`docs/viva_preparation.md`** (Standalone quick-reference viva guide)
- **`reports/technical_report.md`** (Comprehensive Section 3 chapter)

Key viva highlights:
1. **EcoRoute Formulation:** Objective combines load-dependent carbon emissions $\gamma(w) = 0.08 + 0.0012w$, distance, and arrival lateness under vehicle payload, time window, and battery SoC constraints.
2. **Correctness Verification:** Tested via numerical finite differences ($\| \nabla - \nabla_{\text{num}} \| < 10^{-4}$), 100% identical outputs for compiler and DBMS queries, and exhaustive search verification on small benchmark instances.
3. **Quantum vs Classical Reality:** Rigorously proves why classical simulation of quantum circuits scales exponentially ($O(2^n)$ runtime and memory), explaining why quantum simulators on CPUs cannot deliver wall-clock speedups over classical parallel programs.
4. **Vibe Coding Failures Documented:** 3 genuine AI errors diagnosed and fixed (quantum parameter-shift shift angle $\pi/4 \to \pi/2$, compiler CSE kill-set invalidation on variable reassignments, and xv6 kernel lock deadlock).

---

## 7. Commit Discipline & Integrity Statement

All commits follow the mandatory semantic prefix protocol:
- `[AI:gemini-3.8]` : AI-assisted implementations, models, and test scaffolding.
- `[Manual:verify]` : Independent verification tests, analytical derivations, and bug fixes conducted by the student.

*No external secrets, private API keys, or fabricated benchmarks have been committed. All reported numbers correspond directly to actual executions on the host system.*
