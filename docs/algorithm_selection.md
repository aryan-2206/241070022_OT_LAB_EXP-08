# Optimization Techniques (OT) Syllabus Mapping & Algorithm Selection Guide
## Course Syllabus Integration for Experiment 08: "Optimize Anything"
**Student Roll No:** 241070022  
**Course:** Optimization Techniques (OT)  

---

### 1. Syllabus Unit Alignment

The table below maps each syllabus unit from the curriculum to the compulsory experiment tracks in Experiment 08.

| Syllabus Unit | Core Topics | Corresponding Experiment Tracks | Specific Algorithms & Methods Selected |
|---|---|---|---|
| **Unit 1: Linear Programming** | Simplex Method, Duality, Non-simplex Methods | Track B (Discrete), Track E (Case Study), Track F (Compiler), Track H (DBMS), Track J (Software Eng) | • Integer Linear Programming (ILP) for Knapsack and Set Cover<br>• Duality bounds for Branch-and-Bound pruning<br>• Linear relaxations for index selection and test suite minimization |
| **Unit 2: Unconstrained Optimization** | Local Minimizer Conditions, 1D Search, Gradient Methods, Newton's Method | Track A (Continuous Optimization), Track C (Quantum Opt) | • Gradient Descent (GD), SGD, Mini-batch SGD<br>• Momentum and Nesterov Accelerated Gradient (NAG)<br>• Newton's Method with exact Hessian inversion<br>• Gauss-Newton for non-linear least squares<br>• Backpropagation and Curvature Approximations (Fisher, HVP) |
| **Unit 3: Nonlinear Constrained Optimization** | Equality Constraints, Constrained Optimization, Heuristics | Track B (Discrete), Track C (Quantum Opt), Track E (Case Study) | • Penalty methods: Quadratic penalty terms for QUBO/Ising formulations<br>• Lagrange multipliers and KKT optimality conditions in multi-task optimization (MGDA)<br>• Feasibility constraint checking under capacity and time windows |
| **Unit 4: Single-Objective Optimization** | Simulated Annealing, Evolutionary Algorithms, TLBO, PSO, ABC, GWO | Track E (Main Case Study: EcoRoute) | • Simulated Annealing with adaptive cooling schedule for vehicle routing<br>• Local search with 2-opt and swap neighborhood operators |
| **Unit 5: Multiobjective Optimization** | Non-Elitist & Elitist MOEAs, Constrained Multiobjective Optimization | Track D (Parallel & Distributed), Track E (Case Study) | • Multiple Gradient Descent Algorithm (MGDA) for Pareto stationarity<br>• GradNorm for balanced gradient descent across conflicting objectives<br>• Pareto front exploration for emissions vs delivery delay |
| **Unit 6: Optimization in ML and Parallel Programs** | Continuous/Discrete, Submodular Functions, Load Balancing, Locality, Sync/Comm | Track A (Continuous), Track C (Quantum), Track D (Parallel/Dist), Track I (Quantum Adv) | • Submodular greedy optimization with $(1 - 1/e)$ approximation guarantee<br>• Dynamic load balancing, work stealing, data/pipeline parallelism<br>• Amdahl's Law and Gustafson's Law strong/weak scaling<br>• Federated Averaging (FedAvg) over non-IID data<br>• Quantum variational circuits, parameter-shift rule, quantum kernels |

---

### 2. Track-by-Track Algorithm Selection Rationale

#### Track A: Continuous Optimization
- **Problem Class:** Continuous unconstrained and ill-conditioned non-linear optimization.
- **Methods Selected:**
  1. *First-Order:* GD, SGD, Momentum, NAG, AdaGrad, RMSProp, Adam.  
     *Rationale:* Highlights the evolution of momentum (dampening oscillations in ill-conditioned ravines) and adaptive coordinate-wise learning rates (handling sparse/varying gradient scales).
  2. *Second-Order:* Newton's Method, Gauss-Newton.  
     *Rationale:* Showcases quadratic convergence rates ($O(\|\nabla f\|^2)$) near local minima at the expense of $O(p^3)$ Hessian inversion costs.
  3. *Curvature Approximations:* Diagonal Hessian, Pearlmutter Hessian-Vector Products (HVP), Empirical Fisher.  
     *Rationale:* Demonstrates scalable curvature information without explicit $O(p^2)$ memory storage.
  4. *Landscapes:* Convex quadratic bowl vs Non-convex Saddle Point ($f(x,y) = x^2 - y^2$ and Rastrigin).  
     *Rationale:* Proves why first-order GD gets trapped at zero-gradient saddle points whereas Momentum and noise enable escape.

#### Track B: Discrete Optimization
- **Problem Class:** Combinatorial NP-hard discrete optimization.
- **Methods Selected:**
  1. *Greedy Heuristics:* Value-to-weight ratio ordering ($O(N \log N)$).
  2. *Dynamic Programming:* Exact pseudo-polynomial knapsack solver ($O(N \cdot W)$).
  3. *Branch-and-Bound:* Exact tree search with linear programming upper-bound pruning.
  4. *QUBO / Ising Formulation:* Transformation of Max-Cut and subset selection into $H = \sum Q_{ij} x_i x_j$ with penalty terms $P(x) = \lambda (\sum x_i - K)^2$, mapped to spins $s_i \in \{-1, +1\}$ via $x_i = \frac{s_i + 1}{2}$.  
     *Rationale:* Serves as the mathematical bridge connecting classical discrete NP-hard problems to quantum Hamiltonian annealing and QAOA in Track C.

#### Track C: Quantum Optimization
- **Problem Class:** Variational Quantum Algorithms (VQA) on NISQ simulators.
- **Methods Selected:**
  1. *QAOA (Quantum Approximate Optimization Algorithm):* Alternating problem cost Hamiltonian $e^{-i \gamma C}$ and transverse mixer Hamiltonian $e^{-i \beta B}$ across $p$ layers.
  2. *VQE (Variational Quantum Eigensolver):* Parameterized ansatz $|\psi(\theta)\rangle$ to minimize $\langle \psi(\theta)| H |\psi(\theta)\rangle$.
  3. *Gradient Strategies:* Parameter-shift rule (exact analytic gradient on hardware), SPSA (finite difference along random vector), Quantum Natural Gradient (QNG via Fubini-Study metric).
  4. *NISQ Realities:* Barren plateau scaling ($Var[\nabla_\theta] \sim 2^{-n}$) and depolarizing noise degradation.

#### Track D: Submodular, Parallel, and Distributed Optimization
- **Problem Class:** Combinatorial coverage with diminishing marginal returns, high-performance concurrency, and distributed multi-task learning.
- **Methods Selected:**
  1. *Submodular Maximization:* Greedy sensor placement / facility location satisfying $f(A \cup \{e\}) - f(A) \ge f(B \cup \{e\}) - f(B)$ for $A \subseteq B$.
  2. *Parallel Computing:* Shared-memory multiprocessing with work stealing and chunked data parallelism.
  3. *Theoretical Laws:* Amdahl's Law (serial fraction limits) vs Gustafson's Law (scaled problem scaling).
  4. *Distributed ML:* FedAvg with local SGD epochs $E$ and communication rounds $R$.
  5. *Multi-task Optimization:* MGDA (finding minimum norm vector in convex hull of gradients) and GradNorm.

#### Track E: Main Case Study — EcoRoute (Campus Logistics)
- **Problem Class:** Capacitated Vehicle Routing Problem with Time Windows and Battery Limits (CVRPTW-B).
- **Methods Selected:**
  1. *Baseline:* Greedy Nearest-Neighbor with earliest-deadline insertion.
  2. *Metaheuristic:* Simulated Annealing with 2-opt swap and Metropolis acceptance criterion $P = e^{-\Delta E / T}$.
  3. *Exact Solver:* Branch-and-Bound / Mixed-Integer Linear Program for small-scale benchmarks.

#### Track F: Compiler Optimization
- **Problem Class:** Code generation and intermediate representation minimization.
- **Methods Selected:**
  1. *Constant Folding & Propagation:* Compile-time static expression evaluation.
  2. *Dead Code Elimination (DCE):* Liveness analysis removing unreachable/unused IR instructions.
  3. *Common Sub-expression Elimination (CSE):* Available expressions analysis eliminating redundant calculations.
  4. *Strength Reduction:* Replacing expensive operations ($x \times 2$) with bitwise shifts ($x \ll 1$).
  5. *Register Allocation:* Interference graph node coloring formulated as an ILP / Chaitin heuristic.

#### Track G: xv6 Operating System Optimization
- **Problem Class:** OS kernel scheduling, latency, and throughput under contending workloads.
- **Methods Selected:**
  1. *Priority & Stride Scheduling:* Deterministic stride calculation $\text{stride}_i = \frac{L}{\text{priority}_i}$ ensuring tickets-proportional CPU allocation without random jitter.
  2. *Aging Mechanism:* Dynamic priority elevation to prevent CPU starvation of background batch jobs.

#### Track H: DBMS Optimization
- **Problem Class:** Relational query latency minimization and index storage allocation.
- **Methods Selected:**
  1. *Join Order Enumeration:* System-R Dynamic Programming exploring left-deep join trees using Selinger cost model ($C = \text{I/O} + W \cdot \text{CPU}$).
  2. *Index Selection Problem:* 0/1 Knapsack ILP maximizing total workload speedup $\sum \Delta \text{Cost}_q$ subject to disk budget $\sum \text{Size}_i \le B$.

#### Track I: Quantum Algorithms and QML Advantage
- **Problem Class:** Quantum algorithmic complexity vs Classical computing limits.
- **Methods Selected:**
  1. *Exact Oracular Algorithms:* Deutsch-Jozsa (1 query vs $2^{n-1}+1$), Bernstein-Vazirani (1 query vs $n$).
  2. *Amplitude Amplification:* Grover's search ($O(\sqrt{N})$ vs classical $O(N)$).
  3. *Transform & Phase:* QFT, Quantum Phase Estimation (QPE), Shor's period finding.
  4. *QML:* Quantum Kernel SVM vs Classical RBF SVM, highlighting the data loading bottleneck and simulation exponential wall ($O(2^n)$).

#### Track J: Software-Engineering Optimization
- **Problem Class:** Test-suite minimization under continuous integration time budgets.
- **Methods Selected:**
  1. *Formulation:* 0/1 Knapsack Integer Linear Program:
     $$\max \sum_{j} c_j y_j \quad \text{s.t.} \quad \sum_{i} t_i x_i \le T_{\text{budget}}, \quad y_j \le \sum_{i: j \in \text{Cov}(i)} x_i$$
  2. *Algorithms:* Greedy additional coverage vs Exact ILP vs Naive random execution.
