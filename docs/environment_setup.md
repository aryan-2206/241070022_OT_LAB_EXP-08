# Environment Setup & Dependency Documentation
## OT Experiment 08 — "Optimize Anything"
**Roll Number:** 241070022  
**Host Platform:** Windows 11 (x86_64)  
**Python Runtime:** Python 3.12.4  

---

### 1. Installed Environment & Toolchains

The primary runtime is standard Python 3.12.4 with essential numerical, scientific, and testing packages:
- **NumPy & SciPy:** Core vectorization, linear algebra, sparse matrices, and optimization routines.
- **PyTorch:** Used for automatic differentiation, tensor operations, backpropagation, and deep multi-task gradient comparison (MGDA/GradNorm).
- **SQLite3:** Built-in standard library database engine for Track H DBMS optimization.
- **Pytest:** Automated testing framework for unit and correctness testing.
- **Matplotlib & Seaborn:** Figure and plot rendering for experimental results.

### 2. Specialized Track Runtimes & Windows Portability Design

1. **Quantum Circuits & Statevectors (Track C & Track I):**
   - High-fidelity, self-contained pure-Python/NumPy statevector quantum simulator implemented in `src/track_c_quantum_opt/simulator.py` and `src/track_i_quantum_adv/simulator.py`.
   - Supports arbitrary single-qubit rotations ($R_x, R_y, R_z$), Hadamard ($H$), Pauli ($X, Y, Z$), Phase ($S, T$), and two-qubit gates (CNOT, CZ, SWAP).
   - Computes exact analytic expectation values $\langle \psi | H | \psi \rangle$, finite-shot Poisson/multinomial sampling, depolarizing noise channels, and Fubini-Study Quantum Natural Gradient tensors.
   - Ensures 100% reproducible execution on Windows without requiring network-dependent wheels or C++ compiler toolchains.

2. **Operating System Kernel & xv6 (Track G):**
   - Official RISC-V xv6 kernel codebase and clean diff patch (`src/track_g_xv6_os/xv6_patch.diff`).
   - A bit-accurate, tick-level OS kernel scheduler simulator (`src/track_g_xv6_os/xv6_kernel_sim.py`) that emulates xv6 process control blocks (`struct proc`), state transitions (`RUNNABLE`, `RUNNING`, `SLEEPING`, `ZOMBIE`), system calls (`setpriority`, `getpinfo`), and tick interrupts.
   - Provides exact verification and before-and-after benchmark metrics for Round-Robin vs Priority/Stride scheduling on Windows, while preserving the kernel patch for direct QEMU compilation on Linux/WSL.

3. **Compiler Passes & Toy Language VM (Track F):**
   - Fully functional lexer, recursive-descent parser, 3-address code intermediate representation (IR) generator, optimization pass pipeline, and virtual machine interpreter in `src/track_f_compiler/`.
   - Runs directly on Python 3.12 with zero external C++ dependencies, while generating verifiable assembly-like IR and machine cycle reductions.

4. **Database Query Engine (Track H):**
   - Relational database workload running on SQLite3 with a synthetic TPC-H schema (Customers, Orders, LineItem).
   - Cost-based query optimizer generating SQLite-compatible plans and verifying plan modifications via `EXPLAIN QUERY PLAN`.

---

### 3. Execution Instructions

To execute all unit and correctness tests across all 10 tracks:
```powershell
python -m pytest tests/ -v
```

To run individual track experiments:
```powershell
python src/track_a_continuous/runner.py
python src/track_b_discrete/runner.py
python src/track_c_quantum_opt/runner.py
python src/track_d_parallel_dist/runner.py
python src/track_e_case_study/runner.py
python src/track_f_compiler/runner.py
python src/track_g_xv6_os/runner.py
python src/track_h_dbms/runner.py
python src/track_i_quantum_adv/runner.py
python src/track_j_software_eng/runner.py
```
