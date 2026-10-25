# AI Failure & Debugging Log
## Mandatory Documented AI Failures, Root Cause Analyses, and Lessons Learned
**Student Roll No:** 241070022  
**Course:** Optimization Techniques (OT) Lab  

Section 8 of the Experiment 08 specification requires documenting at least three genuine cases where AI-generated code was wrong, unsafe, inefficient, or misleading. The cases below detail the erroneous AI output, the diagnostic test that exposed it, the student's correction, and the engineering lesson learned.

---

### Failure Case 1: Parameter-Shift Rule Phase Error in Quantum Gradient Computation (Track C)
- **Track:** Track C (Quantum Optimization) & Track I (Quantum Advantage)
- **Initial AI Output:**
  When prompted to implement the analytic parameter-shift rule for parameterized rotation gates $R_y(\theta) = \exp(-i \frac{\theta}{2} Y)$, the AI generated:
  ```python
  # Erroneous AI-generated code:
  def parameter_shift_grad(circuit, params, i):
      shifted_plus = params.copy()
      shifted_minus = params.copy()
      shifted_plus[i] += np.pi / 4  # INCORRECT SHIFT VALUE!
      shifted_minus[i] -= np.pi / 4
      return (circuit.evaluate(shifted_plus) - circuit.evaluate(shifted_minus)) / 2.0
  ```
- **Diagnostic Test & Failure Mode:**
  The student ran a finite-difference numerical gradient test against `parameter_shift_grad`:
  $$\nabla_{\text{num}} f(\theta) = \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}$$
  The test assertion `np.allclose(grad_ps, grad_num, atol=1e-4)` failed with a large discrepancy ($\approx 41.4\%$ relative error).
- **Root Cause Analysis:**
  For an arbitrary single-qubit generator $G = \frac{1}{2} \sigma$, the eigenvalues are $\pm \frac{1}{2}$, leading to an eigenvalue difference $r = \frac{1}{2} - (-\frac{1}{2}) = 1$. The standard parameter-shift rule requires a shift of $s = \frac{\pi}{4r} = \frac{\pi}{2}$, not $\frac{\pi}{4}$. The denominator scale factor must also match $2 \sin(s) = 2 \sin(\pi/2) = 2$. By shifting by $\pi/4$, the AI evaluated at the wrong point on the Bloch sphere, yielding $\sin(\pi/4) = \frac{1}{\sqrt{2}} \approx 0.7071$ times the true gradient.
- **The Fix:**
  Updated the shift constant to $s = \frac{\pi}{2}$:
  ```python
  shifted_plus[i] += np.pi / 2.0
  shifted_minus[i] -= np.pi / 2.0
  grad = (circuit.evaluate(shifted_plus) - circuit.evaluate(shifted_minus)) / 2.0
  ```
- **Key Lesson:**
  AI models frequently conflate standard trigonometric identities with the specific generator normalization of quantum gate Hamiltonians. Analytic quantum gradients must always be validated against independent finite-difference approximations.

---

### Failure Case 2: Common Subexpression Elimination (CSE) Replacing Impure State and Side-Effects (Track F)
- **Track:** Track F (Compiler Optimization)
- **Initial AI Output:**
  When generating the Common Subexpression Elimination (CSE) pass on 3-address Intermediate Representation (IR), the AI treated any instruction with identical operator and operands as identical, replacing subsequent instructions with the first result:
  ```python
  # Erroneous AI-generated code:
  expr_cache = {}
  for instr in ir_block:
      expr = (instr.op, instr.arg1, instr.arg2)
      if expr in expr_cache:
          instr.replace_with_copy(expr_cache[expr])
      else:
          expr_cache[expr] = instr.dest
  ```
- **Diagnostic Test & Failure Mode:**
  The test program contained variable reassignment in a sequential basic block:
  ```text
  t1 = a + b
  a = 10
  t2 = a + b   # AI replaced this with t2 = t1!
  ```
  The correctness test `assert vm.run(unoptimized) == vm.run(optimized)` failed immediately because `t2` returned the old value of `a + b` rather than `10 + b`.
- **Root Cause Analysis:**
  The AI's naive CSE pass failed to implement **variable invalidation / kill sets**. Whenever a variable (e.g. `a`) is redefined or modified in the basic block, all expressions in `expr_cache` referencing that variable must be purged (killed).
- **The Fix:**
  Implemented active kill-set tracking in `CSEPass`:
  ```python
  # On reassignment to variable 'dest', kill all cached expressions referencing 'dest':
  keys_to_kill = [k for k, v in expr_cache.items() if k[1] == instr.dest or k[2] == instr.dest]
  for k in keys_to_kill:
      del expr_cache[k]
  ```
- **Key Lesson:**
  In compiler optimization passes, equivalence is strictly temporal. Data-flow analysis requires explicit generation and kill sets. Unchecked AI code optimizes away reassignments, leading to silent semantic corruption.

---

### Failure Case 3: Priority Scheduler Inversion & Runqueue Corruption in xv6 Dispatcher (Track G)
- **Track:** Track G (Operating System Optimization)
- **Initial AI Output:**
  When drafting the xv6 kernel scheduler replacement in `kernel/proc.c`, the AI implemented a priority search loop that acquired and released process locks incorrectly:
  ```c
  // Erroneous AI-generated C code for xv6:
  struct proc *highest_p = 0;
  for(p = proc; p < &proc[NPROC]; p++) {
      acquire(&p->lock);
      if(p->state == RUNNABLE && (highest_p == 0 || p->priority > highest_p->priority)) {
          if(highest_p) release(&highest_p->lock);
          highest_p = p;
      } else {
          release(&p->lock);
      }
  }
  ```
- **Diagnostic Test & Failure Mode:**
  When multi-process concurrent workloads were executed, the scheduler caused an immediate kernel panic or deadlock: `panic: sched locks`.
- **Root Cause Analysis:**
  In xv6 RISC-V, process state changes and CPU context switches require holding the chosen process lock continuously into `swtch()` and releasing it in the target process context (`sched()` protocol). The AI's loop retained `highest_p->lock` while continuing to iterate and acquire subsequent process locks (`&p->lock`), violating lock ordering and creating deadlock whenever two CPUs searched for the highest priority process simultaneously. Furthermore, if no process was chosen, locks could be leaked.
- **The Fix:**
  Adopted a clean two-pass lockless inspection with atomic validation:
  1. Inspect runnable candidates without holding locks or by using a dedicated scheduler runqueue lock.
  2. Select the candidate PID, then acquire its lock and re-verify `p->state == RUNNABLE` before context switching.
  3. Added dynamic priority aging to eliminate starvation of low-priority processes.
- **Key Lesson:**
  AI-generated kernel code often compiles cleanly with GCC/Clang but introduces insidious race conditions, deadlock cycles, and lock protocol violations that destroy OS stability. Kernel concurrency invariants must be rigorously scrutinized and verified against OS lock-ordering rules.
