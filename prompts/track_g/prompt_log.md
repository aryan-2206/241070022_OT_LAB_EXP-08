# Track G Prompt & Engineering Audit Log
## xv6 Operating System Optimization (Priority & Stride Scheduling, Dynamic Aging)
**Track:** Track G  
**AI Assistant:** Gemini 3.8 Flash  
**Date:** 2026-10-25  

---

### Original Prompt
```text
Implement Track G: xv6 Operating System Optimization:
1. Kernel modification: Add Priority & Stride scheduling with dynamic priority aging to prevent process starvation in xv6 RISC-V.
2. System calls: Implement sys_setpriority(int priority) and sys_getpinfo(struct pstat *ps).
3. Generate git patch: xv6_patch.diff applying modifications to kernel/proc.c, kernel/proc.h, kernel/syscall.c, kernel/sysproc.c.
4. Kernel simulation test harness: Model xv6 process control blocks, states, and tick interrupts to benchmark Round-Robin baseline vs Priority/Stride.
5. Benchmark over 3 distinct workloads: CPU-bound, I/O-bound, and mixed contention.
```

### AI Response Summary
The AI generated `xv6_patch.diff` implementing the Stride scheduler algorithm ($L / \text{priority}$) with lock ordering and dynamic pass aging. It built `XV6KernelSimulator` accurately emulating xv6 process scheduling and measured turnaround, waiting, response, and fairness metrics.

### Revision & Refinement
Diagnosed and prevented multiple lock retention cycles during the highest-priority scan in `scheduler()` (Failure Log Case 3), ensuring lock acquisition happens strictly upon candidate selection.

### Verification Notes
- Unit test `test_xv6_patch_file_integrity` verified patch syntax and syscall definitions.
- `test_stride_scheduler_zero_starvation_invariant` proved all low-priority processes terminate under aging.
- Under mixed contention, the interactive task completion turnaround time dropped from 77.0 ticks to 37.0 ticks (a 51.9% latency improvement).

### Linked Commit
Milestone 8: `[AI:gemini-3.8] Track G: xv6 kernel priority stride scheduler patch, simulator, and multi-workload benchmark`
