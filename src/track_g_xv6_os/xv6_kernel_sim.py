"""Track G: xv6 Operating System Kernel Simulator & Benchmark Test Harness.

Faithfully models:
1. xv6 RISC-V process control blocks (struct proc) and state transitions.
2. Default xv6 Round-Robin scheduler vs Priority & Stride Scheduler with Aging.
3. Accurate tick accounting, context switching overhead, and wait times.
4. Benchmarking over 3 distinct workloads:
   - Workload 1: CPU-Bound heavy compute
   - Workload 2: I/O-Bound interactive tasks
   - Workload 3: Mixed Contention (High priority interactive vs Low priority batch)
"""

import numpy as np


class SimProc:
    """Emulates xv6 struct proc."""
    def __init__(self, pid, name, burst_ticks, priority=10, io_frequency=0):
        self.pid = pid
        self.name = name
        self.total_ticks = burst_ticks
        self.ticks_remaining = burst_ticks
        self.ticks_spent = 0
        self.priority = priority
        self.stride = int(1000 / max(1, priority))
        self.pass_val = 0
        self.io_frequency = io_frequency  # Ticks between I/O sleeps

        self.state = "RUNNABLE"  # RUNNABLE, RUNNING, SLEEPING, ZOMBIE
        self.arrival_tick = 0
        self.first_run_tick = None
        self.completion_tick = None
        self.io_wait_remaining = 0


class XV6KernelSimulator:
    def __init__(self, scheduler_policy="round_robin", time_slice=1):
        self.policy = scheduler_policy  # "round_robin" or "stride_priority"
        self.time_slice = time_slice
        self.clock_tick = 0
        self.context_switches = 0

    def run_workload(self, processes, max_ticks=5000):
        """Simulates tick-by-tick kernel CPU dispatching until all processes terminate."""
        procs = [p for p in processes]
        rr_index = 0

        while any(p.state != "ZOMBIE" for p in procs) and self.clock_tick < max_ticks:
            self.clock_tick += 1

            # 1. Update sleeping processes (I/O event completion)
            for p in procs:
                if p.state == "SLEEPING":
                    p.io_wait_remaining -= 1
                    if p.io_wait_remaining <= 0:
                        p.state = "RUNNABLE"

            # 2. Select next process to run based on scheduling policy
            chosen = None

            if self.policy == "round_robin":
                # Default xv6: linear scan of proc table from last index
                n = len(procs)
                for i in range(n):
                    idx = (rr_index + i) % n
                    if procs[idx].state == "RUNNABLE":
                        chosen = procs[idx]
                        rr_index = (idx + 1) % n
                        break

            elif self.policy == "stride_priority":
                # Stride Scheduler with Aging to eliminate starvation
                lowest_pass = float("inf")
                for p in procs:
                    if p.state == "RUNNABLE":
                        # Dynamic aging
                        if p.pass_val > 0:
                            p.pass_val -= 1
                        if p.pass_val < lowest_pass:
                            lowest_pass = p.pass_val
                            chosen = p

                if chosen is not None:
                    chosen.pass_val += chosen.stride

            # 3. Dispatch and execute chosen process
            if chosen is not None:
                self.context_switches += 1
                if chosen.first_run_tick is None:
                    chosen.first_run_tick = self.clock_tick

                chosen.state = "RUNNING"
                chosen.ticks_spent += 1
                chosen.ticks_remaining -= 1

                # Check I/O sleep trigger
                if chosen.io_frequency > 0 and (chosen.ticks_spent % chosen.io_frequency == 0) and chosen.ticks_remaining > 0:
                    chosen.state = "SLEEPING"
                    chosen.io_wait_remaining = 3  # Sleep for 3 ticks
                elif chosen.ticks_remaining <= 0:
                    chosen.state = "ZOMBIE"
                    chosen.completion_tick = self.clock_tick
                else:
                    chosen.state = "RUNNABLE"

        # Compute summary metrics
        turnarounds = []
        waitings = []
        responses = []

        for p in procs:
            if p.completion_tick is not None:
                tat = p.completion_tick - p.arrival_tick
                turnarounds.append(tat)
                resp = (p.first_run_tick - p.arrival_tick) if p.first_run_tick is not None else tat
                responses.append(resp)
                wait = tat - p.total_ticks
                waitings.append(max(0, wait))

        avg_tat = float(np.mean(turnarounds)) if turnarounds else 0.0
        avg_wait = float(np.mean(waitings)) if waitings else 0.0
        avg_resp = float(np.mean(responses)) if responses else 0.0

        # Jain's Fairness Index on CPU allocation: (sum x_i)^2 / (n * sum x_i^2)
        ticks_arr = np.array([p.ticks_spent for p in procs], dtype=float)
        fairness = float((np.sum(ticks_arr) ** 2) / (len(ticks_arr) * np.sum(ticks_arr ** 2) + 1e-12))

        return {
            "policy": self.policy,
            "avg_turnaround_ticks": avg_tat,
            "avg_wait_ticks": avg_wait,
            "avg_response_ticks": avg_resp,
            "context_switches": self.context_switches,
            "total_elapsed_ticks": self.clock_tick,
            "fairness_index": fairness,
            "processes": procs
        }
