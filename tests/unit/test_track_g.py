"""Unit and Correctness Tests for Track G: xv6 Operating System Optimization."""

import os
import pytest

from src.track_g_xv6_os.xv6_kernel_sim import SimProc, XV6KernelSimulator


def test_xv6_patch_file_integrity():
    """Validates that the xv6 kernel patch file exists and contains valid diff headers."""
    patch_path = os.path.join(os.path.dirname(__file__), "../../src/track_g_xv6_os/xv6_patch.diff")
    assert os.path.exists(patch_path), "xv6_patch.diff is missing"
    with open(patch_path, "r") as f:
        content = f.read()
    assert "diff --git a/kernel/proc.c" in content
    assert "setpriority" in content
    assert "stride" in content


def test_stride_scheduler_zero_starvation_invariant():
    """All runnable processes must eventually complete under dynamic aging (zero starvation)."""
    procs = [
        SimProc(1, "low_prio_1", burst_ticks=20, priority=1),
        SimProc(2, "low_prio_2", burst_ticks=20, priority=1),
        SimProc(3, "high_prio", burst_ticks=40, priority=20),
    ]
    sim = XV6KernelSimulator(scheduler_policy="stride_priority")
    res = sim.run_workload(procs)

    # Every process must reach ZOMBIE state
    for p in res["processes"]:
        assert p.state == "ZOMBIE", f"Process {p.name} starved!"
        assert p.completion_tick is not None


def test_priority_scheduling_proportional_share():
    """High priority process must complete faster than low priority processes."""
    procs = [
        SimProc(1, "interactive_high", burst_ticks=15, priority=20),
        SimProc(2, "batch_low", burst_ticks=40, priority=2),
    ]
    sim = XV6KernelSimulator(scheduler_policy="stride_priority")
    res = sim.run_workload(procs)

    p_high = next(p for p in res["processes"] if p.priority == 20)
    p_low = next(p for p in res["processes"] if p.priority == 2)

    assert p_high.completion_tick < p_low.completion_tick, "High priority process did not complete first"
