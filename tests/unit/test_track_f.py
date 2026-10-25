"""Unit and Correctness Tests for Track F: Compiler Optimization."""

import pytest

from src.track_f_compiler.compiler import ToyCompiler, ToyVM, Instruction
from src.track_f_compiler.passes import (
    ConstantFoldingPropagationPass, CommonSubexpressionEliminationPass,
    DeadCodeEliminationPass, StrengthReductionPass, OptimizationPipeline
)
from src.track_f_compiler.reg_alloc import RegisterAllocator


def test_compiler_output_equivalence():
    """Unoptimized IR and Optimized IR must produce bit-for-bit identical results on the VM."""
    code = """
    a = 15
    b = 3
    c = a * 4
    d = a + b
    e = a + b
    unused = c * 10
    res = c + d + e
    return res
    """
    compiler = ToyCompiler()
    vm = ToyVM()
    pipeline = OptimizationPipeline()

    unopt_ir = compiler.parse_and_generate_ir(code)
    opt_ir, _ = pipeline.optimize(unopt_ir)

    unopt_res, _, _ = vm.run(unopt_ir)
    opt_res, _, _ = vm.run(opt_ir)

    assert unopt_res == opt_res, f"Output mismatch: {unopt_res} != {opt_res}"
    assert len(opt_ir) < len(unopt_ir), "Optimization pipeline failed to reduce instruction count"


def test_cse_kill_set_reassignment():
    """Verifies Failure Case 2 fix: CSE must NOT reuse cached expressions if operands are redefined."""
    # t1 = a + b
    # a = 100
    # t2 = a + b  (MUST NOT be replaced by t1)
    ir = [
        Instruction("assign", arg1=5, dest="a"),
        Instruction("assign", arg1=10, dest="b"),
        Instruction("+", arg1="a", arg2="b", dest="t1"),
        Instruction("assign", arg1=100, dest="a"),  # Kills (a + b)
        Instruction("+", arg1="a", arg2="b", dest="t2"),
        Instruction("ret", arg1="t2")
    ]
    vm = ToyVM()
    orig_res, _, _ = vm.run(ir)

    cse = CommonSubexpressionEliminationPass()
    opt_ir, _ = cse.run(ir)
    opt_res, _, _ = vm.run(opt_ir)

    assert orig_res == 110
    assert opt_res == 110, f"CSE incorrectly reused killed expression! Expected 110, got {opt_res}"


def test_register_allocation_graph_coloring_validity():
    """Interfering variables must never be assigned the same physical register."""
    # Overlapping variables: a, b, c all simultaneously live
    ir = [
        Instruction("assign", arg1=1, dest="a"),
        Instruction("assign", arg1=2, dest="b"),
        Instruction("+", arg1="a", arg2="b", dest="c"),
        Instruction("ret", arg1="c")
    ]
    allocator = RegisterAllocator(num_registers=3)
    var_list, adj_mat, _ = allocator.compute_liveness_and_interference(ir)
    res = allocator.solve_ilp(var_list, adj_mat)

    assert res["spill_count"] == 0
    assigned = res["assignment"]
    assert assigned["a"] != assigned["b"]
    assert assigned["a"] != assigned["c"]
    assert assigned["b"] != assigned["c"]
