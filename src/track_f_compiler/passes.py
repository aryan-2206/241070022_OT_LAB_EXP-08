"""Track F: Compiler Optimization Passes.

Implements 4 optimization passes:
1. Constant Folding & Constant Propagation
2. Common Sub-expression Elimination (CSE) with kill-set tracking
3. Dead-Code Elimination (DCE) via backward liveness analysis
4. Strength Reduction (replaces multiplication/division by powers of 2 with bitwise shifts)
"""

import copy
from src.track_f_compiler.compiler import Instruction


class ConstantFoldingPropagationPass:
    """Folds arithmetic on known literals and propagates constant bindings."""
    def run(self, ir):
        new_ir = []
        constants = {}
        changed = False

        for instr in ir:
            if instr.op == "ret":
                arg1 = constants.get(instr.arg1, instr.arg1)
                new_ir.append(Instruction("ret", arg1=arg1))
                continue

            # Propagate known constants into arguments
            arg1 = constants.get(instr.arg1, instr.arg1)
            arg2 = constants.get(instr.arg2, instr.arg2)

            if instr.op == "assign":
                if isinstance(arg1, int):
                    constants[instr.dest] = arg1
                else:
                    constants.pop(instr.dest, None)
                new_ir.append(Instruction("assign", arg1=arg1, dest=instr.dest))

            elif isinstance(arg1, int) and isinstance(arg2, int):
                # Constant fold
                res = None
                if instr.op == "+":
                    res = arg1 + arg2
                elif instr.op == "-":
                    res = arg1 - arg2
                elif instr.op == "*":
                    res = arg1 * arg2
                elif instr.op == "/" and arg2 != 0:
                    res = arg1 // arg2

                if res is not None:
                    constants[instr.dest] = res
                    new_ir.append(Instruction("assign", arg1=res, dest=instr.dest))
                    changed = True
                else:
                    constants.pop(instr.dest, None)
                    new_ir.append(Instruction(instr.op, arg1=arg1, arg2=arg2, dest=instr.dest))
            else:
                constants.pop(instr.dest, None)
                new_ir.append(Instruction(instr.op, arg1=arg1, arg2=arg2, dest=instr.dest))

        return new_ir, changed


class CommonSubexpressionEliminationPass:
    """Eliminates redundant expressions using active available expressions with kill sets."""
    def run(self, ir):
        new_ir = []
        expr_cache = {}  # (op, arg1, arg2) -> dest_var
        changed = False

        for instr in ir:
            if instr.op in ["ret", "assign"]:
                if instr.dest:
                    # Invalidate (kill) all expressions using instr.dest
                    keys_to_kill = [k for k in expr_cache if k[1] == instr.dest or k[2] == instr.dest]
                    for k in keys_to_kill:
                        del expr_cache[k]
                new_ir.append(instr)
                continue

            expr_key = (instr.op, instr.arg1, instr.arg2)

            if expr_key in expr_cache:
                # Subexpression already available! Replace with assignment
                prev_var = expr_cache[expr_key]
                new_ir.append(Instruction("assign", arg1=prev_var, dest=instr.dest))
                changed = True
            else:
                expr_cache[expr_key] = instr.dest
                new_ir.append(instr)

            # Invalidate any cached expressions where destination is modified
            keys_to_kill = [k for k in expr_cache if (k[1] == instr.dest or k[2] == instr.dest) and expr_cache[k] != instr.dest]
            for k in keys_to_kill:
                del expr_cache[k]

        return new_ir, changed


class StrengthReductionPass:
    """Transforms multiplication and division by powers of 2 into fast bitwise shift instructions."""
    def run(self, ir):
        new_ir = []
        changed = False

        for instr in ir:
            if instr.op == "*" and isinstance(instr.arg2, int) and instr.arg2 > 0 and (instr.arg2 & (instr.arg2 - 1)) == 0:
                shift_amt = int(instr.arg2).bit_length() - 1
                new_ir.append(Instruction("<<", arg1=instr.arg1, arg2=shift_amt, dest=instr.dest))
                changed = True
            elif instr.op == "/" and isinstance(instr.arg2, int) and instr.arg2 > 0 and (instr.arg2 & (instr.arg2 - 1)) == 0:
                shift_amt = int(instr.arg2).bit_length() - 1
                new_ir.append(Instruction(">>", arg1=instr.arg1, arg2=shift_amt, dest=instr.dest))
                changed = True
            else:
                new_ir.append(instr)

        return new_ir, changed


class DeadCodeEliminationPass:
    """Removes unused variable assignments via backward liveness analysis."""
    def run(self, ir):
        # Scan backward from return statement
        live_vars = set()
        new_ir_rev = []
        changed = False

        for instr in reversed(ir):
            if instr.op == "ret":
                if isinstance(instr.arg1, str) and not instr.arg1.isdigit():
                    live_vars.add(instr.arg1)
                new_ir_rev.append(instr)
                continue

            # If destination is never read later, instruction is dead code!
            if instr.dest not in live_vars:
                changed = True
                continue

            # Destination was live, so keep instruction and remove dest from live_vars (killed)
            live_vars.remove(instr.dest)

            # Mark arguments as live (gen)
            for arg in [instr.arg1, instr.arg2]:
                if isinstance(arg, str) and not arg.isdigit():
                    live_vars.add(arg)

            new_ir_rev.append(instr)

        new_ir = list(reversed(new_ir_rev))
        return new_ir, changed


class OptimizationPipeline:
    """Runs all 4 optimization passes repeatedly until a fixed point is reached."""
    def __init__(self):
        self.passes = [
            ConstantFoldingPropagationPass(),
            CommonSubexpressionEliminationPass(),
            StrengthReductionPass(),
            DeadCodeEliminationPass()
        ]

    def optimize(self, ir, max_iterations=10):
        current_ir = copy.deepcopy(ir)
        total_changed = False

        for _ in range(max_iterations):
            iter_changed = False
            for p in self.passes:
                current_ir, changed = p.run(current_ir)
                if changed:
                    iter_changed = True
                    total_changed = True
            if not iter_changed:
                break

        return current_ir, total_changed
