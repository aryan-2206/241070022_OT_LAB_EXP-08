"""Track F: Compiler Optimization - Toy Language Compiler & Virtual Machine.

Implements:
1. Lexer and Parser for a toy algorithmic language.
2. 3-Address Code (TAC) Intermediate Representation (IR).
3. Stack/Register Virtual Machine interpreter to execute TAC instructions
   and verify output equivalence between unoptimized and optimized code.
"""

import re


class Instruction:
    """Represents a single 3-Address Code (TAC) instruction: dest = arg1 op arg2."""
    def __init__(self, op, arg1=None, arg2=None, dest=None):
        self.op = op          # '+', '-', '*', '/', '<<', '>>', 'assign', 'ret'
        self.arg1 = arg1      # variable name or literal number
        self.arg2 = arg2      # variable name or literal number
        self.dest = dest      # variable name

    def __repr__(self):
        if self.op == "ret":
            return f"ret {self.arg1}"
        elif self.op == "assign":
            return f"{self.dest} = {self.arg1}"
        else:
            return f"{self.dest} = {self.arg1} {self.op} {self.arg2}"


class ToyCompiler:
    """Compiles high-level arithmetic programs into 3-Address Code (TAC)."""
    def __init__(self):
        self.temp_counter = 0

    def new_temp(self):
        self.temp_counter += 1
        return f"t{self.temp_counter}"

    def parse_and_generate_ir(self, source_code):
        """
        Parses source lines of the form:
        var = expr
        where expr can contain numbers, variables, and operators (+, -, *, /).
        Returns list of Instruction objects.
        """
        ir = []
        lines = [line.strip() for line in source_code.strip().split("\n") if line.strip() and not line.strip().startswith("#")]

        for line in lines:
            if line.startswith("return "):
                val = line.split("return ")[1].strip()
                ir.append(Instruction("ret", arg1=val))
                continue

            # Target = Expr
            if "=" in line:
                target, expr = [x.strip() for x in line.split("=", 1)]
                tokens = re.findall(r'[a-zA-Z_]\w*|\d+|[+\-*/()]', expr)
                # Parse simple binary operations
                if len(tokens) == 1:
                    # direct assignment: x = y or x = 5
                    val = int(tokens[0]) if tokens[0].isdigit() else tokens[0]
                    ir.append(Instruction("assign", arg1=val, dest=target))
                elif len(tokens) == 3 and tokens[1] in ["+", "-", "*", "/"]:
                    arg1 = int(tokens[0]) if tokens[0].isdigit() else tokens[0]
                    op = tokens[1]
                    arg2 = int(tokens[2]) if tokens[2].isdigit() else tokens[2]
                    ir.append(Instruction(op, arg1=arg1, arg2=arg2, dest=target))
                else:
                    # Compound expression: break into temporary 3-address steps
                    # Sequential left-to-right evaluation
                    curr_val = int(tokens[0]) if tokens[0].isdigit() else tokens[0]
                    idx = 1
                    while idx < len(tokens):
                        op = tokens[idx]
                        nxt = int(tokens[idx+1]) if tokens[idx+1].isdigit() else tokens[idx+1]
                        is_last = (idx + 2 >= len(tokens))
                        temp_dest = target if is_last else self.new_temp()
                        ir.append(Instruction(op, arg1=curr_val, arg2=nxt, dest=temp_dest))
                        curr_val = temp_dest
                        idx += 2
        return ir


class ToyVM:
    """Simulates CPU execution of 3-Address Code and measures cycle counts."""
    def run(self, ir_instructions, initial_env=None):
        env = {} if initial_env is None else initial_env.copy()
        cycles = 0

        def resolve(val):
            if isinstance(val, (int, float)):
                return val
            if isinstance(val, str):
                if val.isdigit():
                    return int(val)
                return env.get(val, 0)
            return 0

        for instr in ir_instructions:
            cycles += 1
            op = instr.op

            if op == "assign":
                env[instr.dest] = resolve(instr.arg1)
            elif op == "+":
                env[instr.dest] = resolve(instr.arg1) + resolve(instr.arg2)
            elif op == "-":
                env[instr.dest] = resolve(instr.arg1) - resolve(instr.arg2)
            elif op == "*":
                cycles += 2  # Multiplications take 3 cycles
                env[instr.dest] = resolve(instr.arg1) * resolve(instr.arg2)
            elif op == "/":
                cycles += 4  # Divisions take 5 cycles
                d = resolve(instr.arg2)
                env[instr.dest] = resolve(instr.arg1) // d if d != 0 else 0
            elif op == "<<":
                env[instr.dest] = resolve(instr.arg1) << resolve(instr.arg2)
            elif op == ">>":
                env[instr.dest] = resolve(instr.arg1) >> resolve(instr.arg2)
            elif op == "ret":
                return resolve(instr.arg1), cycles, env

        return env.get("result", 0), cycles, env
