"""Evaluates an SFU unit combinationally in yosys (PIPE = 0: stage registers are wires),
one `eval` per input vector, in batches. Shared by the test_*.py scripts."""
import os
import re
import subprocess

RTL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rtl")
BATCH = 400


def evaluate(modules, top, vectors, output, pipelined=True):
    """`output` of `top` for every vector ({input name: int}); rtl/<module>.v for each
    module. A pipelined unit is evaluated with PIPE = 0. Returns a list of ints."""
    files = " ".join(os.path.join(RTL, m + ".v") for m in modules)
    head = [f"read_verilog -sv {files}"]
    if pipelined:
        head.append(f"chparam -set PIPE 0 {top}")
    head += [f"hierarchy -top {top}", "proc", "flatten", "opt_clean", f"rename -top {top}",
             f"select {top}"]
    results = []
    for start in range(0, len(vectors), BATCH):
        lines = list(head)
        for vec in vectors[start:start + BATCH]:
            sets = " ".join(f"-set {k} {v}" for k, v in vec.items())
            lines.append(f"eval {sets} -show {output}")
        out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
        found = re.findall(rf"Eval result: \\{output} = (\d+)'([01x]+)\.", out)
        if len(found) != len(vectors[start:start + BATCH]):
            raise RuntimeError(f"{top}: {len(found)} results for {len(vectors[start:start + BATCH])} vectors\n"
                               + out[-2000:])
        for _, bits in found:
            if "x" in bits:
                raise RuntimeError(f"{top}: undefined output bits {bits}")
            results.append(int(bits, 2))
    return results
