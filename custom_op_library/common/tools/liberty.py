"""Per-cell area, leakage and input-pin capacitance from the NanGate45 liberty file."""
import os
import re

LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib", "NangateOpenCellLibrary_typical.lib")


def parse(path=LIB):
    cells, cur, pin_dir = {}, None, None
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*cell\s*\(\s*\"?(\w+)\"?\s*\)", line)
            if m:
                cur = m.group(1)
                cells[cur] = {"area": 0.0, "leak": 0.0, "cap": 0.0}
                continue
            if cur is None:
                continue
            m = re.match(r"\s*area\s*:\s*([\d.]+)", line)
            if m and cells[cur]["area"] == 0.0:
                cells[cur]["area"] = float(m.group(1))
            m = re.match(r"\s*cell_leakage_power\s*:\s*([-\d.eE+]+)", line)
            if m:
                cells[cur]["leak"] = float(m.group(1))
            if re.match(r"\s*pin\s*\(", line):
                pin_dir = None
            m = re.match(r"\s*direction\s*:\s*\"?(\w+)\"?", line)
            if m:
                pin_dir = m.group(1)
            m = re.match(r"\s*capacitance\s*:\s*([\d.eE+-]+)", line)
            if m and pin_dir == "input":
                cells[cur]["cap"] += float(m.group(1))
                pin_dir = "counted"
    return cells


CELLS = parse()
