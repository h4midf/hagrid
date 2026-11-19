"""CACTI 6.5 for HyFlexPIM's input and output registers (Table 2 IR / OR rows), and for
ISAAC's IR and OR at 32 nm to check the tool and settings against the source of those
rows. Settings and access rates: default/components.json "registers". Writes
results/registers.json.

  power = leakage + read energy x access rate

Usage (from the repo root, after building CACTI; see cacti_run.py):
  python3 targets/hyflexpim/periphery/characterize_registers.py
"""
import json
import os

from cacti_run import smallest

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = json.load(open(os.path.join(HERE, "..", "default", "components.json")))
DESIGN = json.load(open(os.path.join(HERE, "..", "default", "design.json")))
REG = COMP["registers"]
CFG = REG["config"]
KW = {"cell": CFG["cell"], "peripheral": CFG["peripheral"], "ecc": CFG["ecc"]}


def main():
    out = {"validation": {}, "hyflexpim": {}}
    for name, v in REG["validation"].items():
        r = smallest(v["bytes"], v["node_nm"], CFG["access_widths_bytes"], **KW)
        r["source_area_mm2"] = v["area_mm2"]
        r["source_power_mW"] = v["power_mW"]
        out["validation"][name] = r
        print(f"  {name} {v['bytes']} B at {v['node_nm']} nm: {r['area_mm2']:.5f} mm^2 (source {v['area_mm2']}), "
              f"leakage {r['leakage_mW']:.3f} mW, read {r['read_energy_pJ']:.2f} pJ", flush=True)

    for module in ("analog_module", "digital_module"):
        regs = DESIGN[module]["registers"]
        rate_MHz = REG["access_rate_MHz"][module]
        for reg, key in (("ir", "input_register_bytes"), ("or", "output_register_bytes")):
            r = smallest(regs[key], REG["node_nm"], CFG["access_widths_bytes"], **KW)
            r["access_rate_MHz"] = rate_MHz
            r["power_mW"] = r["leakage_mW"] + r["read_energy_pJ"] * 1e-12 * rate_MHz * 1e6 * 1e3
            out["hyflexpim"][f"{module}.{reg}"] = r
            print(f"  {module} {reg} {regs[key]} B at {REG['node_nm']} nm: {r['area_mm2']:.5f} mm^2, "
                  f"{r['power_mW']:.3f} mW (leakage {r['leakage_mW']:.3f}, read {r['read_energy_pJ']:.2f} pJ "
                  f"x {rate_MHz} MHz)", flush=True)

    path = os.path.join(HERE, "results", "registers.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
