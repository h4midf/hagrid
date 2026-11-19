"""Stillmaker and Baas factors from NanGate45 synthesis to the chip's 65 nm at 1.2 V
(default/components.json), and a synthesized circuit's cost at the chip's clock."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "..")
COMP = json.load(open(os.path.join(TARGET, "default", "components.json")))
DESIGN = json.load(open(os.path.join(TARGET, "default", "design.json")))


def sram_logic_rtl() -> str:
    """The SRAM macro logic's RTL: clock-gated or with enable multiplexers, by
    design.json model_assumptions.sram_logic.clock_gated."""
    gated = DESIGN["model_assumptions"]["sram_logic"]["clock_gated"]
    return os.path.join(HERE, "rtl", "rse_sram_logic_gated.v" if gated else "rse_sram_logic.v")


def _poly(coeffs, v):
    return coeffs[0] * v * v + coeffs[1] * v + coeffs[2]


def factors() -> dict:
    src, dst, scaling = COMP["synthesis"], COMP["target"], COMP["node_scaling"]
    s_key, d_key = f"{src['node_nm']}_{src['type']}", f"{dst['node_nm']}_{dst['type']}"
    g_src, g_dst = scaling["area_geometry"][str(src["node_nm"])], scaling["area_geometry"][str(dst["node_nm"])]
    area = ((g_dst[0] / g_src[0]) ** 2 * (g_dst[1] / g_src[1]) ** 2 * (g_dst[2] / g_src[2])) ** (1 / 3)

    def ratio(table):
        return _poly(scaling[table][d_key], dst["vdd_V"]) / _poly(scaling[table][s_key], src["vdd_V"])
    return {"area": area, "energy": ratio("energy_coefficients"), "power": ratio("power_coefficients")}


def at_chip(row: dict) -> dict:
    """A report.py row at 65 nm and the chip's clock: report.py's dynamic estimate is at
    1 GHz and activity 0.2. Leakage is on the chip's low-power devices (the factor from
    spice/logic/inverter.py)."""
    f = factors()
    clock_GHz = DESIGN["clock"]["freq_MHz"] * 1e-3
    dynamic = row["dynamic_mW_est"] * clock_GHz * f["energy"]
    logic_path = os.path.join(TARGET, "default", "results", "logic.json")
    if not os.path.exists(logic_path):
        raise FileNotFoundError(f"{logic_path}: run spice/logic/inverter.py first")
    leakage = row["leakage_uW"] * 1e-3 * json.load(open(logic_path))["leakage_factor"]
    return {"area_um2": row["area_um2_45nm"] * f["area"], "dynamic_mW": dynamic, "leakage_mW": leakage,
            "power_mW": dynamic + leakage, "clock_GHz": clock_GHz, "factors": f}
