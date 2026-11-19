"""Target description of the default 1F1C macro (Laleni et al. 2024 design point).

RMU = the FeFETs, WMU = the capacitors (CMoM); both are the same 128 x 128 cells.

Sources (this folder):
  design.json          array, cell, clock, ADC
  results/column.json  FeFET -> capacitor load      (spice/column/column_sim.py)
  results/mac.json     RMU MAC, WMU MAC, capacitor read / write  (spice/mac/rmu_mac.py, wmu_mac.py, cap_rw.py)
  organization.json    arrays per PE, PEs per tile, routers (ISAAC organization)
  area.json            column ADC and front-end (buffer, sample-and-hold) area
  results/nvsim.json   cell array area and leakage (run_nvsim.py)
  results/column.json, efficiency.json  column charging and ADC conversion energy
  energy.json          scaling of synthesized circuits to the target node
  results/logic.json   macro-logic leakage on low-power devices (spice/logic/inverter.py)
  macro_logic/results/*.json  input / phase control, shift-and-add, WMU readout
                       (synthesized, macro_logic/characterize_*.py; the shift-and-add's
                       accumulator SRAM banks with NVSim, "acc_sram")
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from targets.target_loader import (  # noqa: E402
    ArrayArea, ArrayEnergy, Geometry, MemoryUnit, Organization, SingleModuleTarget, Timing,
)

MACRO_LOGIC = HERE / "macro_logic" / "results"


def _macro_logic_row(name: str) -> dict:
    rows = json.loads((MACRO_LOGIC / f"{name}.json").read_text())
    if len(rows) != 1:
        raise ValueError(f"macro_logic/results/{name}.json has {len(rows)} rows, expected 1")
    return rows[0]


def _acc_sram(shift_add: dict, node_nm: int) -> dict:
    """The shift-and-add's accumulator SRAM banks (NVSim, already at their own node)."""
    sram = shift_add["acc_sram"]
    if sram["node_nm"] != node_nm:
        raise ValueError(f"shift_add acc_sram is NVSim at {sram['node_nm']} nm, but the target is {node_nm} nm")
    return sram


def _macro_logic_leakage_factor(rows: list[dict], clock_MHz: float) -> float:
    """Leakage of the macro logic on low-power devices over Nangate45 (results/logic.json,
    spice/logic/inverter.py). Fails if a macro-logic critical path, slowed to the
    low-power devices, no longer fits the clock."""
    inv = json.loads((HERE / "results" / "logic.json").read_text())["inverter"]
    period_ps = 1e6 / clock_MHz
    for row in rows:
        if row["delay_ps"] * inv["slowdown"] > period_ps:
            raise ValueError(f"{row['unit']}: critical path {row['delay_ps']:.0f} ps x {inv['slowdown']:.2f} "
                             f"(low-power devices) exceeds the {period_ps:.0f} ps clock period")
    return inv["leakage_factor"]


def _array_energy(design: dict, mac: dict, node_nm: int) -> ArrayEnergy:
    """Per-event energies of one array (results/*.json SPICE and NVSim, macro_logic
    synthesis scaled with energy.json; see energy.json's _provenance)."""
    energy = json.loads((HERE / "energy.json").read_text())
    scaling = energy["cmos_scaling"]
    if scaling["target_node_nm"] != node_nm:
        raise ValueError(f"energy.json is scaled to {scaling['target_node_nm']} nm, but the target is {node_nm} nm")
    e_factor, p_factor = scaling["deepscale_energy_factor"], scaling["deepscale_power_factor"]
    column = json.loads((HERE / "results" / "column.json").read_text())["bl_charging"]
    efficiency = json.loads((HERE / "results" / "efficiency.json").read_text())["macro"]
    nvsim = json.loads((HERE / "results" / "nvsim.json").read_text())["array"]
    writes = [case["write_energy_from_reference_fJ"] for case in mac["cap_write"]["by_level"]]
    ctrl, shift_add, readout = (_macro_logic_row(n) for n in ("macro_ctrl", "shift_add", "wmu_readout"))
    sram = _acc_sram(shift_add, node_nm)
    if energy["macro_logic_devices"]["result"] != "results/logic.json inverter":
        raise ValueError(f"energy.json macro_logic_devices: unknown result {energy['macro_logic_devices']['result']!r}")
    leak_factor = _macro_logic_leakage_factor([ctrl, shift_add, readout], design["clock"]["freq_MHz"])
    phases = design["clock"]["column_phase_cycles"]
    # synthesized dynamic power is estimated at 1 GHz (report.py), so its value in mW is
    # the energy of one cycle in pJ. The SRAM banks are NVSim at node_nm, not scaled:
    # every shift-and-add cycle (slot) reads and writes one word in each bank.
    return ArrayEnergy(
        adc_conversion_pJ=efficiency["adc_energy_pJ"],
        column_mac_pJ=column["energy_per_column_per_mac_fJ"] / 1000,
        cell_load_pJ=column["energy_per_cell_per_mac_fJ"] / 1000,
        wmu_write_cell_pJ=sum(writes) / len(writes) / 1000,
        wmu_read_conversions=mac["cap_read"]["conversions"],
        wmu_mac_conversions=mac["wmu_mac"]["conversions"],
        mac_cycles=phases["reset"] + phases["multiply"] + phases["accumulate"] + design["adc"]["cycles_per_conversion"],
        conversion_cycles=design["adc"]["cycles_per_conversion"],
        shift_add_cycles_per_mac=shift_add["params"]["OUTS"] // shift_add["params"]["UNITS"],
        unit_column_mac_pJ=0.0,   # Laleni's macro has no weight-flip encoding, so no unit column
        macro_ctrl_cycle_pJ=ctrl["dynamic_mW_est"] / e_factor,
        shift_add_cycle_pJ=(shift_add["dynamic_mW_est"] / e_factor
                            + sram["banks"] * (sram["read_pJ"] + sram["write_pJ"])),
        wmu_readout_cycle_pJ=readout["dynamic_mW_est"] / e_factor,
        array_leakage_uW=(nvsim["leakage_nW"] / 1000
                          + (ctrl["leakage_uW"] + shift_add["leakage_uW"] + readout["leakage_uW"])
                          * leak_factor / p_factor
                          + sram["banks"] * sram["leakage_uW"]),
        cmos_energy_factor=e_factor,
        cmos_power_factor=p_factor,
    )


def _array_area(node_nm: int) -> ArrayArea:
    """Area of one array and its circuits at node_nm (area.json, nvsim.json,
    macro_logic results; see area.json's _provenance)."""
    area = json.loads((HERE / "area.json").read_text())
    adc = area["adc"]
    if adc["target_node_nm"] != node_nm:
        raise ValueError(f"area.json ADC is scaled to {adc['target_node_nm']} nm, but the target is {node_nm} nm")
    adc_um2 = adc["source_area_um2"] / adc["deepscale_area_factor"]

    # column front end: capacitors at the MoM density, transistors with NeuroSim's
    # single-transistor layout area at their own node, scaled to node_nm
    fe = area["column_frontend"]
    rules = fe["transistor_layout"]
    f_um = fe["transistor_node_nm"] / 1000
    transistors_um2 = 0.0
    for w_nm, l_nm in fe["transistors_W_L_nm"]:
        width = 2 * (l_nm / 1000 + rules["gate_poly_gap_F"] * f_um)
        height = (w_nm / 1000 + rules["p_n_diffusion_gap_F"] * f_um
                  + 2 * (rules["poly_extension_F"] + rules["field_poly_gap_F"] / 2) * f_um)
        transistors_um2 += width * height
    frontend_um2 = (sum(fe["capacitance_fF"].values()) / fe["mom_density_fF_per_um2"]
                    + transistors_um2 / fe["transistor_deepscale_area_factor"])

    def macro_logic(name):
        return _macro_logic_row(name)[f"area_um2_{node_nm}nm"]

    shift_add = _macro_logic_row("shift_add")
    sram = _acc_sram(shift_add, node_nm)

    return ArrayArea(
        cells_um2=json.loads((HERE / "results" / "nvsim.json").read_text())["array"]["array_area_um2"],
        adcs_um2=adc["per_array"] * adc_um2,
        column_frontends_um2=fe["per_array"] * frontend_um2,
        macro_ctrl_um2=macro_logic("macro_ctrl"),
        shift_add_um2=shift_add[f"area_um2_{node_nm}nm"] + sram["banks"] * sram["area_um2"],
        wmu_readout_um2=macro_logic("wmu_readout"),
    )


def load() -> SingleModuleTarget:
    design = json.loads((HERE / "design.json").read_text())
    mac = json.loads((HERE / "results" / "mac.json").read_text())
    org = json.loads((HERE / "organization.json").read_text())
    scaling = org["scaling"]
    if scaling["target_node_nm"] != design["process"]["node_nm"]:
        raise ValueError(f"organization.json is scaled to {scaling['target_node_nm']} nm, "
                         f"but the target is {design['process']['node_nm']} nm")

    def area_um2(part):
        return part["source_area_mm2"] * 1e6 / scaling["deepscale_area_factor"]

    def access_pJ(part):
        """One access at full activity: scaled power / ISAAC's clock (mW / GHz = pJ)."""
        return part["source_power_mW"] / scaling["deepscale_power_factor"] / scaling["source_clock_GHz"]

    router = org["chip"]["router"]
    input_register = org["pe"]["input_register"]
    bus = org["tile"]["bus"]

    t_clk_ns = 1e3 / design["clock"]["freq_MHz"]

    # RMU read: FeFET -> capacitor load = the multiply phase (WL pulse width; the
    # stored level never settles, see results/column.json -> fet_to_cap)
    rmu_read_ns = design["clock"]["column_phase_cycles"]["multiply"] * t_clk_ns

    # RMU MAC: FeFET -> CMoM load, then one ADC conversion (spice/mac/rmu_mac.py)
    rmu_mac_ns = mac["rmu_mac"]["latency_ns"]

    # WMU MAC: selected rows' CMoMs shared onto the bitline, then one ADC conversion
    # (spice/mac/wmu_mac.py)
    wmu_mac_ns = mac["wmu_mac"]["latency_ns"]

    # WMU read: two ADC conversions (bitline reference, then the cell shared onto it)
    # (spice/mac/cap_rw.py)
    wmu_read_ns = mac["cap_read"]["latency_ns"]

    # WMU write: the clocked write window; the write settles inside it (spice/mac/cap_rw.py)
    wmu_write_ns = mac["cap_write"]["latency_ns"]

    unit = MemoryUnit(rows=design["array"]["rows"], cols=design["array"]["cols"],
                      bits_per_cell=design["cell"]["bits_per_cell"])
    return SingleModuleTarget(
        name="1fefet1c/default",
        architecture=design["architecture"],
        node_nm=design["process"]["node_nm"],
        geometry=Geometry(rmu=unit, wmu=unit),
        timing=Timing(rmu_mac_ns=rmu_mac_ns, wmu_mac_ns=wmu_mac_ns, rmu_read_ns=rmu_read_ns, rmu_write_ns=math.inf,
                      wmu_read_ns=wmu_read_ns, wmu_write_ns=wmu_write_ns),
        # router, PE input register and tile bus scaled from ISAAC's node:
        # value / DeepScaleTool factor
        organization=Organization(
            arrays_per_pe=org["pe"]["arrays"],
            pes_per_tile=org["tile"]["pes"],
            tiles_per_router=org["chip"]["tiles_per_router"],
            router_area_um2=area_um2(router),
            router_flit_bits=router["flit_bits"],
            # full activity moves one flit through each port per cycle
            router_flit_hop_pJ=access_pJ(router) / router["ports"],
            pe_input_register_area_um2=area_um2(input_register),
            pe_input_register_access_bits=input_register["access_bits"],
            pe_input_register_access_pJ=access_pJ(input_register),
            tile_bus_area_um2=area_um2(bus),
            tile_bus_bits=bus["wires"],
            tile_bus_transfer_pJ=access_pJ(bus),
        ),
        array_area=_array_area(design["process"]["node_nm"]),
        array_energy=_array_energy(design, mac, design["process"]["node_nm"]),
    )
