"""Target description of a sized 1F1C config (targets/1fefet1c/<config>, config = 64,
128): the default macro (Laleni et al. 2024) at another array size, with the ADC
resolution of ISAAC's Eq. 2 and ISAAC's weight-flip encoding (one unit column per
array, one flip bit per column); see <config>/design.json.

Each config's <config>/target_loader.py calls load_config(<its folder>).

Sources, in the config folder:
  design.json          array, encoding, cell, clock, ADC
  area.json            column ADC (scaled to the config's resolution) and front end
  results/nvsim.json   cell array area (data columns + unit column) and leakage (run_nvsim.py)
  results/column.json  column charging energy, FeFET -> capacitor load (spice/column/column_sim.py)
  results/adc.json     ADC energy per conversion: bias_sweep at design.json's buffer bias
                       (spice/adc/run_full.py, with the config's sized front end)
  results/mac.json     capacitor read / write (spice/mac/cap_rw.py)
The MAC latencies are clock counts (_mac_latencies_ns), as spice/mac/rmu_mac.py and
wmu_mac.py time them; those two simulations only check the codes, so they are not run.
  macro_logic/results/*.json  input / phase control, shift-and-add with the flip
                       correction, WMU readout (macro_logic/characterize_*.py)
and shared with the default (no dependence on the array size):
  ../default/organization.json   arrays per PE, PEs per tile, routers (ISAAC)
  ../default/energy.json         scaling of synthesized circuits to the target's node
  ../default/results/logic.json  macro-logic leakage on low-power devices
"""
import json
import math
import sys
from pathlib import Path

TARGET_DIR = Path(__file__).resolve().parent
DEFAULT = TARGET_DIR / "default"
sys.path.insert(0, str(TARGET_DIR.parents[1]))
from targets.target_loader import (  # noqa: E402
    ArrayArea, ArrayEnergy, Geometry, MemoryUnit, Organization, SingleModuleTarget, Timing,
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _macro_logic_row(config_dir: Path, name: str) -> dict:
    rows = _json(config_dir / "macro_logic" / "results" / f"{name}.json")
    if len(rows) != 1:
        raise ValueError(f"{config_dir.name}/macro_logic/results/{name}.json has {len(rows)} rows, expected 1")
    return rows[0]


def _acc_sram(shift_add: dict, node_nm: int) -> dict:
    """The shift-and-add's accumulator SRAM banks (NVSim, already at their own node)."""
    sram = shift_add["acc_sram"]
    if sram["node_nm"] != node_nm:
        raise ValueError(f"shift_add acc_sram is NVSim at {sram['node_nm']} nm, but the target is {node_nm} nm")
    return sram


def _check_encoding(config_dir: Path, design: dict, shift_add: dict) -> int:
    """The unit columns per array; fails unless the config has ISAAC's encoding and its
    shift-and-add is the flip-correcting one."""
    enc = design["encoding"]
    if enc["scheme"] != "ISAAC weight flip":
        raise ValueError(f"{config_dir.name}: unknown encoding {enc['scheme']!r}")
    if shift_add["rtl"] != "shift_add_flip":
        raise ValueError(f"{config_dir.name}: the encoding needs shift_add_flip, the results are {shift_add['rtl']}")
    return enc["unit_columns"]


def _adc_energy_pJ(config_dir: Path, design: dict) -> float:
    """One conversion (core, buffer, Vref, VCM) of the config's ADC with its sized front
    end: results/adc.json bias_sweep at design.json's buffer bias (run_full.py)."""
    bias = design["adc"]["buffer_bias_uA"]
    rows = [r for r in _json(config_dir / "results" / "adc.json")["bias_sweep"]["by_bias"]
            if abs(r["ibias_uA"] - bias) < 1e-9]
    if len(rows) != 1:
        raise ValueError(f"{config_dir.name}: results/adc.json bias_sweep has {len(rows)} runs at {bias} uA, expected 1")
    return rows[0]["e_total_pJ"]


def _mac_latencies_ns(design: dict) -> tuple[float, float]:
    """(RMU MAC, WMU MAC) latency in ns, the clock counts of spice/mac/rmu_mac.py and
    wmu_mac.py: an RMU MAC runs from the multiply phase (FeFET -> CMoM load) to the end
    of its conversion; a WMU MAC from the row switches closing (1 ns before the second
    reset releases) to the end of its signal conversion."""
    t_clk_ns = 1e3 / design["clock"]["freq_MHz"]
    t_conv_ns = design["adc"]["cycles_per_conversion"] * t_clk_ns
    rmu = design["clock"]["column_phase_cycles"]["multiply"] * t_clk_ns + t_conv_ns
    wmu = t_conv_ns + 1.0
    return rmu, wmu


# ADC conversions per column of a WMU MAC (spice/mac/wmu_mac.py CONVERSIONS: the
# reference conversion is not needed)
WMU_MAC_CONVERSIONS = 1


def _macro_logic_leakage_factor(rows: list[dict], clock_MHz: float) -> float:
    """Leakage of the macro logic on low-power devices over Nangate45 (default
    results/logic.json, spice/logic/inverter.py). Fails if a macro-logic critical path,
    slowed to the low-power devices, no longer fits the clock."""
    inv = _json(DEFAULT / "results" / "logic.json")["inverter"]
    period_ps = 1e6 / clock_MHz
    for row in rows:
        if row["delay_ps"] * inv["slowdown"] > period_ps:
            raise ValueError(f"{row['unit']}: critical path {row['delay_ps']:.0f} ps x {inv['slowdown']:.2f} "
                             f"(low-power devices) exceeds the {period_ps:.0f} ps clock period")
    return inv["leakage_factor"]


def _array_energy(config_dir: Path, design: dict, mac: dict, node_nm: int) -> ArrayEnergy:
    energy = _json(DEFAULT / "energy.json")
    scaling = energy["cmos_scaling"]
    if scaling["target_node_nm"] != node_nm:
        raise ValueError(f"energy.json is scaled to {scaling['target_node_nm']} nm, but the target is {node_nm} nm")
    e_factor, p_factor = scaling["deepscale_energy_factor"], scaling["deepscale_power_factor"]
    results = config_dir / "results"
    column = _json(results / "column.json")["bl_charging"]
    adc_pJ = _adc_energy_pJ(config_dir, design)
    nvsim = _json(results / "nvsim.json")["array"]
    writes = [case["write_energy_from_reference_fJ"] for case in mac["cap_write"]["by_level"]]
    ctrl, shift_add, readout = (_macro_logic_row(config_dir, n) for n in ("macro_ctrl", "shift_add", "wmu_readout"))
    unit_columns = _check_encoding(config_dir, design, shift_add)
    sram = _acc_sram(shift_add, node_nm)
    if energy["macro_logic_devices"]["result"] != "results/logic.json inverter":
        raise ValueError(f"energy.json macro_logic_devices: unknown result {energy['macro_logic_devices']['result']!r}")
    leak_factor = _macro_logic_leakage_factor([ctrl, shift_add, readout], design["clock"]["freq_MHz"])
    phases = design["clock"]["column_phase_cycles"]
    column_mac_pJ = column["energy_per_column_per_mac_fJ"] / 1000
    # synthesized dynamic power is estimated at 1 GHz (report.py), so its value in mW is
    # the energy of one cycle in pJ. The SRAM banks are NVSim at node_nm, not scaled:
    # every shift-and-add cycle (slot) reads and writes one word in each bank.
    return ArrayEnergy(
        adc_conversion_pJ=adc_pJ,
        column_mac_pJ=column_mac_pJ,
        cell_load_pJ=column["energy_per_cell_per_mac_fJ"] / 1000,
        wmu_write_cell_pJ=sum(writes) / len(writes) / 1000,
        wmu_read_conversions=mac["cap_read"]["conversions"],
        wmu_mac_conversions=WMU_MAC_CONVERSIONS,
        mac_cycles=phases["reset"] + phases["multiply"] + phases["accumulate"] + design["adc"]["cycles_per_conversion"],
        conversion_cycles=design["adc"]["cycles_per_conversion"],
        shift_add_cycles_per_mac=shift_add["params"]["OUTS"] // shift_add["params"]["UNITS"],
        # the unit column charges and converts in every MAC, like a data column
        unit_column_mac_pJ=unit_columns * (column_mac_pJ + adc_pJ),
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


def _array_area(config_dir: Path, design: dict, node_nm: int) -> ArrayArea:
    """Area of one array (data columns and the unit column) and its circuits at node_nm
    (area.json, nvsim.json, macro_logic results; see area.json's _provenance)."""
    area = _json(config_dir / "area.json")
    adc = area["adc"]
    if adc["target_node_nm"] != node_nm:
        raise ValueError(f"area.json ADC is scaled to {adc['target_node_nm']} nm, but the target is {node_nm} nm")
    if adc["bits"] != design["adc"]["bits"]:
        raise ValueError(f"area.json ADC is {adc['bits']}-bit, design.json's ADC is {design['adc']['bits']}-bit")
    shift_add = _macro_logic_row(config_dir, "shift_add")
    columns = design["array"]["cols"] + _check_encoding(config_dir, design, shift_add)
    for part in ("adc", "column_frontend"):
        if area[part]["per_array"] != columns:
            raise ValueError(f"area.json {part}.per_array is {area[part]['per_array']}, the array has {columns} columns")
    # each bit above the source ADC's doubles its area (area.json _provenance)
    adc_um2 = adc["source_area_um2"] * 2 ** (adc["bits"] - adc["source_bits"]) / adc["deepscale_area_factor"]

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

    sram = _acc_sram(shift_add, node_nm)
    return ArrayArea(
        cells_um2=_json(config_dir / "results" / "nvsim.json")["array"]["array_area_um2"],
        adcs_um2=adc["per_array"] * adc_um2,
        column_frontends_um2=fe["per_array"] * frontend_um2,
        macro_ctrl_um2=_macro_logic_row(config_dir, "macro_ctrl")[f"area_um2_{node_nm}nm"],
        shift_add_um2=shift_add[f"area_um2_{node_nm}nm"] + sram["banks"] * sram["area_um2"],
        wmu_readout_um2=_macro_logic_row(config_dir, "wmu_readout")[f"area_um2_{node_nm}nm"],
    )


def load_config(config_dir: Path) -> SingleModuleTarget:
    design = _json(config_dir / "design.json")
    mac = _json(config_dir / "results" / "mac.json")
    org = _json(DEFAULT / "organization.json")
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
    rmu_mac_ns, wmu_mac_ns = _mac_latencies_ns(design)
    unit = MemoryUnit(rows=design["array"]["rows"], cols=design["array"]["cols"],
                      bits_per_cell=design["cell"]["bits_per_cell"])
    return SingleModuleTarget(
        name=f"1fefet1c/{config_dir.name}",
        architecture=design["architecture"],
        node_nm=design["process"]["node_nm"],
        geometry=Geometry(rmu=unit, wmu=unit),
        # as the default (default/target_loader.py): RMU read = the multiply phase; MAC
        # latencies from the clock (_mac_latencies_ns), read and write from cap_rw.py
        timing=Timing(rmu_mac_ns=rmu_mac_ns, wmu_mac_ns=wmu_mac_ns,
                      rmu_read_ns=design["clock"]["column_phase_cycles"]["multiply"] * t_clk_ns,
                      rmu_write_ns=math.inf,
                      wmu_read_ns=mac["cap_read"]["latency_ns"], wmu_write_ns=mac["cap_write"]["latency_ns"]),
        organization=Organization(
            arrays_per_pe=org["pe"]["arrays"],
            pes_per_tile=org["tile"]["pes"],
            tiles_per_router=org["chip"]["tiles_per_router"],
            router_area_um2=area_um2(router),
            router_flit_bits=router["flit_bits"],
            router_flit_hop_pJ=access_pJ(router) / router["ports"],
            pe_input_register_area_um2=area_um2(input_register),
            pe_input_register_access_bits=input_register["access_bits"],
            pe_input_register_access_pJ=access_pJ(input_register),
            tile_bus_area_um2=area_um2(bus),
            tile_bus_bits=bus["wires"],
            tile_bus_transfer_pJ=access_pJ(bus),
        ),
        array_area=_array_area(config_dir, design, design["process"]["node_nm"]),
        array_energy=_array_energy(config_dir, design, mac, design["process"]["node_nm"]),
    )
