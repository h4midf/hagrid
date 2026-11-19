"""Target description of the default RRAM/SRAM ensemble chip (Krishnan et al., TCAD 2022):
returns an EnsembleTarget (targets/target_loader.py).

  RRAM macro  64 x 64 1T1R crossbar holding the weights; 8 cycles per MAC per input bit
  SRAM macro  32 x 64 SRAM feeding 16 PEs

The paper's values and our own assumptions (design.json "model_assumptions", which
include running dynamic matmuls on the SRAM macro and the share of every app's weights the
SRAM correction keeps) are both read from design.json.

  organization.json   PEs, tiles, routers (ISAAC organization), scaled to 65 nm with
                      components.json node_scaling (Stillmaker and Baas)
  components.json     NanGate45 -> 65 nm factors for synthesized circuits
                      (../periphery/scaling.py)
  results/            the macros' area and energy (EnsembleCosts), checked against the
                      paper by ../report.py:
    rram_macro_area.json         RRAM macro area (../rram_macro_area.py)
    rram_macro_power.json        RRAM macro power by rows on (../rram_macro_power.py)
    sram_over_rram.json          SRAM macro area, logic + array (../sram_over_rram.py)
    sram_array_spice.json        SRAM read and write energy (../spice/sram)
  ../periphery/results/sram_logic_power_real.json   SRAM macro logic power on real data
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from targets.target_loader import (  # noqa: E402
    EnsembleCosts, EnsembleTarget, MemoryUnit, Organization, RramMacro, SramPeMacro,
)

sys.path.insert(0, str(HERE.parent / "periphery"))
from scaling import factors  # noqa: E402  (NanGate45 -> 65 nm, Stillmaker and Baas)


def _isaac_factors(org: dict, comp: dict) -> tuple[float, float]:
    """(area, energy) factors from ISAAC's node to the chip's (organization.json scaling,
    components.json node_scaling): Stillmaker and Baas Sec. 4.1 and Eq. (6)."""
    src, dst, table = org["scaling"]["source"], comp["target"], comp["node_scaling"]
    g_src, g_dst = table["area_geometry"][str(src["node_nm"])], table["area_geometry"][str(dst["node_nm"])]
    area = ((g_dst[0] / g_src[0]) ** 2 * (g_dst[1] / g_src[1]) ** 2 * (g_dst[2] / g_src[2])) ** (1 / 3)

    def ef(key, v):
        a2, a1, a0 = table["energy_coefficients"][key]
        return a2 * v * v + a1 * v + a0
    energy = (ef(f"{dst['node_nm']}_{dst['type']}", dst["vdd_V"])
              / ef(f"{src['node_nm']}_{src['type']}", src["vdd_V"]))
    return area, energy


def _organization(org: dict, comp: dict) -> Organization:
    area_f, energy_f = _isaac_factors(org, comp)
    clock = org["scaling"]["source_clock_GHz"]

    def area_um2(part):
        return part["source_area_mm2"] * 1e6 * area_f

    def access_pJ(part):
        """One access at full activity: scaled power / ISAAC's clock (mW / GHz = pJ)."""
        return part["source_power_mW"] * energy_f / clock

    router, input_register, bus = org["chip"]["router"], org["pe"]["input_register"], org["tile"]["bus"]
    return Organization(
        arrays_per_pe=org["pe"]["macros"],
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
    )


def _result(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path}: run its script first (see this loader's docstring)")
    return json.loads(path.read_text())


def _costs(design: dict, rram: RramMacro, sram: SramPeMacro) -> EnsembleCosts:
    """EnsembleCosts from the macro results. Power (mW) x time (ns) = energy (pJ); leakage
    (the programming muxes') is left out."""
    results = HERE / "results"
    power = _result(results / "rram_macro_power.json")
    mac_ns = rram.cycles_per_mac * rram.cycle_ns
    rows_on = str(design["model_assumptions"]["rram_mac_rows_on"])
    if rows_on not in power["by_rows_on"] or "1" not in power["by_rows_on"]:
        raise KeyError(f"rram_macro_power.json has no {rows_on}- or 1-row point")

    def split(p: dict, shift_add_mW: float) -> tuple[float, float]:
        macro_mW = p["wl_mW"] + p["decoder_mW"] + p["mux_select_mW"] + shift_add_mW
        column_mW = p["adc_mW"] + p["column_supply_mW"]
        return macro_mW * mac_ns, column_mW * mac_ns / rram.array.cols

    mac_macro, mac_column = split(power["by_rows_on"][rows_on], power["shift_add_mW_not_in_share"])
    # a one-hot read needs no shift-and-add across input bits
    read_macro, read_column = split(power["by_rows_on"]["1"], 0.0)
    array = _result(results / "sram_array_spice.json")
    logic = _result(HERE.parent / "periphery" / "results" / "sram_logic_power_real.json")
    return EnsembleCosts(
        rram_macro_area_um2=_result(results / "rram_macro_area.json")["total_um2"],
        sram_macro_area_um2=_result(results / "sram_over_rram.json")["sram_macro"]["area_um2"],
        rram_mac_macro_pJ=mac_macro, rram_mac_column_pJ=mac_column,
        rram_read_macro_pJ=read_macro, rram_read_column_pJ=read_column,
        sram_logic_cycle_pJ=logic["dynamic_mW"] * sram.cycle_ns,
        sram_read_pJ=array["read_pJ"], sram_write_pJ=array["write_pJ"],
        sram_accumulator_bits=design["model_assumptions"]["sram_logic"]["accumulator_bits"],
    )


def load() -> EnsembleTarget:
    design = json.loads((HERE / "design.json").read_text())
    assumed = design["model_assumptions"]
    cycle_ns = 1000.0 / design["clock"]["freq_MHz"]

    rm = design["rram_macro"]
    bits_per_cell = assumed["rram_bits_per_cell"]
    if bits_per_cell > rm["cell"]["max_bits_per_cell"]:
        raise ValueError(f"rram_bits_per_cell {bits_per_cell} is over the cell's {rm['cell']['max_bits_per_cell']}")
    rram = RramMacro(
        array=MemoryUnit(rows=rm["array"]["rows"], cols=rm["array"]["cols"], bits_per_cell=bits_per_cell),
        cycles_per_mac=rm["readout"]["cycles_per_mac"],
        cycle_ns=cycle_ns,
    )

    sm, sa = design["sram_macro"], assumed["sram_macro"]
    if sa["read_bits_per_pe"] * sm["pe_count"] != sm["read_word_bits"]:
        raise ValueError(f"{sa['read_bits_per_pe']} bits to each of {sm['pe_count']} PEs "
                         f"is not the {sm['read_word_bits']}-bit read word")
    weight_bits = assumed["sram_logic"]["weight_bits"]
    pe_bits = sm["array"]["rows"] * sm["array"]["cols"] // sm["pe_count"]
    # a weight's reads, then (no overlap) one cycle adding it for the input bit (periphery/rtl)
    reads = -(-weight_bits // sa["read_bits_per_pe"])
    if sa["load_overlaps_compute"]:
        raise NotImplementedError("load_overlaps_compute: the validated macro logic does not overlap them")
    sram = SramPeMacro(
        array=MemoryUnit(rows=sm["array"]["rows"], cols=sm["array"]["cols"], bits_per_cell=1),
        pe_count=sm["pe_count"],
        read_word_bits=sm["read_word_bits"],
        write_word_bits=sa["write_word_bits"],
        read_bits_per_pe=sa["read_bits_per_pe"],
        load_overlaps_compute=sa["load_overlaps_compute"],
        weight_bits=weight_bits,
        weights_per_pe=pe_bits // weight_bits,
        cycles_per_weight=reads + 1,
        cycle_ns=cycle_ns,
    )

    comp = json.loads((HERE / "components.json").read_text())
    org = json.loads((HERE / "organization.json").read_text())
    if comp["target"]["node_nm"] != design["process"]["node_nm"]:
        raise ValueError(f"components.json scales to {comp['target']['node_nm']} nm, "
                         f"but the target is {design['process']['node_nm']} nm")
    share = assumed["sram_correction_weight_pct"]
    for app, pct in share.items():
        if not 0 < pct <= 100:
            raise ValueError(f"sram_correction_weight_pct[{app!r}] = {pct} is not in (0, 100]")
    cmos = factors()

    return EnsembleTarget(
        name="rram-sram-ensemble/default",
        architecture=design["architecture"],
        node_nm=design["process"]["node_nm"],
        rram=rram,
        sram=sram,
        sram_correction_weight_pct=dict(share),
        sram_macros_per_rram_macro=assumed["sram_macros_per_rram_macro"],
        organization=_organization(org, comp),
        costs=_costs(design, rram, sram),
        cmos_area_factor=cmos["area"],
        cmos_energy_factor=cmos["energy"],
        cmos_power_factor=cmos["power"],
    )
