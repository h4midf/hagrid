"""Target description of the default HyFlexPIM chip (Song et al., ISCA 2025), a
dual-module architecture: returns a DualModuleTarget (targets/target_loader.py).

  analog module   512 arrays of 64 x 128, SLC or 2-bit MLC, holding the weights; one input
                  bit per 100 ns cycle, an ADC and a shift-and-add per array
  digital module  256 arrays of 1024 x 1024 SLC computing with NOR stateful logic on data
                  written at run time (Q, K, V, scores), with its SFU

Sources (this folder):
  design.json             configuration, as printed in the paper (see its _provenance)
  components.json         values from cited sources the paper does not give (floatpim, the
                          ISAAC buses and router, scaled to HyFlexPIM's bandwidths)
  results/table2.json     area and power of every module component (../table2.py), checked
                          against Table 2 by ../report.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from targets.target_loader import (  # noqa: E402
    AnalogModule, ComponentCost, DigitalModule, DualModuleTarget, MemoryUnit,
)

sys.path.insert(0, str(HERE.parent))
from table2 import area_factor, energy_factor, sfu_factors  # noqa: E402  (Stillmaker and Baas)


def _scaled(isaac: dict, bandwidth_ratio: float, comp: dict) -> ComponentCost:
    """An ISAAC interconnect part widened to bandwidth_ratio x its bandwidth and scaled to
    65 nm with node_scaling (components.json "interconnect")."""
    scaling = comp["node_scaling"]
    power = isaac["power_mW"] * bandwidth_ratio * energy_factor(scaling)
    # ISAAC gives one power at full activity
    return ComponentCost(area_um2=isaac["area_mm2"] * 1e6 * bandwidth_ratio * area_factor(scaling),
                         power_mW=power, dynamic_mW=power, power_split="source_total")


def _components(table2: dict, module: str) -> dict[str, ComponentCost]:
    return {name: ComponentCost(area_um2=c["area_mm2"] * 1e6, power_mW=c["power_mW"],
                                dynamic_mW=c["dynamic_mW"], power_split=c["power_split"])
            for name, c in table2[module]["components"].items()}


def load() -> DualModuleTarget:
    design = json.loads((HERE / "design.json").read_text())
    comp = json.loads((HERE / "components.json").read_text())
    path = HERE / "results" / "table2.json"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run targets/hyflexpim/table2.py first")
    table2 = json.loads(path.read_text())
    if "digital_module" not in table2:
        raise ValueError(f"{path} has no SFU row: run the sfu/ characterization, then table2.py")
    adc_path = HERE / "results" / "adc.json"
    adc = json.loads(adc_path.read_text())
    if "energy_pJ_per_conversion_6bit" not in adc:
        raise ValueError(f"{adc_path} has no 6-bit mode: run spice/adc/characterize_adc.py")

    am, dm = design["analog_module"], design["digital_module"]
    slc_bits, mlc_bits = min(am["bits_per_cell"]), max(am["bits_per_cell"])
    analog = AnalogModule(
        arrays=am["arrays"],
        array=MemoryUnit(rows=am["array"]["rows"], cols=am["array"]["cols"], bits_per_cell=slc_bits),
        slc_bits=slc_bits,
        mlc_bits=mlc_bits,
        slc_rate_pct=design["mapping"]["slc_rate_pct"],
        adc_bits_slc=am["readout"]["adc_bits"]["slc"],
        adc_bits_mlc=am["readout"]["adc_bits"]["mlc"],
        adc_conversion_pJ_slc=adc["energy_pJ_per_conversion_6bit"]["total"],
        adc_conversion_pJ_mlc=adc["energy_pJ_per_conversion"]["total"],
        cycle_ns=design["clock"]["analog_cycle_ns"],
        # a row read is one input bit (the one-hot row) through the same crossbar + ADC cycle
        read_ns=design["clock"]["analog_cycle_ns"],
        components=_components(table2, "analog_module"),
    )
    digital = DigitalModule(
        arrays=dm["arrays"],
        array=MemoryUnit(rows=dm["array"]["rows"], cols=dm["array"]["cols"], bits_per_cell=dm["bits_per_cell"]),
        logic_op="NOR",
        logic_op_cycle_ns=comp["floatpim"]["nor_ns"],
        logic_op_columns=dm["nor_columns"],
        int8_mul_logic_ops=dm["int8_mul_nor_ops"],
        add_logic_ops_per_bit=dm["add_nor_ops_per_bit"],
        row_write_cycles=dm["row_processing_cycles"]["write"],
        row_read_cycles=dm["row_processing_cycles"]["read"],
        row_read_ns=dm["row_processing_cycles"]["read"] * comp["floatpim"]["nor_ns"],
        row_write_ns=dm["row_processing_cycles"]["write"] * comp["floatpim"]["nor_ns"],
        sfu_lanes=dm["sfu"]["inputs_per_cycle"],
        sfu_freq_MHz=design["clock"]["sfu_freq_MHz"],
        components=_components(table2, "digital_module"),
    )
    pu, chip = design["processing_unit"], design["chip"]
    ic = comp["interconnect"]
    if ic["isaac_node_nm"] != comp["node_scaling"]["from"]["node_nm"]:
        raise ValueError("the interconnect source node is not node_scaling's source node")
    clock = ic["isaac_clock_GHz"]
    inner_bw = chip["inner_unit_bus"]["bandwidth_GB_per_s"]
    global_bw = chip["global_bus"]["bandwidth_GB_per_s"]
    cmos = sfu_factors()
    return DualModuleTarget(
        name="hyflexpim/default",
        architecture=design["architecture"],
        node_nm=design["process"]["node_nm"],
        processing_units=chip["processing_units"],
        analog_modules_per_pu=pu["analog_modules"],
        digital_modules_per_pu=pu["digital_modules"],
        analog=analog,
        digital=digital,
        inner_bus_GB_per_s=inner_bw,
        global_bus_GB_per_s=global_bw,
        inner_bus=_scaled(ic["isaac_tile_bus"], inner_bw / (ic["isaac_tile_bus"]["wires"] * clock / 8), comp),
        # ISAAC's router at full activity: one flit through each port per cycle
        global_router=_scaled(ic["isaac_router"], global_bw / (ic["isaac_router"]["flit_bits"]
                                                              * ic["isaac_router"]["ports"] * clock / 8), comp),
        pus_per_router=ic["pus_per_router"],
        cmos_area_factor=cmos["area"],
        cmos_energy_factor=cmos["energy"],
        cmos_power_factor=cmos["power"],
    )
