"""The uniform description of a hardware target, and how it is loaded.

Every architecture is described with the same two memory units:

  RMU  resident memory unit: holds the stationary operands (weights)
       (1F1C: the FeFETs)
  WMU  working memory unit: holds temporary data (1F1C: the capacitors)

and the same six operations, each with a latency in ns:

  rmu_mac    one MAC from the RMU over the whole array, one input bit
  wmu_mac    one MAC from the WMU over the whole array, one input bit
  rmu_read   read one row of the RMU
  rmu_write  write one row of the RMU
  wmu_read   read one row of the WMU
  wmu_write  write one row of the WMU

math.inf = the architecture does not support that operation.

Each target folder (e.g. targets/1fefet1c/default) has its own target_loader.py
with a load() that fills these classes from that target's files and returns a
SingleModuleTarget. load_target("1fefet1c/default") runs it.

Dual-module architectures (HyFlexPIM) are described by DualModuleTarget instead: the
weights and the dynamic data live in two different modules with their own arrays,
counts, timing and costs (an analog module computing the weight MVMs, a digital module
computing on data written at run time, with its SFU), grouped into processing units
joined by buses. Their load() returns a DualModuleTarget.

The RRAM/SRAM ensemble is described by EnsembleTarget: an analog RRAM macro and a digital
SRAM macro with its PEs. Its load() returns an EnsembleTarget; a mapping takes it bound
to one app (ensemble_for_app), for that app's SRAM correction share.
"""
import importlib.util
from dataclasses import dataclass
from pathlib import Path

TARGETS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MemoryUnit:
    rows: int
    cols: int
    bits_per_cell: int


@dataclass(frozen=True)
class Geometry:
    rmu: MemoryUnit   # resident memory unit
    wmu: MemoryUnit   # working memory unit


@dataclass(frozen=True)
class Timing:
    rmu_mac_ns: float
    wmu_mac_ns: float
    rmu_read_ns: float
    rmu_write_ns: float
    wmu_read_ns: float
    wmu_write_ns: float


@dataclass(frozen=True)
class Organization:
    """How the arrays are grouped on the chip (ISAAC-style): arrays per PE, PEs per tile,
    tiles on a mesh with one router per tiles_per_router tiles; each PE has an input
    register, each tile a bus to its PEs. Areas and energies are at the target's node;
    energies are per access: a router flit per hop, a bus transfer, an input-register
    access, each of the given width in bits."""
    arrays_per_pe: int
    pes_per_tile: int
    tiles_per_router: int
    router_area_um2: float
    router_flit_bits: int
    router_flit_hop_pJ: float
    pe_input_register_area_um2: float
    pe_input_register_access_bits: int
    pe_input_register_access_pJ: float
    tile_bus_area_um2: float
    tile_bus_bits: int
    tile_bus_transfer_pJ: float


@dataclass(frozen=True)
class ArrayArea:
    """Area of one array and the circuits that come with it, in um^2 at the target's
    node: the cell array, its column ADCs and their front ends (buffer and
    sample-and-hold), its input and phase control, shift-and-add and WMU readout."""
    cells_um2: float
    adcs_um2: float
    column_frontends_um2: float
    macro_ctrl_um2: float
    shift_add_um2: float
    wmu_readout_um2: float


@dataclass(frozen=True)
class ArrayEnergy:
    """Energy of one array's operations (pJ) and its static power (uW), at the target's
    node, and how to scale a synthesized CMOS circuit (synthesis node) to it.

      adc_conversion_pJ        one column ADC conversion (incl. its front end)
      column_mac_pJ            one column's load and bitline charging in an RMU MAC, per
                               input bit (the ADC conversion is counted separately)
      cell_load_pJ             one cell's FeFET -> capacitor load (an RMU read)
      wmu_write_cell_pJ        one cell's capacitor write, averaged over its levels
      wmu_read_conversions     ADC conversions per cell of a WMU read
      wmu_mac_conversions      ADC conversions per column of a WMU MAC
      mac_cycles               clock cycles of one MAC (phases + conversion)
      conversion_cycles        clock cycles of one ADC conversion
      shift_add_cycles_per_mac shift-and-add cycles per MAC (its time-shared slots)
      unit_column_mac_pJ       one array's unit column in one MAC (ISAAC's weight-flip
                               encoding: its charging and ADC conversion); 0 without it
      *_cycle_pJ               one clock cycle of that macro-logic block
      array_leakage_uW         one array's cells and macro logic
      cmos_energy_factor       a synthesized circuit's energy / this = energy here
      cmos_power_factor        a synthesized circuit's leakage / this = leakage here
    """
    adc_conversion_pJ: float
    column_mac_pJ: float
    cell_load_pJ: float
    wmu_write_cell_pJ: float
    wmu_read_conversions: int
    wmu_mac_conversions: int
    mac_cycles: int
    conversion_cycles: int
    shift_add_cycles_per_mac: int
    unit_column_mac_pJ: float
    macro_ctrl_cycle_pJ: float
    shift_add_cycle_pJ: float
    wmu_readout_cycle_pJ: float
    array_leakage_uW: float
    cmos_energy_factor: float
    cmos_power_factor: float


@dataclass(frozen=True)
class SingleModuleTarget:
    """A target with one module type: the RMU and WMU are the same arrays (1F1C)."""
    name: str
    architecture: str     # hardware type; target-specific compiler paths branch on it
    node_nm: int          # process node; the areas and powers here are at this node
    geometry: Geometry
    timing: Timing
    organization: Organization
    array_area: ArrayArea
    array_energy: ArrayEnergy


@dataclass(frozen=True)
class ComponentCost:
    """Area and power of one component of a module (all its units), at the target's node.
    dynamic_mW: its dynamic power where we model the split ("synthesis": our synthesized
    circuit; "spice_per_conversion": our ADC's energy per conversion x rate), else the
    cited source's one total, dynamic and leakage together ("source_total")."""
    area_um2: float
    power_mW: float
    dynamic_mW: float
    power_split: str


@dataclass(frozen=True)
class AnalogModule:
    """A module of analog arrays holding the weights. One input bit is applied to every
    array per cycle (bit-serial inputs); each array's columns are converted by its ADC
    and combined by its shift-and-add. Cells store slc_bits or mlc_bits per cell."""
    arrays: int
    array: MemoryUnit             # bits_per_cell = slc_bits
    slc_bits: int
    mlc_bits: int
    slc_rate_pct: float           # share of each weight matrix's output columns stored in SLC
    adc_bits_slc: int
    adc_bits_mlc: int
    adc_conversion_pJ_slc: float  # one conversion of an SLC column (the ADC's 6-bit mode)
    adc_conversion_pJ_mlc: float  # one conversion of an MLC column (its 7-bit mode)
    cycle_ns: float               # one input bit over all arrays (array read + ADC scan)
    read_ns: float                # one row read out (a LUT lookup: a one-hot input)
    components: dict[str, ComponentCost]


@dataclass(frozen=True)
class DigitalModule:
    """A module of digital arrays computing on data written at run time, with stateful
    row-parallel logic operations (one operation per cycle on every row of an array),
    and its special function unit (SFU)."""
    arrays: int
    array: MemoryUnit
    logic_op: str                 # e.g. "NOR"
    logic_op_cycle_ns: float      # one logic operation, all rows of an array
    logic_op_columns: int         # columns one operation uses (operands + output)
    int8_mul_logic_ops: int       # logic operations of one 8-bit multiply
    add_logic_ops_per_bit: int    # logic operations of an addition, per operand bit
    row_write_cycles: int
    row_read_cycles: int
    row_read_ns: float            # row_read_cycles logic-op cycles
    row_write_ns: float           # row_write_cycles logic-op cycles
    sfu_lanes: int
    sfu_freq_MHz: float
    components: dict[str, ComponentCost]   # includes "sfu"


@dataclass(frozen=True)
class DualModuleTarget:
    """A chip of processing units (PUs), each with analog_modules_per_pu analog modules and
    digital_modules_per_pu digital modules on the PU's inner bus; PUs talk over the
    global bus."""
    name: str
    architecture: str
    node_nm: int
    processing_units: int
    analog_modules_per_pu: int
    digital_modules_per_pu: int
    analog: AnalogModule
    digital: DigitalModule
    inner_bus_GB_per_s: float
    global_bus_GB_per_s: float
    inner_bus: ComponentCost      # one PU's inner bus, at its bandwidth, at node_nm
    global_router: ComponentCost  # one router of the global network, at its bandwidth
    pus_per_router: int
    cmos_area_factor: float       # a synthesized (Nangate45) circuit's area x this = area here
    cmos_energy_factor: float     # ... its energy x this = energy here
    cmos_power_factor: float      # ... its leakage x this = leakage here


@dataclass(frozen=True)
class RramMacro:
    """An analog RRAM crossbar holding weights. One MAC per input bit (bit-serial inputs)
    takes cycles_per_mac cycles: its columns share the ADCs through a column mux."""
    array: MemoryUnit
    cycles_per_mac: int
    cycle_ns: float


@dataclass(frozen=True)
class SramPeMacro:
    """A digital SRAM array feeding an array of PEs, each a multiplier and adder computing
    one output (output stationary, bit-serial inputs). Each read delivers read_bits_per_pe
    to every PE; with load_overlaps_compute, a PE loads its next value while computing.
    A PE holds weights_per_pe weights of weight_bits (its share of the array); each takes
    cycles_per_weight cycles per input bit."""
    array: MemoryUnit
    pe_count: int
    read_word_bits: int
    write_word_bits: int
    read_bits_per_pe: int
    load_overlaps_compute: bool
    weight_bits: int
    weights_per_pe: int
    cycles_per_weight: int
    cycle_ns: float


@dataclass(frozen=True)
class EnsembleCosts:
    """Area of each macro and the energy of its events, at the target's node (dynamic
    only). An RRAM MAC (one input bit over the macro, cycles_per_mac cycles) costs
    mac_macro_pJ per macro (word lines, decoder, column-mux select, shift-and-add) plus
    mac_column_pJ per column it converts (ADC, column supply); a read (one row on, a LUT
    lookup) likewise. An SRAM macro costs logic_cycle_pJ per cycle of a pass (all PEs
    busy), read_pJ per read word and write_pJ per written word."""
    rram_macro_area_um2: float
    sram_macro_area_um2: float
    rram_mac_macro_pJ: float
    rram_mac_column_pJ: float
    rram_read_macro_pJ: float
    rram_read_column_pJ: float
    sram_logic_cycle_pJ: float
    sram_read_pJ: float
    sram_write_pJ: float
    sram_accumulator_bits: int


@dataclass(frozen=True)
class EnsembleTarget:
    """RRAM macros and SRAM macros working side by side, placed ISAAC-style (organization:
    a PE's arrays_per_pe slots each hold one macro of either kind).

    The SRAM macros have two roles:
      correction  stationary: a layer's SRAM correction weights, sram_correction_weight_pct
                  of its weights, on up to sram_macros_per_rram_macro macros per
                  RRAM macro holding the layer, computing with its RRAM macros on the same
                  input bits; the shifter adds both outputs
      compute     volatile: the dynamic matmuls' operands written at run time"""
    name: str
    architecture: str
    node_nm: int
    rram: RramMacro
    sram: SramPeMacro
    sram_correction_weight_pct: dict[str, float]   # per app (kernel name)
    sram_macros_per_rram_macro: float         # the most correction macros per RRAM macro
    organization: Organization
    costs: EnsembleCosts
    cmos_area_factor: float       # a synthesized (Nangate45) circuit's area x this = area here
    cmos_energy_factor: float     # ... its energy x this = energy here
    cmos_power_factor: float      # ... its leakage x this = leakage here


@dataclass(frozen=True)
class EnsembleAppTarget(EnsembleTarget):
    """An EnsembleTarget bound to one app (ensemble_for_app): the share of its weights the
    SRAM correction keeps and computes."""
    app: str
    sram_share_pct: float


def ensemble_for_app(target: EnsembleTarget, app: str) -> EnsembleAppTarget:
    """target bound to app: its sram_correction_weight_pct entry. Raises if it has none."""
    if app not in target.sram_correction_weight_pct:
        raise KeyError(f"{target.name}: no sram_correction_weight_pct for app {app!r} "
                       f"(has {sorted(target.sram_correction_weight_pct)})")
    fields = {f: getattr(target, f) for f in EnsembleTarget.__dataclass_fields__}
    return EnsembleAppTarget(**fields, app=app, sram_share_pct=target.sram_correction_weight_pct[app])


def load_target(target_name: str) -> SingleModuleTarget | DualModuleTarget | EnsembleTarget:
    path = TARGETS_DIR / target_name / "target_loader.py"
    spec = importlib.util.spec_from_file_location(f"target_{target_name.replace('/', '_').replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load()
