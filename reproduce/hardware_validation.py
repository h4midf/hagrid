"""The paper's hardware validation table (figures/evaluation/hardware_validation_wide.tex):
each target's model against the numbers its source paper reports. One script per target
runs its part (hardware_validation_1f1c.py, _hyflexpim.py, _rse.py): compiles the
simulators, runs every simulation the target's values come from, and writes
reproduce/results/hardware_validation_<target>.csv, one row per value, checked against
what the paper prints (exit status 1 if any row differs).

A step whose output already exists is skipped; --force deletes the target's results and
reruns every step, compilation included.

  (a) 1F1C, Laleni et al. (JXCDC 2024)
      NVSim (compiled from targets/1fefet1c/NVSim)   bitline geometry and capacitance
      spice/column/column_sim.py (ngspice)           column charging, kappa, VBL
      spice/adc/run_enob.py (ngspice)                SAR ADC ENOB and conversion energy
      compute_efficiency.py                          macro TOPS/W
  (b) HyFlexPIM, Song et al. (ISCA 2025), Table 2
      spice/adc/characterize_adc.py (ngspice)        the 7-bit SAR ADC
      periphery/characterize_shift_add.py (yosys)    the S&A unit
      sfu/characterize_sfu.py, power_sfu.py,         the SFU's area, energy per operation
        sfu_average_power.py (yosys)                   and average power
      table2.py                                      Table 2 rebuilt
  (c) RSE, Krishnan et al. (TCAD 2022), Sec. V-F
      CACTI (compiled from targets/hyflexpim/CACTI)  the SRAM macro's arrays
      spice/logic/inverter.py, spice/readout,        logic devices, flash readout, WL driver,
        spice/wl, spice/sram (ngspice)                 SRAM array
      periphery/characterize_rram_decoder.py,        RRAM decoder and shift-add, SRAM macro
        characterize_rram_shift_add.py,                logic and arrays, its power on real
        characterize_sram_macro.py,                    weights (the int8 ResNet-50 from
        power_sram_logic_real.py (yosys)               build_resnet50_int8.py)
      rram_macro_area.py, rram_macro_power.py,       the macros and SRAM over RRAM
        sram_over_rram.py

Usage (from the repo root):
  python3 reproduce/hardware_validation_1f1c.py [--force]
  python3 reproduce/hardware_validation_hyflexpim.py [--force]
  python3 reproduce/hardware_validation_rse.py [--force]
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoints import checkpoint_steps  # noqa: E402
from common import REPO, Step, check_rows, file_exists, json_has, run_steps  # noqa: E402

PY = sys.executable

# Yosys 0.68 and ngspice-47 from setup.sh, first on PATH for every step
TOOLS_BIN = REPO / ".tools" / "bin"
for _tool in ("yosys", "yosys-abc", "ngspice"):
    if not (TOOLS_BIN / _tool).exists():
        raise SystemExit(f"missing {TOOLS_BIN / _tool}: run ./setup.sh first")
os.environ["PATH"] = f"{TOOLS_BIN}{os.pathsep}{os.environ['PATH']}"

# ---- (a) 1F1C

F1C = REPO / "targets" / "1fefet1c"
F1C_RESULTS = F1C / "default" / "results"
F1C_ENV = {"F1C_CONFIG": "default"}
# Laleni et al., Fig. 6: the FeFET drain plateau at Vth 0.3 V (not in design.json)
LALENI_FIG6_PLATEAU_V = 0.70


def f1c_steps(force: bool) -> list[Step]:
    nvsim = F1C / "NVSim" / "nvsim"
    # C++98: NVSim's enum value `data` collides with std::data under C++17 (GCC 11's default)
    build = f"make -C {F1C / 'NVSim'} CXXFLAGS='-Wall -std=c++98'"
    return [
        Step("1F1C: compile NVSim", ["sh", "-c", f"make -C {F1C / 'NVSim'} clean && {build}" if force else build],
             REPO, file_exists(nvsim), {}, []),
        Step("1F1C: NVSim array", [PY, str(F1C / "run_nvsim.py")], REPO,
             json_has(F1C_RESULTS / "nvsim.json", "array"), F1C_ENV,
             [F1C_RESULTS / "nvsim.json", F1C_RESULTS / "nvsim_output.txt"]),
        Step("1F1C: column SPICE", [PY, str(F1C / "spice" / "column" / "column_sim.py")], REPO,
             json_has(F1C_RESULTS / "column.json", "fig6", "kappa", "bl_charging", "fig10b"), F1C_ENV,
             [F1C_RESULTS / "column.json"]),
        # adc.json also holds other ADC experiments (noise, core_dc, ...): only run_enob.py's entry
        Step("1F1C: ADC ENOB SPICE", [PY, str(F1C / "spice" / "adc" / "run_enob.py")], REPO,
             json_has(F1C_RESULTS / "adc.json", "enob"), F1C_ENV, [(F1C_RESULTS / "adc.json", "enob")]),
        Step("1F1C: efficiency", [PY, str(F1C / "compute_efficiency.py")], REPO,
             json_has(F1C_RESULTS / "efficiency.json", "macro"), F1C_ENV, [F1C_RESULTS / "efficiency.json"]),
    ]


def f1c_rows() -> list[dict]:
    """The table's (a) rows, computed as targets/1fefet1c/report.py computes them."""
    design = json.loads((F1C / "default" / "design.json").read_text())
    laleni = design["reported_results"]
    col = json.loads((F1C_RESULTS / "column.json").read_text())
    adc = json.loads((F1C_RESULTS / "adc.json").read_text())["enob"]
    eff = json.loads((F1C_RESULTS / "efficiency.json").read_text())["macro"]
    f_conv = design["clock"]["freq_MHz"] * 1e6 / design["adc"]["cycles_per_conversion"]
    v_cell = col["bl_charging"]["cell_voltage_V"]
    rows = [
        ("FeFET charging plateau (V)", max(col["fig6"]["drain_V_by_vs"].values()), LALENI_FIG6_PLATEAU_V,
         2, "0.71", "0.70"),
        ("Charge-sharing ratio kappa", col["kappa"]["kappa"], laleni["kappa"], 2, "0.73", "0.69"),
        ("Full-scale BL voltage (V)", col["kappa"]["vbl_full_scale_V"], design["adc"]["vrefp_V"],
         2, "0.48", "0.50"),
        ("BL charging power/cell (nW)", design["cell"]["cmom_fF"] * 1e-15 * v_cell ** 2 * f_conv * 1e9,
         laleni["bl_charging_power_nW"], 2, "0.77", "0.86"),
        ("Maximum BL voltage (V)", max(v["mean"] for v in col["fig10b"]["vbl_by_mac_V"].values()),
         col["fig10b"]["laleni_eq3_max_V"], 2, "0.74", "0.90"),
        ("ADC conversion energy (pJ)", adc["energy_per_conversion_pJ"],
         laleni["adc_walden_fom_fJ"] * 2 ** laleni["adc_enob_bits"] / 1000, 1, "2.2", "2.2"),
        ("ADC ENOB (bits)", adc["enob_bits"], laleni["adc_enob_bits"], 1, "6.1", "6.4"),
        ("Efficiency (TOPS/W)", eff["tops_per_w"], laleni["efficiency_TOPS_per_W"], 1, "112.2", "111.6"),
    ]
    return _rows("(a) 1F1C", rows)


def _rows(table: str, rows: list[tuple]) -> list[dict]:
    """rows: (quantity, model, reported, decimals, paper_model, paper_reported), the model
    and reported values printed at the same precision; or (..., model decimals, reported
    decimals, ...) with one more field where the paper prints them differently."""
    out = []
    for row in rows:
        if len(row) == 6:
            q, m, r, d, pm, pr = row
            dm = dr = d
        else:
            q, m, r, dm, dr, pm, pr = row
        out.append({"table": table, "quantity": q, "model": m, "reported": r, "model_decimals": dm,
                    "reported_decimals": dr, "paper_model": pm, "paper_reported": pr})
    return out


# ---- (b) HyFlexPIM

HFP = REPO / "targets" / "hyflexpim"


def _script(name: str, script: Path, *outputs: Path) -> Step:
    """A Python script that writes `outputs` whole: done when they all exist, and --force
    deletes them first."""
    return Step(name, [PY, str(script)], REPO, lambda: all(p.exists() for p in outputs), {}, list(outputs))


def hfp_steps() -> list[Step]:
    return [
        _script("HFP: ADC SPICE", HFP / "spice" / "adc" / "characterize_adc.py",
                HFP / "default" / "results" / "adc.json"),
        _script("HFP: S&A synthesis", HFP / "periphery" / "characterize_shift_add.py",
                HFP / "periphery" / "results" / "shift_add.json"),
        _script("HFP: SFU synthesis", HFP / "sfu" / "characterize_sfu.py", HFP / "sfu" / "results" / "sfu.json"),
        _script("HFP: SFU energy per operation", HFP / "sfu" / "power_sfu.py",
                HFP / "sfu" / "results" / "sfu_power.json"),
        _script("HFP: SFU average power", HFP / "sfu" / "sfu_average_power.py",
                HFP / "sfu" / "results" / "sfu_average_power.json"),
        _script("HFP: Table 2", HFP / "table2.py", HFP / "default" / "results" / "table2.json"),
    ]


def hfp_rows() -> list[dict]:
    """The table's (b) rows: the paper prints each as area (mm^2) / power (mW)."""
    t2 = json.loads((HFP / "default" / "results" / "table2.json").read_text())
    rep = json.loads((HFP / "default" / "design.json").read_text())["reported_results"]
    ra, rd = rep["table2_analog_module"], rep["table2_digital_module"]
    an, dg, dg_no_sfu = t2["analog_module"], t2["digital_module"], t2["digital_module_without_sfu"]
    rd_no_sfu = {k: rd["module_sum"][k] - rd["components"]["sfu"][k] for k in ("area_mm2", "power_mW")}
    # (row, ours, Table 2's, area decimals, the paper's model and reported strings "area / power")
    pairs = [
        ("Analog RRAM array", an["components"]["rram_array"], ra["components"]["rram_array"], 3,
         "0.045 / 60.78", "0.048 / 60.78"),
        ("Analog ADC", an["components"]["adc"], ra["components"]["adc"], 2, "0.28 / 511.00", "0.30 / 512.00"),
        ("Analog S&A", an["components"]["s_and_a"], ra["components"]["s_and_a"], 2,
         "0.10 / 56.27", "0.10 / 59.54"),
        ("Analog total", an["module_sum"], ra["module_sum"], 2, "0.45 / 926.43", "0.47 / 930.69"),
        ("Digital S&A", dg["components"]["s_and_a"], rd["components"]["s_and_a"], 2,
         "0.20 / 105.79", "0.21 / 119.08"),
        ("Digital SFU", dg["components"]["sfu"], rd["components"]["sfu"], 2, "5.33 / 127.20", "4.79 / 138.89"),
        ("Digital total", dg["module_sum"], rd["module_sum"], 2, "8.55 / 6507.10", "8.01 / 6532.05"),
        ("Digital total excl. SFU", dg_no_sfu["module_sum"], rd_no_sfu, 2, "3.21 / 6379.90", "3.22 / 6393.16"),
    ]
    rows = []
    for name, ours, paper, area_dec, pm, pr in pairs:
        (pm_area, pm_power), (pr_area, pr_power) = pm.split(" / "), pr.split(" / ")
        rows.append((f"{name} area (mm2)", ours["area_mm2"], paper["area_mm2"], area_dec, pm_area, pr_area))
        rows.append((f"{name} power (mW)", ours["power_mW"], paper["power_mW"], 2, pm_power, pr_power))
    return _rows("(b) HFP", rows)


# ---- (c) RSE

RSE = REPO / "targets" / "rram-sram-ensemble"
RSE_RESULTS = RSE / "default" / "results"
RSE_PERIPHERY = RSE / "periphery" / "results"
CACTI = REPO / "targets" / "hyflexpim" / "CACTI"
# targets/hyflexpim/periphery/cacti_run.py's build
CACTI_BUILD = 'mkdir -p obj_opt && make -f cacti.mk TAG=opt CXX="clang++ -std=c++98" CC=clang OPT="-O2 -DNTHREADS=8 -w"'


def rse_steps(force: bool) -> list[Step]:
    build = f"rm -rf obj_opt cacti && {CACTI_BUILD}" if force else CACTI_BUILD
    return [
        Step("RSE: compile CACTI", ["sh", "-c", build], CACTI, file_exists(CACTI / "cacti"), {}, []),
        _script("RSE: logic devices SPICE", RSE / "spice" / "logic" / "inverter.py", RSE_RESULTS / "logic.json"),
        _script("RSE: readout SPICE", RSE / "spice" / "readout" / "characterize_readout.py",
                RSE_RESULTS / "readout.json"),
        _script("RSE: WL driver SPICE", RSE / "spice" / "wl" / "characterize_wl.py", RSE_RESULTS / "wl.json"),
        _script("RSE: SRAM array SPICE", RSE / "spice" / "sram" / "characterize_sram_array.py",
                RSE_RESULTS / "sram_array_spice.json"),
        _script("RSE: RRAM decoder synthesis", RSE / "periphery" / "characterize_rram_decoder.py",
                RSE_PERIPHERY / "rram_decoder.json"),
        _script("RSE: RRAM shift-add synthesis", RSE / "periphery" / "characterize_rram_shift_add.py",
                RSE_PERIPHERY / "rram_shift_add.json"),
        _script("RSE: SRAM macro synthesis", RSE / "periphery" / "characterize_sram_macro.py",
                RSE_PERIPHERY / "sram_logic.json", RSE_PERIPHERY / "sram_array.json"),
        # the int8 ResNet-50 whose activations drive the power run
        *checkpoint_steps(["resnet50"]),
        _script("RSE: SRAM logic power", RSE / "periphery" / "power_sram_logic_real.py",
                RSE_PERIPHERY / "sram_logic_power_real.json"),
        _script("RSE: RRAM macro area", RSE / "rram_macro_area.py", RSE_RESULTS / "rram_macro_area.json"),
        _script("RSE: RRAM macro power", RSE / "rram_macro_power.py", RSE_RESULTS / "rram_macro_power.json"),
        _script("RSE: SRAM over RRAM", RSE / "sram_over_rram.py", RSE_RESULTS / "sram_over_rram.json"),
    ]


def _rse_rows_on() -> str:
    """targets/rram-sram-ensemble/report.py's ROWS_ON (rows on per MAC for the power rows)."""
    spec = importlib.util.spec_from_file_location("rse_report", RSE / "report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ROWS_ON


def rse_rows() -> list[dict]:
    """The table's (c) rows: SRAM over RRAM area and power (%) per network, and the RRAM
    macro's power shares (%), at report.py's rows on."""
    rows_on = _rse_rows_on()
    sor = json.loads((RSE_RESULTS / "sram_over_rram.json").read_text())["networks"]
    power = json.loads((RSE_RESULTS / "rram_macro_power.json").read_text())
    p, share = power["by_rows_on"][rows_on], power["paper_share_pct"]
    # (network, area decimals, the paper's model / reported strings "area / power")
    nets = [("ResNet-20 CIFAR-10", 1, "2.6 / 0.5", "3.7 / 0.5"),
            ("VGG-16 CIFAR-10", 1, "2.6 / 0.5", "2.3 / 0.3"),
            ("MobileNet-v2 ImageNet", 0, "16 / 3.0", "20 / 2.6")]
    rows = []
    for net, area_dec, pm, pr in nets:
        (pm_area, pm_power), (pr_area, pr_power) = pm.split(" / "), pr.split(" / ")
        n = sor[net]
        rows.append((f"SRAM overhead {net} area (%)", n["area_pct"]["ours"], n["area_pct"]["paper"],
                     area_dec, pm_area, pr_area))
        rows.append((f"SRAM overhead {net} power (%)", n["power_pct"]["ours_by_rows_on"][rows_on],
                     n["power_pct"]["paper"], 1, pm_power, pr_power))
    rows += [
        ("RRAM power share: header and column supply (%)", 100 * p["column_supply_mW"] / p["total_mW"], None,
         1, "83.3", "---"),
        ("RRAM power share: ADC (%)", 100 * p["adc_mW"] / p["total_mW"], None, 1, "3.0", "---"),
        ("RRAM power share: combined (%)", p["share_supply_view_pct"], share, 1, 0, "86.3", "90"),
    ]
    return _rows("(c) RSE", rows)


# target -> (its steps (given --force), its rows of the table)
TABLES = {
    "1f1c": (f1c_steps, f1c_rows),
    "hyflexpim": (lambda force: hfp_steps(), hfp_rows),
    "rse": (rse_steps, rse_rows),
}


def run(target: str, description: str) -> None:
    """One target's part of the table: its steps, then reproduce/results/
    hardware_validation_<target>.csv checked against the paper."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="delete this target's results and rerun every step, compilation included")
    args = parser.parse_args()
    steps, rows = TABLES[target]
    log = run_steps(steps(args.force), args.force)
    check_rows(f"hardware_validation_{target}", rows(), log)
