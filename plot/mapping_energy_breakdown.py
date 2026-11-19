"""Create an absolute stacked energy breakdown without in-bar annotations from saved greedy results.

python3 plot/mapping_energy_breakdown.py --extract  # snapshot results and draw
python3 plot/mapping_energy_breakdown.py            # redraw the saved snapshot
python3 plot/mapping_energy_breakdown.py --refresh-gpu-mobile  # add only the mobile GPU, then draw

Each GPU's energy per inference (TensorRT INT8, whole board) is a horizontal reference line:
RTX 3090 dashed, RTX 3060 Laptop dotted, in different colors.

No compiler, mapper, or energy simulation is rerun. Original figures are untouched.
"""
import argparse
import json
import math
from pathlib import Path

import figure_data
from mapping_latency import MODELS, LABELS, TARGETS

GROUP_STRIDE = 3.1  # Use less space between applications to separate their budget bars.

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/submission/figures/evaluation"
DATA = Path(__file__).resolve().parent / "meta_data/mapping_energy_breakdown.json"
GPU_DATA = ROOT / "dump/gpu_baseline/latency_rtx3090.json"
GPU_MOBILE_DATA = ROOT / "dump/gpu_baseline/latency_rtx3060_laptop.json"
# GPU reference lines: (snapshot key, legend label, color, dash pattern)
GPUS = (("gpu", "3090", "energyGPU", "on 2pt off 1.2pt"),
        ("gpu_mobile", "3060L", "energyGPUMobile", "on 0.8pt off 0.8pt"))
CATEGORIES = (("Array", "energyArray", "31,120,180"),
              ("Interconnect/Buffer", "energyInterconnect", "166,206,227"),
              ("Digital module", "energyDigital", "51,160,44"),
              ("Accel units", "energyAccel", "178,223,138"))
INTERCONNECT = {"PE input registers", "routers", "tile buses", "inner buses"}


def category(target, name):
    if name.startswith("accelerator "):
        return "Accel units"
    if name in INTERCONNECT:
        return "Interconnect/Buffer"
    if target == "hyflexpim":
        if name.startswith("digital "):
            return "Digital module"
        if name.startswith("analog "):
            return "Array"
    elif target == "fefet1c" and name.startswith(("MAC:", "RMU ", "WMU ")):
        return "Array"
    elif target == "rse" and name.startswith(("RRAM ", "SRAM ", "SRAM:")):
        return "Array"
    raise ValueError(f"Unclassified energy component: {target}/{name}")


def extract():
    targets, sources = {}, [GPU_DATA, GPU_MOBILE_DATA]
    for target, _ in TARGETS:
        targets[target] = {}
        for model in MODELS:
            files = list((ROOT / "run_scripts/results" / f"{target}_energy" / model).glob("cap_*.json"))
            files.sort(key=lambda p: int(p.stem.split("_")[1]))
            if len(files) != 3:
                raise ValueError(f"{target}/{model}: expected three energy results, found {len(files)}")
            rows = []
            for path in files:
                record = json.loads(path.read_text())
                total = float(record["total_mJ"])
                if not math.isfinite(total) or total <= 0:
                    raise ValueError(f"{path}: invalid total energy")
                if record["model"] not in (model, model + "-w3"):
                    raise ValueError(f"{path}: model mismatch")
                grouped = {name: 0.0 for name, _, _ in CATEGORIES}
                for name, value in record["parts_mJ"].items():
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"{path}: invalid energy for {name}")
                    grouped[category(target, name)] += value
                if not math.isclose(sum(grouped.values()), total, rel_tol=1e-8, abs_tol=1e-10):
                    raise ValueError(f"{path}: component energies do not sum to total")
                cap = int(record["cap"])
                if cap <= 0 or cap & (cap - 1):
                    raise ValueError(f"{path}: cap must be a power of two")
                rows.append({"cap": cap, "total_mJ": total, "groups_mJ": grouped,
                             "groups_pct": {k: 100 * v / total for k, v in grouped.items()},
                             "parts_mJ": record["parts_mJ"], "tiles_file": record["tiles_file"]})
                sources.append(path)
            targets[target][model] = rows
    figure_data.write(DATA, __file__, sources, {"method": "greedy", "targets": targets, "gpu": gpu_energy(GPU_DATA),
        "gpu_mobile": gpu_energy(GPU_MOBILE_DATA),
        "grouping": {"Array": "Array computation, read/write and local peripheral energy; includes RSE SRAM macros and their MACs",
                     "Interconnect/Buffer": "Registers, buses and routers",
                     "Digital module": "HFP digital RRAM computation and reads/writes",
                     "Accel units": "Digital functional units"}})


def gpu_energy(path):
    record = json.loads(path.read_text())
    energy = {model: record["measurement_summaries"][model]["trt_int8"]["mJ_per_inference"] for model in MODELS}
    for model, value in energy.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{path}: invalid GPU energy for {model}: {value}")
    return {"configuration": "trt_int8", "source": str(path.relative_to(ROOT)),
            "source_commit": record.get("source_commit", record.get("environment", {}).get("kernels_commit")),
            "device": record.get("device"),
            "scope": "whole-board energy per inference, including static power",
            "energy_mJ": energy}


def refresh_gpu_mobile():
    """Add only the mobile GPU's energy to the saved snapshot; every other value stays."""
    record = json.loads(DATA.read_text())
    record["data"]["gpu_mobile"] = gpu_energy(GPU_MOBILE_DATA)
    record["meta"]["gpu_mobile_refresh"] = figure_data.meta(__file__, [GPU_MOBILE_DATA])
    DATA.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")


def family_axis(target, data, indices, family, width, offset, legend, gpu):
    models = [MODELS[i] for i in indices]
    rows = [(i * GROUP_STRIDE + j, r) for i, model in enumerate(models) for j, r in enumerate(data[model])]
    maximum = max(max(r["total_mJ"] for _, r in rows), max(g[m] for g in gpu.values() for m in models))
    step = 10 ** math.floor(math.log10(maximum))
    ymax = math.ceil(maximum * 1.08 / step) * step
    ticks = ",".join(f"{x:g}" for x, _ in rows)
    labels = ",".join(f"{{$2^{{{int(math.log2(r['cap']))}}}$}}" for _, r in rows)
    lines = [r"\begin{scope}", r"\begin{axis}[",
             f"at={{({offset}\\energyPlotUnit,0)}},anchor=south west,scale only axis,",
             (r"width=0.20\energyPlotUnit,height=0.204545\energyPlotUnit," if family == "CNNs"
              else r"width=0.33\energyPlotUnit,height=0.204545\energyPlotUnit,"),
             f"title={{{family}}},title style={{font=\\tiny}},",
             r"ybar stacked,bar width=3pt,bar shift=0pt,",
             f"xmin=-0.7,xmax={(len(models)-1)*GROUP_STRIDE+2.7:g},ymin=0,ymax={ymax:g},",
             (r"ylabel={Energy (mJ)}," if family == "CNNs" and target == TARGETS[0][0] else "") + r"scaled y ticks=false,",
             r"ylabel style={font=\tiny,at={(0.35,0.5)}},",
             r"tick label style={font=\tiny},x tick label style={rotate=60,anchor=north east},",
             r"ymajorgrids=true,major grid style={dotted,gray!45},axis x line*=bottom,",
             f"xtick={{{ticks}}},xticklabels={{{labels}}},",
             r"legend style={draw=none,fill=none,font=\tiny,at={(1.65,1.28)},anchor=south},legend columns=5,]"]
    for name, color, _ in CATEGORIES:
        coords = " ".join(f"({x:g},{r['groups_mJ'][name]:.9f})" for x, r in rows)
        lines.append(f"\\addplot+[fill={color},draw=none,forget plot] coordinates {{{coords}}};")
        if legend:
            lines += [f"\\addlegendimage{{area legend,fill={color},draw=none}}", f"\\addlegendentry{{{name}}}"]
    if legend:
        for key, label, color, dash in GPUS:
            lines += [f"\\addlegendimage{{line legend,no markers,dash pattern={dash},draw={color},line width=0.5pt}}",
                      f"\\addlegendentry{{{label}}}"]
    for i, model in enumerate(models):
        base = i * GROUP_STRIDE
        left = -0.7 if i == 0 else base + (2 - GROUP_STRIDE) / 2
        right = ((len(models) - 1) * GROUP_STRIDE + 2.7 if i == len(models) - 1
                 else (i + 1) * GROUP_STRIDE + (2 - GROUP_STRIDE) / 2)
        for key, _, color, dash in GPUS:
            lines.append(f"\\draw[dash pattern={dash},{color},line width=0.5pt] (axis cs:{left:g},{gpu[key][model]:.9f}) -- (axis cs:{right:g},{gpu[key][model]:.9f});")
    for i, _ in enumerate(models):
        if i:
            boundary = i * GROUP_STRIDE + (2 - GROUP_STRIDE) / 2
            lines.append(f"\\draw[dashed,gray!45] (axis cs:{boundary:g},0) -- (axis cs:{boundary:g},{ymax:g});")
        lines.append(f"\\coordinate ({family}model{i}) at (axis cs:{i*GROUP_STRIDE+1:g},0);")
    lines.append(r"\end{axis}")
    for i, index in enumerate(indices):
        lines.append(f"\\node[rotate=30,anchor=north,font=\\fontsize{{5}}{{5.5}}\\selectfont,align=center] at ([yshift=-2.3em]{family}model{i}) {{{LABELS[index]}}};")
    lines.append(r"\end{scope}")
    return "\n".join(lines)


def panel(target, title, data, legend, gpu):
    lines = [r"\begin{subfigure}[t]{0.325\textwidth}\centering\vspace{0pt}", r"\begin{tikzpicture}"]
    lines.append(family_axis(target, data, range(3), "CNNs", 0.30, 0, legend, gpu))
    lines.append(family_axis(target, data, range(3, 8), "Transformers", 0.54, 0.30, False, gpu))
    lines += [r"\end{tikzpicture}", f"\\caption{{{title}}}\\label{{fig:mapping-energy-{target}}}", r"\end{subfigure}"]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--refresh-gpu-mobile", action="store_true", help="add only the mobile GPU to the saved snapshot")
    args = parser.parse_args()
    if args.extract:
        extract()
    if args.refresh_gpu_mobile:
        refresh_gpu_mobile()
    snapshot = figure_data.read(DATA)
    data = snapshot["targets"]
    gpu = {key: snapshot[key]["energy_mJ"] for key, *_ in GPUS}
    lines = ["% Generated by plot/mapping_energy_breakdown.py; saved measurements only.",
             r"\begin{figure*}[t]\centering\begingroup"]
    lines += [f"\\definecolor{{{color}}}{{RGB}}{{{rgb}}}" for _, color, rgb in CATEGORIES]
    lines.append(r"\definecolor{energyGPU}{RGB}{70,70,70}")
    lines.append(r"\definecolor{energyGPUMobile}{RGB}{170,55,130}")
    lines.append(r"\newlength{\energyPlotUnit}\setlength{\energyPlotUnit}{0.44\textwidth}")
    lines.append(r"\newcommand{\energyLegendBox}[1]{\tikz[baseline=-0.4ex]{\fill[#1] (0,0) rectangle (0.18,0.10);}}")
    legend_items = [f"\\energyLegendBox{{{color}}}\\, {name}" for name, color, _ in CATEGORIES]
    legend_items += [f"\\tikz[baseline=-0.4ex]{{\\draw[{color},dash pattern={dash},line width=0.5pt] (0,0.05) -- (0.35,0.05); }}\\, {label}"
                     for key, label, color, dash in GPUS]
    lines.append(r"{\scriptsize " + r"\quad ".join(legend_items) + r"}\par\vspace{1pt}")
    for i, (target, title) in enumerate(TARGETS):
        lines.append(panel(target, title, data[target], False, gpu))
        if i < len(TARGETS) - 1:
            lines.append(r"\hfill%")
    lines += [r"\caption{Energy breakdown of applications mapped using greedy at three array budgets on (a) 1F1C, (b) HFP, and (c) RSE. Dashed and dotted lines show RTX 3090 and RTX 3060 Laptop (3060L) energy per inference (TensorRT INT8).}",
              r"\label{fig:mapping-energy-breakdown}", r"\endgroup\end{figure*}"]
    path = OUT / "mapping_energy_breakdown.tex"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
