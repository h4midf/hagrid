"""Create an absolute stacked area breakdown without in-bar annotations from saved greedy results.

python3 plot/mapping_area_breakdown.py --extract  # snapshot results and draw
python3 plot/mapping_area_breakdown.py            # redraw the saved snapshot

No compiler, mapper, or area simulation is rerun. Original figures are untouched.
"""
import argparse
import json
import math
from pathlib import Path

import figure_data
from mapping_latency import MODELS, LABELS, TARGETS, GROUP_STRIDE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/submission/figures/evaluation"
DATA = Path(__file__).resolve().parent / "meta_data/mapping_area_breakdown.json"
CATEGORIES = (("Array", "areaArray", "31,120,180"),
              ("Interconnect/Buffer", "areaInterconnect", "166,206,227"),
              ("Digital module", "areaDigital", "51,160,44"),
              ("Accel units", "areaAccel", "178,223,138"))
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
    elif target == "fefet1c" and (name.startswith("array ") or name == "added arrays"):
        return "Array"
    elif target == "rse" and (name.startswith(("RRAM ", "SRAM ")) or name == "added macros"):
        return "Array"
    raise ValueError(f"Unclassified area component: {target}/{name}")


def extract():
    targets, sources = {}, []
    for target, _ in TARGETS:
        targets[target] = {}
        for model in MODELS:
            files = list((ROOT / "run_scripts/results" / f"{target}_area" / model).glob("cap_*.json"))
            files.sort(key=lambda p: int(p.stem.split("_")[1]))
            if len(files) != 3:
                raise ValueError(f"{target}/{model}: expected three area results, found {len(files)}")
            rows = []
            for path in files:
                record = json.loads(path.read_text())
                total = float(record["total_mm2"])
                if not math.isfinite(total) or total <= 0:
                    raise ValueError(f"{path}: invalid total area")
                if record["model"] not in (model, model + "-w3"):
                    raise ValueError(f"{path}: model mismatch")
                grouped = {name: 0.0 for name, _, _ in CATEGORIES}
                for name, value in record["parts_mm2"].items():
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"{path}: invalid area for {name}")
                    grouped[category(target, name)] += value
                if not math.isclose(sum(grouped.values()), total, rel_tol=1e-8, abs_tol=1e-10):
                    raise ValueError(f"{path}: component areas do not sum to total")
                cap = int(record["cap"])
                if cap <= 0 or cap & (cap - 1):
                    raise ValueError(f"{path}: cap must be a power of two")
                rows.append({"cap": cap, "total_mm2": total, "groups_mm2": grouped,
                             "groups_pct": {k: 100 * v / total for k, v in grouped.items()},
                             "parts_mm2": record["parts_mm2"], "tiles_file": record["tiles_file"]})
                sources.append(path)
            targets[target][model] = rows
    figure_data.write(DATA, __file__, sources, {"method": "greedy", "targets": targets,
        "grouping": {"Array": "Memory arrays and local peripheral circuitry, including RSE SRAM macros and their MACs",
                     "Interconnect/Buffer": "Registers, buses and routers",
                     "Digital module": "HFP digital RRAM modules",
                     "Accel units": "Digital functional units"}})


def family_axis(target, data, indices, family, width, offset, legend):
    models = [MODELS[i] for i in indices]
    rows = [(i * GROUP_STRIDE + j, r) for i, model in enumerate(models) for j, r in enumerate(data[model])]
    maximum = max(r["total_mm2"] for _, r in rows)
    step = 10 ** math.floor(math.log10(maximum))
    ymax = math.ceil(maximum * 1.08 / step) * step
    area_ticks = [ymax * i / 2 for i in range(3)]
    area_tick_labels = ["{$0$}" if value == 0 else f"{{$ {value / 100000:g}\\times10^{{5}} $}}"
                        for value in area_ticks]
    ticks = ",".join(f"{x:g}" for x, _ in rows)
    labels = ",".join(f"{{$2^{{{int(math.log2(r['cap']))}}}$}}" for _, r in rows)
    lines = [r"\begin{scope}", r"\begin{axis}[",
             f"at={{({offset}\\areaPlotUnit,0)}},anchor=south west,scale only axis,",
             (r"width=0.20\areaPlotUnit,height=0.20\areaPlotUnit," if family == "CNNs"
              else r"width=0.33\areaPlotUnit,height=0.20\areaPlotUnit,"),
             f"title={{{family}}},title style={{font=\\tiny}},",
             r"ybar stacked,bar width=3pt,bar shift=0pt,",
             f"xmin=-0.7,xmax={(len(models)-1)*GROUP_STRIDE+2.7:g},ymin=0,ymax={ymax:g},",
             (r"ylabel={Area (mm$^2$)}," if family == "CNNs" and target == TARGETS[0][0] else "") + r"scaled y ticks=false,",
             "ytick={" + ",".join(f"{value:g}" for value in area_ticks) + "},yticklabels={" + ",".join(area_tick_labels) + "},",
             r"ylabel style={font=\tiny,at={(0.35,0.5)}},",
             r"tick label style={font=\tiny},x tick label style={rotate=60,anchor=north east},",
             r"ymajorgrids=true,major grid style={dotted,gray!45},axis x line*=bottom,",
             f"xtick={{{ticks}}},xticklabels={{{labels}}},",
             r"legend style={draw=none,fill=none,font=\tiny,at={(1.65,1.28)},anchor=south},legend columns=5,]"]
    for name, color, _ in CATEGORIES:
        coords = " ".join(f"({x:g},{r['groups_mm2'][name]:.9f})" for x, r in rows)
        lines.append(f"\\addplot+[fill={color},draw=none,forget plot] coordinates {{{coords}}};")
        if legend:
            lines += [f"\\addlegendimage{{area legend,fill={color},draw=none}}", f"\\addlegendentry{{{name}}}"]
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


def panel(target, title, data, legend):
    lines = [r"\begin{subfigure}[t]{0.325\textwidth}\centering\vspace{0pt}", r"\begin{tikzpicture}"]
    lines.append(family_axis(target, data, range(3), "CNNs", 0.30, 0, legend))
    lines.append(family_axis(target, data, range(3, 8), "Transformers", 0.54, 0.30, False))
    lines += [r"\end{tikzpicture}", f"\\caption{{{title}}}\\label{{fig:mapping-area-{target}}}", r"\end{subfigure}"]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    if parser.parse_args().extract:
        extract()
    snapshot = figure_data.read(DATA)
    data = snapshot["targets"]
    lines = ["% Generated by plot/mapping_area_breakdown.py; saved measurements only.",
             r"\begin{figure*}[t]\centering\begingroup"]
    lines += [f"\\definecolor{{{color}}}{{RGB}}{{{rgb}}}" for _, color, rgb in CATEGORIES]
    lines.append(r"\newlength{\areaPlotUnit}\setlength{\areaPlotUnit}{0.45\textwidth}")
    lines.append(r"\newcommand{\areaLegendBox}[1]{\tikz[baseline=-0.4ex]{\fill[#1] (0,0) rectangle (0.18,0.10);}}")
    legend_items = [f"\\areaLegendBox{{{color}}}\\, {name}" for name, color, _ in CATEGORIES]
    lines.append(r"{\scriptsize " + r"\quad ".join(legend_items) + r"}\par\vspace{1pt}")
    for i, (target, title) in enumerate(TARGETS):
        lines.append(panel(target, title, data[target], False))
        if i < len(TARGETS) - 1:
            lines.append(r"\hfill%")
    lines += [r"\caption{Area breakdown of greedy mappings at three array budgets per application on (a) 1F1C, (b) HFP, and (c) RSE. Stacked heights show total area in mm$^2$, including local peripheral circuitry. CNNs and Transformers use separate linear y-axis scales. Array budgets are labeled as powers of two. The Array category includes RSE SRAM correction and dynamic-computation macros.}",
              r"\label{fig:mapping-area-breakdown}", r"\endgroup\end{figure*}"]
    path = OUT / "mapping_area_breakdown.tex"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
