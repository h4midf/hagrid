"""A new Figure-7-style comparison, independent of latency_bars_all.tex.

Run: python3 plot/mapping_latency.py --extract [--gpu path/to/results.json] [--gpu-mobile path]
Then: python3 plot/mapping_latency.py  (redraw the saved snapshot)
Add only the mobile GPU to the saved snapshot: python3 plot/mapping_latency.py --refresh-gpu-mobile

GPU input: {"latency_ns": {"bert-base": measured_ns, ...}}.
By default uses dump/gpu_baseline/latency_rtx3090.json and
dump/gpu_baseline/latency_rtx3060_laptop.json (both TensorRT INT8).
The original measurements and branch provenance are saved in that folder;
regenerating this figure does not access another branch.
Each GPU's latency is a horizontal reference line across each workload's three budget
groups: RTX 3090 dashed, RTX 3060 Laptop dotted, in different colors.
Missing CP-SAT points are stored as explicit zero placeholders; zero has no
position on a log axis, so these slots are left blank in the chart.
Missing, infeasible, or failed measurements are never replaced with another method.
No compiler or scheduler is run by this script.
"""
import argparse
import json
import math
from pathlib import Path

import figure_data

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "run_scripts/results"
OUT = ROOT / "paper/submission/figures/evaluation"
DATA = Path(__file__).resolve().parent / "meta_data/mapping_latency.json"
MODELS = ("resnet50", "mobilenet-v2", "vgg16", "vit-base", "vit-large", "bert-base", "bert-large", "gpt2")
LABELS = ("resnet", "mobilenet", "vgg", r"vit\_b", r"vit\_l", r"bert\_b", r"bert\_l", "gpt")
KNOWN_INFEASIBLE = {("rse", "resnet50"), ("rse", "mobilenet-v2"),
                    ("fefet1c", "mobilenet-v2")}
KNOWN_INFEASIBLE_CAPS = {("rse", "gpt2", 65536), ("rse", "bert-base", 131072)}
GROUP_STRIDE = 3.5  # Three budget groups plus a half-slot gap between workloads.
TARGETS = (("fefet1c", "1F1C"), ("hyflexpim", "HFP"), ("rse", "RSE"))
METHODS = (("gpu", "GPU", "mappingGPU", 0), ("listsched", "LS", "mappingLS", -2.2),
           ("greedy", "Greedy", "mappingGreedy", 0), ("cpsat", "CP-SAT", "mappingCPSAT", 2.2))
# GPU reference lines: (row key, legend label, color, dash pattern)
GPUS = (("gpu", "3090", "mappingGPU", "on 2pt off 1.2pt"),
        ("gpu_mobile", "3060L", "mappingGPUMobile", "on 0.8pt off 0.8pt"))
GPU_MOBILE = ROOT / "dump/gpu_baseline/latency_rtx3060_laptop.json"


def load_gpu(path):
    """latency_ns by model from a GPU JSON, each a finite positive number."""
    record = json.loads(Path(path).read_text())
    latency = record["latency_ns"]
    if not isinstance(latency, dict):
        raise ValueError(f"{path}: latency_ns must map model names to measured nanoseconds")
    for model, value in latency.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{path}: invalid GPU latency for {model}: {value}")
    return record, latency


def gpu_provenance(record):
    return {k: record[k] for k in ("source_branch", "source_commit", "source_path", "device", "batch_size",
                                   "environment") if k in record}


def valid_latency(point):
    value = point.get("latency_ns", point.get("latency"))
    if point.get("feasible") is False or value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None


def extract(gpu_path, gpu_mobile_path):
    sources, gpu = [], {}
    gpu_record = {}
    if gpu_path:
        gpu_path = Path(gpu_path).resolve()
        gpu_record = json.loads(gpu_path.read_text())
        gpu = gpu_record["latency_ns"]
        sources.append(gpu_path)
        if not isinstance(gpu, dict):
            raise ValueError("GPU latency_ns must map model names to measured nanoseconds")
    gpu_mobile_path = Path(gpu_mobile_path).resolve()
    gpu_mobile_record, gpu_mobile = load_gpu(gpu_mobile_path)
    sources.append(gpu_mobile_path)
    targets = {}
    for target, _ in TARGETS:
        targets[target] = {}
        for model in MODELS:
            by_method = {}
            for method, suffix in (("greedy", "offload_policy"), ("listsched", "listsched"), ("cpsat", "cpsat")):
                path = RESULTS / f"{target}_{suffix}_{model}.json"
                if path.exists():
                    sources.append(path)
                    record = json.loads(path.read_text())
                    if record["model"] not in (model, model + "-w3"):
                        raise ValueError(f"{path}: model mismatch")
                    by_method[method] = record.get("points", {})
                else:
                    by_method[method] = {}
            caps = sorted(map(int, by_method["greedy"]))
            if len(caps) != 3:
                raise ValueError(f"{target}/{model}: expected three greedy budgets, got {caps}")
            rows = []
            for cap in caps:
                if cap <= 0 or cap & (cap - 1):
                    raise ValueError(f"{target}/{model}: cap {cap} is not a power of two")
                row = {"cap": cap}
                for method in ("listsched", "greedy", "cpsat"):
                    point = by_method[method].get(str(cap), {})
                    if method == "cpsat":
                        mapping_path = RESULTS / f"{target}_cpsat" / model / f"cap_{cap}.json"
                        if mapping_path.exists():
                            mapping = json.loads(mapping_path.read_text())
                            if mapping.get("model") not in (model, model + "-w3") or mapping.get("cap") != cap:
                                raise ValueError(f"{mapping_path}: model or cap mismatch")
                            used = mapping.get("arrays", {}).get("total")
                            if used is not None and used > cap:
                                raise ValueError(f"{mapping_path}: uses {used} arrays, exceeding cap {cap}")
                            point = mapping
                            sources.append(mapping_path)
                    row[method] = {"latency_ns": valid_latency(point),
                                   "status": point.get("status"), "feasible": point.get("feasible")}
                    if method == "cpsat" and mapping_path.exists():
                        row[method]["source"] = str(mapping_path.relative_to(ROOT))
                    if method == "cpsat" and row[method]["latency_ns"] is None:
                        row[method].update(latency_ns=0, placeholder=True)
                        if target == "hyflexpim":
                            row[method].update(display_infeasible=True,
                                               note="No feasible result obtained; marker requested by user")
                if (target, model) in KNOWN_INFEASIBLE or (target, model, cap) in KNOWN_INFEASIBLE_CAPS:
                    row["cpsat"] = {"latency_ns": None, "status": "INFEASIBLE",
                                    "feasible": False, "source": "user-confirmed"}
                value = gpu.get(model)
                if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                    raise ValueError(f"Invalid GPU latency for {model}: {value}")
                row["gpu"] = {"latency_ns": None if value is None else float(value)}
                mobile = gpu_mobile.get(model)
                row["gpu_mobile"] = {"latency_ns": None if mobile is None else float(mobile)}
                rows.append(row)
            targets[target][model] = rows
    figure_data.write(DATA, __file__, sources, {"targets": targets, "gpu_source": str(gpu_path) if gpu_path else None,
        "gpu_configuration": gpu_record.get("configuration"),
        "gpu_provenance": gpu_provenance(gpu_record),
        "gpu_mobile_source": str(gpu_mobile_path),
        "gpu_mobile_configuration": gpu_mobile_record.get("configuration"),
        "gpu_mobile_provenance": gpu_provenance(gpu_mobile_record)})


def refresh_gpu_mobile(gpu_mobile_path):
    """Add only the mobile GPU's latency to the saved snapshot; every other measurement stays."""
    gpu_mobile_path = Path(gpu_mobile_path).resolve()
    gpu_mobile_record, gpu_mobile = load_gpu(gpu_mobile_path)
    record = json.loads(DATA.read_text())
    data = record["data"]
    for rows_by_model in data["targets"].values():
        for model, rows in rows_by_model.items():
            mobile = gpu_mobile.get(model)
            for row in rows:
                row["gpu_mobile"] = {"latency_ns": None if mobile is None else float(mobile)}
    data["gpu_mobile_source"] = str(gpu_mobile_path)
    data["gpu_mobile_configuration"] = gpu_mobile_record.get("configuration")
    data["gpu_mobile_provenance"] = gpu_provenance(gpu_mobile_record)
    record["meta"]["gpu_mobile_refresh"] = figure_data.meta(__file__, [gpu_mobile_path])
    DATA.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")


def infeasible_marker(point):
    return point.get("status") == "INFEASIBLE" or point.get("display_infeasible", False)


def refresh_hfp_cpsat():
    """Update only HFP CP-SAT; preserve all other saved measurements."""
    record = json.loads(DATA.read_text())
    data = record["data"]
    sources = []
    for model in MODELS:
        path = RESULTS / f"hyflexpim_cpsat_{model}.json"
        points = {}
        if path.exists():
            sources.append(path)
            summary = json.loads(path.read_text())
            if summary["model"] != model:
                raise ValueError(f"{path}: model mismatch")
            points = summary.get("points", {})
        for row in data["targets"]["hyflexpim"][model]:
            point = points.get(str(row["cap"]), {})
            latency = valid_latency(point)
            row["cpsat"] = {"latency_ns": latency, "status": point.get("status"),
                            "feasible": point.get("feasible"),
                            "source": str(path.relative_to(ROOT)) if point else None}
            if latency is None:
                row["cpsat"].update(latency_ns=0, placeholder=True, display_infeasible=True,
                                    note="No feasible result obtained; marker requested by user")
    provenance = figure_data.meta(__file__, sources)
    record["meta"]["hfp_cpsat_refresh"] = provenance
    DATA.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")


def panel(target, title, data, ymin, ymax, legend):
    rows = [(i * GROUP_STRIDE + j, model, row) for i, model in enumerate(MODELS) for j, row in enumerate(data[model])]
    ticks, tick_labels, gpu_rows = [], [], []
    for i, model in enumerate(MODELS):
        base = i * GROUP_STRIDE
        for key, *_ in GPUS:
            gpu_values = {row.get(key, {}).get("latency_ns") for row in data[model]}
            if len(gpu_values) != 1:
                raise ValueError(f"{target}/{model}: {key} measurement must be independent of the cap")
        gpu_rows.append((base, model, data[model][0]))
        for j, row in enumerate(data[model]):
            ticks.append(f"{base+j:g}")
            tick_labels.append(f"{{$2^{{{int(math.log2(row['cap']))}}}$}}")
    ticks, caps = ",".join(ticks), ",".join(tick_labels)
    lines = [r"\begin{subfigure}{\columnwidth}", r"\centering", r"\begin{tikzpicture}", r"\begin{axis}[",
             r"width=1.08\columnwidth,height=0.364\columnwidth,",
             r"ybar,bar width=2.2pt,ymode=log,log basis y=10,",
             f"ymin={ymin},ymax={ymax},xmin=-0.7,xmax={(len(MODELS)-1)*GROUP_STRIDE+2.7:g},",
             r"ylabel={Latency (ns)},ylabel style={font=\tiny,at={(0.07,0.5)}},",
             r"ymajorgrids=true,major grid style={dotted,gray!45},axis x line*=bottom,",
             r"tick label style={font=\tiny},x tick label style={rotate=60,anchor=north east},",
             f"xtick={{{ticks}}},xticklabels={{{caps}}},",
             r"legend style={draw=none,fill=none,font=\scriptsize,at={(0.5,1.04)},anchor=south},legend columns=5,",
             r"clip=true,]"]
    for method, label, color, shift in METHODS:
        if method == "gpu":
            if legend:
                lines += [f"\\addlegendimage{{line legend,no markers,dash pattern=on 2pt off 1.2pt,draw={color},line width=0.5pt}}", f"\\addlegendentry{{{label}}}"]
            continue
        method_rows = rows
        coords = " ".join(f"({x:g},{row[method]['latency_ns']:.9g})" for x, _, row in method_rows if row[method].get("feasible") is not False and row[method]["latency_ns"] is not None and row[method]["latency_ns"] > 0)
        lines.append(f"\\addplot+[fill={color},draw=none,bar shift={shift}pt,forget plot] coordinates {{{coords}}};")
        if legend:
            lines += [f"\\addlegendimage{{area legend,fill={color},draw=none}}", f"\\addlegendentry{{{label}}}"]
    infeasible_coords = " ".join(f"({x:g},{ymax})" for x, model, row in rows
                                  if infeasible_marker(row["cpsat"]))
    if infeasible_coords:
        lines.append(f"\\addplot+[fill=gray!30,draw=gray,dashed,bar shift=2.2pt,forget plot] coordinates {{{infeasible_coords}}};")
    if legend:
        lines += [r"\addlegendimage{area legend,fill=gray!30,draw=gray,dashed}",
                  r"\addlegendentry{CP-SAT (infeasible)}"]
    for i, (base, model, row) in enumerate(gpu_rows):
        left = -0.7 if i == 0 else base + (2 - GROUP_STRIDE) / 2
        right = ((len(MODELS) - 1) * GROUP_STRIDE + 2.7 if i == len(MODELS) - 1
                 else (i + 1) * GROUP_STRIDE + (2 - GROUP_STRIDE) / 2)
        for key, _, color, dash in GPUS:
            latency = row.get(key, {}).get("latency_ns")
            if latency is not None and latency > 0:
                lines.append(f"\\draw[dash pattern={dash},{color},line width=0.5pt] (axis cs:{left:g},{latency:.9g}) -- (axis cs:{right:g},{latency:.9g});")
    for i, label in enumerate(LABELS):
        if i:
            lines.append(f"\\draw[dashed,gray!45] (axis cs:{i*GROUP_STRIDE+(2-GROUP_STRIDE)/2:g},{ymin}) -- (axis cs:{i*GROUP_STRIDE+(2-GROUP_STRIDE)/2:g},{ymax});")
        lines.append(f"\\coordinate (model{i}) at (axis cs:{i*GROUP_STRIDE+1:g},{ymin});")
    lines.append(r"\end{axis}")
    for i, label in enumerate(LABELS):
        lines.append(f"\\node[anchor=north,yshift=-2.8em,font=\\fontsize{{5}}{{5.5}}\\selectfont,align=center] at (model{i}) {{{label}}};")
    lines += [r"\end{tikzpicture}", f"\\caption{{{title}}}", f"\\label{{fig:mapping-latency-{target}}}", r"\end{subfigure}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--refresh-hfp-cpsat", action="store_true", help="refresh only HFP CP-SAT in the saved snapshot")
    parser.add_argument("--gpu", default=str(ROOT / "dump/gpu_baseline/latency_rtx3090.json"), help="GPU JSON containing latency_ns by model")
    parser.add_argument("--gpu-mobile", default=str(GPU_MOBILE), help="mobile GPU JSON containing latency_ns by model")
    parser.add_argument("--refresh-gpu-mobile", action="store_true", help="add only the mobile GPU to the saved snapshot")
    args = parser.parse_args()
    if args.extract:
        extract(args.gpu, args.gpu_mobile)
    if args.refresh_hfp_cpsat:
        refresh_hfp_cpsat()
    if args.refresh_gpu_mobile:
        refresh_gpu_mobile(args.gpu_mobile)
    data = figure_data.read(DATA)["targets"]
    for target, model in KNOWN_INFEASIBLE:
        for row in data[target][model]:
            row["cpsat"] = {"latency_ns": None, "status": "INFEASIBLE",
                            "feasible": False, "source": "user-confirmed"}
    for target, model, cap in KNOWN_INFEASIBLE_CAPS:
        for row in data[target][model]:
            if row["cap"] == cap:
                row["cpsat"] = {"latency_ns": None, "status": "INFEASIBLE",
                                "feasible": False, "source": "user-confirmed"}
    keys = [m for m, *_ in METHODS] + [k for k, *_ in GPUS if k not in {m for m, *_ in METHODS}]
    values = [row[m]["latency_ns"] for target in data.values() for rows in target.values() for row in rows
              for m in keys if m in row and row[m]["latency_ns"] is not None and row[m]["latency_ns"] > 0]
    ymin = 10 ** math.floor(math.log10(min(values)))
    ymax = 10 ** math.ceil(math.log10(max(values)*1.3))
    missing = []
    for target, title in TARGETS:
        for method, label, *_ in METHODS:
            count = sum((row[method]["latency_ns"] is None or row[method]["latency_ns"] == 0) for rows in data[target].values() for row in rows)
            if count:
                missing.append(f"{title} {label}: {count}/24 points missing or infeasible")
    tex = ["% Generated by plot/mapping_latency.py from plot/meta_data/mapping_latency.json.",
           *["% " + line for line in missing], r"\begin{figure}[t]", r"\centering\begingroup",
           r"\definecolor{mappingGPU}{RGB}{70,70,70}", r"\definecolor{mappingGPUMobile}{RGB}{170,55,130}",
           r"\definecolor{mappingLS}{RGB}{141,160,203}",
           r"\definecolor{mappingGreedy}{RGB}{102,194,165}", r"\definecolor{mappingCPSAT}{RGB}{252,141,98}"]
    tex.append(r"\newcommand{\mappingLegendBox}[1]{\tikz[baseline=-0.4ex]{\fill[#1] (0,0) rectangle (0.18,0.10);}}")
    legend_items = [f"\\tikz[baseline=-0.4ex]{{\\draw[{color},dash pattern={dash},line width=0.5pt] (0,0.05) -- (0.35,0.05); }}\\, {label}"
                    for key, label, color, dash in GPUS]
    legend_items += [f"\\mappingLegendBox{{{color}}}\\, {label}"
                     for method, label, color, _ in METHODS if method != "gpu"]
    has_infeasible = any(infeasible_marker(row["cpsat"])
                         for target in data.values() for rows in target.values() for row in rows)
    if has_infeasible:
        legend_items.append(r"\tikz[baseline=-0.4ex]{\filldraw[fill=gray!30,draw=gray,dashed] (0,0) rectangle (0.18,0.10);}\, CP-SAT (infeasible)")
    tex.append(r"{\scriptsize " + r"\quad ".join(legend_items) + r"}\par\vspace{1pt}")
    tex += [panel(t, title, data[t], ymin, ymax, False) for t, title in TARGETS]
    infeasible_caption = " Gray bars with dashed outlines mark CP-SAT cases with no feasible result obtained." if has_infeasible else ""
    tex += [r"\caption{Application latency for LS, Greedy, and CP-SAT across three array budgets per application on (a) 1F1C, (b) HFP, and (c) RSE. Array budgets are labeled as powers of two. Horizontal dashed and dotted lines show RTX 3090 and RTX 3060 Laptop (3060L) latency (TensorRT INT8) for each workload."
            + infeasible_caption + r"}",
            r"\label{fig:mapping-latency-comparison}", r"\endgroup\end{figure}"]
    path = OUT / "mapping_latency_comparison.tex"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tex) + "\n")
    print(f"wrote {path}")
    for line in missing:
        print(line)


if __name__ == "__main__":
    main()
