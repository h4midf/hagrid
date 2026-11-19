"""Writes the mapping example's panels (implementation, Fig. 6b and 6c) in
paper/submission/figures/implementation/:

  attention_liveness_gantt.pdf  6b: when each value of the layer is generated (its op's
                                schedule, first to last output) and, for a staged dynamic
                                rhs (k, v), retained in WMUs until its matvec ends (staged,
                                then in the matvec's copies); values between the named
                                ones are collapsed into "..."
  attention_array_boxes.pdf     6c: the arrays in groups (consecutive stationary copies
                                whose arrays run the same ops), each with its RMU contents
                                above the WMU placement over t1-t3: which op runs in which
                                arrays, and (hatched) staged operands and partial-result buffers

It draws from plot/meta_data/attention_mapping.json (via plot/figure_data.py),
which --extract writes from the mapping and
placement run_scripts/fefet1c_attention_mapping.py saved (RESULTS_PATH): the phase bounds,
the groups, every box (addresses and times) and every chart row. Everything about the
look is here, not in the data.

The style is the earlier figure's (paper/algorithm/attention_device_mapping.py, which
drew the old compiler's mapping). t1-t3 are three phases of equal width, each linear in
time: up to the end of the projections, up to the start of the last op, up to the end.
"_p" marks partial results (a tiled matvec's outputs before their reduction).
Intermediate buffers and backpressure are a figure-only approximation with uniform
batch timing and a 4096-array cap, recorded in the metadata. Compiler source is unchanged.
Text is in Biolinum (Libertine's sans family, through LaTeX).

Run from the repo root:
    python3 plot/attention_mapping.py             # draw from DATA_PATH
    python3 plot/attention_mapping.py --extract   # rewrite DATA_PATH from RESULTS_PATH, then draw
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_data  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "run_scripts/results/fefet1c_attention_mapping_bert-attention-1layer.json"
OUT_DIR = REPO_ROOT / "paper/submission/figures/implementation"
DATA_PATH = Path(__file__).resolve().parent / "meta_data/attention_mapping.json"

# ---- what the figure shows of the mapping (used by --extract)
# working ops of 6c, by fx_name
RUNS = {
    "_quantized_layers_0_attn_wq": "Q", "_quantized_layers_0_attn_wk": "K",
    "_quantized_layers_0_attn_wv": "V", "_quantized_layers_0_attn_wo": "O",
    "matmul": "S",  # QK^T
    "matmul_1": "P",  # PV
    "softmax_exp": "E", "softmax_recip": "R",
    "e": "E", "r": "R",
}
# stationary operands of 6c, by their consumer's fx_name
WEIGHTS = {"_quantized_layers_0_attn_wq": "WQ", "_quantized_layers_0_attn_wk": "WK",
           "_quantized_layers_0_attn_wv": "WV", "_quantized_layers_0_attn_wo": "WO",
           "softmax_exp": "Texp", "softmax_recip": "Trecip"}
# named rows of 6b: (op, fx_name) -> value name; every other op is collapsed into "..."
VALUES = {
    ("cim.quantize", "quantize_per_tensor"): "x_q",
    ("cim.MatvecOp", "_quantized_layers_0_attn_wq"): "q_p",
    ("cim.vec_add", "_quantized_layers_0_attn_wq"): "q_b",
    ("cim.vec_add", "_quantized_layers_0_attn_wk"): "k_b",
    ("cim.vec_add", "_quantized_layers_0_attn_wv"): "v_b",
    # q, k, v: the values S = QK^T and PV read (requantized per head)
    ("cim.quantize", "quantize_per_tensor_1"): "q",
    ("cim.quantize", "quantize_per_tensor_2"): "k",
    ("cim.quantize", "quantize_per_tensor_3"): "v",
    ("cim.MatvecOp", "matmul"): "s",
    ("cim.lut_search", "softmax_exp"): "e",
    ("cim.reduce_vec", "softmax"): "d",
    ("cim.lut_search", "softmax_recip"): "r",
    ("cim.vec_el_mul", "softmax"): "p",
    ("cim.quantize", "quantize_per_tensor_4"): "p_q",
    ("cim.MatvecOp", "matmul_1"): "z_p",
    ("cim.dequantize", "dequantize_5"): "o",
}
# ops that only rename their input (no schedule of their own in the chart)
NOT_CHARTED = {"const.ref", "cim.save_stationary", "cim.view", "cim.permute", "cim.reshape"}
# Main computation outputs and substantial temporary values only. This filters
# the presentation after all lifetimes and buffer allocations are calculated.
GANTT_VALUES = {"x_q", "q_p", "q_b", "k_b", "v_b", "q", "k", "v", "s", "e", "r", "z_p", "o"}

# ---- the look
INK, EDGE = "#263746", "#8294A2"
RMU_BAND, WMU_BG = "#E2EAF1", "#F1F3F5"
GENERATING, RETAINED = "#4C8E88", "#C5DDD2"
STORED = "#EBCB9D"  # warm sand for temporary values; retain the hatching
RUN_COLORS = dict.fromkeys(("Q", "K", "V", "O", "S", "P", "E", "R"), "#86AFC6")
# value names as drawn (LaTeX text); any other name has its underscores escaped
VALUE_TEXT = {"x_q": r"x\_q"}
RC = {
    "text.usetex": True,
    "text.latex.preamble": (r"\usepackage[T1]{fontenc}"
                            r"\usepackage[tt=false,type1=true]{libertine}"
                            r"\renewcommand{\familydefault}{\sfdefault}"),
    "font.family": "sans-serif", "pdf.fonttype": 42,
}


def fx_name(op: dict) -> str:
    name = op["attributes"].get("fx_name", "")
    if name == "softmax" and op["op"] == "cim.lut_search":
        return "softmax_recip" if op["shape"][-1] == 1 else "softmax_exp"
    return name


def phase_bounds(results: dict) -> list[float]:
    """[0, end of the working ops that end before a dynamic-rhs op starts, start of the
    last working op, the latency] (ns): t1-t3 start at the first three."""
    working = results["placement"]["working"]
    first_dynamic = min(w["live_start_ns"] for w in working if w["rhs"] == "dynamic")
    bounds = [0.0, max(w["live_end_ns"] for w in working if w["live_end_ns"] < first_dynamic),
              max(w["live_start_ns"] for w in working), results["mapping"]["latency_ns"]]
    if bounds != sorted(set(bounds)):
        raise ValueError(f"phase boundaries not increasing: {bounds}")
    return bounds


def groups_of(results: dict, by_id: dict) -> list[dict]:
    """{first, last, rmu} per stationary operand's copies. Consecutive copies are one group
    when every array in them runs the same ops (a copy whose arrays run different ones is
    a group of its own). The arrays past them, up to the cap, are not shown: raises if
    any op uses one."""
    ops_at: dict[int, frozenset] = {}
    for w in results["placement"]["working"]:
        for first, last in w["addresses"]:
            for a in range(first, last + 1):
                ops_at[a] = ops_at.get(a, frozenset()) | {w["op"]}
    groups = []
    for s in results["placement"]["stationary"]:
        name = WEIGHTS[fx_name(by_id[s["consumer"]])]
        lo, n = s["addresses"][0][0], s["arrays_per_copy"]
        if [[lo, lo + s["copies"] * n - 1]] != s["addresses"]:
            raise ValueError(f"stationary {s['value']} is not one run of addresses")
        runs = []  # [first copy, last copy, the ops of its arrays (None if they differ)]
        for c in range(s["copies"]):
            kinds = {ops_at.get(a, frozenset()) for a in range(lo + c * n, lo + (c + 1) * n)}
            same = next(iter(kinds)) if len(kinds) == 1 else None
            if runs and same is not None and runs[-1][2] == same:
                runs[-1][1] = c
            else:
                runs.append([c, c, same])
        for c0, c1, _ in runs:
            groups.append({"first": lo + c0 * n, "last": lo + (c1 + 1) * n - 1, "rmu": name})
    shown = groups[-1]["last"]
    if shown < results["mapping"]["arrays"]["total"] - 1:
        groups.append({"first": shown + 1, "last": results["mapping"]["arrays"]["total"] - 1,
                       "rmu": "empty"})
        shown = groups[-1]["last"]
    for w in results["placement"]["working"]:
        for first, last in w["addresses"] + w.get("held_addresses", []):
            if last > shown:
                raise ValueError(f"op {w['op']} uses arrays up to {last}, past the stationary ones (to {shown})")
    return groups


def rows_of(results: dict) -> list[dict]:
    """6b's rows in block order: {value, start_ns, end_ns, retained_until_ns} for a named
    value (retained_until_ns null unless it is a staged rhs), {value: "..."} for each
    stretch of the others."""
    # a staged rhs stays in WMUs until its matvec ends: staged until it starts, then in its copies
    retained_until = {w["rhs_value"]: w["live_end_ns"] for w in results["placement"]["working"]
                      if "held_addresses" in w}
    rows = []
    for op in results["mapping"]["ops"]:
        if op["op"] in NOT_CHARTED:
            continue
        name = VALUES.get((op["op"], fx_name(op)))
        if name is not None:
            rows.append({"value": name, "start_ns": op["schedule"]["start_ns"], "end_ns": op["schedule"]["end_ns"],
                         "retained_until_ns": retained_until.get(op["id"])})
        elif not rows or rows[-1]["value"] != "...":
            rows.append({"value": "..."})
    missing = set(VALUES.values()) - {r["value"] for r in rows}
    if missing:
        raise ValueError(f"no op for {sorted(missing)} (the IR changed: update VALUES)")
    return rows


def extract() -> None:
    """DATA_PATH from RESULTS_PATH."""
    results = json.loads(RESULTS_PATH.read_text())
    by_id = {o["id"]: o for o in results["mapping"]["ops"]}
    boxes = []
    for w in results["placement"]["working"]:
        if "held_addresses" in w:
            rhs = by_id[w["rhs_value"]]
            boxes.append({"kind": "staged", "label": VALUES[(rhs["op"], fx_name(rhs))], "addresses": w["held_addresses"],
                          "start_ns": w["rhs_hold_start_ns"], "end_ns": w["rhs_hold_end_ns"]})
        boxes.append({"kind": "run", "label": RUNS[fx_name(by_id[w["op"]])], "addresses": w["addresses"],
                      "start_ns": w["live_start_ns"], "end_ns": w["live_end_ns"]})
    data = {
        "implementation_flags": results["flags"],
        "stationary_placement_policy": "Dedicated arrays for narrow LUT tiles; no sharing between lookup consumers",
        "placed_stationary_arrays": sum(last - first + 1 for s in results["placement"]["stationary"]
                                        for first, last in s["addresses"]),
        "cap": results["cap"], "arrays_used": results["mapping"]["arrays"]["total"],
        "array": results["array"], "latency_ns": results["mapping"]["latency_ns"],
        "phase_bounds_ns": phase_bounds(results),
        "groups": groups_of(results, by_id),
        "boxes": boxes,
        "rows": rows_of(results),
    }
    add_partial_storage(data, results)
    simplified = []
    for row in data["rows"]:
        if row["value"] in GANTT_VALUES:
            simplified.append(row)
        elif not simplified or simplified[-1]["value"] != "...":
            simplified.append({"value": "..."})
    # Group the three projection outputs visually without changing their timing.
    bias_values = {"q_b", "k_b", "v_b"}
    first_bias = next(i for i, row in enumerate(simplified) if row["value"] == "q_b")
    last_bias = next(i for i, row in enumerate(simplified) if row["value"] == "v_b")
    data["rows"] = (simplified[:first_bias]
                    + [row for row in simplified[first_bias:last_bias + 1]
                       if row["value"] in bias_values]
                    + simplified[last_bias + 1:])
    figure_data.write(DATA_PATH, __file__, [RESULTS_PATH], data)


def add_partial_storage(data: dict, results: dict) -> None:
    """Apply the figure-only live-FIFO placement and backpressure approximation."""
    from attention_buffer_placement import allocate

    names = {op["id"]: VALUES[(op["op"], fx_name(op))] for op in results["mapping"]["ops"]
             if (op["op"], fx_name(op)) in VALUES}
    allocation = allocate(results, 4096, names)
    data["compiler_cap"] = results["cap"]
    data["cap"] = 4096
    data["compiler_arrays_used"] = results["mapping"]["arrays"]["total"]
    data["compiler_latency_ns"] = results["mapping"]["latency_ns"]
    data["boxes"] = allocation["boxes"]
    for box in data["boxes"]:
        if box["kind"] == "run":
            box["label"] = RUNS[box["label"]]
    q_b = next(box for box in data["boxes"] if box["label"] == "q_b")
    buffer_split = max(last for _, last in q_b["addresses"])
    k_b = next(box for box in data["boxes"] if box["label"] == "k_b")
    kv_split = max(last for _, last in k_b["addresses"])
    data["groups"].extend([
        {"first": data["compiler_arrays_used"], "last": buffer_split, "rmu": "empty"},
        {"first": buffer_split + 1, "last": kv_split, "rmu": "empty"},
        {"first": kv_split + 1, "last": data["cap"] - 1, "rmu": "empty"},
    ])
    # Keep the two-row layout readable after adding the two LUT groups.
    # WMU rectangles retain their exact address extents within a merged group.
    if len(data["groups"]) > 10:
        merged = []
        for group in data["groups"]:
            if (merged and group["rmu"] != "empty" and
                    merged[-1]["rmu"] == group["rmu"] and
                    merged[-1]["last"] + 1 == group["first"]):
                merged[-1]["last"] = group["last"]
            else:
                merged.append(dict(group))
        data["groups"] = merged
    data["arrays_used"] = max(hi for box in data["boxes"] for _, hi in box["addresses"]) + 1
    times = allocation["times"]
    for row in data["rows"]:
        if row["value"] == "...":
            continue
        idx = next(idx for idx, label in names.items() if label == row["value"])
        row["start_ns"], row["end_ns"] = float(times[idx][0]), float(times[idx][-1])
        if row["retained_until_ns"] is not None:
            consumer = next(w for w in results["placement"]["working"] if w.get("rhs_value") == idx)
            row["retained_until_ns"] = float(times[consumer["op"]][-1])
        buffer_ends = [box["end_ns"] for box in allocation["requests"] if box["producer"] == idx]
        if buffer_ends and max(buffer_ends) > row["end_ns"]:
            row["retained_until_ns"] = max(row["retained_until_ns"] or row["end_ns"], max(buffer_ends))
    runs = [box for box in data["boxes"] if box["kind"] == "run"]
    data["latency_ns"] = max(float(t[-1]) for t in times.values())
    data["phase_bounds_ns"] = [0.0, max(box["end_ns"] for box in runs if box["label"] in {"Q", "K", "V"}),
                               next(box["start_ns"] for box in runs if box["label"] == "O"), data["latency_ns"]]
    data["buffer_storage_policy"] = {
        "source": "figure-only approximation; compiler source unchanged",
        "arrival": "uniform exported batches, with producer stalls and input-readiness propagation",
        "release": "logical-prefix consumption at consumer batch completion; aliases share allocations",
        "allocation": "peak live FIFO size reserved over its live envelope; reuse idle WMUs",
        "staged_rhs": "existing dynamic RHS copies counted once",
        "projection_stall_ns": allocation["stall_ns"],
        "schedule_limit": "export omits exact first-batch and operand-read times; this is not a compiler result",
    }
    for box in allocation["requests"]:
        print(f"buffer {box['producer']} {box['label']}: {box['buffer_arrays']} WMUs at {box['addresses']}")
    print(f"Figure-only Q/K/V pacing delay: {allocation['stall_ns']/1000:g} us; {data['arrays_used']}/{data['cap']} addresses")


def warp_to(bounds: list[float], positions: list[float]):
    """Time (ns) -> position, bounds[i] at positions[i], linear between."""
    def warp(t: float) -> float:
        i = min(len(bounds) - 2, max(0, bisect.bisect_right(bounds, t) - 1))
        return positions[i] + (positions[i + 1] - positions[i]) * (t - bounds[i]) / (bounds[i + 1] - bounds[i])
    return warp


def value_text(name: str) -> str:
    return VALUE_TEXT.get(name, name.replace("_", r"\_"))


def var_text(name: str) -> str:
    """A value as an IR variable (%name), as in 6a; "..." stays as is."""
    return name if name == "..." else r"\%" + value_text(name)


def plot(data: dict) -> None:
    bounds = data["phase_bounds_ns"]
    groups = data["groups"]
    per_row = -(-len(groups) // 2)
    sx = 5 / per_row  # the earlier figure had 5 boxes per row: same figure width with per_row
    box_warp = warp_to(bounds, [0, 1 / 3, 2 / 3, 1])

    fig = plt.figure(figsize=(13, 9.3), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, .72, 1.25], left=.12, right=.98, top=.97, bottom=.08, hspace=.24)
    ax = fig.add_subplot(gs[0])
    ax.set(xlim=(0, 9.33), ylim=(0, 5.9))
    ax.axis("off")
    pos = ax.get_position()
    device_width = .41472 * 9.33 / 9.3
    ax.set_position([pos.x0 + (1 - device_width) / 2 * pos.width, pos.y0, device_width * pos.width, pos.height])
    ax.legend(handles=[
        Patch(facecolor=RUN_COLORS["Q"], edgecolor="none", label="Computation"),
        Patch(facecolor=STORED, edgecolor="white", hatch="///", label="Temporary storage"),
    ], loc="lower center", bbox_to_anchor=(.5, .835), ncol=2, frameon=False,
        fontsize=8, handletextpad=.5, columnspacing=1.5, borderpad=.1, borderaxespad=.1)

    # ---- 6c: array groups
    for j, g in enumerate(groups):
        lo, hi = g["first"], g["last"]
        x = .17 + (j % per_row) * 1.83 * sx
        y = 3.0 - (j // per_row) * 1.918
        w = 2.10 * sx
        box_width = 1.65 * sx
        ax.add_patch(Rectangle((x, y + .10), box_width, 1.608, fc="white", ec="black", lw=.8))
        ax.text(x + box_width / 2, y + 1.848, f"Arrays {lo}–{hi}", fontsize=8, color="black", ha="center", va="center")
        rmu_width = (w - .16 * sx) * .8 * .9 * .9 * 1.1
        ax.add_patch(Rectangle((x + (box_width - rmu_width) / 2, y + 1.447295), rmu_width, .21141, fc=RMU_BAND, ec="none"))
        ax.text(x + box_width / 2, y + 1.553, "RMUs: " + g["rmu"], fontsize=8, color=INK, ha="center", va="center")
        left = x + .34
        bottom = y + .15
        bw = box_width - .34 - .12 * sx
        bh = 1.078
        ax.add_patch(Rectangle((left, bottom), bw, bh, fc=WMU_BG, ec=EDGE, lw=.7))
        for box in data["boxes"]:
            if box["kind"] in {"partial", "temporary", "staged"} and sum(
                    last - first + 1 for first, last in box["addresses"]) < 5:
                continue
            if box["kind"] in {"staged", "partial", "temporary"}:
                fc, hatch, text = STORED, "///", var_text(box["label"])
            else:
                fc, hatch, text = RUN_COLORS[box["label"]], None, f"run\n{box['label']}"
            xx = left + bw * box_warp(box["start_ns"])
            ww = bw * (box_warp(box["end_ns"]) - box_warp(box["start_ns"]))
            for first, last in box["addresses"]:
                aa, bb = max(first, lo), min(last, hi)
                if aa > bb:
                    continue
                yy = bottom + bh * (hi - bb) / (hi - lo + 1)
                hh = bh * (bb - aa + 1) / (hi - lo + 1)
                ax.add_patch(Rectangle((xx, yy), max(.001, ww), hh, fc=fc, ec="white", lw=.5, hatch=hatch))
                if hh > .09 and ww > .1:
                    ax.text(xx + ww / 2, yy + hh / 2, text, fontsize=7.1, color=INK, ha="center", va="center", zorder=10)
        for region in range(3):
            ax.text(left + bw * region / 3, bottom + bh + .11, f"$t_{{{region + 1}}}$", fontsize=8,
                    ha="center", va="center", color=INK, weight="bold")
        for region in (1, 2):
            xx = left + bw * region / 3
            ax.plot([xx, xx], [bottom, bottom + bh], color=EDGE, lw=.8, ls=(0, (2, 2)))
        ax.annotate("", xy=(left + bw, bottom + bh), xytext=(left, bottom + bh),
                    arrowprops=dict(arrowstyle="->", color=EDGE, lw=.8), zorder=5)
        ax.annotate("", xy=(left, bottom), xytext=(left, bottom + bh),
                    arrowprops=dict(arrowstyle="->", color=EDGE, lw=.8), zorder=5)
        ax.text(x + .15, bottom + bh / 2, "WMU placement", rotation=90, ha="center", va="center", fontsize=7.1, color=INK)

    # ---- 6b: generation chart
    rows = data["rows"]
    phase = 2.6 * .95  # t1-t2 and t2-t3, 5% shorter than t3-end
    symbolic = [0, phase, 2 * phase, 2 * phase + 2.6]
    chart_warp = warp_to(bounds, symbolic)
    axis = fig.add_subplot(gs[1:])
    cb = axis.get_position()
    bar_height = .360671781888
    row_pitch = bar_height + (1 - bar_height) * .5 * .9 * .6
    last_row = (len(rows) - 1) * row_pitch
    chart_height = .80 * cb.height * (last_row + 1.4) / (len(rows) + .4) * (len(rows) + .4) / (16 + .4)
    x_right = symbolic[3] + .3
    # same scale per unit as with the earlier xlim (-1.4, 8.1)
    axis.set_position([cb.x0, cb.y0 + cb.height - chart_height,
                       cb.width * .118098 * (x_right + 1.4) / 9.5, chart_height])
    axis.set(xlim=(-1.4, x_right), ylim=(-.6, last_row + .8))
    axis.axis("off")
    for i, row in enumerate(rows):
        y = (len(rows) - 1 - i) * row_pitch
        axis.text(-.35, y, var_text(row["value"]), ha="right", va="center", fontsize=8.5, color=INK)
        if row["value"] == "...":
            continue
        x0, x1 = chart_warp(row["start_ns"]), chart_warp(row["end_ns"])
        axis.add_patch(Rectangle((x0, y - bar_height / 2), x1 - x0, bar_height, fc=GENERATING, ec=INK, lw=.4))
        if row["retained_until_ns"] is not None:
            x2 = chart_warp(row["retained_until_ns"])
            axis.add_patch(Rectangle((x1, y - bar_height / 2), x2 - x1, bar_height, fc=RETAINED, ec=INK, lw=.4))
    for i, xx in enumerate(symbolic[:3]):
        axis.plot([xx, xx], [-.2, last_row + bar_height / 2 + .05], ls=(0, (2, 3)), lw=.8, color=EDGE)
        axis.text(xx, -.65, f"$t_{{{i + 1}}}$", ha="center", va="top", fontsize=8, color=INK)
    axis.legend(handles=[Patch(facecolor=GENERATING, label="Generating"),
                         Patch(facecolor=RETAINED, label="Retained")],
                loc="lower center", bbox_to_anchor=(.5, .94), ncol=2, frameon=False, fontsize=8,
                handlelength=1.4, columnspacing=.6,
                handletextpad=.4, labelspacing=.2, borderpad=.1, borderaxespad=.1)
    axis.annotate("", xy=(symbolic[3] + .15, -.35), xytext=(0, -.35),
                  arrowprops=dict(arrowstyle="->", color=EDGE, shrinkA=0, shrinkB=0))
    axis.text(symbolic[3] / 2 + .1, -1.2, "Execution progress", ha="center", va="top", fontsize=8, color=INK)

    # each panel exported alone, cropped to what it draws (the sizes the paper scales)
    for exported, name in [(ax, "attention_array_boxes"), (axis, "attention_liveness_gantt")]:
        for candidate in fig.axes:
            candidate.set_visible(candidate is exported)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        artists = [*exported.patches, *exported.lines, *exported.texts]
        if exported.get_legend() is not None:
            artists.append(exported.get_legend())
        extents = [a.get_window_extent(renderer) for a in artists if a.get_visible()]
        bbox = Bbox.union(extents).transformed(fig.dpi_scale_trans.inverted()).padded(.02)
        out = OUT_DIR / f"{name}.pdf"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches=bbox)
        print(f"wrote {out}")
    plt.close(fig)
    print("t1-t3 start at (ms): " + ", ".join(f"{t / 1e6:.3g}" for t in bounds[:3])
          + f"; end {bounds[3] / 1e6:.3g} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extract", action="store_true", help="rewrite the data file from the mapping first")
    if parser.parse_args().extract:
        extract()
    with plt.rc_context(RC):
        plot(figure_data.read(DATA_PATH))


if __name__ == "__main__":
    main()
