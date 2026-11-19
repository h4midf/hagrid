"""Writes the array-usage panels of the paper's mapping motivation figure (background,
Fig. 3b-d): paper/submission/figures/background_motivation/array_usage_<target>_attention.pdf,
the working arrays in use over time for the 12-layer attention app, list scheduling above
CP-SAT, at each target's middle cap of the five (the sweeps' caps sorted, the third).

It draws from DATA_PATH (plot/figure_data.py), which --extract writes from the mappings
the motivation sweeps saved (run_scripts/{fefet1c,hyflexpim,rse}_motivation_sweep.py):
run_scripts/results/<target>_motivation/<method>/cap_<cap>.json. Each op's working arrays
("volatile": arrays, live_start_ns, live_end_ns; held from start to end, as xdsl_hardware's
peak count) are summed over time into a step curve per method. The dashed line is the
cap, the dotted line each method's completion time. Text is in Biolinum (Libertine's sans
family, through LaTeX).

Run from the repo root:
    python3 plot/motivation_array_usage.py             # draw from DATA_PATH
    python3 plot/motivation_array_usage.py --extract   # rewrite DATA_PATH from the mappings, then draw
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_data  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "run_scripts" / "results"
OUT_DIR = REPO_ROOT / "paper/submission/figures/background_motivation"
DATA_PATH = Path(__file__).resolve().parent / "meta_data/motivation_array_usage.json"
APP = "bert-attention-12layer"
# results prefix, output name, panel width (pt, as the figure lays them out)
TARGETS = (("fefet1c", "1fefet1c", 130.207), ("hyflexpim", "hyflexpim", 101.18), ("rse", "rse", 101.18))
PANEL_HEIGHT_PT = 101.091888  # Four successive 10% reductions; fonts and ticks unchanged
TRIM_PT = 6.0  # mapping_motivation_all.tex trims 5pt off each side of the panel
TITLED_PANEL = "hyflexpim"  # the middle panel, the only one that names the methods
# paper/submission/colors.tex: blind_safe_three_scheme_three_colors, blind_safe_two_scheme_three_colors
METHODS = (("listsched", "List scheduling", (141 / 255, 160 / 255, 203 / 255)),
           ("cpsat", "CP-SAT", (252 / 255, 141 / 255, 98 / 255)))
CAP_COLOR = (0.533, 0.541, 0.522)  # aluminium4

RC = {
    "text.usetex": True,
    "text.latex.preamble": (r"\usepackage[T1]{fontenc}"
                            r"\usepackage[tt=false,type1=true]{libertine}"
                            r"\renewcommand{\familydefault}{\sfdefault}"),
    "font.family": "sans-serif", "font.size": 7,
    "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
}


def power_of_two(n: int) -> str:
    """'2^e' in text (Biolinum), for an exact power of two."""
    e = n.bit_length() - 1
    if 1 << e != n:
        raise ValueError(f"{n} is not a power of two")
    return rf"2\textsuperscript{{{e}}}"


def power_of_ten(value: float, _pos=None) -> str:
    e = round(math.log10(value))
    return rf"10\textsuperscript{{{e}}}"


def scientific(value: float, _pos=None) -> str:
    """A time tick as 'm x 10^e' in text (Biolinum)."""
    if value == 0:
        return "0"
    e = int(f"{value:e}".split("e")[1])
    m = value / 10 ** e
    m_text = f"{m:g}"
    return rf"{m_text}\,\texttimes\,10\textsuperscript{{{e}}}" if m_text != "1" else rf"10\textsuperscript{{{e}}}"


def usage(mapping_path: Path) -> tuple[list, list]:
    """(times, arrays in use from each time on) of a mapping's working arrays."""
    events = []
    for op in json.loads(mapping_path.read_text())["ops"]:
        vol = op.get("volatile")
        if vol and vol["arrays"] > 0:
            events.append((vol["live_start_ns"], vol["arrays"]))
            events.append((vol["live_end_ns"], -vol["arrays"]))
    # at equal times, starts before ends (the peak count's closed intervals)
    events.sort(key=lambda ev: (ev[0], ev[1] < 0))
    times, levels, level = [0.0], [0], 0
    for t, delta in events:
        level += delta
        times.append(t)
        levels.append(level)
    return times, levels


def middle_cap(points: dict) -> int:
    caps = sorted(int(c) for c in points)
    if len(caps) != 5:
        raise ValueError(f"{len(caps)} caps, the figure takes 5")
    return caps[2]


def extract() -> None:
    """DATA_PATH from the mappings: per target, its middle cap and, per method, the usage
    curve (times in ns, arrays in use from each time on) and the completion time."""
    data, sources = {}, []
    for prefix, _, _ in TARGETS:
        path = RESULTS_DIR / f"{prefix}_motivation_{APP}.json"
        points = json.loads(path.read_text())["points"]
        cap = middle_cap(points)
        sources.append(path)
        methods = {}
        for method, _, _ in METHODS:
            entry = points[str(cap)][method]
            if "mapping" not in entry:
                raise ValueError(f"{prefix} cap {cap}: no {method} mapping ({entry.get('error')})")
            mapping = figure_data.repo_path(entry["mapping"])
            sources.append(mapping)
            times, levels = usage(mapping)
            methods[method] = {"times_ns": times, "arrays": levels, "latency_ns": entry["latency"]}
        data[prefix] = {"cap": cap, "methods": methods}
    figure_data.write(DATA_PATH, __file__, sources, data)


def plot_target(prefix: str, out_name: str, width_pt: float, data: dict) -> None:
    cap = data[prefix]["cap"]
    curves = {method: ((m["times_ns"], m["arrays"]), m["latency_ns"])
              for method, m in data[prefix]["methods"].items()}
    t_max = max(lat for _, lat in curves.values())
    y_max = max(max(levels) for (_, levels), _ in curves.values())
    # the axis starts at the lowest decade the usage reaches while it runs
    y_min = 10 ** math.floor(math.log10(min(lv for (_, levels), _ in curves.values() for lv in levels if lv > 0)))

    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(width_pt / 72, PANEL_HEIGHT_PT / 72))
        for ax, (method, title, color) in zip(axes, METHODS):
            (times, levels), latency = curves[method]
            ax.fill_between(times, levels, step="post", color=color, linewidth=0)
            ax.step(times, levels, where="post", color=color, linewidth=0.4)
            ax.axhline(cap, color=CAP_COLOR, linestyle="--", linewidth=0.5)
            ax.text(t_max, cap, rf"cap = {power_of_two(cap)}", ha="right", va="bottom", fontsize=6.5,
                    color="black")
            ax.axvline(latency, color=CAP_COLOR, linestyle=":", linewidth=0.6)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max * 40)
            ax.set_xlim(0, t_max * 1.04)
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
            ax.yaxis.set_major_formatter(FuncFormatter(power_of_ten))
            ax.yaxis.set_minor_formatter(NullFormatter())
            # the method names only on the middle panel (once in the figure); the others
            # keep an invisible title of the same height so the three panels line up
            ax.set_title(title if prefix == TITLED_PANEL else rf"\phantom{{{title}}}", fontsize=7, loc="center", pad=2)
            ax.grid(axis="y", color=(0.827, 0.843, 0.812), linewidth=0.4)
            if prefix == "fefet1c":
                ax.set_ylabel("\\#array\nin use")
        axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=3))
        axes[-1].xaxis.set_major_formatter(FuncFormatter(scientific))
        axes[-1].set_xlabel("time (ns)")
        # TRIM_PT clear at both sides: the figure includes each panel with trim=5pt 0 5pt 0
        side = TRIM_PT / width_pt
        fig.tight_layout(pad=0.2, h_pad=0.4, rect=(side, 0, 1 - side, 1))
        out = OUT_DIR / f"array_usage_{out_name}_attention.pdf"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
    print(f"wrote {out} (cap {cap} = 2^{cap.bit_length() - 1}; completion list scheduling "
          f"{curves['listsched'][1]:.0f} ns, CP-SAT {curves['cpsat'][1]:.0f} ns)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extract", action="store_true", help="rewrite the data file from the mappings first")
    if parser.parse_args().extract:
        extract()
    data = figure_data.read(DATA_PATH)
    for prefix, out_name, width in TARGETS:
        plot_target(prefix, out_name, width, data)


if __name__ == "__main__":
    main()
