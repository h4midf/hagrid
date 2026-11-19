"""Writes the latency panel of the paper's mapping motivation figure (background, Fig. 3a):
paper/submission/figures/background_motivation/pf_results_list_schedule_plot.tex, a
pgfplots bar chart of list scheduling against CP-SAT for the 12-layer attention app, five
array caps per target.

It draws from DATA_PATH (plot/figure_data.py), which --extract writes from the three
motivation sweeps (run_scripts/{fefet1c,hyflexpim,rse}_motivation_sweep.py),
run_scripts/results/<target>_motivation_bert-attention-12layer.json: each cap's
"listsched" and "cpsat" latency. A CP-SAT point whose "source" is "greedy" is greedy's
mapping standing in for CP-SAT's (listed in the generated file's header). The y-axis
spans the whole decades around the data; the tick labels are the caps as powers of two,
in text so they take the panel's Biolinum (\\sffamily) and not the math font. The
figure's layout (sizes, fonts, y ticks) is set by mapping_motivation_all.tex.

Run from the repo root:
    python3 plot/motivation_latency.py             # draw from DATA_PATH
    python3 plot/motivation_latency.py --extract   # rewrite DATA_PATH from the sweep results, then draw
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_data  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "run_scripts" / "results"
OUT_PATH = REPO_ROOT / "paper/submission/figures/background_motivation/pf_results_list_schedule_plot.tex"
DATA_PATH = Path(__file__).resolve().parent / "meta_data/motivation_latency.json"
APP = "bert-attention-12layer"
# (symbolic x prefix, results file prefix, label above the group)
TARGETS = (("f", "fefet1c", "1F1C"), ("h", "hyflexpim", "HFP"), ("r", "rse", "RSE"))
# the cap panels (b)-(d) show: the middle of the five (plot/motivation_array_usage.middle_cap)
MIDDLE_CAP_INDEX = 2


def extract() -> None:
    """DATA_PATH from the sweep results: per target, its five caps' latencies (ns)."""
    import json
    data, sources = {}, []
    for _, target, _ in TARGETS:
        path = RESULTS_DIR / f"{target}_motivation_{APP}.json"
        sources.append(path)
        points = json.loads(path.read_text())["points"]
        rows = [points[c] for c in sorted(points, key=int)]
        if len(rows) != 5:
            raise ValueError(f"{path}: {len(rows)} caps, the figure takes 5")
        for p in rows:
            for method in ("listsched", "cpsat"):
                if "latency" not in p[method]:
                    raise ValueError(f"{path}: cap {p['cap']} has no {method} latency ({p[method].get('error')})")
        data[target] = [{"cap": int(p["cap"]),
                         "listsched": {"latency": p["listsched"]["latency"]},
                         "cpsat": {"latency": p["cpsat"]["latency"], "source": p["cpsat"].get("source", "cpsat")}}
                        for p in rows]
    figure_data.write(DATA_PATH, __file__, sources, data)


def coordinates(rows_by_prefix: dict, method: str) -> str:
    return "\n".join(f"({prefix}{k}, {p[method]['latency']:.9f})"
                     for prefix, rows in rows_by_prefix.items() for k, p in enumerate(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extract", action="store_true", help="rewrite the data file from the sweep results first")
    if parser.parse_args().extract:
        extract()
    data = figure_data.read(DATA_PATH)
    rows_by_prefix = {prefix: data[target] for prefix, target, _ in TARGETS}
    latencies = [p[m]["latency"] for rows in rows_by_prefix.values() for p in rows for m in ("listsched", "cpsat")]
    ymin = f"1e{math.floor(math.log10(min(latencies)))}"
    ymax = f"1e{math.ceil(math.log10(max(latencies)))}"

    coords = []
    for i, (prefix, _, _) in enumerate(TARGETS):
        coords += [f"{prefix}{k}" for k in range(5)]
        if i < len(TARGETS) - 1:
            coords.append(f"sp{i}")
    ticks = [c for c in coords if not c.startswith("sp")]
    # in text (Biolinum, \sffamily), not math
    tick_labels = [f"2\\textsuperscript{{{int(p['cap']).bit_length() - 1}}}"
                   for rows in rows_by_prefix.values() for p in rows]
    stand_ins = [f"{label} cap 2^{int(p['cap']).bit_length() - 1}"
                 for (prefix, _, label) in TARGETS for p in rows_by_prefix[prefix]
                 if p["cpsat"].get("source") == "greedy"]
    sources = figure_data.relative(DATA_PATH)
    # a border around each target's middle cap, the one panels (b)-(d) show
    # (plot/motivation_array_usage.py), from the axis to above its taller bar
    middle = MIDDLE_CAP_INDEX
    borders = "\n".join(
        f"\\draw [red, dashed, line width=0.5pt] "
        f"([xshift={{-(\\latencybarshift+0.5*\\latencybarwidth+1pt)}}]axis cs:{prefix}{middle},{ymin}) rectangle "
        f"([xshift={{\\latencybarshift+0.5*\\latencybarwidth+1pt}}]axis cs:{prefix}{middle},"
        f"{max(rows[middle]['listsched']['latency'], rows[middle]['cpsat']['latency']) * 2.5:.9f});"
        for prefix, rows in rows_by_prefix.items())

    tex = f"""% Generated by plot/motivation_latency.py from {sources}.
% CP-SAT points that are greedy's mapping standing in for CP-SAT's: {", ".join(stand_ins) or "none"}.
\\begin{{tikzpicture}}
\\begin{{axis}}
[
    width=\\latencyplotwidth,
    height=\\latencyplotheight,
    ybar=0.2pt,
    enlargelimits=0.03,
    enlarge y limits={{upper, value=0.12}},
    ylabel style={{align=center}},
    y label style={{at={{(-0.09,0.5)}}, font=\\sffamily\\footnotesize}},
    ylabel=Latency (ns),
    legend style={{draw=none, fill=none}},
    bar width=\\latencybarwidth,
    legend columns=2,
    ymode=log,
    log basis y={{10}},
    ymin={ymin}, ymax={ymax},
    ymajorgrids=true,
    grid style=dashed,
    axis x line*=bottom,
    x tick label style={{xshift=0em, yshift=-.2em, rotate=0, anchor=center, font=\\sffamily\\tiny}},
    yminorticks=true,
    legend style={{at={{(0.5,1.05)}}, anchor=south}},
    xlabel={{}},
    symbolic x coords={{{",".join(coords)}}},
    xtick={{{",".join(ticks)}}},
    /pgfplots/latency panel style,
    xticklabels={{{",".join(tick_labels)}}},
]

\\addplot+ [blind_safe_three_scheme_three_colors, fill=blind_safe_three_scheme_three_colors, draw=none, bar shift=-\\latencybarshift] coordinates {{
{coordinates(rows_by_prefix, "listsched")}
}};

\\addplot+ [blind_safe_two_scheme_three_colors, fill=blind_safe_two_scheme_three_colors, draw=none, bar shift=\\latencybarshift] coordinates {{
{coordinates(rows_by_prefix, "cpsat")}
}};

\\legend{{List scheduling, CP-SAT (optimal)}}

% the caps panels (b)-(d) show
{borders}

\\draw [dashed, aluminium4] (axis cs:sp0,{ymin}) -- (axis cs:sp0,{ymax});
\\draw [dashed, aluminium4] (axis cs:sp1,{ymin}) -- (axis cs:sp1,{ymax});

\\coordinate (lbl-fefet1c) at (axis cs:f2,{ymax});
\\coordinate (lbl-hyflexpim) at (axis cs:h2,{ymax});
\\coordinate (lbl-rse) at (axis cs:r2,{ymax});
\\end{{axis}}
\\node [anchor=south, yshift=\\latencytargetlabelshift, font=\\latencytargetlabelfont] at (lbl-fefet1c) {{1F1C}};
\\node [anchor=south, yshift=\\latencytargetlabelshift, font=\\latencytargetlabelfont] at (lbl-hyflexpim) {{HFP}};
\\node [anchor=south, yshift=\\latencytargetlabelshift, font=\\latencytargetlabelfont] at (lbl-rse) {{RSE}};
\\end{{tikzpicture}}
"""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(tex)
    print(f"wrote {OUT_PATH} (y {ymin}..{ymax}; greedy stand-ins: {', '.join(stand_ins) or 'none'})")


if __name__ == "__main__":
    main()
