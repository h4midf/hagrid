"""bert-base's variants on 1F1C with 64 x 64 and 128 x 128 arrays, two plots: energy per
inference vs latency, and chip area vs latency; one point per variant (each at its
smallest array cap), marker = array size, color = the variant's pseudo-perplexity (the
same color scale in both).

The variants are those with masked-LM pseudo-perplexity below the sweep's --ppl-max
(run_scripts/fefet1c_bert_base_variant_sweep.py; its results under
run_scripts/results/fefet1c_bert_base_variants/<config>/<variant>.json).

Output goes directly to paper/submission/figures/evaluation/dse/.

Run from the repo root:
    python3 plot/fefet1c_bert_base_variants.py
Writes fefet1c_bert_base_variants_energy.pdf and _area.pdf in that folder.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "run_scripts/results/fefet1c_bert_base_variants"
OUT_DIR = ROOT / "paper/submission/figures/evaluation/dse"
PPL_MAX = 14.4   # the sweep's --ppl-max; every variant read is checked to be below it
CONFIGS = (("64", r"$64\times64$", "o"), ("128", r"$128\times128$", "^"))
PANELS = (("energy_mJ", "Energy per inference (mJ)", "energy"), ("area_mm2", r"Area (mm$^2$)", "area"))


def load(config: str) -> list[dict]:
    paths = sorted((RESULTS / config).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no results in {RESULTS / config}: run the variant sweep first")
    rows = []
    for path in paths:
        r = json.loads(path.read_text())
        if r["target"] != f"1fefet1c/{config}":
            raise ValueError(f"{path} is for {r['target']}, not 1fefet1c/{config}")
        if r["ppl"] >= PPL_MAX:
            raise ValueError(f"{path}: ppl {r['ppl']} is not below {PPL_MAX}")
        rows.append(r)
    return rows


def main() -> None:
    data = {config: load(config) for config, _, _ in CONFIGS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Linux Biolinum (the libertine package's sans face) for all text, and through
    # mathastext for the letters and digits in math too (tick labels, 64 x 64)
    with plt.rc_context({"font.family": "sans-serif", "font.size": 20,
                         "text.usetex": True,
                         "text.latex.preamble": (
                             r"\usepackage[T1]{fontenc}"
                             r"\usepackage[tt=false,type1=true]{libertine}"
                             r"\usepackage[libertine]{newtxmath}"
                             r"\renewcommand{\familydefault}{\sfdefault}"
                             r"\usepackage{mathastext}"
                         )}):
        ppls = [r["ppl"] for rows in data.values() for r in rows]
        norm = Normalize(vmin=min(ppls), vmax=max(ppls))
        handles = [Line2D([], [], linestyle="none", marker=marker, color="0.35", markersize=9,
                          label=label) for _, label, marker in CONFIGS]
        for key, ylabel, name in PANELS:
            fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=150)
            for config, label, marker in CONFIGS:
                pts = data[config]
                sc = ax.scatter([r["latency_ns"] / 1e6 for r in pts], [r[key] for r in pts],
                                c=[r["ppl"] for r in pts], cmap="viridis", norm=norm,
                                s=64, marker=marker, alpha=0.8, linewidths=0, label=label)
            ax.set_xscale("log")
            ax.set_xlabel("Latency (ms)")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="major", linestyle=":", color="0.75")
            ax.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
            fig.colorbar(sc, ax=ax, label="Pseudo-perplexity" if name == "energy" else "", pad=0.02)
            out = OUT_DIR / f"fefet1c_bert_base_variants_{name}.pdf"
            fig.savefig(out, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
