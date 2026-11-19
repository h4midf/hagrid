"""Monte Carlo ENOB of the full SAR ADC (buffer + bootstrapped switch + core) with
device mismatch and transient noise. Same coherent sine test as run_enob.py.

Mismatch (drawn in Python, per instance, reproducible from the instance seed):
  - CDAC: unit capacitor sigma(dC/C) = A_C / sqrt(C_unit), A_C = 0.85 %*sqrt(fF)
    (Tripathi & Murmann, TCAS-I 2014, measured 32 nm SOI MOM). A capacitor of n
    units gets n*cu*(1 + N(0, sigma/sqrt(n))).
  - Comparator: every transistor gets delvto ~ N(0, A_VT / sqrt(2*W*L)),
    A_VT = 2.5 mV*um (Pelgrom pair definition; Kuhn et al., TED 2011, Fig. 25,
    HiK/MG typical). The comparator is inlined with per-device delvto so
    sar_core.lib stays untouched.
Noise (seeded piecewise-constant sources, drawn in Python per instance):
  - input-referred sampling noise in series with the buffer output:
    buffer 0.55 mV (ngspice .noise, worst case) and kT/C 0.13 mV -> 0.565 mV rms,
    one value per conversion
  - comparator noise in series with its VCM input, 0.4 mV rms (typical StrongARM,
    no paper), new value every 50 ns clock cycle (each comparison sees a fresh sample)

Results append to work/mc/results.jsonl; finished instances are skipped on rerun.

Usage: python3 targets/1fefet1c/spice/adc/run_mc.py [n_instances] [workers] [N] [M]
       python3 targets/1fefet1c/spice/adc/run_mc.py --test      (short checks)
       python3 targets/1fefet1c/spice/adc/run_mc.py --summary   (rewrite results/adc_mc.json)
"""
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from tb import FS, LALENI, T_CLK, T_CONV, save_result, testbench

HERE = Path(__file__).resolve().parent
WORK = HERE / "work" / "mc"
VMID, VAMP = 0.25, 0.24
IBUF = 0.7e-6
CU = 2e-15

A_C = 0.0085          # 0.85 % * sqrt(fF)
A_VT = 2.5e-3 * 1e-6  # 2.5 mV*um, in V*m
V_N_SAMPLE = 0.565e-3
V_N_CMP = 0.4e-3

# comparator devices: name, drain, gate, source, bulk, type, W, L (as in sar_core.lib)
CMP_DEVICES = [
    ("Mtail", "t", "clkcb", "vdd", "vdd", "pmos", 2e-6, 32e-9),
    ("M1", "p", "inp", "t", "vdd", "pmos", 2e-6, 64e-9),
    ("M2", "q", "inn", "t", "vdd", "pmos", 2e-6, 64e-9),
    ("M3", "outn", "outp", "p", "vdd", "pmos", 1e-6, 32e-9),
    ("M4", "outp", "outn", "q", "vdd", "pmos", 1e-6, 32e-9),
    ("M5", "outn", "outp", "0", "0", "nmos", 500e-9, 32e-9),
    ("M6", "outp", "outn", "0", "0", "nmos", 500e-9, 32e-9),
    ("M7", "outn", "clkcb", "0", "0", "nmos", 250e-9, 32e-9),
    ("M8", "outp", "clkcb", "0", "0", "nmos", 250e-9, 32e-9),
    ("M9", "p", "clkcb", "0", "0", "nmos", 250e-9, 32e-9),
    ("M10", "q", "clkcb", "0", "0", "nmos", 250e-9, 32e-9),
]
CAP_UNITS = {"C6": 64, "C5": 32, "C4": 16, "C3": 8, "C2": 4, "C1": 2, "C0": 1, "CD": 1}


def draw(seed: int, mismatch: bool) -> dict:
    rng = np.random.default_rng(seed)
    sig_u = A_C / np.sqrt(CU * 1e15)
    caps = {k: n * CU * (1 + (rng.normal(0, sig_u / np.sqrt(n)) if mismatch else 0.0))
            for k, n in CAP_UNITS.items()}
    dvt = {d[0]: (rng.normal(0, A_VT / np.sqrt(2 * d[6] * d[7])) if mismatch else 0.0)
           for d in CMP_DEVICES}
    return {"caps": caps, "dvt": dvt}


def comparator_mc(dvt: dict) -> str:
    lines = [".subckt comparator_mc inp inn clkc cmp vdd", "Xic clkc clkcb vdd inv"]
    for name, d, g, s, b, typ, w, l in CMP_DEVICES:
        lines.append(f"{name} {d} {g} {s} {b} {typ} W={w:.4g} L={l:.4g} delvto={dvt[name]:.6e}")
    lines += ["Xn1 outp cmpb cmp vdd nor2", "Xn2 outn cmp cmpb vdd nor2", ".ends"]
    return "\n".join(lines)


def build(seed, n, m, mismatch=True, noise=True, offset_test=0.0):
    fin = m / n * FS
    t_sample0 = T_CONV + 51e-9
    t_end = (n + 1) * T_CONV + 50e-9
    p = draw(seed, mismatch)
    if offset_test:
        p["dvt"]["M1"] += offset_test
    net = testbench("sar_full_tb.sp")
    net = net.replace(".param ibuf=1u", f".param ibuf={IBUF}\n{comparator_mc(p['dvt'])}")
    phase_deg = (-360.0 * fin * t_sample0) % 360.0
    net = net.replace("VBL  bl   0 PWL(__PWL__)",
                      f"VBL  bl   0 SIN({VMID} {VAMP} {fin:.6e} 0 0 {phase_deg:.6f})")
    # mismatched CDAC
    for k, c in p["caps"].items():
        net = re.sub(rf"^{k} (\S+) (\S+) \{{[^}}]*\}}", rf"{k} \1 \2 {c:.6e}", net, flags=re.M)
    # noise sources: buffer output -> vbufn -> bootstrapped switch; comparator VCM input
    # Noise as seeded piecewise-constant PWL sources (ngspice trnoise aborts when its
    # steps coincide with clock edges). Values change midway between clock edges:
    #   sampling noise: one value per conversion, switched 4.75 clock cycles into the
    #     conversion (switch is off then; held through the next sample phase)
    #   comparator noise: one value per clock cycle, switched a quarter cycle after the
    #     rising edge
    if noise:
        rng = np.random.default_rng(10_000 + seed)

        def steps(period, offset, sigma):
            pts, v, k = ["0 0"], 0.0, 0
            while offset + k * period < t_end:
                t = offset + k * period
                v_new = rng.normal(0, sigma)
                pts += [f"{t:.6e} {v:.6e}", f"{t + 0.5e-9:.6e} {v_new:.6e}"]
                v, k = v_new, k + 1
            return "PWL(" + " ".join(pts) + ")"

        # step times must not coincide with clock edges (multiples of half a cycle) or
        # with each other: comparator steps at 1/4 cycle, sampling steps at 3/4 cycle
        src_s = steps(T_CONV, 4.75 * T_CLK, V_N_SAMPLE)
        src_c = steps(T_CLK, 0.25 * T_CLK, V_N_CMP)
    else:
        src_s = src_c = "0"
    net = net.replace("Xbs vbuf vin samp sampb vdd bootsw",
                      f"VNS vbufn vbuf {src_s}\n"
                      "Xbs vbufn vin samp sampb vdd bootsw")
    net = net.replace("Xcmp top vcm clkc cmp vdd comparator",
                      f"VNC vcmn vcm {src_c}\n"
                      "Xcmp top vcmn clkc cmp vdd comparator_mc")
    meas = [f"meas tran code{k} find code at={(k + 1) * T_CONV + 40e-9:.5e}" for k in range(1, n + 1)]
    net = net.replace("__TEND__", f"{t_end:.5e}").replace("__MEAS__", "\n".join(meas))
    # keep only the bit outputs in memory: saving every node needs ~30 GB per 64-sample run
    net = net.replace(".control\nrun", ".save v(b6) v(b5) v(b4) v(b3) v(b2) v(b1) v(b0)\n"
                      f".control\nset rndseed={seed + 1}\nrun")
    return net, p


def enob(codes, m):
    spec = np.abs(np.fft.rfft(codes - codes.mean())) ** 2
    return (10 * np.log10(spec[m] / (spec[1:].sum() - spec[m])) - 1.76) / 6.02


def run_one(seed, n, m, tag, **kw):
    WORK.mkdir(parents=True, exist_ok=True)
    net, p = build(seed, n, m, **kw)
    path = WORK / f"{tag}_s{seed}_N{n}_M{m}.sp"
    path.write_text(net)
    o = subprocess.run(["ngspice", "-b", str(path)], capture_output=True, text=True)
    out = o.stdout
    (WORK / (path.stem + ".log")).write_text(o.stdout + "\n--- stderr ---\n" + o.stderr)
    codes = []
    for k in range(1, n + 1):
        hit = re.search(rf"^code{k}\s*=\s*([-+\d.eE]+)", out, re.M)
        if not hit:
            raise RuntimeError(f"seed {seed}: code{k} missing, see {path.stem}.log")
        codes.append(float(hit.group(1)))
    codes = np.array(codes)
    sig_off = p["dvt"]["M1"] - p["dvt"]["M2"]
    return {"seed": seed, "N": n, "M": m, "enob": float(enob(codes, m)),
            "codes": codes.astype(int).tolist(), "dvt_M1_minus_M2_mV": sig_off * 1e3,
            "cap_err_pct": {k: (c / (CAP_UNITS[k] * CU) - 1) * 100 for k, c in p["caps"].items()}}


def test():
    n, m = 16, 3
    print("check 1: nominal (no mismatch, no noise), should match run_enob-style codes")
    a = run_one(0, n, m, "test_nom", mismatch=False, noise=False)
    print("   codes", a["codes"])
    print("check 2: +20 mV on M1 only (~5 LSB offset), codes should shift by ~5")
    b = run_one(0, n, m, "test_off", mismatch=False, noise=False, offset_test=20e-3)
    print("   codes", b["codes"], " mean shift", np.mean(np.array(b["codes"]) - a["codes"]))
    print("check 3: one random instance, mismatch + noise")
    c = run_one(1, n, m, "test_mc")
    print("   codes", c["codes"], " diff vs nominal", (np.array(c["codes"]) - a["codes"]).tolist(),
          f" M1-M2 dVt {c['dvt_M1_minus_M2_mV']:+.2f} mV")


def main():
    n_inst = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    m = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    res_file = WORK / "results.jsonl"
    WORK.mkdir(parents=True, exist_ok=True)
    done = set()
    if res_file.exists():
        for line in res_file.read_text().splitlines():
            r = json.loads(line)
            if r["N"] == n and r["M"] == m:
                done.add(r["seed"])
    todo = [s for s in range(1, n_inst + 1) if s not in done]
    print(f"{len(done)} done, {len(todo)} to run, {workers} workers", flush=True)

    def job(seed):
        try:
            r = run_one(seed, n, m, "mc")
        except RuntimeError as err:  # one failed instance must not stop the batch
            print(f"seed {seed}: FAILED ({err})", flush=True)
            return None
        with res_file.open("a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"seed {seed}: ENOB {r['enob']:.3f}, comparator dVt(M1-M2) {r['dvt_M1_minus_M2_mV']:+.2f} mV",
              flush=True)
        return r

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(job, todo))
    summarize(n, m)


def summarize(n=64, m=7):
    """Summary of work/mc/results.jsonl -> results/adc_mc.json ("summary")."""
    allr = [json.loads(l) for l in (WORK / "results.jsonl").read_text().splitlines()]
    allr = [r for r in allr if r["N"] == n and r["M"] == m]
    e = np.array([r["enob"] for r in allr])
    path = save_result("adc_mc", "summary", {
        "instances": len(e), "samples": n, "cycles_in_record": m,
        "enob_mean": float(e.mean()), "enob_std": float(e.std(ddof=1)),
        "enob_min": float(e.min()), "enob_max": float(e.max()),
        "enob_by_seed": {str(r["seed"]): r["enob"] for r in sorted(allr, key=lambda r: r["seed"])},
        "comparator_offset_mV_by_seed": {str(r["seed"]): r["dvt_M1_minus_M2_mV"]
                                         for r in sorted(allr, key=lambda r: r["seed"])},
        "mismatch": {"A_C_pct_sqrt_fF": A_C * 100, "A_VT_mV_um": A_VT * 1e9, "unit_cap_fF": CU * 1e15,
                     "sources": "A_C: Tripathi & Murmann, TCAS-I 2014 (32 nm SOI MOM); "
                                "A_VT: Kuhn et al., TED 2011, Fig. 25 (HiK/MG typical)"},
        "noise": {"sampling_rms_mV": V_N_SAMPLE * 1e3, "comparator_rms_mV": V_N_CMP * 1e3,
                  "sources": "sampling: buffer .noise (worst case 0.55 mV) + kT/C 0.13 mV; "
                             "comparator: typical StrongARM value, no paper"},
        "buffer_bias_uA": IBUF * 1e6, "raw": "spice/adc/work/mc/results.jsonl",
        "laleni_enob_bits": LALENI["adc_enob_bits"]})
    print(f"ENOB over {len(e)} instances: mean {e.mean():.3f}, std {e.std(ddof=1):.3f}, "
          f"min {e.min():.3f}, max {e.max():.3f}  (Laleni {LALENI['adc_enob_bits']}) -> {path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test()
    elif len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summarize()
    else:
        main()
