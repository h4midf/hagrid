"""Dynamic power of the SRAM macro's logic on real data: the same gate-level method as
power_sram_logic.py, with the macro fed real bit streams instead of random bits. Writes
results/sram_logic_power_real.json.

Data, from the int8 ResNet-50 (kernels/resnet50.py) on a real ImageNet image, at the 3x3
convs after a ReLU:
  inputs   real activations of one output position (im2col), cut to 5 bits (the paper's
           ResNet-18 precision, Table III), one macro pass per bit plane
  weights  a proxy: the paper's trained correction weights are not available. For 16 output
           channels, the 42 input positions of largest weight norm (what pruning would
           keep), quantized to 3 bits signed over the layer's range

Each pass's mapped-netlist output is checked against the Python sum of weight x input bit.

The netlist is scaling.sram_logic_rtl(): with clock_gated (design.json), each register bank
is clocked through a clock gate on its enable, so the simulated clock pins count only the
cycles a bank is enabled.

Usage (from the repo root, after power_sram_logic.py):
  python3 targets/rram-sram-ensemble/periphery/power_sram_logic_real.py
"""
import json
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import torch
from torchvision.models import ResNet50_Weights

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..", "..")
sys.path[:0] = [REPO, HERE]
from kernels.resnet50 import ResNet50Int8  # noqa: E402
from run_scripts.measure_accuracy.build_resnet50_int8 import load_images  # noqa: E402

from power_sram_logic import CYCLES, LIB, PASS_END, PASS_START, mapped_netlist, toggle_energy  # noqa: E402
from scaling import DESIGN, factors  # noqa: E402

LAYERS = ("stage1.1.conv2", "stage2.1.conv2", "stage3.1.conv2", "stage4.1.conv2")
POSITIONS = 2
ACT_BITS = 5
W, ACC, NW, PES = 3, 9, 42, 16
IMAGE_SEED = 7
CAPTURE_TIMEOUT_S = 600

TB = """module tb(input clk, output [143:0] out_flat);
    reg [9:0] t = 0;
    wire rst = (t < 2);
    wire in_we = (t >= 2) && (t < 66);
    wire [5:0] in_addr = t - 10'd2;
    wire start = (t == 67);
    reg [0:0] in_rom [0:63];
    reg [15:0] w_rom [0:127];
    initial begin
        $readmemh("in.hex", in_rom);
        $readmemh("w.hex", w_rom);
    end
    wire [6:0] addr;
    wire ren, done;
    rse_sram_logic dut (.clk(clk), .rst(rst), .start(start), .in_we(in_we), .in_addr(in_addr),
                        .in_bit(in_rom[in_addr]), .sram_rdata(w_rom[addr]), .sram_addr(addr), .sram_ren(ren),
                        .done(done), .out_flat(out_flat));
    always @(posedge clk) t <= t + 10'd1;
endmodule
"""


def capture() -> dict:
    """capture_in_process, run in a child process that is terminated once it has sent its
    result: the dataset library's streaming thread pool can deadlock when it shuts down."""
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_capture_child, args=(queue,))
    proc.start()
    try:
        kind, payload = queue.get(timeout=CAPTURE_TIMEOUT_S)
    finally:
        proc.terminate()
        proc.join()
    if kind == "error":
        raise RuntimeError(f"capture failed:\n{payload}")
    return payload


def _capture_child(queue) -> None:
    try:
        queue.put(("ok", capture_in_process()))
    except Exception:
        queue.put(("error", traceback.format_exc()))


def capture_in_process() -> dict:
    """{layer: (int8 weights [out, in * 9], uint8 im2col activations [in * 9, positions])}."""
    model = ResNet50Int8()._quantized
    modules = dict(model.named_modules())
    inputs = {}
    handles = [modules[n].register_forward_pre_hook(lambda _m, a, n=n: inputs.__setitem__(n, a[0])) for n in LAYERS]
    images, _ = load_images(1, IMAGE_SEED, ResNet50_Weights.IMAGENET1K_V2)
    with torch.no_grad():
        model(images[0])
    for h in handles:
        h.remove()
    out = {}
    for n in LAYERS:
        x = inputs[n]
        if x.q_zero_point() != 0:
            raise RuntimeError(f"{n}: input zero point {x.q_zero_point()}, expected a post-ReLU input")
        cols = torch.nn.functional.unfold(x.int_repr().float(), 3, padding=1)[0].numpy().astype(np.uint8)
        w = modules[n].weight().int_repr().numpy().astype(np.int32)
        out[n] = (w.reshape(w.shape[0], -1), cols)
    return out


def proxy_weights(w: np.ndarray) -> tuple:
    """(kept input indices [NW], 3-bit signed weights [NW, PES]) of the first 16 channels."""
    block = w[:PES]
    keep = np.argsort(-np.linalg.norm(block, axis=0))[:NW]
    scale = np.abs(w).max() / 3
    return keep, np.clip(np.round(block[:, keep].T / scale), -4, 3).astype(int)


def words(q: np.ndarray) -> list:
    """SRAM words: address k * W + b, bit p = bit b (MSB first) of weight k of PE p."""
    out = []
    for k in range(NW):
        for b in range(W):
            word = 0
            for p in range(PES):
                word |= (((int(q[k, p]) & 7) >> (W - 1 - b)) & 1) << p
            out.append(word)
    return out + [0] * (128 - len(out))


def run_pass(mapped: str, tmp: str, q: np.ndarray, x_bits: np.ndarray) -> tuple:
    with open(os.path.join(tmp, "in.hex"), "w") as f:
        f.write("\n".join(str(int(b)) for b in list(x_bits) + [0] * (64 - NW)))
    with open(os.path.join(tmp, "w.hex"), "w") as f:
        f.write("\n".join(f"{v:04x}" for v in words(q)))
    with open(os.path.join(tmp, "tb.v"), "w") as f:
        f.write(TB)
    script = (f"read_liberty -ignore_miss_func {LIB}; read_verilog {mapped}; read_verilog -sv tb.v; "
              f"hierarchy -top tb; proc; memory; flatten; sim -clock clk -zinit -n {CYCLES} -vcd sim.vcd")
    subprocess.run(["yosys", "-q", "-p", script], cwd=tmp, check=True)
    vcd_path = os.path.join(tmp, "sim.vcd")
    vcd = open(vcd_path).read()
    ident = re.search(r"\$var \w+ \d+ (\S+) out_flat ", vcd).group(1)
    last = re.findall(rf"^b([01x]+) {re.escape(ident)}$", vcd, flags=re.M)[-1]
    if "x" in last:
        raise RuntimeError("out_flat is undefined")
    got = int(last, 2)
    for p in range(PES):
        want = int((q[:, p] * x_bits).sum()) & ((1 << ACC) - 1)
        if (got >> (p * ACC)) & ((1 << ACC) - 1) != want:
            raise RuntimeError(f"PE {p}: mapped netlist gives the wrong sum")
    return toggle_energy(mapped, vcd_path)


def main() -> None:
    data = capture()
    rng = np.random.default_rng(IMAGE_SEED)
    samples, logic_sum, clock_sum, cycles_sum = [], 0.0, 0.0, 0
    with tempfile.TemporaryDirectory() as tmp:
        mapped = os.path.join(tmp, "mapped.v")
        mapped_netlist(mapped)
        for name, (w, cols) in data.items():
            keep, q = proxy_weights(w)
            w_bits = np.mean([bin(int(v) & 7).count("1") / W for v in q.ravel()])
            for pos in rng.integers(0, cols.shape[1], POSITIONS):
                acts = cols[keep, pos] >> (8 - ACT_BITS)
                for plane in range(ACT_BITS):
                    x_bits = (acts >> (ACT_BITS - 1 - plane)) & 1
                    logic, clock, cycles = run_pass(mapped, tmp, q, x_bits.astype(int))
                    logic_sum, clock_sum, cycles_sum = logic_sum + logic, clock_sum + clock, cycles_sum + cycles
                    samples.append({"layer": name, "plane": plane, "input_density": float(x_bits.mean()),
                                    "weight_bit_density": float(w_bits), "logic_pJ_per_cycle": logic / cycles})
            print(f"  {name}: weight bit density {w_bits:.3f}, zero weights {np.mean(q == 0):.2f}", flush=True)

    f = factors()
    clock_GHz = DESIGN["clock"]["freq_MHz"] * 1e-3
    logic_mW = logic_sum / cycles_sum * f["energy"] * clock_GHz
    clock_mW = clock_sum / cycles_sum * f["energy"] * clock_GHz
    leakage = json.load(open(os.path.join(HERE, "results", "sram_logic.json")))[0]["chip_65nm"]["leakage_mW"]
    density = float(np.mean([s["input_density"] for s in samples]))
    out = {"passes": len(samples), "mean_input_density": density, "logic_mW": logic_mW, "clock_mW": clock_mW,
           "clock_gated": DESIGN["model_assumptions"]["sram_logic"]["clock_gated"],
           "dynamic_mW": logic_mW + clock_mW, "leakage_mW": leakage, "power_mW": logic_mW + clock_mW + leakage,
           "samples": samples}
    path = os.path.join(HERE, "results", "sram_logic_power_real.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"{len(samples)} passes correct; mean input bit density {density:.3f}")
    print(f"dynamic {logic_mW + clock_mW:.4f} mW (logic {logic_mW:.4f} + clock {clock_mW:.4f}); leakage {leakage:.4f}; "
          f"total {out['power_mW']:.4f} mW")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
