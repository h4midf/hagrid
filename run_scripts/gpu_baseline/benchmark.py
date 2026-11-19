"""GPU baseline for the CIM comparison: latency and energy of the kernels on one GPU,
batch 1, FP32 and INT8.

Configurations, per model:
  torch_fp32  PyTorch eager, strict FP32 (TF32 off), cuDNN autotuning on (reference)
  trt_fp32    TensorRT engine, strict FP32 (TF32 off)
  trt_int8    TensorRT engine, INT8 (implicit quantization, random calibration data) with
              strict-FP32 fallback for layers that have no INT8 implementation

Workloads are the float modules of kernels/*.py: the structure the compiler maps (the
registry's *Int8 classes wrap these modules with CPU-only int8 checkpoints). Batch 1;
BERT encoder, 512 tokens; GPT-2 decoder stack and ln_f, one 1024-token causal pass (no
embeddings or LM head); ViT encoder, 197 tokens; CNNs, 224 x 224. Weights are random:
latency and energy do not depend on their values. The INT8 calibration data is random
too, which fixes the quantization scales but not the speed.

Phases:
  build    export ONNX and build the TensorRT engines, cached in
           results/gpu_baseline/trt_engines/ (not committed)
  measure  per model and configuration, `repeats` times: idle power, warm-up, latency
           (one inference at a time: CUDA events around it, synchronize after it), energy
           (inferences back to back for `energy_s` seconds, NVML total-energy counter).
           A monitor thread samples NVML every 0.2 s: other processes on the GPU, SM
           clock, temperature, power, and the reasons the clock is below maximum
           (power cap, thermal, ...). Clocks are not locked: GPU Boost, the board's
           default power cap. A repeat is valid only if no other compute process was on
           the GPU during it (the desktop's graphics processes are recorded, not
           counted); an invalid repeat is retried.

Usage (on the GPU machine, from the repo root):
  python run_scripts/gpu_baseline/benchmark.py build   [--models M ...]
  python run_scripts/gpu_baseline/benchmark.py measure [--models M ...] [--quick] [--cpus 8-11]
Writes run_scripts/results/gpu_baseline/build_<host>.json and benchmark_<host>.json
(rewritten after every model, so a partial run keeps what it measured).
"""
import argparse
import importlib
import json
import os
import platform
import socket
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pynvml
import tensorrt as trt
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
RESULTS = REPO / "run_scripts" / "results" / "gpu_baseline"
ENGINES = RESULTS / "trt_engines"
HOST = socket.gethostname()

# model -> (kernels module, float class)
SPECS = {
    "gpt2": ("gpt2", "GPT2Kernel"),
    "bert-large": ("bert_large", "BertLargeKernel"),
    "bert-base": ("bert_base", "BertBaseKernel"),
    "vit-large": ("vit", "ViTEncoder"),
    "vit-base": ("vit_base", "ViTEncoder"),
    "vgg16": ("vgg16", "VGG16Kernel"),
    "resnet50": ("resnet50", "ResNet50Kernel"),
    "mobilenet-v2": ("mobilenet_v2", "MobileNetV2Kernel"),
}
CONFIGS = ("torch_fp32", "trt_fp32", "trt_int8")
FULL = {"idle_s": 10.0, "warmup_s": 10.0, "latency_runs": 500, "energy_s": 20.0, "repeats": 3, "retries": 2}
QUICK = {"idle_s": 2.0, "warmup_s": 2.0, "latency_runs": 50, "energy_s": 3.0, "repeats": 1, "retries": 0}
CALIBRATION_BATCHES = 16
LOGGER = trt.Logger(trt.Logger.WARNING)
# NVML clock event reasons (why the GPU runs below its maximum clock)
CLOCK_EVENT_REASONS = {0x1: "gpu_idle", 0x2: "applications_clocks", 0x4: "sw_power_cap",
                       0x8: "hw_slowdown", 0x10: "sync_boost", 0x20: "sw_thermal_slowdown",
                       0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake", 0x100: "display_clock"}


def strict_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True


def load_model(name):
    modname, cls = SPECS[name]
    mod = importlib.import_module(f"kernels.{modname}")
    shape = tuple(mod.INPUT_SHAPE) if hasattr(mod, "INPUT_SHAPE") else (mod.MAX_SEQ_LEN, mod.DIM)
    torch.manual_seed(0)
    return getattr(mod, cls)().eval(), (mod.BATCH_SIZE, *shape)


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


# ---------------------------------------------------------------- build

class RandomCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, shape, batches):
        super().__init__()
        self.shape, self.left = shape, batches
        self.gen = torch.Generator().manual_seed(1)
        self.buf = torch.empty(shape, device="cuda")

    def get_batch_size(self):
        return self.shape[0]

    def get_batch(self, names):
        if self.left == 0:
            return None
        self.left -= 1
        self.buf.copy_(torch.randn(self.shape, generator=self.gen))
        return [int(self.buf.data_ptr())]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass


def engine_summary(serialized):
    engine = trt.Runtime(LOGGER).deserialize_cuda_engine(serialized)
    info = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    text = json.dumps(info)
    return {"layers": len(info.get("Layers", [])), "mentions_Int8": text.count("Int8"),
            "mentions_Half": text.count("Half"), "mentions_Float": text.count("Float")}


def build_engine(onnx_path, shape, mode):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.clear_flag(trt.BuilderFlag.TF32)
    if mode == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = RandomCalibrator(shape, CALIBRATION_BATCHES)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"engine build failed: {onnx_path} {mode}")
    return bytes(serialized), time.time() - t0


def phase_build(models, rebuild):
    ENGINES.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"build_{HOST}.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report["environment"] = environment()
    report.setdefault("models", {})
    for name in models:
        model, shape = load_model(name)
        onnx_path = ENGINES / f"{name}.onnx"
        row = {"input": list(shape)}
        if rebuild or not onnx_path.exists():
            t0 = time.time()
            torch.onnx.export(model, torch.randn(shape), str(onnx_path), opset_version=17,
                              input_names=["x"], output_names=["y"], do_constant_folding=True)
            row["onnx_export_s"] = round(time.time() - t0, 1)
        for mode in ("fp32", "int8"):
            plan = ENGINES / f"{name}_{mode}.plan"
            if rebuild or not plan.exists():
                serialized, build_s = build_engine(onnx_path, shape, mode)
                plan.write_bytes(serialized)
                row[f"{mode}_build_s"] = round(build_s, 1)
            row[f"{mode}_engine"] = engine_summary(plan.read_bytes())
            row[f"{mode}_engine_MB"] = round(plan.stat().st_size / 2**20, 1)
        report["models"][name] = row
        write_json(path, report)
        print(json.dumps({name: row}), flush=True)
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- measure

class Monitor(threading.Thread):
    """Samples NVML every `period` s. Other compute processes (someone else's CUDA work)
    make a repeat invalid; graphics processes (the desktop: Xorg, gnome-shell) are only
    recorded, their small constant draw is part of the measured idle power."""

    def __init__(self, handle, period=0.2):
        super().__init__(daemon=True)
        self.h, self.period, self.own = handle, period, os.getpid()
        self.stop_event = threading.Event()
        self.foreign, self.graphics, self.sm, self.temp, self.power, self.util = set(), set(), [], [], [], []
        self.reasons = {}  # clock event reason -> samples it was active in
        self.get_reasons = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons")

    def run(self):
        while not self.stop_event.is_set():
            try:
                compute = pynvml.nvmlDeviceGetComputeRunningProcesses(self.h)
                self.foreign.update(p.pid for p in compute if p.pid != self.own)
                graphics = pynvml.nvmlDeviceGetGraphicsRunningProcesses(self.h)
                self.graphics.update(p.pid for p in graphics if p.pid != self.own)
                self.sm.append(pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM))
                self.temp.append(pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU))
                self.power.append(pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000)
                self.util.append(pynvml.nvmlDeviceGetUtilizationRates(self.h).gpu)
                bits = self.get_reasons(self.h)
                for bit, label in CLOCK_EVENT_REASONS.items():
                    if bits & bit:
                        self.reasons[label] = self.reasons.get(label, 0) + 1
            except pynvml.NVMLError:
                pass
            self.stop_event.wait(self.period)

    def finish(self):
        self.stop_event.set()
        self.join()

        def span(v):
            return {"min": min(v), "median": statistics.median(v), "max": max(v)} if v else None
        return {"foreign_pids": sorted(self.foreign), "graphics_pids": sorted(self.graphics),
                "sm_clock_MHz": span(self.sm),
                "temperature_C": span(self.temp), "power_W": span(self.power), "util_pct": span(self.util),
                "samples": len(self.sm), "clock_event_reasons": self.reasons}


class TorchRunner:
    def __init__(self, name, stream):
        model, shape = load_model(name)
        self.model = model.to("cuda")
        self.x = torch.randn(shape, device="cuda")
        self.stream = stream

    def step(self):
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            self.model(self.x)


class TrtRunner:
    def __init__(self, name, mode, stream):
        plan = ENGINES / f"{name}_{mode}.plan"
        if not plan.exists():
            raise FileNotFoundError(f"{plan}: run the build phase first")
        self.engine = trt.Runtime(LOGGER).deserialize_cuda_engine(plan.read_bytes())
        self.ctx = self.engine.create_execution_context()
        to_torch = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int8: torch.int8}
        self.buffers = []
        for i in range(self.engine.num_io_tensors):
            tname = self.engine.get_tensor_name(i)
            buf = torch.randn(tuple(self.engine.get_tensor_shape(tname)), device="cuda").to(
                to_torch[self.engine.get_tensor_dtype(tname)])
            self.ctx.set_tensor_address(tname, buf.data_ptr())
            self.buffers.append(buf)
        self.stream = stream

    def step(self):
        self.ctx.execute_async_v3(self.stream.cuda_stream)


def energy_mJ(handle):
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)


def one_repeat(runner, handle, p):
    stream = runner.stream
    monitor = Monitor(handle)
    monitor.start()
    out = {}
    # idle power
    torch.cuda.synchronize()
    e0, t0 = energy_mJ(handle), time.perf_counter()
    time.sleep(p["idle_s"])
    e1, t1 = energy_mJ(handle), time.perf_counter()
    idle_W = (e1 - e0) / 1000 / (t1 - t0)
    out["idle_W"] = round(idle_W, 2)
    # warm-up
    t0, n = time.perf_counter(), 0
    while time.perf_counter() - t0 < p["warmup_s"]:
        runner.step()
        n += 1
        if n % 32 == 0:
            stream.synchronize()
    stream.synchronize()
    # latency: one inference at a time
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(p["latency_runs"])]
    wall = []
    for start, end in ev:
        w0 = time.perf_counter()
        start.record(stream)
        runner.step()
        end.record(stream)
        end.synchronize()
        wall.append((time.perf_counter() - w0) * 1000)
    gpu = sorted(s.elapsed_time(e) for s, e in ev)
    wall.sort()

    def pct(v, q):
        return v[min(len(v) - 1, int(q * len(v)))]
    out["latency_ms"] = {"median": round(statistics.median(gpu), 4), "p10": round(pct(gpu, 0.1), 4),
                         "p90": round(pct(gpu, 0.9), 4), "mean": round(statistics.fmean(gpu), 4)}
    out["wall_latency_ms"] = {"median": round(statistics.median(wall), 4), "p90": round(pct(wall, 0.9), 4)}
    # energy: back to back
    stream.synchronize()
    e0, t0, n = energy_mJ(handle), time.perf_counter(), 0
    while time.perf_counter() - t0 < p["energy_s"]:
        runner.step()
        n += 1
        if n % 32 == 0:
            stream.synchronize()
    stream.synchronize()
    e1, t1 = energy_mJ(handle), time.perf_counter()
    total_mJ, secs = e1 - e0, t1 - t0
    out["energy"] = {
        "inferences": n, "seconds": round(secs, 3), "throughput_per_s": round(n / secs, 2),
        "avg_power_W": round(total_mJ / 1000 / secs, 2),
        "mJ_per_inference": round(total_mJ / n, 4),
        "dynamic_mJ_per_inference": round((total_mJ - idle_W * secs * 1000) / n, 4),
    }
    out["monitor"] = monitor.finish()
    out["valid"] = not out["monitor"]["foreign_pids"]
    return out


def summarize(repeats):
    valid = [r for r in repeats if r["valid"]]
    if not valid:
        return {"valid_repeats": 0}

    def med(f):
        return round(statistics.median(f(r) for r in valid), 4)
    return {
        "valid_repeats": len(valid),
        "latency_ms_median": med(lambda r: r["latency_ms"]["median"]),
        "latency_ms_p90": med(lambda r: r["latency_ms"]["p90"]),
        "wall_latency_ms_median": med(lambda r: r["wall_latency_ms"]["median"]),
        "mJ_per_inference": med(lambda r: r["energy"]["mJ_per_inference"]),
        "dynamic_mJ_per_inference": med(lambda r: r["energy"]["dynamic_mJ_per_inference"]),
        "avg_power_W": med(lambda r: r["energy"]["avg_power_W"]),
        "idle_W": med(lambda r: r["idle_W"]),
        "throughput_per_s": med(lambda r: r["energy"]["throughput_per_s"]),
        "sm_clock_MHz_median": med(lambda r: (r["monitor"]["sm_clock_MHz"] or {"median": 0})["median"]),
    }


def environment():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    drv = pynvml.nvmlSystemGetDriverVersion()
    src = REPO / "SOURCE"
    return {
        "host": HOST, "date": datetime.now().isoformat(timespec="seconds"), "python": platform.python_version(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "tensorrt": trt.__version__,
        "driver": drv.decode() if isinstance(drv, bytes) else drv, "gpu": torch.cuda.get_device_name(0),
        "power_limit_W": pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000,
        "max_sm_clock_MHz": pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM),
        "tf32": False, "cudnn_benchmark": True,
        "kernels_source": src.read_text().strip() if src.exists() else None,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "loadavg": os.getloadavg(),
    }


def phase_measure(models, p):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    path = RESULTS / f"benchmark_{HOST}{'_quick' if p is QUICK else ''}.json"
    report = json.loads(path.read_text()) if path.exists() else {"models": {}}
    report["environment"] = environment()
    report["protocol"] = p
    for name in models:
        report["models"].setdefault(name, {})
        for cfg in CONFIGS:
            stream = torch.cuda.Stream()
            runner = (TorchRunner(name, stream) if cfg == "torch_fp32"
                      else TrtRunner(name, cfg.split("_")[1], stream))
            repeats, attempts = [], 0
            while len([r for r in repeats if r["valid"]]) < p["repeats"] and attempts < p["repeats"] + p["retries"]:
                repeats.append(one_repeat(runner, handle, p))
                attempts += 1
            report["models"][name][cfg] = {"summary": summarize(repeats), "repeats": repeats}
            write_json(path, report)
            print(json.dumps({"model": name, "config": cfg, **report["models"][name][cfg]["summary"]}), flush=True)
            del runner
            torch.cuda.empty_cache()


def parse_cpus(text):
    cpus = set()
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=("build", "measure"))
    ap.add_argument("--models", nargs="+", default=list(SPECS), choices=list(SPECS))
    ap.add_argument("--rebuild", action="store_true", help="re-export ONNX and rebuild engines")
    ap.add_argument("--quick", action="store_true", help="short protocol, for a dry run")
    ap.add_argument("--cpus", help="pin this process to these CPUs, e.g. 8-11")
    args = ap.parse_args()
    if args.cpus:
        os.sched_setaffinity(0, parse_cpus(args.cpus))
    strict_fp32()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.phase == "build":
        phase_build(args.models, args.rebuild)
    else:
        phase_measure(args.models, QUICK if args.quick else FULL)


if __name__ == "__main__":
    main()
