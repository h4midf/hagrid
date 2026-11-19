"""The compiler's access point to the hardware target.

TARGET_NAME (environment variable, required) names the target folder, e.g.
TARGET_NAME=1fefet1c/default. It is an environment variable so that forked worker
processes inherit it. TARGET is that target (see targets/target_loader.py).

Latencies in scheduler ticks (1 tick = 1 ns, truncated):
  AcceleratorTiming  the CMOS accelerator units from custom_op_library, which do not
                     depend on the target (every target type uses the same ones)
  TargetTiming       AcceleratorTiming plus a single-module target's own operations
                     (RMU / WMU)
  DualModuleTiming   AcceleratorTiming plus a dual-module target's array reads and
                     writes, under the same RMU / WMU names
  EnsembleTiming     AcceleratorTiming plus the RRAM/SRAM ensemble's macro reads and
                     writes, under the same RMU / WMU names
"""
import os

from custom_op_library.common.tools import load_results
from targets.target_loader import DualModuleTarget, EnsembleTarget, SingleModuleTarget, load_target

TARGET_NAME = os.environ["TARGET_NAME"]
TARGET = load_target(TARGET_NAME)


def ticks(ns: float) -> int:
    return int(ns)


class AcceleratorTiming:
    """The CMOS accelerator units (custom_op_library); n = lanes, w = data width."""

    def reduce_accel(self, n, w):
        return ticks(load_results.reduce_accel_ns(n, w))

    def argmax_accel(self, n, w):
        return ticks(load_results.argmax_accel_ns(n, w))

    def vecadd_accel(self, n, w):
        return ticks(load_results.vecadd_accel_ns(n, w))

    def vecmul_accel(self, n, w):
        return ticks(load_results.vecmul_accel_ns(n, w))

    def vecselect_accel(self, n, w):
        return ticks(load_results.vecselect_accel_ns(n, w))

    def absmax_tree(self, n, w):
        """One pass of the per-token abs-max tree (custom_op_library/token_quant)."""
        return ticks(load_results.absmax_tree_ns(n, w))

    def token_scale(self, w, mw):
        """The per-row scale of per-token quantisation (custom_op_library/token_quant)."""
        return ticks(load_results.token_scale_ns(w, mw))

    def softmax_row_pipeline_ns(self, variant, row_width):
        """Latency of one softmax row of row_width elements, in ns (not truncated)."""
        return load_results.softmax_row_pipeline_ns(variant, row_width)

    def gelu_cmos(self, variant, w):
        return ticks(load_results.gelu_accel_ns(variant, w))

    def relu_accel(self, w):
        return ticks(load_results.relu_accel_ns(w))

    def rsqrt_cmos(self, w):
        return ticks(load_results.rsqrt_accel_ns(w))

    def silu_cmos(self, w):
        return ticks(load_results.silu_accel_ns(w))

    def elu_cmos(self, w, alpha):
        return ticks(load_results.elu_accel_ns(w, alpha))

    def leaky_relu_accel(self, w, alpha):
        return ticks(load_results.leaky_relu_accel_ns(w, alpha))

    def layer_norm_cmos(self, variant, row, w, lanes):
        return ticks(load_results.layer_norm_accel_ns(variant, row, w, lanes))

    def rms_norm_cmos(self, n, w):
        return ticks(load_results.rms_norm_accel_ns(n, w))

    def quantize_cmos(self, variant, w, mw=None):
        return ticks(load_results.quantize_accel_ns(variant, w, mw))

    def dequantize_cmos(self, variant, w, mw=None):
        return ticks(load_results.dequantize_accel_ns(variant, w, mw))


class TargetTiming(AcceleratorTiming):
    """A single-module target's own operations, plus the accelerator units."""

    def __init__(self, target: SingleModuleTarget):
        self._t = target.timing

    def rmu_mac(self, input_bits: int) -> int:
        """One MAC from the RMU over the whole array per input bit, input_bits times."""
        return ticks(input_bits * self._t.rmu_mac_ns)

    def wmu_mac(self, input_bits: int) -> int:
        """One MAC from the WMU over the whole array per input bit, input_bits times."""
        return ticks(input_bits * self._t.wmu_mac_ns)

    def rmu_read(self) -> int:
        return ticks(self._t.rmu_read_ns)

    def rmu_write(self) -> int:
        return ticks(self._t.rmu_write_ns)

    def wmu_read(self) -> int:
        return ticks(self._t.wmu_read_ns)

    def wmu_write(self) -> int:
        return ticks(self._t.wmu_write_ns)


class DualModuleTiming(AcceleratorTiming):
    """A dual-module target's array operations under the names the timing model calls:
    the RMU is the analog module (stored values), the WMU the digital module (run-time
    data). Plus the accelerator units."""

    def __init__(self, target: DualModuleTarget):
        self._analog = target.analog
        self._digital = target.digital

    def rmu_read(self) -> int:
        """One row read out of an analog array."""
        return ticks(self._analog.read_ns)

    def wmu_read(self) -> int:
        """One digital-array row read."""
        return ticks(self._digital.row_read_ns)

    def wmu_write(self) -> int:
        """One digital-array row write."""
        return ticks(self._digital.row_write_ns)

    def digital_row_ns(self) -> float:
        """One digital-array row processed: its writes and its read (not truncated: a
        matvec round adds many rows)."""
        return self._digital.row_write_ns + self._digital.row_read_ns


class EnsembleTiming(AcceleratorTiming):
    """The RRAM/SRAM ensemble's macro operations under the names the timing model calls:
    the RMU is the RRAM macro (stored values), the WMU the SRAM macro (run-time data).
    Plus the accelerator units."""

    def __init__(self, target: EnsembleTarget):
        self._rram = target.rram
        self._sram = target.sram

    def rmu_read(self) -> int:
        """One RRAM row read out (a LUT lookup: a one-hot input): one MAC of one input bit."""
        return ticks(self._rram.cycles_per_mac * self._rram.cycle_ns)

    def wmu_read(self) -> int:
        """One SRAM row read, read_word_bits per cycle."""
        s = self._sram
        return ticks(-(-s.array.cols // s.read_word_bits) * s.cycle_ns)

    def wmu_write(self) -> int:
        """One SRAM row write, write_word_bits per cycle."""
        s = self._sram
        return ticks(-(-s.array.cols // s.write_word_bits) * s.cycle_ns)


def timing_of(target: SingleModuleTarget | DualModuleTarget | EnsembleTarget
              ) -> TargetTiming | DualModuleTiming | EnsembleTiming:
    """The timing object of a target, by its type."""
    match target:
        case SingleModuleTarget():
            return TargetTiming(target)
        case DualModuleTarget():
            return DualModuleTiming(target)
        case EnsembleTarget():
            return EnsembleTiming(target)
        case _:
            raise TypeError(f"no timing for target type {type(target).__name__}")
