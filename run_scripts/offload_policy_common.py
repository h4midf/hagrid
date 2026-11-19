"""The fixed offload policy every target's greedy / LS / CP-SAT sweep maps under
(offload_policy_flags), the caps those sweeps pick (select_min_median_max_caps), and
the check that a sweep's results file is already complete (load_if_complete)."""
import json
import os

# Set before torch is imported (spawned workers inherit it): one thread for torch's
# OpenMP / MKL pools, so tracing in parallel workers does not oversubscribe the cores.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from compiler.array_caps import build_cap_sequence  # noqa: E402
from variant_generators.common import (  # noqa: E402
    ARGMAX_WIDTHS, DATA_WIDTHS, LN_STREAM_LANES, POLICY_DEQUANT_MANTISSA_WIDTH, POLICY_QUANT_IMPL,
    POLICY_QUANT_MANTISSA_WIDTH, REDUCE_IMPLS, SMOOTH_IMPLS,
)


def select_min_median_max_caps(caps: list) -> list:
    """The smallest, the median (by index; the sequence is log-spaced) and the largest
    cap of the sequence. Returns caps unchanged if there are 3 or fewer."""
    if len(caps) <= 3:
        return list(caps)
    return [caps[0], caps[(len(caps) - 1) // 2], caps[-1]]


def load_if_complete(out_path: str, cap_selector, code_fp: str) -> dict | None:
    """The data in out_path if it already has every cap that cap_selector picks from the
    sequence its stored min / max cap generate, else None. Needs no model build, so a
    finished kernel is skipped before tracing it. A file whose "code_fingerprint" differs
    from code_fp counts as incomplete (its points came from older code)."""
    if not os.path.exists(out_path):
        return None
    with open(out_path) as f:
        data = json.load(f)
    if data.get("code_fingerprint") != code_fp:
        return None
    points = data.get("points", {})
    min_cap = data.get("min_feasible_array_cap")
    max_cap = data.get("max_cap")
    if min_cap is None or max_cap is None:
        return None
    caps = cap_selector(build_cap_sequence(min_cap, max_cap))
    expected = {str(c) for c in caps}
    if expected and expected <= set(points):
        return data
    return None


def offload_policy_flags() -> dict:
    """The variant flags of the offload policy."""
    return {
        "data_width": DATA_WIDTHS[0],
        "sm_impl": "exact", "gelu_impl": "exact", "relu_impl": "exact",
        "silu_impl": "exact", "elu_impl": "exact", "leaky_relu_impl": "exact",
        # the streamed exact layer-norm circuit (custom_op_library/layer_norm/
        # layer_norm_stream.v): the decomposed "rsqrt" / "lut" sum and square at the data
        # width, which saturates on bert-base's outlier rows
        "ln_impl": "exact", "ln_lanes": LN_STREAM_LANES[0], "rms_norm_impl": "lut",
        # SmoothQuant's x / s as an explicit vec_el_mul (the other form folds it into the
        # quantize: compiler/xdsl_fold_smoothing.py)
        "smooth_impl": SMOOTH_IMPLS[0],
        "reduce_impl": REDUCE_IMPLS[0],
        "argmax_width": ARGMAX_WIDTHS[0],
        # CMOS circuits (custom_op_library/quantize, dequantize), not array LUTs
        "quantize_impl": POLICY_QUANT_IMPL, "quantize_mantissa_width": POLICY_QUANT_MANTISSA_WIDTH,
        "dequantize_impl": POLICY_QUANT_IMPL, "dequantize_mantissa_width": POLICY_DEQUANT_MANTISSA_WIDTH,
    }
