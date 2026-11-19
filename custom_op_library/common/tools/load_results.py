"""The one reader of every custom_op_library/<unit>/results/<unit>.json: the synthesized
delay, area and power of each circuit, for targets/target_support.py and the compiler.

Every function takes each hardware axis its circuit has as a required argument (lane
count n, data width w, alpha, variant, ...): nothing is defaulted. A point that is not
characterized raises; _CHARACTERIZED_N / _CHARACTERIZED_W / _CHARACTERIZED_ALPHA list
each unit's characterized points, checked before its rows are read.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_ROOT = os.path.dirname(os.path.dirname(_HERE))  # custom_op_library/

# each unit's characterized lane counts and widths
_CHARACTERIZED_N = {
    # one reduce tree per input count; a reduce op uses the smallest that fits its
    # reduction (compiler/xdsl_accel_width_stamp.py)
    "reduce_vec": (8, 16, 32, 64, 128, 256, 512, 1024),
    "argmax": (512, 1024),
    # the compiler's vec_add uses the 1-lane point; reduce_vec's "vec_add" reduce_impl
    # reuses this circuit at half the reduce tree's input count
    "vec_add": (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
    # the compiler's vec_el_mul uses the 1-lane point
    "vec_el_mul": (1, 2, 4, 8, 16, 32, 64),
    # rms_norm.v is synthesized at N = 4 only (rms_norm/characterize.py): a real model's
    # hidden size raises
    "rms_norm": (4,),
    # absmax_reduce_tree: softmax's tree width; a wider row takes ceil(width / 256) passes
    "token_quant": (256,),
    # the same lane counts as vec_add
    "vec_select": (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
}
_CHARACTERIZED_W = {
    "reduce_vec": (32, 16),
    "argmax": (32, 16),
    "vec_add": (32, 16),
    "vec_el_mul": (32, 16),
    "relu": (32, 16),
    "rsqrt": (32, 16),
    "gelu": (32, 16),
    "silu": (32, 16),
    "elu": (32, 16),
    "leaky_relu": (32, 16),
    "rms_norm": (32, 16),
    "quark_shared": (32, 16),
    "token_quant": (32,),
    "vec_select": (32, 16),
}
# elu's alpha and leaky_relu's negative_slope are Verilog parameters (ALPHA): one circuit
# per value, so an op's value must be one of these
_CHARACTERIZED_ALPHA = {
    "elu": (1.0, 0.5),
    "leaky_relu": (0.01, 0.125, 0.0078125),
}


def characterized_sizes(unit: str) -> tuple:
    """The vector sizes (N) a unit's circuit is characterized at."""
    return _CHARACTERIZED_N[unit]


def characterized_alphas(kind: str) -> tuple:
    """The alpha (elu) / negative_slope (leaky_relu) values a circuit is characterized at."""
    return _CHARACTERIZED_ALPHA[kind]


_rows_cache: dict = {}  # unit -> parsed rows, loaded once per process


def _rows(unit):
    cached = _rows_cache.get(unit)
    if cached is not None:
        return cached
    path = os.path.join(_LIB_ROOT, unit, "results", f"{unit}.json")
    with open(path) as f:
        rows = json.load(f)
    _rows_cache[unit] = rows
    return rows


def _check(unit, n=None, w=None):
    if n is not None and n not in _CHARACTERIZED_N[unit]:
        raise ValueError(f"n={n!r} is not a characterized lane count for {unit} -- must be one of "
                          f"{_CHARACTERIZED_N[unit]}")
    if w is not None and w not in _CHARACTERIZED_W[unit]:
        raise ValueError(f"w={w!r} is not a characterized data width for {unit} -- must be one of "
                          f"{_CHARACTERIZED_W[unit]}")


def _row_at_n_w(unit, module, n, w):
    _check(unit, n=n, w=w)
    rows = [r for r in _rows(unit)
            if r["unit"] == module and r["params"].get("N") == n and r["params"].get("W") == w]
    if not rows:
        raise ValueError(f"no row for {unit}/{module} (N={n}, W={w}) in results/{unit}.json")
    if len(rows) > 1:
        raise ValueError(f"{unit}/{module} (N={n}, W={w}) has {len(rows)} rows -- ambiguous")
    return rows[0]


def _delay_ns_at_n_w(unit, module, n, w):
    return _row_at_n_w(unit, module, n, w)["delay_ps"] / 1000.0


def _row_at_w(unit, module, w):
    _check(unit, w=w)
    rows = [r for r in _rows(unit) if r["unit"] == module and r["params"].get("W") == w]
    if not rows:
        raise ValueError(f"no row for {unit}/{module} (W={w}) in results/{unit}.json")
    if len(rows) > 1:
        raise ValueError(f"{unit}/{module} (W={w}) has {len(rows)} rows -- ambiguous")
    return rows[0]


def _delay_ns_at_w(unit, module, w):
    return _row_at_w(unit, module, w)["delay_ps"] / 1000.0


def _row_alpha(raw) -> float:
    # a row's alpha is "1.0" or labeled ("0.01_generic": what synthesis made of that
    # value); the number is the part before the first "_"
    return float(str(raw).split("_")[0])


def _row_at_w_alpha(unit, module, w, alpha):
    _check(unit, w=w)
    if alpha not in _CHARACTERIZED_ALPHA[unit]:
        raise ValueError(f"alpha={alpha!r} is not a characterized alpha for {unit} -- must be one of "
                          f"{_CHARACTERIZED_ALPHA[unit]}")
    rows = [r for r in _rows(unit)
            if r["unit"] == module and r["params"].get("W") == w and _row_alpha(r["params"].get("alpha")) == float(alpha)]
    if not rows:
        raise ValueError(f"no row for {unit}/{module} (W={w}, alpha={alpha}) in results/{unit}.json")
    if len(rows) > 1:
        raise ValueError(f"{unit}/{module} (W={w}, alpha={alpha}) has {len(rows)} rows -- ambiguous")
    return rows[0]


def _delay_ns_at_w_alpha(unit, module, w, alpha):
    return _row_at_w_alpha(unit, module, w, alpha)["delay_ps"] / 1000.0


def reduce_accel_ns(n, w):
    return _delay_ns_at_n_w("reduce_vec", "reduce_tree", n, w)


def argmax_accel_ns(n, w):
    return _delay_ns_at_n_w("argmax", "argmax_tree", n, w)


def vecadd_accel_ns(n, w):
    return _delay_ns_at_n_w("vec_add", "vec_add_vv", n, w)


def vecmul_accel_ns(n, w):
    return _delay_ns_at_n_w("vec_el_mul", "vec_el_mul_vv_round", n, w)


# area / power: the same rows as the delays above. vec_add / vec_el_mul are the vector x
# vector ("vv") circuits: a scalar operand is made a stored vector before costing
# (xdsl_tiling's run_convert_static_values_canon)
def reduce_accel_area_power(n, w):
    return _row_at_n_w("reduce_vec", "reduce_tree", n, w)


def argmax_accel_area_power(n, w):
    return _row_at_n_w("argmax", "argmax_tree", n, w)


def vecadd_accel_area_power(n, w):
    return _row_at_n_w("vec_add", "vec_add_vv", n, w)


def vecmul_accel_area_power(n, w):
    return _row_at_n_w("vec_el_mul", "vec_el_mul_vv_round", n, w)


def vecselect_accel_ns(n, w):
    """out = mask ? x : fill over n lanes of w bits (custom_op_library/vec_select)."""
    return _delay_ns_at_n_w("vec_select", "vec_select", n, w)


def vecselect_accel_area_power(n, w):
    return _row_at_n_w("vec_select", "vec_select", n, w)


def lane_register_area_power(bits):
    """The register of one shared 1-lane unit: its input operands and result, `bits`
    in total (custom_op_library/lane_register)."""
    return _row_matching("lane_register", "lane_register", {"BITS": bits})


def relu_accel_area_power(w):
    return _row_at_w("relu", "relu", w)


def relu_accel_ns(w):
    return _delay_ns_at_w("relu", "relu", w)


def rsqrt_accel_area_power(w):
    """1/sqrt(x): custom_op_library/rsqrt (the rsqrt circuit of ../layer_norm)."""
    return _row_at_w("rsqrt", "rsqrt", w)


def rsqrt_accel_ns(w):
    return _delay_ns_at_w("rsqrt", "rsqrt", w)


# circuit variant -> (unit folder, module in its results file). QUARK's circuits share
# primitives, so they are all in quark_shared/, not their own kind's folder.
_GELU_VARIANT_MODULE = {
    "exact": ("gelu", "gelu"),
    # the tanh form (GPT-2's gelu_new): gelu.v's circuit with a tanh-form table
    "exact_tanh": ("gelu", "gelu_tanh"),
    "quark": ("quark_shared", "quark_gelu"),
    "wang_tcad": ("gelu", "wang_tcad_gelu24"),
    "gqa_lut": ("gelu", "gqa_lut_gelu8"),
    "ibert": ("gelu", "ibert_gelu"),
}
# each gelu variant's characterized widths (per variant: quark and ibert synthesize only
# at W = 16, gelu/characterize.py, quark_shared/characterize.py)
_GELU_VARIANT_W = {
    "exact": (32, 16), "exact_tanh": (32, 16), "quark": (16,), "wang_tcad": (32, 16), "gqa_lut": (32, 16),
    "ibert": (16,),
}
# layer_norm circuit impl -> its streamed circuit (custom_op_library/layer_norm/
# characterize_stream.py): N lanes per cycle over a row of ROW elements, rows keyed by
# (ROW, N, W), delay_ps the clock period and cycles_per_row the cycles one row takes
_LAYER_NORM_VARIANT_MODULE = {
    "exact": ("layer_norm", "layer_norm_stream"),
    "sole": ("layer_norm", "ailayernorm_stream"),
}


def _layer_norm_row(variant, row, w, lanes):
    """The streamed layer-norm circuit's results row at a row of `row` elements, `lanes`
    lanes, data width `w`; raises if that point is not characterized."""
    if variant not in _LAYER_NORM_VARIANT_MODULE:
        raise ValueError(f"variant={variant!r} is not a real layer_norm circuit -- must be one of "
                          f"{sorted(_LAYER_NORM_VARIANT_MODULE)}")
    unit, module = _LAYER_NORM_VARIANT_MODULE[variant]
    rows = [r for r in _rows(unit) if r["unit"] == module and r["params"].get("ROW") == row
            and r["params"].get("N") == lanes and r["params"].get("W") == w]
    if not rows:
        have = sorted((r["params"]["ROW"], r["params"]["N"], r["params"]["W"])
                      for r in _rows(unit) if r["unit"] == module)
        raise ValueError(f"layer_norm/{variant}: {module} is not characterized at ROW={row}, N={lanes}, "
                          f"W={w} (characterized (ROW, N, W): {have})")
    if len(rows) > 1:
        raise ValueError(f"layer_norm/{variant}: {len(rows)} rows at ROW={row}, N={lanes}, W={w} -- ambiguous")
    return rows[0]


def layer_norm_accel_area_power(variant, row, w, lanes):
    """Area/power of one streamed layer-norm circuit (the row its timing uses)."""
    return _layer_norm_row(variant, row, w, lanes)


def _gelu_variant_row(variant, w):
    if variant not in _GELU_VARIANT_MODULE:
        raise ValueError(f"variant={variant!r} is not a real gelu circuit -- must be one of "
                          f"{sorted(_GELU_VARIANT_MODULE)}")
    if w not in _GELU_VARIANT_W[variant]:
        raise ValueError(f"w={w!r} is not a characterized width for gelu/{variant} -- must be one of "
                          f"{_GELU_VARIANT_W[variant]}")
    unit, module = _GELU_VARIANT_MODULE[variant]
    return _row_at_w(unit, module, w)


def gelu_accel_ns(variant, w):
    return _gelu_variant_row(variant, w)["delay_ps"] / 1000.0


def gelu_accel_area_power(variant, w):
    return _gelu_variant_row(variant, w)


def _row_matching(unit, module, params: dict):
    """The one row of unit/module whose params include `params` (the softmax stages'
    circuits each have their own parameters: N and W, W and FRAC, SUM_W, or none)."""
    rows = [r for r in _rows(unit) if r["unit"] == module
            and all(r["params"].get(k) == v for k, v in params.items())]
    if not rows:
        raise ValueError(f"no row for {unit}/{module} (params matching {params}) in results/{unit}.json")
    if len(rows) > 1:
        raise ValueError(f"{unit}/{module} (params matching {params}) has {len(rows)} rows -- ambiguous")
    return rows[0]


# Every non-"lut" softmax variant's row pipeline: each stage is (unit folder, module
# name, lookup params, per_element), in order. per_element=True: a 1-element circuit
# that runs once per row element (the 1-lane adder and multiplier, the exp
# approximations). per_element=False: a circuit that runs once per row (the trees,
# the reciprocal, quark_appro_ln). The trees have N=256 inputs (HyFlexPIM's SFU
# width: "we configured each SFU to process 256 inputs per cycle", the only
# paper-sourced nonlinear-op width); a longer row takes ceil(row_width / 256)
# passes. The adder, multiplier and sum tree are the same circuits vec_add,
# vec_el_mul and reduce_vec use (compiler/xdsl_accel_concurrency.py pools them).
_SOFTMAX_ROW_PIPELINE_STAGES = {
    "exact": (
        ("softmax", "max_reduce_tree", {"W": 32, "N": 256}, False),
        # x_i - max (exp_pwl16_standalone takes the delta)
        ("vec_add", "vec_add_vv", {"W": 32, "N": 1}, True),
        ("softmax", "exp_pwl16_standalone", {}, True),
        ("reduce_vec", "reduce_tree", {"W": 32, "N": 256}, False),
        ("softmax", "q8_8_reciprocal", {"SUM_W": 20}, False),
        ("vec_el_mul", "vec_el_mul_vv_round", {"W": 32, "N": 1}, True),
    ),
    "sole": (
        ("softmax", "max_reduce_tree", {"W": 32, "N": 256}, False),
        ("softmax", "log2exp_standalone", {}, True),
        ("reduce_vec", "reduce_tree", {"W": 32, "N": 256}, False),
        ("softmax", "aldivision_standalone", {"SUM_W": 24}, True),
    ),
    "gqa_lut": (
        ("softmax", "max_reduce_tree", {"W": 32, "N": 256}, False),
        ("softmax", "gqa_lut_exp8", {"W": 32, "FRAC": 20}, True),
        ("reduce_vec", "reduce_tree", {"W": 32, "N": 256}, False),
        ("softmax", "q8_8_reciprocal", {"SUM_W": 20}, False),
        ("vec_el_mul", "vec_el_mul_vv_round", {"W": 32, "N": 1}, True),
    ),
    "quark": (
        ("softmax", "max_reduce_tree", {"W": 32, "N": 256}, False),
        ("quark_shared", "quark_appro_exp", {"W": 32, "FRAC": 20}, True),
        ("reduce_vec", "reduce_tree", {"W": 32, "N": 256}, False),
        ("quark_shared", "quark_appro_ln", {"W": 32, "FRAC": 20}, False),
        # QUARK's log-sum-exp form applies its exp approximator twice per element: on
        # x' = x - max, then on z = x' - ln(S)
        ("quark_shared", "quark_appro_exp", {"W": 32, "FRAC": 20}, True),
    ),
    "ibert": (
        ("softmax", "max_reduce_tree", {"W": 32, "N": 256}, False),
        ("softmax", "ibert_exp_pwl_standalone", {}, True),
        ("reduce_vec", "reduce_tree", {"W": 32, "N": 256}, False),
        ("softmax", "q8_8_reciprocal", {"SUM_W": 20}, False),
        ("vec_el_mul", "vec_el_mul_vv_round", {"W": 32, "N": 1}, True),
    ),
}


def softmax_stages(variant, row_width):
    """[(unit, module, row, per_element, passes)] of a softmax variant's row pipeline,
    in order (_SOFTMAX_ROW_PIPELINE_STAGES). passes: how many times a once-per-row
    stage runs per row (ceil(row_width / N) for a tree, else 1); 1 for a per-element
    stage."""
    if variant not in _SOFTMAX_ROW_PIPELINE_STAGES:
        raise ValueError(f"variant={variant!r} has no softmax row pipeline; must be one of "
                         f"{sorted(_SOFTMAX_ROW_PIPELINE_STAGES)}")
    stages = []
    for unit, module, params, per_element in _SOFTMAX_ROW_PIPELINE_STAGES[variant]:
        row = _row_matching(unit, module, params)
        passes = -(-row_width // params["N"]) if not per_element and "N" in params else 1
        stages.append((unit, module, row, per_element, passes))
    return stages


def softmax_row_pipeline_ns(variant, row_width):
    """Latency of one softmax row: the stage delays in sequence, a tree stage once
    per pass. A per-element stage counts one element delay (the elements wait in a
    buffer for the shared units; that wait is not modeled)."""
    return sum(row["delay_ps"] * passes for _, _, row, _, passes in softmax_stages(variant, row_width)) / 1000.0


def silu_accel_ns(w):
    return _delay_ns_at_w("silu", "silu", w)


def silu_accel_area_power(w):
    return _row_at_w("silu", "silu", w)


def elu_accel_ns(w, alpha):
    return _delay_ns_at_w_alpha("elu", "elu", w, alpha)


def leaky_relu_accel_ns(w, alpha):
    return _delay_ns_at_w_alpha("leaky_relu", "leaky_relu", w, alpha)


def elu_accel_area_power(w, alpha):
    return _row_at_w_alpha("elu", "elu", w, alpha)


def leaky_relu_accel_area_power(w, alpha):
    return _row_at_w_alpha("leaky_relu", "leaky_relu", w, alpha)


def layer_norm_accel_ns(variant, row, w, lanes):
    """One row of `row` elements through the streamed layer-norm circuit: its
    cycles_per_row times its clock period (ns)."""
    r = _layer_norm_row(variant, row, w, lanes)
    return r["cycles_per_row"] * r["delay_ps"] / 1000.0


def rms_norm_accel_ns(n, w):
    return _delay_ns_at_n_w("rms_norm", "rms_norm", n, w)


def _quant_dequant_row(unit, variant, w, mw):
    """The row of quantize / dequantize circuit `variant` (compiler/ops/quant_encoding.
    CIRCUIT_IMPLS; module "quant_" / "dequant_" + variant) at data width w and, for a
    mant_* variant, mantissa width mw; raises if there is none. The variant generator
    checks the combinations first."""
    prefix = "quant_" if unit == "quantize" else "dequant_"
    module = prefix + variant
    uses_mant = "pow2" not in variant
    rows = [r for r in _rows(unit)
            if r["unit"] == module and r["params"].get("XW") == w
            and r["params"].get("MW") == (mw if uses_mant else None)]
    if not rows:
        raise ValueError(f"no row for {unit}/{module} (XW={w}, MW={mw if uses_mant else None}) in results/{unit}.json")
    if len(rows) > 1:
        raise ValueError(f"{unit}/{module} (XW={w}, MW={mw}) has {len(rows)} rows -- ambiguous")
    return rows[0]


def quantize_accel_ns(variant, w, mw=None):
    return _quant_dequant_row("quantize", variant, w, mw)["delay_ps"] / 1000.0


def dequantize_accel_ns(variant, w, mw=None):
    return _quant_dequant_row("dequantize", variant, w, mw)["delay_ps"] / 1000.0


def quantize_accel_area_power(variant, w, mw=None):
    return _quant_dequant_row("quantize", variant, w, mw)


def dequantize_accel_area_power(variant, w, mw=None):
    return _quant_dequant_row("dequantize", variant, w, mw)


# per-token (dynamic) int8 quantisation (custom_op_library/token_quant): the row
# abs-max tree and the per-row scale; the per-element rounding is quantize's
# quant_mant_round (quantize_accel_ns("mant_round", w, mw))
def absmax_tree_ns(n, w):
    """One pass of the abs-max tree over n lanes of w bits."""
    return _delay_ns_at_n_w("token_quant", "absmax_reduce_tree", n, w)


def absmax_tree_area_power(n, w):
    return _row_at_n_w("token_quant", "absmax_reduce_tree", n, w)


def _token_scale_row(w, mw):
    _check("token_quant", w=w)
    return _row_matching("token_quant", "token_scale", {"W": w, "MW": mw})


def token_scale_ns(w, mw):
    """The per-row scale (m, sh of 127 / row max) for quant_mant_round at mantissa width mw."""
    return _token_scale_row(w, mw)["delay_ps"] / 1000.0


def token_scale_area_power(w, mw):
    return _token_scale_row(w, mw)
