"""Shared by every per-kernel variant generator: finding the ops in a traced graph
that take a config entry (discover_alt_ops), the value sets of every hardware axis,
building one variant's per-op config (build_config_for_variant), and writing variant
files to disk.

Which axes a kernel sweeps is decided in its own variant_generators/<kernel>.py,
from the ops its graph actually contains.
"""
import hashlib
import itertools
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from compiler.ops.quant_encoding import (
    DEQUANTIZE, QUANTIZE, encode_mantissa, encode_pow2, uses_mantissa,
)
from compiler.support_strings import TENSOR_META
from compiler.xdsl_bridge import (
    elu_alpha, function_config_kind, is_quantize_per_tensor_static, is_w3_call, resolve_scalar,
    static_quantize_params, w3_output_quantization,
)
from custom_op_library.common.tools import load_results

# module type -> the kind of its config entry
ALT_MODULE_KIND = {
    nn.Linear: "linear",
    torch.ao.nn.quantized.Linear: "linear",
    torch.ao.nn.intrinsic.quantized.LinearReLU: "linear",
    nn.LayerNorm: "layer_norm",
    nn.ReLU: "relu",
    # convolutions lower to the same weighted matmul as Linear
    nn.Conv2d: "linear",
    torch.ao.nn.quantized.Conv2d: "linear",
    torch.ao.nn.intrinsic.quantized.ConvReLU2d: "linear",
    nn.Conv3d: "linear",
    nn.ConvTranspose3d: "linear",
    nn.Conv1d: "linear",
    nn.SiLU: "silu",
    # GroupNorm/InstanceNorm share one lowering: layer_norm's mean/variance chain, per group
    nn.GroupNorm: "group_norm",
    nn.InstanceNorm1d: "group_norm",
    nn.InstanceNorm2d: "group_norm",
    nn.InstanceNorm3d: "group_norm",
    nn.MaxPool2d: "max_pool2d",
    # AdaptiveAvgPool2d((1,1)): sum then scale (reduce_vec + vec_el_mul)
    nn.AdaptiveAvgPool2d: "avg_pool2d",
    nn.LeakyReLU: "leaky_relu",
}
# functions whose __name__ is the kind of their config entry ("add"/"mul" are every
# plain + / *, e.g. residual adds)
ALT_FUNCTION_NAMES = {
    "softmax", "gelu", "relu", "elu", "leaky_relu", "silu", "matmul",
    "add", "mul", "neg", "topk", "truediv",
}
# neg and truediv by a scalar become a vec_el_mul, so they take the "mul" kind
ALT_FUNCTION_KIND = {"neg": "mul", "truediv": "mul"}
# torch.ops.quantized.mul's __name__ is not reliably "mul": matched by identity
_QUANTIZED_MUL_TARGET = torch.ops.quantized.mul

# the table parameters of every LUT implementation
LUT_TABLE_DEFAULTS = {
    "softmax": {"xmin": -12.0, "xmax": 0.0, "nbins": 256, "rcp_nbins": 1024},
    "gelu": {"xmin": -6.0, "xmax": 6.0, "nbins": 256},
    "relu": {"xmin": -8.0, "xmax": 8.0, "nbins": 256},
    "elu": {"xmin": -8.0, "xmax": 8.0, "nbins": 256},
    "leaky_relu": {"xmin": -8.0, "xmax": 8.0, "nbins": 256},
    "silu": {"xmin": -8.0, "xmax": 8.0, "nbins": 256},
    # rsqrt of variance + eps
    "layer_norm": {"xmin": 1e-6, "xmax": 8.0, "nbins": 256},
    "group_norm": {"xmin": 1e-6, "xmax": 8.0, "nbins": 256},
    # reciprocal of sqrt(mean square) + eps: a narrower range than a variance
    "rms_norm": {"xmin": 1e-6, "xmax": 4.0, "nbins": 256},
}


def _match_rms_norm_shape_fx(mul_node):
    """The fx-graph form of compiler/xdsl_gen_lowering.py's _match_rms_norm_shape
    (weight * x / (norm(x) / sqrt(dim) + eps)). Returns (x_node, weight_node,
    add_node, eps) or None; add_node is returned so the caller can drop its generic
    "add" tag."""
    if mul_node.op != "call_function" or getattr(mul_node.target, "__name__", "") != "mul":
        return None
    if len(mul_node.args) != 2:
        return None
    a0, a1 = mul_node.args
    if not (isinstance(a0, torch.fx.Node) and isinstance(a1, torch.fx.Node)):
        return None
    if a0.op == "get_attr" and a1.op != "get_attr":
        weight_node, div2 = a0, a1
    elif a1.op == "get_attr" and a0.op != "get_attr":
        weight_node, div2 = a1, a0
    else:
        return None

    if div2.op != "call_function" or getattr(div2.target, "__name__", "") != "truediv":
        return None
    if len(div2.args) != 2 or not isinstance(div2.args[0], torch.fx.Node) or not isinstance(div2.args[1], torch.fx.Node):
        return None  # x / (rms+eps): both real tensor nodes, no python-scalar rhs
    x_node, add_node = div2.args

    if add_node.op != "call_function" or getattr(add_node.target, "__name__", "") != "add":
        return None
    if len(add_node.args) != 2 or not isinstance(add_node.args[0], torch.fx.Node) or isinstance(add_node.args[1], torch.fx.Node):
        return None  # rms + eps: tensor lhs, python-scalar rhs
    div1_node, eps_val = add_node.args

    if div1_node.op != "call_function" or getattr(div1_node.target, "__name__", "") != "truediv":
        return None
    if len(div1_node.args) != 2 or not isinstance(div1_node.args[0], torch.fx.Node) or isinstance(div1_node.args[1], torch.fx.Node):
        return None  # norm / sqrt(DIM): tensor lhs, python-scalar rhs
    norm_node = div1_node.args[0]

    if norm_node.op != "call_method" or norm_node.target != "norm":
        return None
    if len(norm_node.args) < 2 or norm_node.args[1] != 2:
        return None
    if norm_node.args[0] is not x_node:
        return None
    return x_node, weight_node, add_node, float(eps_val)


def discover_alt_ops(traced: torch.fx.GraphModule) -> tuple:
    """Returns (found, quant_data).

    found: {fx_name: kind} for every node that takes a config entry: every node that
    reaches an accelerator (reduce_vec, vec_el_mul, vec_add, argmax, max) or has an
    implementation choice (softmax, gelu, layer_norm, linear, ...).

    quant_data: {fx_name: {...}}, per-op data known from the trace, used to validate or
    encode an implementation:
      quantize_per_tensor / dequantize   {"scale", "zero_point"} (calibrated values)
      layer_norm / rms_norm              {"hidden_size"} (validates impl "exact")
      leaky_relu                         {"negative_slope"}
      elu                                {"alpha"} (the call's, as the bridge reads it)
    """
    modules = dict(traced.named_modules())
    found = {}
    quant_data = {}
    for node in traced.graph.nodes:
        if node.op == "call_module":
            sub = modules[node.target]
            kind = ALT_MODULE_KIND.get(type(sub))
            if kind is not None:
                found[node.name] = kind
                if kind == "layer_norm":
                    quant_data[node.name] = {"hidden_size": int(tuple(sub.normalized_shape)[0])}
                elif kind == "leaky_relu":
                    quant_data[node.name] = {"negative_slope": float(sub.negative_slope)}
        elif node.op == "call_function":
            if node.target is _QUANTIZED_MUL_TARGET:
                found[node.name] = "mul"
                continue
            if node.target is torch.quantize_per_tensor:
                # args = (tensor, scale, zero_point, dtype); scale/zero_point are get_attr buffers
                found[node.name] = "quantize_per_tensor"
                quant_data[node.name] = {
                    "scale": float(resolve_scalar(traced, node.args[1])),
                    "zero_point": int(resolve_scalar(traced, node.args[2])),
                }
                continue
            if is_quantize_per_tensor_static(node):
                # kernels/gpt2.py's static per-tensor int8 (V before P.V): the frontend makes it
                # a quantize_per_tensor (scale x_max / 127, zero point 0)
                scale, zero_point = static_quantize_params(traced, node)
                found[node.name] = "quantize_per_tensor"
                quant_data[node.name] = {"scale": scale, "zero_point": zero_point}
                continue
            fn_kind = function_config_kind(node)
            if fn_kind is not None:
                # kernels/gpt2.py's int8 functions and torch.where (the frontend maps them
                # by name)
                found[node.name] = fn_kind
                continue
            fn_name = getattr(node.target, "__name__", "")
            if fn_name == "mul":
                rms_shape = _match_rms_norm_shape_fx(node)
                if rms_shape is not None:
                    x_node, weight_node, add_node, eps = rms_shape
                    tm = x_node.meta.get(TENSOR_META)
                    if tm is None or getattr(tm, "shape", None) is None:
                        raise ValueError(f"{x_node.name}: no tensor_meta -- run ShapeProp first")
                    found[node.name] = "rms_norm"
                    quant_data[node.name] = {"hidden_size": int(tuple(tm.shape)[-1])}
                    # the eps add inside the chain was tagged "add" earlier: rms_norm owns it
                    found.pop(add_node.name, None)
                    continue
            if fn_name in ALT_FUNCTION_NAMES:
                found[node.name] = ALT_FUNCTION_KIND.get(fn_name, fn_name)
                if fn_name == "elu":
                    quant_data[node.name] = {"alpha": elu_alpha(node)}
        elif node.op == "call_method" and node.target == "mean":
            found[node.name] = "mean"
        elif node.op == "call_method" and node.target == "dequantize":
            # GenDequantizeOp carries no scale/zero_point of its own: they
            # are the output scale/zero_point of whatever produced the value.
            # Producers: quantize_per_tensor, a quantized Linear or Conv2d module
            # (ConvReLU2d is a quantized Conv2d), and quantized.matmul. Any other
            # producer raises.
            producer = node.args[0]
            if producer.op == "call_function" and producer.target is torch.quantize_per_tensor:
                scale, zero_point = resolve_scalar(traced, producer.args[1]), resolve_scalar(traced, producer.args[2])
            elif producer.op == "call_function" and producer.target is torch.ops.quantized.matmul:
                scale, zero_point = resolve_scalar(traced, producer.args[2]), resolve_scalar(traced, producer.args[3])
            elif is_w3_call(producer):
                scale, zero_point = w3_output_quantization(producer)
            elif producer.op == "call_module" and isinstance(
                traced.get_submodule(producer.target), (torch.ao.nn.quantized.Linear, torch.ao.nn.quantized.Conv2d)
            ):
                sub = traced.get_submodule(producer.target)
                scale, zero_point = sub.scale, sub.zero_point
            else:
                raise NotImplementedError(
                    f"{node.name}: dequantize's producer {producer} is not quantize_per_tensor, "
                    "a quantized Linear / Conv2d, a w3 Linear / conv or quantized.matmul -- its "
                    "scale/zero_point is unknown"
                )
            found[node.name] = "dequantize"
            quant_data[node.name] = {"scale": float(scale), "zero_point": int(zero_point)}
    return found, quant_data


# Implementation choices per op kind; each name is a characterized circuit in
# custom_op_library (or "lut", a lookup table), handled in compiler/ops/<kind>.py.
SOFTMAX_IMPLS = ("exact", "lut", "sole", "quark", "gqa_lut", "ibert")
GELU_IMPLS = ("exact", "lut", "quark", "wang_tcad", "gqa_lut", "ibert")
# gelu impls with no 32-bit circuit: generators skip them at data_width 32
GELU_IMPLS_W16_ONLY = frozenset({"ibert", "quark"})
RELU_IMPLS = ("exact", "lut")
SILU_IMPLS = ("exact", "lut")
ELU_IMPLS = ("exact", "lut")
LEAKY_RELU_IMPLS = ("exact", "lut")
LN_IMPLS = ("exact", "lut", "rsqrt", "sole")
# layer_norm impls decomposed into vector ops (any hidden size): "lut" (1/sqrt by a LUT in
# the arrays) and "rsqrt" (1/sqrt by the CMOS rsqrt circuit). The others are streamed
# circuits (custom_op_library/layer_norm/characterize_stream.py), characterized at the
# row lengths and lane counts their results rows list.
LN_DECOMPOSED_IMPLS = ("lut", "rsqrt")
# lanes per cycle of a streamed layer_norm circuit
LN_STREAM_LANES = (4, 8)

# lane widths of the argmax accelerator: the two characterized widths (a reduction's
# tree size comes from the IR; vec_add and vec_el_mul are built from 1-lane units,
# compiler/xdsl_accel_concurrency.py)
# reduce_vec circuit: "tree" (a dedicated adder tree) or "vec_add" (the vec_add unit
# reused over log2(width) halving rounds)
REDUCE_IMPLS = ("tree", "vec_add")
# fixed-point width of the non-matmul ops: 32 = Q12.20, 16 = Q8.8
DATA_WIDTHS = (32, 16)
# argmax and max share one comparator-tree circuit
ARGMAX_WIDTHS = (512, 1024)
# the quantize / dequantize every offload-policy sweep uses (all targets): the CMOS
# circuit with a mantissa scale, round to nearest and zero point (the activations
# are affine), not the array LUT; it also requantizes every int8 Linear / conv
POLICY_QUANT_IMPL = "mant_round_zp"
# SmoothQuant's per-channel x * (1 / s) before a quantize (kernels/bert_base.py):
# "vec_el_mul" (an explicit vec_el_mul by the stored 1 / s) or "per_channel_mantissa"
# (folded into the quantize circuit: one mantissa / shift per channel, no extra op)
SMOOTH_IMPLS = ("vec_el_mul", "per_channel_mantissa")
POLICY_QUANT_MANTISSA_WIDTH = 16
# dequantize's mantissa is scale * 2^frac (at least 20.0M for bert-large's largest
# activation scale, 19.08, in Q12.20), so it needs more bits than quantize's 1 / scale
POLICY_DEQUANT_MANTISSA_WIDTH = 31


def _require_exact_valid(fx_name: str, kind: str, characterized: tuple, actual, label: str,
                          variant: str) -> None:
    """Raises at generation time if a circuit impl (`variant`) is not characterized
    for this op's value of `label` (e.g. hidden_size)."""
    if actual not in characterized:
        raise ValueError(
            f"{fx_name}: {kind} impl={variant!r} needs {label}={actual!r}, but custom_op_library only "
            f"characterizes {characterized} -- this combination doesn't exist for this op"
        )


def _quant_entry(kind: str, fx_name: str, real: dict, impl: str, fl: dict, mantissa_key: str,
                 data_width: int) -> dict:
    """The config entry of one quantize / dequantize op: impl, plus mantissa_width for a
    mantissa impl (read from `fl` only then). The (sh, m, zp) encoding is not stored:
    the compiler computes it from the op's scale / zero point in the IR. It is computed
    here only to reject a variant whose circuit cannot represent this op's scale."""
    entry = {"kind": kind, "impl": impl}
    if impl == "lut":
        return entry
    if uses_mantissa(impl):
        mantissa_width = fl[mantissa_key]
        encode_mantissa(kind, fx_name, real["scale"], real["zero_point"], impl, mantissa_width, data_width)
        entry["mantissa_width"] = mantissa_width
    else:
        encode_pow2(kind, fx_name, real["scale"], real["zero_point"], impl, data_width)
    return entry


def build_config_for_variant(alt_ops: dict, fl: dict, quant_data: dict) -> dict:
    """{fx_name: {per-op config}} for one variant `fl`. Each entry has its kind, its
    implementation choice where the kind has one, the lane widths of the accelerators
    its lowering uses, and data_width (the fixed-point width, Q12.20 or Q8.8):

      softmax                     reduce_impl
      layer_norm / rms_norm / group_norm   reduce_impl
      silu                        none
      gelu / relu / elu / leaky_relu    none (one LUT)
      linear / matmul             reduce_impl (tiles)
      avg_pool2d / mean           reduce_impl
      max_pool2d / topk           argmax_width
      add / mul                   none
      quantize / dequantize       impl, plus mantissa_width for a mantissa impl
      row_absmax / quantize_per_token / dequantize_per_token / where   none (one circuit)

    A reduction's tree size is not a config field: each reduce op uses the smallest
    tree that fits it (compiler/xdsl_accel_width_stamp.py). vec_add and vec_el_mul have
    no lane width: they are built from 1-lane units. vec_el_mul always rounds to
    nearest. Only the kinds in alt_ops read their fields of `fl`.
    """
    xw = fl["argmax_width"]
    ri = fl["reduce_impl"]
    dw = fl["data_width"]
    cfg = {}
    for fx_name, kind in alt_ops.items():
        if kind == "softmax":
            cfg[fx_name] = {"kind": "softmax", "impl": fl["sm_impl"], "reduce_impl": ri,
                             **LUT_TABLE_DEFAULTS["softmax"]}
        elif kind == "gelu":
            if fl["gelu_impl"] in GELU_IMPLS_W16_ONLY and dw != 16:
                raise ValueError(
                    f"{fx_name}: gelu impl={fl['gelu_impl']!r} is only characterized at data_width=16, "
                    f"got data_width={dw!r}"
                )
            cfg[fx_name] = {"kind": "gelu", "impl": fl["gelu_impl"], **LUT_TABLE_DEFAULTS["gelu"]}
        elif kind == "layer_norm":
            cfg[fx_name] = {"kind": "layer_norm", "impl": fl["ln_impl"], "reduce_impl": ri,
                             **LUT_TABLE_DEFAULTS["layer_norm"]}
            if fl["ln_impl"] not in LN_DECOMPOSED_IMPLS:
                # a streamed circuit: raises if it is not characterized at this row
                # length, lane count and width
                load_results.layer_norm_accel_ns(fl["ln_impl"], quant_data[fx_name]["hidden_size"], dw,
                                                 fl["ln_lanes"])
                cfg[fx_name]["ln_lanes"] = fl["ln_lanes"]
        elif kind == "rms_norm":
            if fl["rms_norm_impl"] == "exact":
                _require_exact_valid(
                    fx_name, "rms_norm", load_results.characterized_sizes("rms_norm"),
                    quant_data[fx_name]["hidden_size"], "hidden_size", variant="exact",
                )
            cfg[fx_name] = {"kind": "rms_norm", "impl": fl["rms_norm_impl"], "reduce_impl": ri,
                             **LUT_TABLE_DEFAULTS["rms_norm"]}
        elif kind == "silu":
            cfg[fx_name] = {"kind": kind, "impl": fl["silu_impl"], **LUT_TABLE_DEFAULTS[kind]}
        elif kind == "relu":
            cfg[fx_name] = {"kind": kind, "impl": fl["relu_impl"], **LUT_TABLE_DEFAULTS[kind]}
        elif kind == "elu":
            if fl["elu_impl"] == "exact":
                _require_exact_valid(fx_name, "elu", load_results.characterized_alphas("elu"),
                                     quant_data[fx_name]["alpha"], "alpha", variant="exact")
            cfg[fx_name] = {"kind": kind, "impl": fl["elu_impl"], **LUT_TABLE_DEFAULTS[kind]}
        elif kind == "leaky_relu":
            if fl["leaky_relu_impl"] == "exact":
                _require_exact_valid(
                    fx_name, "leaky_relu", load_results.characterized_alphas("leaky_relu"),
                    quant_data[fx_name]["negative_slope"], "negative_slope", variant="exact",
                )
            cfg[fx_name] = {"kind": kind, "impl": fl["leaky_relu_impl"], **LUT_TABLE_DEFAULTS[kind]}
        elif kind == "linear" or kind == "matmul":
            # the int8 matmul is fixed by the quantized model; its config holds the
            # settings of the units after it: tile reduce, scale (vec_el_mul), and the
            # requantize to int8, which is the quantize impl (requantize_impl)
            cfg[fx_name] = {"kind": kind, "reduce_impl": ri, "requantize_impl": fl["quantize_impl"]}
            if fl["quantize_impl"] != "lut" and uses_mantissa(fl["quantize_impl"]):
                cfg[fx_name]["requantize_mantissa_width"] = fl["quantize_mantissa_width"]
        elif kind == "group_norm":
            cfg[fx_name] = {"kind": "group_norm", "impl": "lut", "reduce_impl": ri,
                             **LUT_TABLE_DEFAULTS["group_norm"]}
        elif kind in ("avg_pool2d", "mean"):
            cfg[fx_name] = {"kind": kind, "reduce_impl": ri}
        elif kind in ("max_pool2d", "topk"):
            cfg[fx_name] = {"kind": kind, "argmax_width": xw}
        elif kind == "add":
            cfg[fx_name] = {"kind": "add"}
        elif kind == "mul":
            # smooth_impl: how a SmoothQuant x * (1 / s) is computed (SMOOTH_IMPLS); a mul
            # by a scalar constant ignores it
            cfg[fx_name] = {"kind": "mul", "smooth_impl": fl["smooth_impl"]}
        elif kind in ("row_absmax", "quantize_per_token", "dequantize_per_token", "where"):
            # one circuit each (token_quant, vec_el_mul / vec_add, vec_select): no choice
            cfg[fx_name] = {"kind": kind}
        elif kind == QUANTIZE:
            cfg[fx_name] = _quant_entry(QUANTIZE, fx_name, quant_data[fx_name], fl["quantize_impl"], fl,
                                        "quantize_mantissa_width", dw)
        elif kind == DEQUANTIZE:
            cfg[fx_name] = _quant_entry(DEQUANTIZE, fx_name, quant_data[fx_name], fl["dequantize_impl"], fl,
                                        "dequantize_mantissa_width", dw)
        else:
            raise NotImplementedError(f"{fx_name}: no config rule for kind {kind!r}")
    for entry in cfg.values():
        entry["data_width"] = dw
    return cfg


_SHARD_HEX_DIGITS = 2  # 256 shard directories


def _shard_dir(name: str) -> str:
    """Shard directory of a variant: the first hex digits of md5(name)."""
    return hashlib.md5(name.encode()).hexdigest()[:_SHARD_HEX_DIGITS]


def _generate_one_shard_worker(args) -> str:
    """Process-pool worker: builds and writes one variant's config file
    {name: {"config": ...}}. An invalid combination raises (the kernel's all_variants()
    must only yield valid ones)."""
    name, fl, alt_ops, quant_data, out_dir = args
    cfg = build_config_for_variant(alt_ops, fl, quant_data)
    shard = os.path.join(out_dir, _shard_dir(name))
    os.makedirs(shard, exist_ok=True)
    path = os.path.join(shard, f"{name}.json")
    with open(path, "w") as f:
        json.dump({name: {"config": cfg}}, f)
    return name


def _process_chunk(chunk: list) -> list:
    """Process-pool worker: _generate_one_shard_worker over one batch."""
    return [_generate_one_shard_worker(args) for args in chunk]


def generate_sharded(kernel: nn.Module, example_input: torch.Tensor, out_dir: str, variants_fn,
                     workers: int) -> None:
    """Traces the kernel once, then writes one file per variant of
    `variants_fn(alt_ops)` in a process pool (sharded directories under out_dir).
    Submission keeps at most workers * 4 batches in flight: pool.map would create every
    future up front."""
    traced = symbolic_trace(kernel)
    ShapeProp(traced).propagate(example_input)
    alt_ops, quant_data = discover_alt_ops(traced)

    kinds = {}
    for kind in alt_ops.values():
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"discovered {len(alt_ops)} alternative-capable ops: {kinds}")

    os.makedirs(out_dir, exist_ok=True)

    def _task_iter():
        for name, fl in variants_fn(alt_ops):
            yield (name, fl, alt_ops, quant_data, out_dir)

    CHUNK = 200
    MAX_IN_FLIGHT = workers * 4

    def _chunks():
        it = _task_iter()
        while True:
            batch = list(itertools.islice(it, CHUNK))
            if not batch:
                return
            yield batch

    written = 0
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        chunk_iter = _chunks()
        pending = set()
        for chunk in itertools.islice(chunk_iter, MAX_IN_FLIGHT):
            pending.add(pool.submit(_process_chunk, chunk))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                # re-raises a worker's error: an invalid variant stops the run
                written += len(fut.result())
                if written % 10000 == 0:
                    elapsed = time.time() - start
                    rate = written / elapsed if elapsed > 0 else 0.0
                    print(f"  {written} written -- {rate:.0f}/s", flush=True)
            for chunk in itertools.islice(chunk_iter, len(done)):
                pending.add(pool.submit(_process_chunk, chunk))
    elapsed = time.time() - start
    rate = written / elapsed if elapsed > 0 else 0.0
    print(f"done: {written} written, {elapsed:.1f}s, {rate:.0f}/s")
