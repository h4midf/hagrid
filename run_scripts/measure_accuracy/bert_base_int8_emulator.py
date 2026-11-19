"""Runs the int8 bert-base (kernels/bert_base.py, BertBaseKernelInt8) with one variant
config the way the hardware computes it, node by node over its traced graph:

  - every value outside the int8 matmuls is fixed point at the config's data_width
    (cim_dialect.FIXED_POINT_FRAC_BITS: Q12.20 or Q8.8), held as its raw integer and
    saturated to data_width bits;
  - quantize / dequantize: the custom_op_library circuit of the config's impl, bit-exact
    to its RTL (custom_op_library/{quantize,dequantize}/rtl), with the (sh, m) encoding
    the compiler stamps (compiler/ops/quant_encoding.py); fc2's static symmetric
    quantize (kernels/gpt2.py quantize_per_tensor_static) is the same circuit at zero
    point 0;
  - int8 Linear / quantized.matmul: the exact integer accumulator, the rescale by the
    combined scale (+ bias) to fixed point, then the requantize on the quantize circuit
    of the config's requantize_impl (compiler/ops/requantize.py);
  - fc2's raw int8 Linear (kernels/gpt2.py int8_linear, dequantize_per_token): the exact
    integer accumulator, then the rescale (+ bias) to fixed point, no requantize;
  - quantize / dequantize / requantize impl "lut": the identity table at index
    round(x / scale + zp), and the table of (q - zp) * scale in fixed point (exact);
  - mul by a constant: vec_el_mul_vs_round.v; SmoothQuant's x * (1 / s) by its mul
    entry's smooth_impl: "vec_el_mul" (vec_el_mul_vv_round.v, 1 / s in fixed point) or
    "per_channel_mantissa" (folded into the next quantize: one (m, sh) per channel,
    encoding scale * s_j); add: a saturating add;
  - softmax / gelu / layer_norm: the config's impl (evaluate_variant_configs.py) on the
    fixed-point input, its output rounded to fixed point.

Modeling choices, not taken from an RTL:
  - The matmul rescale (cim.vec_el_mul by the combined scale, + bias) is exact, then
    rounded to fixed point once: the RTL vec_el_mul takes Q-format operands, and how the
    i32 accumulator enters it is not defined.
  - The int8 values are held as codes 0..255 with a zero point 0..255 (torch's quint8;
    a symmetric int8 value is code q + 128 at zero point 128); the circuits are signed
    int8, so they run on code - 128 with zero point - 128 (the same values).

Only impls "mant_round_zp" and "lut" are emulated for quantize / dequantize / requantize;
any other raises. An op whose scale the circuit cannot represent raises ValueError when the
emulator is built (quant_encoding.encode_mantissa).

With count_saturation, `saturated` counts, per op kind, the values clamped to the
data_width range.
"""
import functools
import operator

import torch
from torch.fx import Interpreter

from compiler.dialects.cim_dialect import FIXED_POINT_FRAC_BITS
from compiler.ops.quant_encoding import DEQUANTIZE, QUANTIZE, encode_mantissa
from compiler.xdsl_accel_width_stamp import reduce_tree_inputs
from compiler.xdsl_bridge import is_quantize_per_tensor_static, static_quantize_params
from custom_op_library.layer_norm.test_layer_norm import SEEDS as RSQRT_SEEDS
from custom_op_library.layer_norm.test_layer_norm import rsqrt_fixed
from kernels.gpt2 import INT8_MAX, int8_linear
from run_scripts.measure_accuracy.evaluate_variant_configs import gelu_impl, layer_norm_impl, softmax_impl

EMULATED_QUANT_IMPLS = ("mant_round_zp", "lut")
# SmoothQuant's x / s (common.SMOOTH_IMPLS): a vec_el_mul by the stored 1 / s, or folded
# into the next quantize as one (m, sh) per channel
SMOOTH_VEC_EL_MUL, SMOOTH_PER_CHANNEL = "vec_el_mul", "per_channel_mantissa"
# layer_norm impls emulated bit for bit as their streamed circuit (custom_op_library)
STREAM_LN_IMPLS = ("exact", "sole")
STREAM_RS = 40   # the row division's reciprocal shift (layer_norm_stream.v's RS)
# layer_norm impls emulated in fixed point as the compiler decomposes them
DECOMPOSED_LN_IMPLS = ("rsqrt",)
_CODE_OFFSET = 128   # quint8 code -> the circuits' signed int8
_VIEW_METHODS = ("view", "permute", "reshape")


def rsqrt_rtl(val: int, w: int) -> int:
    """rsqrt_q12_20.v / rsqrt_q8_8.v for any val >= 0: rsqrt_fixed (the RTL's Python
    model, val > 0), and for val = 0 the RTL's own path (msb_pos(0) = 0)."""
    if val > 0:
        return rsqrt_fixed(val, w)
    if val < 0:
        raise ValueError(f"rsqrt of a negative value {val}")
    frac = 20 if w == 32 else 8
    one_half_q = round(1.5 * (1 << frac))
    e = (0 - frac) >> 1
    parity = (0 - frac) & 1
    y = RSQRT_SEEDS[w][(parity, 0)]
    y = (y >> e) if e >= 0 else (y << (-e))
    y = (y * (one_half_q - 0)) >> frac
    return max(-(1 << (w - 1)), min((1 << (w - 1)) - 1, y))


class QTensor:
    """An int8 value: codes (int64 tensor, 0..255) with its scale and zero point."""

    def __init__(self, codes: torch.Tensor, scale: float, zero_point: int):
        self.codes, self.scale, self.zero_point = codes, scale, zero_point


def _require_impl(fx_name: str, impl: str) -> None:
    if impl not in EMULATED_QUANT_IMPLS:
        raise NotImplementedError(f"{fx_name}: impl {impl!r} is not emulated (only {EMULATED_QUANT_IMPLS})")


def _shift_round(p: torch.Tensor, sh: int) -> torch.Tensor:
    """(p + 2^(sh-1)) >>> sh, as the RTL: round to nearest, ties up. p is an integer
    tensor: >> on a signed integer is the arithmetic (floor) shift."""
    if sh == 0:
        return p
    return (p + (1 << (sh - 1))) >> sh


def _shift_round_vec(p: torch.Tensor, sh: torch.Tensor) -> torch.Tensor:
    """_shift_round with one shift per channel (the last dim)."""
    half = torch.where(sh > 0, 1 << (sh - 1).clamp(min=0), 0)
    return (p + half) >> sh


def _is_smoothing_mul(n) -> bool:
    """SmoothQuant's x * (1 / s): a mul by a stored per-channel vector (kernels/bert_base.py)."""
    return (n.op == "call_function" and n.target is operator.mul and len(n.args) == 2
            and isinstance(n.args[1], torch.fx.Node) and n.args[1].op == "get_attr")


@functools.lru_cache(maxsize=None)
def _encode_mantissa_cached(kind, fx_name, scale, zero_point, impl, mantissa_width, data_width) -> tuple:
    """quant_encoding.encode_mantissa, memoized (as sorted items; raises as it does)."""
    return tuple(sorted(encode_mantissa(kind, fx_name, scale, zero_point, impl, mantissa_width, data_width).items()))


def _to_signed(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >= (1 << (bits - 1)) else v


def _is_int8_linear(n) -> bool:
    return n.op == "call_function" and n.target is int8_linear


class Int8Emulator(Interpreter):
    def __init__(self, module: torch.fx.GraphModule, config: dict, data_width: int, count_saturation: bool):
        """count_saturation: count the clamped values per op kind in `saturated` (a
        diagnostic; the clamping is the same either way)."""
        super().__init__(module)
        self.count_saturation = count_saturation
        self.config = config
        self.data_width = data_width
        self.frac = FIXED_POINT_FRAC_BITS[data_width]
        self.q_max = (1 << (data_width - 1)) - 1
        self.q_min = -(1 << (data_width - 1))
        self._modules = dict(module.named_modules())
        self.saturated = {}
        for n in module.graph.nodes:
            cfg = config.get(n.name)
            if cfg is not None and cfg["data_width"] != data_width:
                raise ValueError(f"{n.name}: data_width {cfg['data_width']} != {data_width}")
        self.encodings = self._encode_all(module)

    # ---- the (sh, m) of every quantize / dequantize / requantize, computed up front

    def _encode(self, kind: str, fx_name: str, scale: float, zero_point: int, impl: str, cfg: dict,
                mantissa_key: str) -> dict:
        """The circuit's (sh, m, zp) (quant_encoding.encode_mantissa; cfg[mantissa_key] is
        its mantissa width), or for impl "lut" the identity table's scale and zero point."""
        _require_impl(fx_name, impl)
        if impl == "lut":
            return {"lut": True, "scale": scale, "zp": zero_point - _CODE_OFFSET}
        return dict(_encode_mantissa_cached(kind, fx_name, scale, zero_point - _CODE_OFFSET, impl, cfg[mantissa_key],
                                            self.data_width))

    def _encode_per_channel(self, n, mul, scale: float, zero_point: int, cfg: dict) -> dict:
        """The quantize `n` with SmoothQuant's 1 / s folded in (smooth_impl
        "per_channel_mantissa"): one (m, sh) per channel encoding scale * s_j."""
        if cfg["impl"] == "lut":
            raise NotImplementedError(f"{n.name}: a per-channel mantissa x / s needs a mantissa circuit, "
                                      "not the quantize lut")
        if len(mul.users) != 1:
            raise NotImplementedError(f"{mul.name}: x / s read by {len(mul.users)} ops, not only its quantize")
        s = (1.0 / self._attr(mul.args[1]).double()).tolist()
        encs = [self._encode(QUANTIZE, n.name, scale * sj, zero_point, cfg["impl"], cfg, "mantissa_width")
                for sj in s]
        return {"m": torch.tensor([e["m"] for e in encs]), "sh": torch.tensor([e["sh"] for e in encs]),
                "zp": encs[0]["zp"], "per_channel": True}

    def _output_qparams(self, n) -> tuple:
        """(scale, zero_point) of an int8 matmul's output (a quantized Linear or
        quantized.matmul node); raises for any other node."""
        if n.op == "call_module" and isinstance(self._modules[n.target], torch.ao.nn.quantized.Linear):
            sub = self._modules[n.target]
            return float(sub.scale), int(sub.zero_point)
        if n.op == "call_function" and n.target is torch.ops.quantized.matmul:
            return float(self._attr(n.args[2])), int(self._attr(n.args[3]))
        raise NotImplementedError(f"{n.name}: not an int8 matmul, its output scale is unknown")

    def _attr(self, node) -> torch.Tensor:
        return self.fetch_attr(node.target)

    def _quantize_qparams(self, n) -> tuple:
        """(scale, zero_point as a code) of a quantize node."""
        if is_quantize_per_tensor_static(n):
            scale, zero_point = static_quantize_params(self.module, n)
            return scale, zero_point + _CODE_OFFSET
        if n.args[3] != torch.quint8:
            raise NotImplementedError(f"{n.name}: dtype {n.args[3]} (only quint8)")
        return float(self._attr(n.args[1])), int(self._attr(n.args[2]))

    def _encode_all(self, module) -> dict:
        enc = {}
        for n in module.graph.nodes:
            cfg = self.config.get(n.name)
            if cfg is None:
                continue
            if cfg["kind"] == QUANTIZE:
                scale, zero_point = self._quantize_qparams(n)
                src = n.args[0]
                if _is_smoothing_mul(src) and self.config[src.name]["smooth_impl"] == SMOOTH_PER_CHANNEL:
                    enc[n.name] = self._encode_per_channel(n, src, scale, zero_point, cfg)
                else:
                    enc[n.name] = self._encode(QUANTIZE, n.name, scale, zero_point, cfg["impl"], cfg,
                                               "mantissa_width")
            elif cfg["kind"] == DEQUANTIZE:
                scale, zero_point = self._output_qparams(n.args[0])
                enc[n.name] = self._encode(DEQUANTIZE, n.name, scale, zero_point, cfg["impl"], cfg, "mantissa_width")
            elif cfg["kind"] in ("linear", "matmul") and not _is_int8_linear(n):
                scale, zero_point = self._output_qparams(n)
                enc[n.name] = self._encode(QUANTIZE, n.name, scale, zero_point, cfg["requantize_impl"], cfg,
                                           "requantize_mantissa_width")
        return enc

    # ---- fixed point

    def _saturate(self, raw: torch.Tensor, kind: str) -> torch.Tensor:
        if self.count_saturation:
            count = self.saturated.setdefault(kind, [0, 0])
            count[0] += int(((raw > self.q_max) | (raw < self.q_min)).sum())
            count[1] += raw.numel()
        return torch.clamp(raw, self.q_min, self.q_max)

    def to_fixed(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        """A real value to its raw fixed-point integer: round to nearest (ties up), saturate."""
        return self._saturate(torch.floor(x.double() * 2.0 ** self.frac + 0.5).long(), kind)

    def to_real(self, raw: torch.Tensor) -> torch.Tensor:
        return raw.double() / 2.0 ** self.frac

    # ---- circuits

    def _quantize(self, raw: torch.Tensor, enc: dict) -> torch.Tensor:
        """quant_mant_round_zp.v: q = clamp_int8(round(x * m * 2^-sh) + zp) (one (m, sh) per
        channel for a folded x / s); "lut": the identity table at index
        round(x / scale + zp) (compiler/ops/quantize_per_tensor.py), exact; returns codes."""
        if "lut" in enc:
            q = torch.round(self.to_real(raw) / enc["scale"]).long() + enc["zp"]
        elif "per_channel" in enc:
            q = _shift_round_vec(raw * enc["m"], enc["sh"]) + enc["zp"]
        else:
            q = _shift_round(raw * enc["m"], enc["sh"]) + enc["zp"]
        return torch.clamp(q, -128, 127) + _CODE_OFFSET

    def _dequantize(self, qt: QTensor, enc: dict) -> torch.Tensor:
        """dequant_mant_round_zp.v: x = clamp(round((q - zp) * m * 2^-sh)); "lut": the
        table of (q - zp) * scale in fixed point (compiler/ops/dequantize.py)."""
        if "lut" in enc:
            return self.to_fixed((qt.codes - qt.zero_point).double() * enc["scale"], DEQUANTIZE)
        return self._saturate(_shift_round((qt.codes - qt.zero_point) * enc["m"], enc["sh"]), DEQUANTIZE)

    def _int8_matmul(self, n, kind: str, acc: torch.Tensor, combined_scale, bias) -> QTensor:
        real = acc.double() * combined_scale
        if bias is not None:
            real = real + bias.double()
        scale, zero_point = self._output_qparams(n)
        return QTensor(self._quantize(self.to_fixed(real, kind), self.encodings[n.name]), scale, zero_point)

    # ---- the graph

    def run_node(self, n):
        if n.op in ("placeholder", "get_attr", "output"):
            return super().run_node(n)
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        if n.op == "call_method" and n.target in _VIEW_METHODS:
            if not isinstance(args[0], torch.Tensor):
                raise TypeError(f"{n.name}: {n.target} on {type(args[0]).__name__}")
            return super().run_node(n)
        cfg = self.config.get(n.name)
        if cfg is None:
            raise NotImplementedError(f"{n.name}: {n.op} {n.target} has no config entry and no emulation")
        kind = cfg["kind"]
        if kind == QUANTIZE:
            scale, zero_point = self._quantize_qparams(n)
            return QTensor(self._quantize(args[0], self.encodings[n.name]), scale, zero_point)
        if kind == DEQUANTIZE:
            _require_impl(n.name, cfg["impl"])
            return self._dequantize(args[0], self.encodings[n.name])
        if kind == "linear" and _is_int8_linear(n):
            # raw: the int32 accumulator, rescaled by the dequantize_per_token after it
            qt, w_int8 = args
            if qt.zero_point != _CODE_OFFSET:
                raise NotImplementedError(f"{n.name}: int8_linear on an asymmetric input")
            return torch.matmul((qt.codes - qt.zero_point).double(), w_int8.double().t())
        if kind == "linear":
            sub = self._modules[n.target]
            qt, w = args[0], sub.weight()
            if int(w.q_per_channel_zero_points().abs().max()) != 0:
                raise NotImplementedError(f"{n.name}: weight zero points are not 0")
            acc = torch.matmul((qt.codes - qt.zero_point).double(), w.int_repr().double().t())
            combined = qt.scale * w.q_per_channel_scales().double()
            return self._int8_matmul(n, kind, acc, combined, sub.bias())
        if kind == "matmul":
            a, b = args[0], args[1]
            acc = torch.matmul((a.codes - a.zero_point).double(), (b.codes - b.zero_point).double())
            return self._int8_matmul(n, kind, acc, a.scale * b.scale, None)
        if kind == "dequantize_per_token":
            # acc * (row_max / 127) * (col_max / 127) + bias, rounded to fixed point once
            acc, row_max, col_max = args[0], args[1], args[2]
            bias = args[3] if len(args) > 3 else kwargs.get("bias")
            real = acc * (row_max.double() / INT8_MAX) * (col_max.double() / INT8_MAX)
            if bias is not None:
                real = real + bias.double()
            return self.to_fixed(real, kind)
        if kind == "mul":
            x, c = args
            if n.target is not operator.mul:
                raise NotImplementedError(f"{n.name}: mul target {n.target}")
            if isinstance(c, float):
                # vec_el_mul_vs_round.v: the constant in fixed point, product >> frac, round
                y = int(self.to_fixed(torch.tensor(c), "mul_const"))
                return self._saturate(_shift_round(x * y, self.frac), "mul")
            if isinstance(c, torch.Tensor) and c.is_floating_point():
                if cfg["smooth_impl"] == SMOOTH_PER_CHANNEL:
                    # folded into the next quantize's per-channel (m, sh) (_encode_per_channel)
                    return x
                if cfg["smooth_impl"] != SMOOTH_VEC_EL_MUL:
                    raise ValueError(f"{n.name}: smooth_impl {cfg['smooth_impl']!r}")
                # vec_el_mul_vv_round.v: the stored vector in fixed point
                y = self.to_fixed(c, "mul_const")
                return self._saturate(_shift_round(x * y, self.frac), "mul")
            raise NotImplementedError(f"{n.name}: mul by {type(c).__name__}")
        if kind == "add":
            return self._saturate(args[0] + args[1], kind)
        if kind == "softmax":
            return self.to_fixed(softmax_impl(self.to_real(args[0]), cfg["impl"]), kind)
        if kind == "gelu":
            return self.to_fixed(gelu_impl(self.to_real(args[0]), cfg["impl"]), kind)
        if kind == "layer_norm":
            sub = self._modules[n.target]
            if cfg["impl"] in STREAM_LN_IMPLS:
                return self._layer_norm_stream(args[0], sub, cfg["impl"], cfg)
            if cfg["impl"] in DECOMPOSED_LN_IMPLS:
                return self._layer_norm_decomposed(args[0], sub, cfg)
            return self.to_fixed(layer_norm_impl(self.to_real(args[0]), sub.weight.double(), sub.bias.double(),
                                                 sub.eps, cfg["impl"]), kind)
        raise NotImplementedError(f"{n.name}: kind {kind!r} is not emulated")

    def _shift_round_t(self, p: torch.Tensor) -> torch.Tensor:
        """clamp((p + 2^(FRAC-1)) >>> FRAC) to the data width."""
        return torch.clamp(_shift_round(p, self.frac), self.q_min, self.q_max)

    def _layer_norm_stream(self, raw: torch.Tensor, sub, impl: str, cfg: dict) -> torch.Tensor:
        """The streamed layer-norm circuit of `impl` (custom_op_library), bit for bit
        (each one's test_*_stream.py model), over every row of the last dim. The row
        sums are divided by the row length through the reciprocal; squares and their
        sums are 2W bits, var is clamped to W bits; gamma / beta are per element, in
        fixed point. eps below one LSB is one LSB (rsqrt needs val > 0).
          exact  layer_norm_stream.v
          sole   ailayernorm_stream.v (cfg["ptf_alpha"]: the per-channel PTF exponents)"""
        w = self.data_width
        frac = self.frac
        row = raw.shape[-1]
        rows = raw.reshape(-1, row)
        mask = (1 << w) - 1
        recip = ((1 << STREAM_RS) + row // 2) // row
        half = 1 << (frac - 1)

        def row_q(sums):
            """floor(sum * recip / 2^RS) per row, as Python ints."""
            return [(int(s) * recip) >> STREAM_RS for s in sums.tolist()]

        def low_w(vals):
            return torch.tensor([_to_signed(v & mask, w) for v in vals], dtype=torch.long)

        def clamp_w(vals):
            return [max(self.q_min, min(self.q_max, v)) for v in vals]

        def clamp_t(t):
            return torch.clamp(t, self.q_min, self.q_max)

        def per_row(vals):
            return torch.tensor(vals, dtype=torch.long)[:, None]

        eps = max(int(self.to_fixed(torch.tensor(sub.eps), "layer_norm")), 1)
        gamma = self.to_fixed(sub.weight.detach(), "layer_norm")
        beta = self.to_fixed(sub.bias.detach(), "layer_norm")

        if impl == "exact":
            mean = low_w(row_q(rows.sum(-1)))
            d = clamp_t(rows - mean[:, None])
            var = clamp_w(row_q(_shift_round(d * d, frac).sum(-1)))
            var_eps = clamp_w([v + eps for v in var])
            inv_std = per_row([rsqrt_fixed(v, w) for v in var_eps])
            norm = self._shift_round_t(d * inv_std)
        elif impl == "sole":
            alpha = torch.tensor(cfg["ptf_alpha"], dtype=torch.long)
            if alpha.numel() != row:
                raise ValueError(f"layer_norm sole: {alpha.numel()} PTF exponents for a row of {row}")
            s = alpha + frac
            pos = s > 0
            sp = torch.clamp(s, min=0)
            sn = torch.clamp(-s, min=0)
            q = torch.where(pos, torch.div(rows + torch.where(pos, 1 << (sp - 1).clamp(min=0), 0), 1 << sp,
                                           rounding_mode="floor"),
                            rows * (1 << sn))
            x_q = torch.clamp(q, -128, 127)
            xs_wide = torch.where(s >= 0, x_q * (1 << sp),
                                  torch.div(x_q + torch.where(sn > 0, 1 << (sn - 1).clamp(min=0), 0), 1 << sn,
                                            rounding_mode="floor"))
            xs = clamp_t(xs_wide)
            mag = x_q.abs()
            high = mag >= 64
            xc = torch.clamp(torch.where(high, mag >> 4, mag >> 2), max=15)
            xc2 = torch.where(high, (xc * xc) << 4, xc * xc)
            sh = 2 * alpha + frac
            shp = torch.clamp(sh, min=0)
            shn = torch.clamp(-sh, min=0)
            xc2s = torch.where(sh >= 0, xc2 * (1 << shp),
                               torch.div(xc2 + torch.where(shn > 0, 1 << (shn - 1).clamp(min=0), 0), 1 << shn,
                                         rounding_mode="floor"))
            ex = low_w(row_q(xs.sum(-1)))
            inv_std = []
            for e, e2 in zip(ex.tolist(), row_q(xc2s.sum(-1))):
                var = max(max(self.q_min, min(self.q_max, (e2 << 4) - ((e * e + half) >> frac))), 1)
                inv_std.append(rsqrt_fixed(max(max(self.q_min, min(self.q_max, var + eps)), 1), w))
            d = clamp_t(xs - ex[:, None])
            norm = self._shift_round_t(d * per_row(inv_std))
        else:
            raise NotImplementedError(f"layer_norm impl {impl!r} has no streamed model")
        scaled = self._shift_round_t(norm * gamma)
        y = self._saturate(scaled + beta, "layer_norm")
        return y.reshape(raw.shape)

    def _reduce(self, v: torch.Tensor, impl: str) -> torch.Tensor:
        """cim.reduce_vec over the last dim, on the reduce tree of the smallest
        characterized size that fits (compiler/xdsl_accel_width_stamp.py), the input
        zero-padded to it. "tree" (reduce_tree.v): the exact sum, saturated to the data
        width at the W-bit result. "vec_add" (vec_add_vv.v over log2 halving rounds):
        each round adds the upper half to the lower half and saturates."""
        n = reduce_tree_inputs(v.shape[-1])
        if v.shape[-1] > n:
            raise NotImplementedError(f"reduce of {v.shape[-1]} values (more than one {n}-input pass)")
        pad = torch.zeros(*v.shape[:-1], n - v.shape[-1], dtype=v.dtype)
        v = torch.cat([v, pad], dim=-1)
        if impl == "tree":
            return self._saturate(v.sum(-1, keepdim=True), "reduce")
        if impl == "vec_add":
            while v.shape[-1] > 1:
                h = v.shape[-1] // 2
                v = self._saturate(v[..., :h] + v[..., h:], "reduce")
            return v
        raise ValueError(f"reduce_impl {impl!r}")

    def _mul_const(self, x: torch.Tensor, c: float) -> torch.Tensor:
        """vec_el_mul_vs_round.v: the constant in fixed point, product >> frac, round, saturate."""
        y = int(self.to_fixed(torch.tensor(c), "mul_const"))
        return self._saturate(_shift_round(x * y, self.frac), "mul")

    def _mul(self, a: torch.Tensor, b: torch.Tensor, kind: str) -> torch.Tensor:
        """vec_el_mul_vv_round.v."""
        return self._saturate(_shift_round(a * b, self.frac), kind)

    def _layer_norm_decomposed(self, raw: torch.Tensor, sub, cfg: dict) -> torch.Tensor:
        """compiler/ops/layer_norm.py's decomposition (impl "rsqrt"), op for op on the
        vector units at the data width: mean = reduce(x) * (1/H), centered = x - mean,
        var = reduce(centered^2) * (1/H), var_eps = var + eps, inv_std = the CMOS rsqrt
        circuit (rsqrt_q12_20.v / rsqrt_q8_8.v), y = centered * inv_std * gamma + beta.
        The reduce circuit is cfg["reduce_impl"]."""
        if cfg["impl"] != "rsqrt":
            raise NotImplementedError(f"layer_norm impl {cfg['impl']!r} is not emulated in fixed point")
        hidden = raw.shape[-1]
        impl = cfg["reduce_impl"]
        mean = self._mul_const(self._reduce(raw, impl), 1.0 / hidden)
        neg_mean = self._mul_const(mean, -1.0)
        centered = self._saturate(raw + neg_mean, "add")
        sq = self._mul(centered, centered, "layer_norm_square")
        var = self._mul_const(self._reduce(sq, impl), 1.0 / hidden)
        eps = int(self.to_fixed(torch.tensor(sub.eps), "layer_norm"))
        var_eps = self._saturate(var + eps, "add")
        inv_std = torch.tensor([rsqrt_rtl(int(v), self.data_width) for v in var_eps.flatten().tolist()],
                               dtype=torch.long).reshape(var_eps.shape)
        y = self._mul(centered, inv_std, "mul")
        y = self._mul(y, self.to_fixed(sub.weight.detach(), "layer_norm"), "mul")
        return self._saturate(y + self.to_fixed(sub.bias.detach(), "layer_norm"), "layer_norm")

    def numeric_signature(self) -> tuple:
        """What this emulator computes beyond its graph and data width: every quantize /
        dequantize / requantize encoding (per channel for a folded x / s), each x / s
        mul's smooth_impl and every config entry's impl. Two emulators of the same graph
        and data width with equal signatures compute the same values."""
        enc = []
        for name in sorted(self.encodings):
            e = self.encodings[name]
            enc.append((name, tuple(sorted((k, tuple(v.tolist()) if isinstance(v, torch.Tensor) else v)
                                           for k, v in e.items()))))
        impls = tuple(sorted((name, cfg.get("impl"), cfg.get("smooth_impl"))
                             for name, cfg in self.config.items()))
        return (self.data_width, tuple(enc), impls)

    def forward_real(self, x: torch.Tensor) -> torch.Tensor:
        """The encoder on a real input (rounded to fixed point); the output as real values."""
        return self.to_real(self.run(self.to_fixed(x, "input")))
