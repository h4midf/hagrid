"""SmoothQuant's per-channel x * (1 / s) (kernels/bert_base.py) folded into the quantize
circuit after it, for a mul whose config entry has smooth_impl "per_channel_mantissa"
(variant_generators/common.py SMOOTH_IMPLS):

  cim.vec_el_mul(x, save_stationary(1 / s)) -> cim.quantize      becomes
  cim.quantize(x, channel_table = save_stationary(1 / s))

The vec_el_mul is gone; the quantize circuit (a mantissa circuit: its m and sh are input
ports, custom_op_library/quantize/rtl) reads one (m, sh) per channel from the stored
table, which encodes scale * s_j (one entry per channel, in place of the 1 / s vector).
smooth_impl "vec_el_mul" leaves the mul as it is. Raises for a per-channel mul that is
not a vector-by-stored-vector product read only by a quantize circuit (e.g. a quantize
"lut", which has no mantissa to fold into).

Runs on the cim module after convert_stationary_to_cim_stationary (compiler/lowering.py).
"""
from compiler.dialects import cim_dialect as cd
from compiler.dialects.rewrite_helpers import shape_of
from compiler.ops.dispatch import fx_name_of

PER_CHANNEL = "per_channel_mantissa"
VEC_EL_MUL = "vec_el_mul"


def fold_per_channel_smoothing(module, config: dict) -> int:
    """In place; returns the number of folded muls."""
    folded = 0
    for op in list(module.walk()):
        if type(op) is not cd.CimVecElMulOp:
            continue
        entry = config.get(fx_name_of(op))
        if entry is None or entry["kind"] != "mul":
            continue
        if entry["smooth_impl"] == VEC_EL_MUL:
            continue
        if entry["smooth_impl"] != PER_CHANNEL:
            raise ValueError(f"{fx_name_of(op)}: smooth_impl {entry['smooth_impl']!r}")
        if len(op.operands) != 2 or type(op.operands[1].owner) is not cd.CimSaveStationaryOp:
            # a mul by a scalar constant (e.g. attention's 1 / sqrt(d)): not an x / s
            continue
        x, table = op.operands[0], op.operands[1]
        channels = shape_of(table.type)
        if len(channels) != 1 or channels[0] != shape_of(x.type)[-1]:
            raise ValueError(f"{fx_name_of(op)}: a per-channel x / s needs a [{shape_of(x.type)[-1]}] "
                             f"table, got {channels}")
        users = [use.operation for use in op.results[0].uses]
        if len(users) != 1 or type(users[0]) is not cd.CimQuantizeOp:
            raise NotImplementedError(f"{fx_name_of(op)}: a per-channel x / s folds only into one quantize "
                                      f"circuit, it is read by {[type(u).__name__ for u in users]}")
        quant = users[0]
        if quant.mantissa_width is None:
            raise NotImplementedError(f"{fx_name_of(quant)}: a per-channel x / s needs a mantissa circuit, "
                                      f"got {quant.variant.data!r}")
        quant.operands = [x, table]
        op.detach()
        op.erase()
        folded += 1
    return folded
