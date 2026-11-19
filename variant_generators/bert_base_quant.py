"""bert-base's variant generator, on the int8 bert-base the compiler maps (kernels/
bert_base.py: SmoothQuant W8A8, fc2's output raw). Each variant is the offload policy's
flags (run_scripts/offload_policy_common.py offload_policy_flags) with these axes
changed, the ones that change the accuracy:

  smooth                      the checkpoint: SmoothQuant on Wq / Wk / Wv, fc1 and fc2
                              ("qkv_fc1_fc2", kernel "bert-base") or on fc1 and fc2 only
                              ("fc1_fc2", kernel "bert-base-ffn-smooth": 12 fewer x / s
                              ops); MODEL_OF_SMOOTH
  smooth_impl                 common.SMOOTH_IMPLS: x / s as an explicit vec_el_mul, or a
                              per-channel quantize mantissa
  data_width                  Q12.20, Q8.8 (common.DATA_WIDTHS)
  sm_impl                     SM_IMPLS
  gelu_impl                   GELU_IMPLS
  ln_impl                     LN_IMPLS: the streamed exact and SOLE circuits
                              (custom_op_library/layer_norm/characterize_stream.py)
  quantize_impl               the mantissa circuit at QUANT_MANTISSA_WIDTH, or "lut" (the
                              identity table); also the requantize at the end of every
                              int8 matmul. A per-channel-mantissa x / s needs the circuit
  dequantize_impl             the mantissa circuit at DEQUANT_MANTISSA_WIDTH[data_width],
                              or "lut"

Left out, as dominated by another option on the same op (larger and slower with no
better accuracy, from the synthesized circuits and the single-axis accuracy runs):
  softmax gqa_lut (its exp approximation goes negative below -5.875, breaking the sum),
          quark (the largest and slowest pipeline, 141k um^2 / 30.5 ns per row, accuracy
          within noise of exact's 128k um^2 / 27.6 ns), ibert (130k um^2 / 28.0 ns,
          slightly worse than exact)
  gelu    quark (+1.1 ppl at Q8.8, 9.2k um^2), gqa_lut (5.5k um^2 / 3.0 ns at Q12.20 vs
          exact's 3.9k um^2 / 2.0 ns, no better), ibert (Q8.8 only: 6.5x exact's area,
          3.8x its delay)
  layer norm  the decomposed "lut" / "rsqrt" (they sum and square at the data width,
          which saturates on bert-base's outlier rows), QUARK and I-BERT (dropped)
  mantissa widths above the narrowest one that represents every op's scale: the
          circuit's area grows ~3x from 8 to 31 bits, the accuracy is flat across the
          representable widths (within +-0.07 at Q12.20)

Every other field is the policy's: ln_lanes and reduce_impl change only the hardware
(the lane count and the reduce circuit compute the same values), and are swept by the
hardware DSE, not here.

The variants are evaluated by run_scripts/measure_accuracy/bert_base_quant_sweep.py.
"""
from run_scripts.offload_policy_common import offload_policy_flags
from variant_generators import common

SM_IMPLS = ("exact", "lut", "sole")
GELU_IMPLS = ("exact", "lut", "wang_tcad")
LN_IMPLS = ("exact", "sole")
# the narrowest mantissa widths that represent every op's scale (bert_base_quant_ppl
# sweep: dequantize 8-16 bits cannot at Q12.20, 8 bits cannot at Q8.8)
QUANT_MANTISSA_WIDTH = 8
DEQUANT_MANTISSA_WIDTH = {32: 24, 16: 12}
# the kernel (kernels/registry.py) of each SmoothQuant checkpoint
MODEL_OF_SMOOTH = {"qkv_fc1_fc2": "bert-base", "fc1_fc2": "bert-base-ffn-smooth"}


def all_variants():
    """A generator of (name, flags) over the axes above."""
    for smooth in MODEL_OF_SMOOTH:
        for smi_x in common.SMOOTH_IMPLS:
            for dw in common.DATA_WIDTHS:
                for smi in SM_IMPLS:
                    for gei in GELU_IMPLS:
                        for lni in LN_IMPLS:
                            for qi in (common.POLICY_QUANT_IMPL, "lut"):
                                if qi == "lut" and smi_x == "per_channel_mantissa":
                                    continue  # the per-channel x / s is a mantissa per channel
                                for di in (common.POLICY_QUANT_IMPL, "lut"):
                                    flags = offload_policy_flags()
                                    flags.update({"smooth": smooth, "smooth_impl": smi_x, "data_width": dw,
                                                  "sm_impl": smi, "gelu_impl": gei, "ln_impl": lni,
                                                  "quantize_impl": qi, "dequantize_impl": di})
                                    # a mantissa width only for the mantissa circuit
                                    del flags["quantize_mantissa_width"], flags["dequantize_mantissa_width"]
                                    q = f"q-{qi}"
                                    if qi != "lut":
                                        flags["quantize_mantissa_width"] = QUANT_MANTISSA_WIDTH
                                        q += str(QUANT_MANTISSA_WIDTH)
                                    q += f"_dq-{di}"
                                    if di != "lut":
                                        flags["dequantize_mantissa_width"] = DEQUANT_MANTISSA_WIDTH[dw]
                                        q += str(DEQUANT_MANTISSA_WIDTH[dw])
                                    yield (f"{smooth}_{smi_x}__dw{dw}__sm-{smi}_ge-{gei}_ln-{lni}__{q}", flags)
