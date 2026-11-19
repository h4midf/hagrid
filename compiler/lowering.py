"""The lowering of a kernel to its tiled cim block, written once: every sweep and
compiler/dump_ir.py run these same steps.

  trace_kernel        kernel -> traced, shape-propagated fx graph
  lower_to_cinm       fx graph -> gen -> cinm (config applied)
  lower_to_tiled_cim  cinm -> cim, fixed-point typed and tiled
  build_tiled_cim_block   all of the above for one model and variant flags
"""
import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp

from compiler.xdsl_accel_width_stamp import stamp_accel_widths
from compiler.xdsl_bridge import gen_fx_to_xdsl
from compiler.xdsl_cinm_canon import run_full_cinm_canon, run_inline_shape_info_canon, run_quantize_lut_canon
from compiler.xdsl_fixed_point import assign_fixed_point_types
from compiler.xdsl_fold_smoothing import fold_per_channel_smoothing
from compiler.xdsl_gen_lowering import run_gen_to_cinm_canon, run_rewrite_cinm_applicable_canon
from compiler.xdsl_lowering import convert_stationary_to_cim_stationary, lower_cinm_to_cim
from compiler.xdsl_tiling import run_convert_static_values_canon, tile_matmuls
from kernels.registry import get_kernel
from targets.target_loader import DualModuleTarget, EnsembleTarget, SingleModuleTarget
from targets.target_support import TARGET
from variant_generators.common import build_config_for_variant, discover_alt_ops

# model -> its traced kernel, one model at a time: tracing is the same for every
# variant, and the lowering never mutates it
_TRACED_KERNEL_CACHE: dict = {}


def trace_kernel(model: str) -> torch.fx.GraphModule:
    """The traced, shape-propagated kernel of `model` (cached)."""
    traced = _TRACED_KERNEL_CACHE.get(model)
    if traced is None:
        _TRACED_KERNEL_CACHE.clear()
        kernel_class, input_shape, batch_size = get_kernel(model)
        traced = symbolic_trace(kernel_class())
        # the int8 model quantizes its own float input
        ShapeProp(traced).propagate(torch.randn(batch_size, *input_shape))
        _TRACED_KERNEL_CACHE[model] = traced
    return traced


def lower_to_cinm(traced: torch.fx.GraphModule, config: dict) -> ModuleOp:
    """The cinm module of `traced` under the variant `config`."""
    module = gen_fx_to_xdsl(traced)
    run_rewrite_cinm_applicable_canon(module)
    run_gen_to_cinm_canon(module)
    run_full_cinm_canon(module, config)
    stamp_accel_widths(module, config)
    run_inline_shape_info_canon(module)
    run_quantize_lut_canon(module, config)
    return module


def lower_to_tiled_cim(module: ModuleOp, config: dict,
                       target: SingleModuleTarget | DualModuleTarget | EnsembleTarget) -> None:
    """Lowers the cinm `module` in place to cim, fixed-point typed and tiled to the
    array rows of the module each matvec runs in."""
    lower_cinm_to_cim(module)
    convert_stationary_to_cim_stationary(module)
    fold_per_channel_smoothing(module, config)
    assign_fixed_point_types(module, config)
    run_convert_static_values_canon(module)
    match target:
        case DualModuleTarget():
            # a matvec with a stored rhs runs in the analog module, one with both
            # operands dynamic in the digital module: each is tiled to its module's rows
            tile_matmuls(module, target.analog.array.rows, dynamic_rhs_rows=target.digital.array.rows)
        case SingleModuleTarget():
            tile_matmuls(module, target.geometry.rmu.rows)
        case EnsembleTarget():
            # a stored rhs on the RRAM macro's rows; a dynamic rhs in the SRAM macros, one
            # PE's weights (its share of the array) per row tile
            tile_matmuls(module, target.rram.array.rows, dynamic_rhs_rows=target.sram.weights_per_pe)
        case _:
            raise TypeError(f"no tiling for target type {type(target).__name__}")
    stamp_accel_widths(module, config)


def func_block(module: ModuleOp):
    """The body block of the module's one func."""
    return next(op for op in module.body.ops if isinstance(op, FuncOp)).body.block


def build_tiled_cim_block(model: str, flags: dict):
    """The tiled cim block of `model` under the variant `flags`, on TARGET."""
    traced = trace_kernel(model)
    alt_ops, quant_data = discover_alt_ops(traced)
    config = build_config_for_variant(alt_ops, flags, quant_data)
    module = lower_to_cinm(traced, config)
    lower_to_tiled_cim(module, config, TARGET)
    return func_block(module)
