"""Repeated structure in a lowered cim block, found from the IR alone (used by
compiler/xdsl_segments.py):

  find_repeat_segments  splits the op sequence (one structural descriptor per op,
                        xdsl_layer_decompose._node_desc) into maximal runs of a repeated
                        period and residual chunks; e.g. bert's 12 identical layers.
                        A stage whose first block differs (resnet50's downsampling
                        Bottleneck) splits into several smaller pieces, still covering
                        every op.
  extract_region        clones one contiguous run of ops into a standalone block
                        (operands from before the run become block arguments, results
                        used after it become its return), so a piece costs exactly
                        what its occurrence in the full block costs.
"""
from __future__ import annotations

from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block, SSAValue

from compiler.dialects import cim_dialect as cd
from compiler.dialects.common_dialect import ConstRefOp
from compiler.xdsl_timing import _defining_op

_MIN_REPEAT_COVERAGE = 6  # L * reps below this is treated as residual, not a real repeat


def find_repeat_segments(descs: list) -> list:
    """Partitions the whole descriptor sequence into maximal segments, each
    (start, period_length, repeat_count): a repeat (repeat_count >= 2 and
    L * reps >= _MIN_REPEAT_COVERAGE, which filters out tiny coincidental repeats) or a
    residual (repeat_count 1). Adjacent residuals are merged into one chunk. Candidate
    periods come only from later positions with the same descriptor."""
    n = len(descs)
    pos_by_desc: dict = {}
    for idx, d in enumerate(descs):
        pos_by_desc.setdefault(d, []).append(idx)

    segments = []
    i = 0
    while i < n:
        best = None
        for j in pos_by_desc.get(descs[i], []):
            length = j - i
            if length <= 0 or i + 2 * length > n:
                continue
            reps = 1
            while i + (reps + 1) * length <= n and descs[i + reps * length: i + (reps + 1) * length] == descs[i: i + length]:
                reps += 1
            if reps >= 2 and length * reps >= _MIN_REPEAT_COVERAGE and (best is None or length * reps > best[0] * best[1]):
                best = (length, reps)
        if best is not None:
            length, reps = best
            segments.append((i, length, reps))
            i += length * reps
        else:
            segments.append((i, 1, 1))
            i += 1

    merged = []
    for seg in segments:
        # a residual (repeat_count 1) joins the residual chunk before it, of any length
        if seg[2] == 1 and merged and merged[-1][2] == 1:
            start, _l, _r = merged[-1]
            merged[-1] = (start, seg[0] + 1 - start, 1)
        else:
            merged.append(seg)
    return merged


def _auxiliary_chain(op) -> list:
    """The ConstRefOp -> CimSaveStationaryOp chains feeding any operand of `op` (a LUT
    table, a matvec's stored weight: the mapping needs the stored value itself in the
    region, not a block argument), in dependency order; [] if none."""
    chain = []
    for operand in op.operands:
        cur_chain = []
        cur = _defining_op(operand)
        while cur is not None and isinstance(cur, (ConstRefOp, cd.CimSaveStationaryOp)):
            cur_chain.append(cur)
            cur = _defining_op(cur.operands[0]) if cur.operands else None
        chain.extend(reversed(cur_chain))
    return chain


def extract_region(values: list) -> tuple:
    """(block, region_values) for `values`, one contiguous occurrence of a piece: a new
    block with a clone of each value's defining op (and their _auxiliary_chain), and
    region_values[j] the clone of values[j], so a pf solved on the block maps back onto
    any occurrence with the same descriptors by position."""
    real_ops = [v.owner for v in values]

    ops_in_range: list = []
    seen_ops: set = set()
    for op in real_ops:
        for aux in _auxiliary_chain(op):
            if id(aux) not in seen_ops:
                seen_ops.add(id(aux))
                ops_in_range.append(aux)
        if id(op) not in seen_ops:
            seen_ops.add(id(op))
            ops_in_range.append(op)

    ops_id_set = {id(op) for op in ops_in_range}
    values_defined_in_range = {result for op in ops_in_range for result in op.results}

    external_inputs: list[SSAValue] = []
    seen_external: set = set()
    for op in ops_in_range:
        for operand in op.operands:
            if operand not in values_defined_in_range and operand not in seen_external:
                seen_external.add(operand)
                external_inputs.append(operand)

    new_block = Block(arg_types=[v.type for v in external_inputs])
    value_mapper = {ext: arg for ext, arg in zip(external_inputs, new_block.args)}

    for op in ops_in_range:
        cloned = op.clone(value_mapper)
        new_block.add_op(cloned)
        for old_r, new_r in zip(op.results, cloned.results):
            value_mapper[old_r] = new_r
    region_values = [value_mapper[v] for v in values]

    # only the real ops' results are outputs: a stored table or weight is shared by
    # other occurrences, it is copied, not an output of this region
    region_outputs: list[SSAValue] = []
    seen_output: set = set()
    for op in real_ops:
        for result in op.results:
            if result in seen_output:
                continue
            for use in result.uses:
                if id(use.operation) not in ops_id_set:
                    seen_output.add(result)
                    region_outputs.append(result)
                    break
    if not region_outputs:
        region_outputs = [values[-1]]

    new_block.add_op(ReturnOp(*[value_mapper[o] for o in region_outputs]))
    return new_block, region_values
