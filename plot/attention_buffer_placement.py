"""Figure-only FIFO approximation from exported batches; no compiler imports.

The export omits exact first-batch times and read transactions. Approximate them
with uniform batches and logical-prefix consumption at consumer completion. Views
share a root allocation. Keep staged dynamic RHS storage already in the mapping.
Reserve each FIFO's peak live size through its live envelope, reusing idle WMUs.
A hardcoded backpressure knob delays Q/K/V producer batches; dependent batch
completion is delayed until its inputs arrive. This is an illustration, not a
new compiler schedule or a cycle-accurate simulator.
"""
import copy
import math

import numpy as np

ALIASES = {'cim.view', 'cim.reshape', 'cim.permute', 'cim.transpose'}
STATIC = {'const.ref', 'cim.save_stationary'}
PROJECTIONS = {'_quantized_layers_0_attn_wq', '_quantized_layers_0_attn_wk', '_quantized_layers_0_attn_wv'}
PARTIAL_NAMES = {'_quantized_layers_0_attn_wq': 'q_p', '_quantized_layers_0_attn_wk': 'k_p',
                 '_quantized_layers_0_attn_wv': 'v_p', 'matmul_1': 'z_p', '_quantized_layers_0_attn_wo': 'o_p'}


def place_buffers(results, cap, stall_ns, names):
    ops = results['mapping']['ops']
    by = {op['id']: op for op in ops}

    def root(idx):
        while idx in by and by[idx]['op'] in ALIASES:
            idx = by[idx]['inputs'][0]['from']
        return idx

    dynamic = {w['op']: w for w in results['placement']['working'] if 'held_addresses' in w}
    staged = {root(w['rhs_value']) for w in dynamic.values()}
    times, amounts = {}, {}
    consumers = {}
    for op in ops:
        idx = op['id']
        if op['op'] in ALIASES:
            continue
        if op['op'] in STATIC:
            continue
        size = math.prod(op['shape'])
        spec = op['schedule']
        out = np.minimum(size, np.arange(1, spec['count'] + 1, dtype=np.int64) * (spec['unit_size'] or size))
        ready = spec['start_ns'] + np.arange(1, spec['count'] + 1) * spec['interval_ns']
        if op['op'] == 'cim.MatvecOp' and op['attributes'].get('fx_name') in PROJECTIONS:
            ready = ready + stall_ns * out / size
        for edge in op['inputs']:
            parent = root(edge['from'])
            if parent not in times:
                continue  # stationary operand or external streamed input
            parent_size = math.prod(by[parent]['shape'])
            needed = np.minimum(parent_size, np.ceil(out * (parent_size / size))).astype(np.int64)
            if idx in dynamic and parent == root(dynamic[idx]['rhs_value']):
                needed[:] = parent_size  # stage the entire dynamic RHS before use
            indices = np.searchsorted(amounts[parent], needed, side='left')
            parent_spec = by[parent]['schedule']
            nominal_input = parent_spec['start_ns'] + (int(indices[0])+1) * parent_spec['interval_ns']
            nominal_output = spec['start_ns'] + spec['interval_ns']
            service = max(0.0, nominal_output - nominal_input)
            ready = np.maximum(ready, times[parent][indices] + service)
            consumers.setdefault(parent, set()).add(idx)
        # Each batch retains its original completion deadline unless inputs or
        # backpressure make it later. Prefix dependencies preserve FIFO order.
        times[idx], amounts[idx] = ready, out

    fixed = []
    for w in results['placement']['working']:
        op = by[w['op']]
        end = max(w['live_end_ns'], float(times[w['op']][-1]))
        run_label = names[op['id']] if op['op'] == 'cim.lut_search' else op['attributes']['fx_name']
        fixed.append({'kind': 'run', 'label': run_label,
                      'op': w['op'], 'addresses': w['addresses'],
                      'start_ns': w['live_start_ns'], 'end_ns': end})
        if 'held_addresses' in w:
            rhs = root(w['rhs_value'])
            fixed.append({'kind': 'staged', 'label': names.get(rhs, '...'),
                          'addresses': w['held_addresses'], 'start_ns': float(times[rhs][0]),
                          'end_ns': max(w['rhs_hold_end_ns'], float(times[rhs][-1]))})

    cell_bits = results['array']['rows'] * results['array']['cols']
    requests = []
    for idx, produced in times.items():
        if idx in staged or idx not in consumers:
            continue
        op = by[idx]
        size = math.prod(op['shape'])
        uses = sorted(consumers[idx])
        events = np.unique(np.concatenate([produced, *[times[c] for c in uses]]))
        counts = np.searchsorted(produced, events, side='right')
        made = np.where(counts > 0, amounts[idx][np.maximum(0, counts-1)], 0)
        read = np.full(len(events), size, dtype=np.int64)
        for c in uses:
            count = np.searchsorted(times[c], events, side='right')
            completed = np.where(count > 0, amounts[c][np.maximum(0, count-1)], 0)
            consumed = np.floor(completed * (size / math.prod(by[c]['shape'])) + 1e-9).astype(np.int64)
            read = np.minimum(read, consumed)
        live = np.maximum(0, made - read)
        positive = np.flatnonzero(live)
        if not len(positive):
            continue
        peak_bits = int(live.max()) * op['bits']
        first, last = int(positive[0]), int(positive[-1])
        assert last + 1 < len(events), f'op {idx}: FIFO never drains'
        fx = op['attributes'].get('fx_name', '')
        partial = op['op'] == 'cim.MatvecOp' and fx in PARTIAL_NAMES
        label = PARTIAL_NAMES[fx] if partial else names.get(idx, '...')
        if op['op'] == 'cim.vec_add' and fx in PROJECTIONS:
            label = {'_quantized_layers_0_attn_wq': 'q_b', '_quantized_layers_0_attn_wk': 'k_b',
                     '_quantized_layers_0_attn_wv': 'v_b'}[fx]
        requests.append({'kind': 'partial' if partial else 'temporary', 'label': label,
                         'producer': idx, 'consumers': uses, 'start_ns': float(events[first]),
                         'end_ns': float(events[last+1]), 'peak_bits': peak_bits,
                         'tensor_bits': size * op['bits'], 'buffer_arrays': math.ceil(peak_bits/cell_bits)})

    boxes = copy.deepcopy(fixed)
    order = range(cap)  # First-fit across every WMU, including idle compute arrays.
    for request in sorted(requests, key=lambda r: (r['start_ns'], r['producer'])):
        busy = {a for box in boxes
                if box['start_ns'] < request['end_ns'] and request['start_ns'] < box['end_ns']
                for lo, hi in box['addresses'] for a in range(lo, hi+1)}
        selected = [a for a in order if a not in busy][:request['buffer_arrays']]
        if len(selected) != request['buffer_arrays']:
            return None, f"op {request['producer']} ({request['label']}) needs {request['buffer_arrays']} WMUs, only {len(selected)} available"
        runs = []
        for a in sorted(selected):
            if runs and a == runs[-1][1]+1:
                runs[-1][1] = a
            else:
                runs.append([a, a])
        request['addresses'] = runs
        boxes.append(request)
    # Check every allocation, including fixed regions if their execution extended.
    for i, a in enumerate(boxes):
        for b in boxes[i+1:]:
            if a['start_ns'] >= b['end_ns'] or b['start_ns'] >= a['end_ns']:
                continue
            if any(max(lo, x) <= min(hi, y) for lo, hi in a['addresses'] for x, y in b['addresses']):
                return None, f"overlapping {a['label']} / {b['label']}"
    return {'boxes': boxes, 'requests': requests, 'times': times, 'stall_ns': stall_ns}, None


def allocate(results, cap, names):
    # Keep the compiler's PFs and resource assignments; only the illustration's
    # Q/K/V output rates and dependent FIFO readiness change.
    result, failure = place_buffers(results, cap, 0.0, names)
    if result is not None:
        return result
    print(f'Without stalls: {failure}')
    high = 10_000.0
    low = 0.0
    for _ in range(12):
        result, failure = place_buffers(results, cap, high, names)
        if result is not None:
            break
        low, high = high, high * 2
    else:
        raise ValueError(f'Cannot place live buffers within {cap} WMUs: {failure}')
    # Reproducible 1-us resolution; placement is conservatively peak-reserved.
    while high-low > 1000:
        mid = (high+low)/2
        candidate, _ = place_buffers(results, cap, mid, names)
        if candidate is None:
            low = mid
        else:
            high, result = mid, candidate
    return result
