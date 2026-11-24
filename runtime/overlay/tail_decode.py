"""Isolated instrumentation and a bounded-geometry policy for one TP2 Inkling request."""
import json
import os
from pathlib import Path
from time import monotonic_ns

TRACE_DIR = os.environ.get("INKLING_TAIL_TRACE_DIR")
TAG = os.environ.get("INKLING_TAIL_TRACE_REQUEST", "")
_stream = None


def select_rows(rows, requested, capacity, real_rows, num_outputs, draft_ids, width=8):
    """Narrow only a capacity-clipped, one-anchor speculative decode."""
    if (num_outputs > 0 and real_rows == 1 and draft_ids and width in (4, 6, 8)
            and requested == width and capacity == rows and 1 < rows < width):
        return 1
    return rows


def capture_widths(stock, max_reqs):
    if max_reqs == 1 and len(stock) == 1 and next(iter(stock)) in (4, 6, 8):
        return {**stock, 1: 1}
    return stock


def matches(request_id):
    # OpenAI runner prefixes the client ID and may append a generated occurrence suffix.
    return any(request_id == x or request_id.startswith(x + "-")
               for x in (TAG, "cmpl-" + TAG, "chatcmpl-" + TAG)) if TAG else False


def emit(kind, request_id, **fields):
    global _stream
    if not TRACE_DIR or not matches(request_id):
        return
    if _stream is None:
        path = Path(TRACE_DIR) / f"tail-{os.getpid()}.jsonl"
        _stream = path.open("x", buffering=1)
    data = dict(ts=monotonic_ns(), pid=os.getpid(), kind=kind, request=request_id, **fields)
    _stream.write(json.dumps(data, separators=(",", ":")) + "\n")


def after_dispatch(runner, batch_state, output, desc, uniform):
    if batch_state is None:
        return
    for request_id, ix in zip(batch_state.req_ids, batch_state.idx_mapping_np, strict=True):
        if matches(request_id):
            emit("dispatch", request_id, gpu=str(runner.device),
                 scheduled=output.num_scheduled_tokens[request_id],
                 draft=len(output.scheduled_spec_decode_tokens.get(request_id, ())),
                 skip_drafts=output.skip_drafts, uniform=uniform, graph=desc.cg_mode.name,
                 physical_rows=desc.num_tokens, computed_cpu=int(runner.req_states.num_computed_tokens_np[ix]))
