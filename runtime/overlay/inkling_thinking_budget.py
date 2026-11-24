"""Budget the thinking blocks of a single Inkling response, including DSpark proposals."""
import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState


def closing_deadline(output_limit):
    """Latest generated position for the first forced closing token."""
    final_tokens = min(256, (output_limit + 3) // 4)
    return max(0, output_limit - 3 - final_tokens)


# Phases: normal=0, thinking=1, need message_model=2, need content_text=3.
@triton.jit
def _consume(token, phase, used, exhausted, OPEN: tl.constexpr, END: tl.constexpr,
             ROLE: tl.constexpr, TEXT: tl.constexpr, TOOL: tl.constexpr):
    if phase == 1:
        if token == END:
            phase = tl.where(exhausted, 2, 0)
        elif token == TOOL:
            phase = 0
        elif token != OPEN:
            used += 1
    elif phase == 2 and token == ROLE:
        phase = 3
    elif token == OPEN:
        phase = 1
    else:
        phase = 0
    return phase, used


@triton.jit
def _committed(Req, Budgets, Deadlines, Tokens, PromptLen, TotalLen, Scan, Phase, Used,
               STRIDE: tl.constexpr, OPEN: tl.constexpr, END: tl.constexpr,
               ROLE: tl.constexpr, TEXT: tl.constexpr, TOOL: tl.constexpr):
    req = tl.load(Req + tl.program_id(0))
    budget = tl.load(Budgets + req)
    if budget < 0:
        return
    deadline = tl.load(Deadlines + req)
    prompt = tl.load(PromptLen + req)
    total = tl.load(TotalLen + req)
    scan = tl.load(Scan + req)
    phase = tl.load(Phase + req)
    used = tl.load(Used + req)
    history = Tokens + req.to(tl.int64) * STRIDE
    if scan == 0 or scan > total:
        # Normally the prompt ends at a message boundary. A continuation of
        # an unfinished assistant thinking block must include its open suffix.
        hi = prompt
        marker_pos = -1
        marker = -1
        while hi > 0 and marker_pos < 0:
            lo = tl.maximum(0, hi - 256)
            pos = lo + tl.arange(0, 256)
            ids = tl.load(history + pos, mask=pos < hi, other=-1)
            # The parser also stops its backward seed scan at the next block
            # header, sampling terminator, raw tool text, or tool error.
            found = ((ids == OPEN) | (ids == END) | (ids == TEXT) | (ids == ROLE)
                     | (ids == TOOL) | (ids == 200006) | (ids == 200057) | (ids == 200022))
            marker_pos = tl.max(tl.where(found, pos, -1), 0)
            marker = tl.max(tl.where(pos == marker_pos, ids, -1), 0)
            hi = lo
        phase = tl.where(marker == OPEN, 1, 0)
        used = tl.where(marker == OPEN, prompt - marker_pos - 1, 0)
        scan = prompt
    for pos in tl.range(scan, total):
        token = tl.load(history + pos)
        exhausted = (used >= budget) | (pos >= deadline)
        phase, used = _consume(token, phase, used, exhausted, OPEN, END, ROLE, TEXT, TOOL)
    tl.store(Used + req, used)
    tl.store(Phase + req, phase)
    tl.store(Scan + req, total)


@triton.jit
def _enforce(Logits, Expanded, Local, Draft, Budgets, Deadlines, TotalLen, Phase, Used,
             STRIDE: tl.constexpr, OPEN: tl.constexpr, END: tl.constexpr,
             ROLE: tl.constexpr, TEXT: tl.constexpr, TOOL: tl.constexpr):
    row = tl.program_id(0)
    req = tl.load(Expanded + row)
    budget = tl.load(Budgets + req)
    if budget < 0:
        return
    local_pos = tl.load(Local + row)
    total = tl.load(TotalLen + req)
    deadline = tl.load(Deadlines + req)
    phase = tl.load(Phase + req)
    used = tl.load(Used + req)
    for i in tl.range(0, local_pos):
        # Draft[0] is the previously committed token; Draft[1] is proposal 0.
        token = tl.load(Draft + row - local_pos + i + 1)
        exhausted = (used >= budget) | (total + i >= deadline)
        phase, used = _consume(token, phase, used, exhausted, OPEN, END, ROLE, TEXT, TOOL)
    exhausted = (used >= budget) | (total + local_pos >= deadline)
    scores = Logits + row.to(tl.int64) * STRIDE
    if phase == 2:
        tl.store(scores + ROLE, 1.e9)
    elif phase == 3:
        # A required tool or allowed-token grammar may disallow a text block.
        # Preserve that constraint and allow the model to pick a legal tool.
        tl.store(scores + OPEN, -float("inf"))
        if tl.load(scores + TEXT) != -float("inf"):
            tl.store(scores + TEXT, 1.e9)
    elif exhausted:
        if phase == 1:
            tl.store(scores + END, 1.e9)
        else:
            # Direct text and tool calls remain available without reopening thinking.
            tl.store(scores + OPEN, -float("inf"))


class InklingThinkingBudgetState(ThinkingBudgetState):
    def __init__(self, req_states, reasoning_config):
        super().__init__(req_states, reasoning_config)
        # These four token IDs and the JSON tool ID are dedicated Inkling-2
        # vocabulary entries. Check the configured tokenizer before using them.
        if (not self.enabled or reasoning_config.reasoning_start_token_ids != [200008]
                or reasoning_config.natural_reasoning_end_token_ids != [200010]
                or reasoning_config.reasoning_end_token_ids != [200010, 200001, 200004]):
            raise ValueError("Inkling thinking budget requires its native block delimiters")
        self.markers = dict(OPEN=200008, END=200010, ROLE=200001, TEXT=200004, TOOL=200049)
        self.deadlines = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.deadlines_dirty = False
        # The base class owns and resets these per-slot arrays whenever a new
        # budgeted request is admitted. Inkling tracks cumulative turns instead.
        self.committed_phase = self.cached_last_end
        self.committed_used = self.cached_last_start

    def add_request(self, req_idx, sampling_params):
        enabled = super().add_request(req_idx, sampling_params)
        if enabled:
            prompt = int(self.req_states.prompt_len.np[req_idx])
            max_output = int(self.req_states.max_seq_len[req_idx]) - prompt
            self.deadlines.np[req_idx] = prompt + closing_deadline(max_output)
            self.deadlines_dirty = True
        return enabled

    def apply_staged_writes(self):
        super().apply_staged_writes()
        if self.deadlines_dirty:
            self.deadlines.copy_to_uva()
            self.deadlines_dirty = False

    def apply(self, logits, ctx):
        if not np.any(self.use_thinking_budget[ctx.idx_mapping_np]):
            return
        history = self.req_states.all_token_ids.gpu
        total = self.req_states.total_len.gpu
        budgets = self.thinking_token_budget.gpu
        deadlines = self.deadlines.gpu
        _committed[(ctx.idx_mapping.shape[0],)](
            ctx.idx_mapping, budgets, deadlines, history, self.req_states.prompt_len.gpu, total,
            self.cached_scan_pos, self.committed_phase, self.committed_used,
            history.stride(0), **self.markers)
        _enforce[(logits.shape[0],)](
            logits, ctx.expanded_idx_mapping, ctx.expanded_local_pos, ctx.input_ids,
            budgets, deadlines, total, self.committed_phase, self.committed_used,
            logits.stride(0), **self.markers)
