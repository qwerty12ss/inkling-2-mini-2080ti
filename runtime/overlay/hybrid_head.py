"""Shorten only the recurrent head after computing the trained eight-row draft backbone."""
import os

import torch

from hybrid_dspark import HybridDSpark
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup


class BudgetDSpark(HybridDSpark):
    def __init__(self, vllm_config, device, req_states):
        super().__init__(vllm_config, device, req_states)
        self.head_width = int(os.environ["INKLING_HEAD_K"])
        self.crop_projection = os.environ.get("INKLING_HEAD_CROP", "0") == "1"
        self.reduce_pairs = os.environ.get("INKLING_HEAD_PAIRS", "0") == "1"
        self.prune_head = os.environ.get("INKLING_HEAD_PRUNE", "1") == "1"
        self.shadow = os.environ.get("INKLING_HEAD_SHADOW", "0") == "1"
        self.differences = torch.zeros(self.head_width, device=device, dtype=torch.int32)
        config = self.speculative_config
        if self.head_width not in (3, 5, 7) or vllm_config.scheduler_config.max_num_seqs != 1:
            raise ValueError("The head experiment supports one request at fixed K3/K5/K7")
        if self.head_width != 7 or config.num_speculative_tokens_per_batch_size is not None:
            schedule = build_dynamic_sd_schedule_lookup(
                config.num_speculative_tokens_per_batch_size, 1, self.num_speculative_steps)
            if list(schedule) != [0, self.head_width]:
                raise ValueError("The scheduler must verify the selected head width")
        if self.use_draft_logits:
            raise ValueError("Use the unmodified speculator for probabilistic drafting")
        if self.head_width < 7 and self.lookup_enabled:
            raise ValueError("Disable lookup in K3/K5 head-only tests")
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size != 2:
            raise ValueError("This profile requires TP2")

    def _sample_sequential(self, num_reqs, head_hidden):
        # K7 vs K3/K5 control can retain the full original head at the same target width.
        if not self.prune_head or (self.head_width == 7 and not self.reduce_pairs):
            return super()._sample_sequential(num_reqs, head_hidden)
        full_width = self.num_speculative_steps
        width = self.head_width
        reference = None
        if self.shadow:
            super()._sample_sequential(num_reqs, head_hidden)
            reference = self.draft_tokens[:num_reqs, :width].clone()
        states = self.sample_indices[:num_reqs * full_width].view(num_reqs, full_width)
        if self.crop_projection:
            states = states[:, :width]
        hidden = head_hidden[states.flatten()]
        aux = getattr(self.model, "_conditional_aux_head", None) if num_reqs == 1 else None
        if aux is None:
            base_logits = self.model.compute_base_logits(hidden)
        else:
            base_logits = aux(hidden.contiguous())
        base_logits = base_logits.view(num_reqs, states.shape[1], -1)
        context = self.model.model.markov_head.context(hidden, self.f_rnn[:num_reqs])
        state = hidden.new_zeros(num_reqs, self.model.config.markov_rank)
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        for i in range(width):
            state, bias = self.model.markov_step(state, prev, context[i])
            if aux is not None:
                from proposal_head import choose
                prev = choose(aux, hidden[i], self.model.lm_head.weight, base_logits[:, i], bias,
                              self.model.config.final_logit_softcapping, self.model.config.vocab_size,
                              self.tp_rank, self.tp_size, tensor_model_parallel_all_gather)
            elif self.reduce_pairs:
                from dspark_argmax import sharded_markov_argmax
                prev = sharded_markov_argmax(
                    base_logits[:, i], bias, self.model.config.final_logit_softcapping,
                    self.model.config.vocab_size, self.tp_rank, self.tp_size,
                    tensor_model_parallel_all_gather,
                )
            else:
                prev = self.model.markov_argmax(base_logits[:, i], bias)
            self.draft_tokens[:num_reqs, i] = prev
        if width < full_width:
            self.draft_tokens[:num_reqs, width:].fill_(-1)
        if reference is not None:
            diff = (reference != self.draft_tokens[:num_reqs, :width]).sum(dim=0)
            self.differences.copy_(diff)

    def propose(self, *args, **kwargs):
        if self.shadow:
            self.differences.zero_()
        result = super().propose(*args, **kwargs)
        if self.shadow and not (kwargs.get("dummy_run") or kwargs.get("is_profile")):
            mismatches = self.differences.tolist()
            if any(mismatches):
                raise RuntimeError(f"Markov prefix differs at K={self.head_width}: {mismatches}")
        return result
