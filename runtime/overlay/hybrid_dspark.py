"""Isolated vLLM V2 speculator for greedy DSpark + suffix A/Bs."""
import json
import os
from pathlib import Path

import torch

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.models.inkling2.speculator import Inkling2DSparkSpeculator

from hybrid_lookup import HistoryLookup


class HybridDSpark(Inkling2DSparkSpeculator):
    def __init__(self, vllm_config, device, req_states):
        super().__init__(vllm_config, device)
        if self.num_speculative_steps != 7:
            raise ValueError("The research profile requires the original K7 draft block")
        self.history_state = req_states
        self.lookup_enabled = os.environ.get("INKLING_LOOKUP", "1") == "1"
        self.lookup = HistoryLookup(
            device=device, rows=self.max_num_reqs, width=self.num_speculative_steps,
            window=int(os.environ.get("INKLING_LOOKUP_WINDOW", "8192")),
            min_match=int(os.environ.get("INKLING_LOOKUP_MIN", "8")),
            max_match=int(os.environ.get("INKLING_LOOKUP_MAX", "16")),
            agree=int(os.environ.get("INKLING_LOOKUP_AGREE", "1")),
        )
        path = os.environ.get("INKLING_LOOKUP_TRACE")
        self.trace = None
        if path and get_tensor_model_parallel_rank() == 0:
            # Per-process file also keeps independently restarted services separate.
            self.trace = Path(path, f"lookup-{os.getpid()}.jsonl").open("x", buffering=1)

    @torch.inference_mode()
    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states,
                aux_hidden_states, num_sampled, num_rejected, last_sampled,
                next_prefill_tokens, temperature, seeds, dp_sync_state=None,
                dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None,
                is_profile=False, num_speculative_tokens=None):
        draft = super().propose(
            input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
            num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds,
            dp_sync_state=dp_sync_state, dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run, mm_inputs=mm_inputs,
            is_profile=is_profile, num_speculative_tokens=num_speculative_tokens,
        )
        if (dummy_run or is_profile or self.context_kv_only or self.draft_logits is not None
                or num_speculative_tokens not in (None, self.num_speculative_steps)):
            return draft
        if not self.lookup_enabled and self.trace is None:
            return draft
        state = self.history_state
        result = self.lookup.propose(
            state.all_token_ids.gpu, state.total_len.gpu, input_batch.idx_mapping,
            temperature, num_sampled, draft, record=self.trace is not None,
        )
        if self.trace is not None:
            # Diagnostics only. Reading individual rounds synchronizes the GPU.
            # Record both choices for offline teacher-forced comparison.
            for batch, request in enumerate(input_batch.req_ids):
                sampled = int(num_sampled[batch].item())
                if sampled == 0:
                    continue
                slot = int(input_batch.idx_mapping[batch].item())
                if slot < 0:
                    continue
                length = int(state.total_len.gpu[slot].item())
                committed = state.all_token_ids.gpu[slot, length-sampled:length].tolist()
                match, changed = self.lookup.hit[batch].tolist()
                record = dict(request=request, length=length, committed=committed,
                              dspark=draft[batch].tolist(), hybrid=result[batch].tolist(),
                              match=match, changed=bool(changed),
                              greedy=bool(temperature[slot].item() == 0),
                              applied=self.lookup_enabled)
                scheduled = input_batch.num_draft_tokens_per_req
                record["previous_scheduled"] = int(scheduled[batch]) if scheduled is not None else 0
                record["previous_rejected"] = int(num_rejected[batch].item())
                if record["greedy"]:
                    self.add_diagnostics(record, batch)
                self.trace.write(json.dumps(record) + "\n")
        return result if self.lookup_enabled else draft

    def add_diagnostics(self, record, batch):
        """Additional diagnostics supplied by a specialized research speculator."""
