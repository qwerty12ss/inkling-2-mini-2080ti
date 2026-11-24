"""Diagnostic-only row fingerprints during fixed, cold, 68-token Inkling calls."""
import os
from contextlib import nullcontext
from functools import wraps
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

def install(cls):
    directory = Path(os.environ["INKLING_LAYER_TRACE_DIR"])
    original_forward = cls.forward
    original_layer = cls._run_layer
    state = {}
    detailed = os.environ.get("INKLING_LAYER_TRACE_DETAIL") == "1"

    class MarlinCapture(TorchDispatchMode):
        """Keep actual route order, inputs and outputs in the first layer."""
        def __init__(self):
            super().__init__()
            self.projects = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            output = func(*args, **(kwargs or {}))
            if func._schema.name == "_moe_C::moe_wna16_marlin_gemm":
                prefix = f"marlin-{self.projects}-"
                self.projects += 1
                # Turing extension includes an additional activation-scale argument.
                params = dict(zip((p.name for p in func._schema.arguments), args))
                for name in ("sorted_token_ids","expert_ids","num_tokens_past_padded","topk_weights"):
                    if name in params:
                        state["audit"][prefix+name] = params[name].detach().clone()
                state["audit"][prefix+"input"] = args[0].detach().clone()
                state["audit"][prefix+"output"] = output.detach().clone()
            return output

    def keep(name, value):
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != 68:
            return
        # Preserve all token positions at fixed columns and the full final row.
        # Clone on the worker stream. The first CPU synchronization is after the model.
        state[name] = (value[:,::71].contiguous().clone(),value[-1].clone())
        if detailed and (name == "embedding" or name.startswith(("00-","01-"))):
            state["audit"][name] = value.detach().clone()

    @wraps(original_forward)
    def forward(self, positions, inputs_embeds):
        trigger = directory / "request"
        rank = inputs_embeds.device.index
        if (positions.numel()!=68 or (rank != 0 and not detailed)
                or torch.cuda.is_current_stream_capturing() or not trigger.exists()):
            return original_forward(self,positions,inputs_embeds)
        tag=trigger.read_text().strip()
        if not tag or not tag.replace("-","").isalnum():
            return original_forward(self,positions,inputs_embeds)
        dest=directory/(tag+("-rank1" if rank == 1 else "")+".pt")
        if dest.exists():
            return original_forward(self,positions,inputs_embeds)
        state.clear()
        state["active"] = True
        state["visit"] = 0
        state["audit"] = {}
        keep("embedding",inputs_embeds)
        try:
            out=original_forward(self,positions,inputs_embeds)
            result={}
            for key,values in state.items():
                if isinstance(values,tuple):
                    result[key]=tuple(x.detach().cpu() for x in values)
            result["positions"]=positions.detach().cpu()
            torch.save(result,dest)
            if detailed:
                torch.save({key:x.cpu() for key,x in state["audit"].items()},
                           directory/(tag+f"-detail-{rank}.pt"))
            return out
        finally:
            state.clear()

    @wraps(original_layer)
    def layer(self,i,pass_idx,stack,num_blocks,inputs,bank,next_query,next_norm,positions,tau,plan):
        if not state.get("active"):
            return original_layer(self,i,pass_idx,stack,num_blocks,inputs,bank,next_query,next_norm,positions,tau,plan)
        visit=state["visit"]
        state["visit"]+=1
        prefix=f"{visit:02d}-{i}-{pass_idx}"
        layer=self.layers[i]
        def attention_hook(module, args, output):
            keep(prefix+"-attention",output)
        def mlp_hook(module, args, output):
            keep(prefix+"-mlp-in",args[0])
            keep(prefix+"-mlp",output)
        hooks=[layer.attn.register_forward_hook(attention_hook),layer.mlp.register_forward_hook(mlp_hook)]
        try:
            with MarlinCapture() if detailed and visit == 0 else nullcontext():
                count,outputs=original_layer(self,i,pass_idx,stack,num_blocks,inputs,bank,next_query,next_norm,positions,tau,plan)
            keep(prefix+"-stack",stack[count])
            return count,outputs
        finally:
            for hook in hooks:
                hook.remove()
    cls.forward=forward
    cls._run_layer=layer
