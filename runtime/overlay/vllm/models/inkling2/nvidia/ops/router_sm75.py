"""SM75 expert router, same three ordered 1024-wide K groups with FP32 outputs."""
import torch


def grouped_cublas(x, weight):
    tokens, hidden = x.shape
    experts = weight.shape[0]
    groups = hidden // 1024
    tail = hidden % 1024
    result = x.new_empty((groups + bool(tail), tokens, experts), dtype=torch.float32)
    if groups:
        xs = x.as_strided((groups,tokens,1024), (1024,hidden,1))
        ws = weight.as_strided((groups,experts,1024), (1024,hidden,1))
        torch.bmm(xs,ws.transpose(1,2),out_dtype=torch.float32,out=result[:groups])
    if tail:
        start = groups*1024
        torch.mm(x[:,start:],weight[:,start:].t(),out_dtype=torch.float32,out=result[-1])
    return result


def whole_cublas(x, weight):
    return torch.mm(x, weight.t(), out_dtype=torch.float32).unsqueeze(0)
