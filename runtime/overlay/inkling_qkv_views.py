"""Use contiguous neighbouring input streams to share a cuBLAS weight read on Turing."""
import torch
import torch.nn.functional as F


def adjacent(xs):
    first=xs[0]
    m,k=first.shape
    pointer=first.untyped_storage().data_ptr()
    origin=first.storage_offset()
    for i,x in enumerate(xs):
        if (x.shape!=(m,k) or not x.is_contiguous() or x.untyped_storage().data_ptr()!=pointer
                or x.storage_offset()!=origin+i*m*k):
            return None
    return first.as_strided((len(xs)*m,k),(k,1))


def project(xq,xk,xv,w,qside,kv,qpad):
    m=xq.shape[0]
    if (m not in (1,4,6,8) or xq.dtype!=torch.float16 or w.shape!=(4304,2304)
            or not w.is_contiguous() or (qside,kv,qpad)!=(3276,512,3280)):
        return None
    if m in (1,4):
        x=adjacent((xq,xk,xv))
        if x is None:
            return None
        y=F.linear(x,w)
        return y[:m,:qpad],y[m:2*m,qside:qside+kv],y[2*m:,qside+kv:qside+2*kv]
    x=adjacent((xk,xv))
    if x is None:
        return None
    q=F.linear(xq,w[:qpad])
    y=F.linear(x,w[qside:qside+2*kv])
    return q,y[:m,:kv],y[m:,kv:]
