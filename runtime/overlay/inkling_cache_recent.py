"""Refresh a bounded set of hash-verified sliding-window blocks after a long request retires."""
from vllm.v1.core.kv_cache_utils import resolve_block_hashes


def boundaries(existing,n,chunk,anchors):
    if n<2*chunk:
        return existing
    last=(n-1)//chunk*chunk
    return (*existing,*range(max(chunk,last-(anchors-1)*chunk),last+1,chunk))


def refresh(coordinator,request,chunk,anchors=3):
    n=request.num_prompt_tokens
    if n<2*chunk or not request.block_hashes:
        return
    # Protect the most recent full-size chunks shared with nearby edits.
    last=(n-1)//chunk*chunk
    ends=range(max(chunk,last-(anchors-1)*chunk),last+1,chunk)
    for cfg,size,group_ids in coordinator.attention_groups:
        window=getattr(cfg.params,'sliding_window',None)
        if not window:
            continue
        manager=coordinator.single_type_managers[group_ids[0]]
        pool=manager.block_pool
        hashes=resolve_block_hashes(request.block_hashes,pool.hash_block_size,size,
                                    supports_fine_grained_hash_lookup=cfg.manager_cls.supports_fine_grained_hash_lookup,
                                    alignment_tokens=coordinator._cache_hit_alignment_tokens)
        required=(window-1+size-1)//size
        for end in ends:
            right=end//size
            if right>len(hashes):
                continue
            pieces=[]
            for index in range(max(0,right-required),right):
                blocks=pool.get_cached_block(hashes[index],group_ids)
                if blocks is None:
                    break
                pieces.extend(b for b in blocks if b.ref_cnt==0 and b.block_hash is not None)
            else:
                # A touch/free pair updates the existing eviction order and restores
                # every reference count. The pages remain ordinary eviction candidates.
                pool.touch(pieces)
                pool.free_blocks(pieces)
