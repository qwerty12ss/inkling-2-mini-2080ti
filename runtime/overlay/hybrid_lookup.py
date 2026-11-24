"""GPU lookup on a request's committed history, with a linear DSpark fallback."""
import torch
import triton
import triton.language as tl


@triton.jit
def _match(History, Length, Mapping, Temperature, Sampled, Draft, Scores,
           HISTORY_STRIDE: tl.constexpr, DRAFT_STRIDE: tl.constexpr,
           K: tl.constexpr, WINDOW: tl.constexpr, MIN_MATCH: tl.constexpr,
           MAX_MATCH: tl.constexpr, AGREE: tl.constexpr, BLOCK: tl.constexpr,
           TILES: tl.constexpr):
    batch = tl.program_id(0)
    tile = tl.program_id(1)
    slot = tl.load(Mapping + batch)
    out = Scores + batch * TILES + tile
    if slot < 0 or tl.load(Sampled + batch) < 1:
        tl.store(out, 0)
        return
    length = tl.load(Length + slot)
    if tl.load(Temperature + slot) != 0 or length > HISTORY_STRIDE or length < K + MIN_MATCH:
        tl.store(out, 0)
        return
    cols = tl.arange(0, 8)
    absent = tl.load(Draft + batch * DRAFT_STRIDE + cols, cols < K, other=0) < 0
    if tl.max(absent.to(tl.int32), 0) != 0:
        tl.store(out, 0)
        return
    lag = tile * BLOCK + tl.arange(0, BLOCK)
    start = length - K - lag
    valid = (lag < WINDOW) & (start >= MIN_MATCH)
    row = History + slot.to(tl.int64) * HISTORY_STRIDE
    for j in tl.static_range(AGREE):
        copied = tl.load(row + start + j, valid, other=-1)
        valid = valid & (copied == tl.load(Draft + batch * DRAFT_STRIDE + j))
    depth = tl.full((BLOCK,), 0, tl.int32)
    for j in tl.static_range(MAX_MATCH):
        valid = valid & (start > j)
        past = tl.load(row + start - 1 - j, valid, other=-1)
        last = tl.load(row + length - 1 - j, mask=(length > j), other=-2)
        valid = valid & (past == last)
        depth += valid.to(tl.int32)
    # Highest score is the longest exact match; nearest history wins ties.
    score = tl.where(depth >= MIN_MATCH, depth * (WINDOW + 1) + WINDOW - lag, 0)
    tl.store(out, tl.max(score, 0))


@triton.jit
def _select(History, Length, Mapping, Draft, Scores, Output, Hit,
            HISTORY_STRIDE: tl.constexpr, DRAFT_STRIDE: tl.constexpr,
            K: tl.constexpr, WINDOW: tl.constexpr, TILES: tl.constexpr,
            TILE_BLOCK: tl.constexpr, K_BLOCK: tl.constexpr, RECORD: tl.constexpr):
    batch = tl.program_id(0)
    tile = tl.arange(0, TILE_BLOCK)
    best = tl.max(tl.load(Scores + batch * TILES + tile, tile < TILES, 0), 0)
    slot = tl.load(Mapping + batch)
    use = (best != 0) & (slot >= 0)
    length = tl.load(Length + slot, use, other=0)
    lag = WINDOW - best % (WINDOW + 1)
    pos = length - K - lag
    col = tl.arange(0, K_BLOCK)
    row_base = slot.to(tl.int64) * HISTORY_STRIDE
    suffix = tl.load(History + row_base + pos + col, use & (col < K), other=0)
    dspark = tl.load(Draft + batch * DRAFT_STRIDE + col, col < K, other=0)
    proposal = tl.where(use, suffix, dspark)
    tl.store(Output + batch * K + col, proposal, col < K)
    if RECORD:
        # A matched suffix identical to all DSpark IDs does not change the round.
        changed = tl.max(((col < K) & (suffix != dspark)).to(tl.int32), 0)
        tl.store(Hit + batch * 2, best // (WINDOW + 1))
        tl.store(Hit + batch * 2 + 1, (use & (changed != 0)).to(tl.int32))


class HistoryLookup:
    def __init__(self, *, device, rows, width, window, min_match, max_match, agree):
        if not (1 <= width <= 7 and 1 <= min_match <= max_match <= 32
                and 1 <= window <= 65536 and 0 <= agree <= width):
            raise ValueError("Invalid lookup width, window, match range or agreement")
        self.width = width
        self.window = window
        self.min_match = min_match
        self.max_match = max_match
        self.agree = agree
        self.block = 128
        self.tiles = triton.cdiv(window, self.block)
        self.scores = torch.empty((rows, self.tiles), dtype=torch.int32, device=device)
        self.output = torch.empty((rows, width), dtype=torch.int64, device=device)
        self.hit = torch.empty((rows, 2), dtype=torch.int32, device=device)

    def propose(self, history, lengths, mapping, temperature, sampled, draft, *, record=False):
        n = draft.shape[0]
        if (draft.shape[1] != self.width or n > self.output.shape[0]
                or history.stride(-1) != 1 or draft.stride(-1) != 1):
            raise ValueError("Unexpected history or proposal layout")
        _match[(n, self.tiles)](
            history, lengths, mapping, temperature, sampled, draft, self.scores,
            history.stride(0), draft.stride(0), self.width, self.window, self.min_match,
            self.max_match, self.agree, self.block, self.tiles, num_warps=4,
        )
        _select[(n,)](
            history, lengths, mapping, draft, self.scores, self.output, self.hit,
            history.stride(0), draft.stride(0), self.width, self.window, self.tiles,
            triton.next_power_of_2(self.tiles), triton.next_power_of_2(self.width),
            record, num_warps=2,
        )
        return self.output[:n]
