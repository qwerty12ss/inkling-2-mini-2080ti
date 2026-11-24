"""Resume long prefixes on their original prefill lattice, checking every hybrid group."""
def find_hit(lookup, hashes, maximum, length, *, chunk):
    if length < 2 * chunk:
        return lookup(hashes, maximum, length)
    bound = maximum - maximum % chunk
    while True:
        blocks, matched, shared = lookup(hashes, bound, length)
        if not 0 <= matched <= bound:
            raise RuntimeError("Invalid reconciled hybrid prefix length")
        if matched % chunk == 0:
            return blocks, matched, shared - shared % chunk
        # Sliding-window pages and all four conv histories must exist at the
        # earlier boundary too. Ask every group to perform its ordinary lookup.
        bound = matched - matched % chunk
