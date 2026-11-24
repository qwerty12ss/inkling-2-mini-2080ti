"""Check the archived measurements using only the Python standard library."""

import csv
from math import isclose
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def rows(name):
    with (DATA / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def require(test, key):
    if not test:
        raise ValueError(f"Inconsistent benchmark record: {key}")


def check():
    pairs = {}
    for row in rows("text-pairs.csv"):
        key = int(row["input_tokens"]), int(row["seed"])
        mode = row["mode"]
        block = pairs.setdefault(key, {})
        require(mode in {"target", "K5"} and mode not in block, key)
        require(int(row["cached_tokens"]) == 0, key)
        require(0 < int(row["first_chunk_tokens"]) < int(row["matched_endpoint_tokens"]) <= int(row["output_tokens"]) <= 256, key)
        require(isclose(float(row["prefill_tok_s"]), key[0] / float(row["ttft_s"]), rel_tol=1e-7), key)
        rounds, accepted, proposed = (int(row[k]) for k in ("draft_rounds", "accepted_draft_tokens", "proposed_draft_tokens"))
        require(0 <= accepted <= proposed and (rounds == proposed == 0 if mode == "target" else proposed == 5 * rounds > 0), key)
        block[mode] = row
    require({n for n, _ in pairs} == {512, 32768, 131072, 524032}, "input lengths")
    for (n, seed), block in pairs.items():
        require(set(block) == {"target", "K5"}, (n, seed))
        for k in ("prompt_ids_sha256", "matched_endpoint_tokens"):
            require(block["target"][k] == block["K5"][k], (n, seed, k))
        expected_seeds = {1729, 1811, 863} if n <= 32768 else {1729}
        require({s for size, s in pairs if size == n} == expected_seeds, n)

    full = rows("text-k5-full.csv")
    expected = {(str(n), str(seed), corpus) for n in (512, 32768, 131072, 524032)
                for seed in (1729, 1811, 863) for corpus in ("canonical", "document")}
    keys = [(r["input_tokens"], r["seed"], r["corpus"]) for r in full]
    require(len(keys) == len(expected) and set(keys) == expected, "full-output sweep")
    for row in full:
        n, seed = int(row["input_tokens"]), int(row["seed"])
        rate = (int(row["output_tokens"]) - int(row["first_chunk_tokens"])) / float(row["postfirst_s"])
        require(int(row["cached_tokens"]) == 0 and isclose(rate, float(row["full_decode_tok_s"]), rel_tol=1e-7), (n, seed))
        if row["corpus"] == "canonical" and (n, seed) in pairs:
            match = pairs[n, seed]["K5"]
            require(row["ttft_s"] == match["ttft_s"] and row["prompt_ids_sha256"] == match["prompt_ids_sha256"], (n, seed))

    vision = rows("vision-cold.csv")
    for row in vision:
        n = int(row["input_tokens"])
        require(0 < int(row["image_tokens"]) <= n and int(row["cached_tokens"]) == 0, row["case"])
        require(0 < float(row["engine_prefill_s"]) <= float(row["engine_ttft_s"]), row["case"])
    long_image = next(row for row in vision if row["case"] == "long-458496")
    require(int(long_image["input_tokens"]) + int(long_image["output_tokens"]) == 458752, "full vision capacity")

    history = rows("vision-prefix.csv")
    require([row["case"] for row in history] == ["cold-json", "repeat-json", "repeat-stream", "changed-suffix-json"], "prefix order")
    for i, row in enumerate(history):
        n, cached, written, computed = (int(row[k]) for k in ("input_tokens", "cached_tokens", "created_cache_tokens", "computed_tokens"))
        require(cached + computed == n and written <= computed and (cached == 0) == (i == 0), row["case"])
        require(int(row["reasoning_tokens"]) <= int(row["output_tokens"]) and int(row["image_tokens"]) == 521, row["case"])
    print(f"Verified {len(pairs)} text pairs, {len(full)} full-output trials, {len(vision)} vision runs and {len(history)} prefix-cache requests.")


if __name__ == "__main__":
    check()
