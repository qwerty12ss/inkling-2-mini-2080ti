"""A speculative server can deliver several output IDs per SSE frame."""
import io
import json
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from run_text import measure


def frames(hits=0, done=True):
    usage = {"prompt_tokens": 4, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": hits}}
    parts = [{"choices": [{"token_ids": [90, 91]}]}, {"choices": [{"token_ids": [15, 16, 17]}]},
             {"usage": usage, "choices": []}]
    lines = ["data: " + json.dumps(obj) for obj in parts]
    if done:
        lines.append("data: [DONE]")
    return io.BytesIO(("\n\n".join(lines) + "\n").encode())


class ReadBenchTests(TestCase):
    def invoke(self, reply):
        with patch("run_text.time.monotonic", side_effect=[100.0, 102.0, 105.0]):
            return measure(lambda *_: reply, "test-model", [1, 2, 3, 4], 1729, 5, 5)

    def test_first_speculative_burst_is_excluded(self):
        result = self.invoke(frames())
        self.assertEqual(result["first_chunk_tokens"], 2)
        self.assertEqual(result["ttft_s"], 2.0)
        self.assertEqual(result["prefill_tok_s"], 2.0)
        self.assertEqual(result["full_decode_tok_s"], 1.0)
        self.assertEqual(result["matched_decode_tok_s"], 1.0)

    def test_warm_input_rejected_by_cold_measurement(self):
        with self.assertRaisesRegex(RuntimeError, "zero prefix hits"):
            self.invoke(frames(hits=2))

    def test_incomplete_stream_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete stream"):
            self.invoke(frames(done=False))
