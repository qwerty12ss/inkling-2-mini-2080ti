"""CPU regressions for public smoke checks and the exact photograph."""
import hashlib
import unittest
from unittest.mock import patch
from smoke import PHOTO, run, validate_usage


def response(cached, answer="42"):
    return {"answer": answer, "usage": {"prompt_tokens": 5040, "completion_tokens": 5,
            "total_tokens": 5045, "prompt_tokens_details": {"cached_tokens": cached}}}


class SmokeTests(unittest.TestCase):
    def test_prefix_read_is_part_of_input(self):
        self.assertEqual(validate_usage(response(4096)["usage"]), 4096)

    def test_total_cannot_count_the_prefix_twice(self):
        usage = response(4096)["usage"]
        usage["total_tokens"] += 4096
        with self.assertRaises(ValueError):
            validate_usage(usage)

    def test_no_hit_on_repeat_fails_check(self):
        with patch("smoke.completion", side_effect=[response(0), response(0)]):
            with self.assertRaisesRegex(ValueError, "repeat"):
                run("", "", image=False, cache=True)

    def test_repeat_uses_same_salt_and_prompt(self):
        with patch("smoke.completion", side_effect=[response(0), response(4096)]) as send:
            rows = run("", "", image=False, cache=True)
        self.assertEqual(send.call_args_list[0], send.call_args_list[1])
        self.assertEqual([r["case"] for r in rows], ["cold", "repeat"])

    def test_photo_is_the_measured_fixture(self):
        self.assertEqual(hashlib.sha256(PHOTO.read_bytes()).hexdigest(),
                         "cfb94a498f9bd611949bd98afbf9951871e71914bfe106ce1a96ed09ff8632c4")
