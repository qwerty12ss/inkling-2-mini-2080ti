"""Compare the independently rebuilt engine's recorded trials to their controls."""
import csv
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "tests"


class RebuiltMeasurements(unittest.TestCase):
    def test_rebuilt_text_uses_same_inputs_and_accepts_same_number_of_drafts(self):
        with (ROOT / "benchmarks/data/text-pairs.csv").open() as source:
            baseline = {int(r["seed"]): r for r in csv.DictReader(source)
                        if r["input_tokens"] == "32768" and r["mode"] == "K5"}
        runs = json.loads((HERE / "rebuilt-text-32k.json").read_text())
        comparable = [r for r in runs if r["phase"] == "three-seed series after warm-up"]
        self.assertCountEqual([r["seed"] for r in comparable], list(baseline))
        for row in comparable:
            prior = baseline[row["seed"]]
            self.assertEqual(row["prompt_ids_sha256"], prior["prompt_ids_sha256"])
            self.assertEqual(row["cached_tokens"], 0)
            self.assertEqual(row["output_tokens"], 256)
            self.assertEqual(row["num_accepted_tokens_total"], int(prior["accepted_draft_tokens"]))
            self.assertEqual(row["num_draft_tokens_total"], int(prior["proposed_draft_tokens"]))
            self.assertEqual(row["num_drafts_total"], int(prior["draft_rounds"]))

    def test_rebuilt_vision_readout_and_accounting(self):
        for filename, length in (("rebuilt-vision-8k.json", 8192), ("clean-install-448k.json", 458496)):
            with self.subTest(filename=filename):
                report = json.loads((HERE / filename).read_text())
                cold, repeat = report["cases"]
                self.assertEqual(report["input_tokens"], length)
                self.assertEqual(cold["cached_tokens"], 0)
                self.assertGreater(repeat["cached_tokens"], 0)
                for row in (cold, repeat):
                    self.assertTrue(row["ocr_ok"])
                    self.assertEqual(row["answer"].splitlines()[0].strip(), "COFFEE Q7M42")
                    self.assertEqual(row["image_tokens"], 521)
                    self.assertEqual(row["cached_tokens"] + row["computed_tokens"], length)
