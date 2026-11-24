"""Exercise the launcher's public arguments without installing CUDA or weights."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import unittest


SERVE = Path(__file__).resolve().parents[1] / "scripts" / "serve.sh"


def launch(*args, **environment):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "PORT": "8042", "NUMA_NODE": "", **environment}
    return subprocess.run([SERVE, *args, "--dry-run"], env=env, capture_output=True, text=True)


class TestLaunchProfiles(unittest.TestCase):
    def test_profile_matrix(self):
        cases = [
            ("text-512k", "524288", "10468982784", "inkling-2-mini-3loop", True),
            ("vision-448k", "458752", "9261023232", "inkling-2-mini-3loop-vision-gpu-448k", False),
        ]
        for profile, window, kv_bytes, name, text_only in cases:
            for use_draft in (True, False):
                with self.subTest(profile=profile, use_draft=use_draft):
                    result = launch(profile, *([] if use_draft else ["--no-drafter"]), MODEL_DIR="/tmp/path with spaces")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    args = shlex.split(result.stdout.strip().splitlines()[-1])
                    for option, value in {
                        "--model": "/tmp/path with spaces",
                        "--served-model-name": name,
                        "--max-model-len": window,
                        "--kv-cache-memory-bytes": kv_bytes,
                        "--port": "8042",
                        "--tensor-parallel-size": "2",
                        "--max-num-seqs": "1",
                        "--max-num-batched-tokens": "2048",
                        "--prefix-cache-retention-interval": "2048",
                    }.items():
                        self.assertEqual(args[args.index(option) + 1], value)
                    self.assertEqual("--language-model-only" in args, text_only)
                    self.assertEqual("--skip-mm-profiling" in args, not text_only)
                    self.assertEqual("--speculative-config" in args, use_draft)
                    if use_draft:
                        spec = json.loads(args[args.index("--speculative-config") + 1])
                        self.assertEqual(spec, {"method": "dspark", "num_speculative_tokens": 7,
                                                "num_speculative_tokens_per_batch_size": [[1, 1, 5]]})

    def test_invalid_inputs(self):
        for cmd, env in [
            (("unknown",), {}), (("text-512k", "--wrong"), {}),
            (("text-512k",), {"CUDA_VISIBLE_DEVICES": "0"}),
            (("vision-448k",), {"PORT": "65536"}),
        ]:
            with self.subTest(cmd=cmd, env=env):
                self.assertEqual(launch(*cmd, **env).returncode, 2)


if __name__ == "__main__":
    unittest.main()
