"""Check that reconstruction works inside the user's Git clone."""
from hashlib import sha256
import importlib.util
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recipe_prepare", ROOT / "runtime/prepare.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PrepareTest(unittest.TestCase):
    def test_patch_and_overlay_applied_in_nested_git_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "-q", root], check=True)
            runtime = root / "runtime"
            overlay = runtime / "overlay"
            overlay.mkdir(parents=True)
            (overlay / "extra.py").write_text("value = 7\n")
            (overlay / "__pycache__").mkdir()
            (overlay / "__pycache__/local.pyc").write_bytes(b"not source")
            subpath = "vllm/models/inkling2/nvidia/model.py"
            upstream_file = root / "base" / subpath
            upstream_file.parent.mkdir(parents=True)
            upstream_file.write_text("# upstream\n")
            archive = root / "upstream.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(root / "base", arcname="vllm-testrev")
            diff = root / "model.patch"
            expected = "from inkling_vision_sm75 import SM75VisionEncoder"
            diff.write_text(f"--- a/{subpath}\n+++ b/{subpath}\n@@ -1 +1,2 @@\n # upstream\n+{expected}\n")
            digest = sha256(archive.read_bytes()).hexdigest()
            work = root / ".work"
            with patch.multiple(prepare, RUNTIME=runtime, PATCH=diff, OVERLAY=overlay, COMMIT="testrev", DIGEST=digest):
                tree = prepare.prepare(work, archive)
                self.assertEqual(prepare.prepare(work, archive), tree)
            self.assertIn(expected, (tree / subpath).read_text())
            self.assertEqual((tree / "extra.py").read_text(), "value = 7\n")
            self.assertFalse((tree / "__pycache__").exists())
