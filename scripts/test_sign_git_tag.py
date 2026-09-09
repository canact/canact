#!/usr/bin/env python3
"""Lock sign-git-tag.sh argument checks."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sign-git-tag.sh"


class SignGitTagTests(unittest.TestCase):
    def test_script_is_executable(self) -> None:
        self.assertTrue(SCRIPT.is_file())
        self.assertTrue(os.access(SCRIPT, os.X_OK))

    def test_rejects_empty_tag(self) -> None:
        proc = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env={**os.environ, "TAG": ""},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("vX.Y.Z", proc.stderr)

    def test_rejects_non_semver_tag(self) -> None:
        proc = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env={**os.environ, "TAG": "v0.2.0-rc.1"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("vX.Y.Z", proc.stderr)


if __name__ == "__main__":
    unittest.main()
