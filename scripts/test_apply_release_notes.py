#!/usr/bin/env python3
"""Tests for scripts/apply-release-notes.sh."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "apply-release-notes.sh"


def run_script(
    env: dict[str, str], cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.pop("GH_TOKEN", None)
    merged.pop("GITHUB_TOKEN", None)
    merged.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=merged,
        cwd=cwd,
        check=False,
    )


class ApplyReleaseNotesTests(unittest.TestCase):
    def test_rejects_bad_tag(self) -> None:
        r = run_script({"TAG": "canact-v0.1.2", "GH_REPO": "canact/canact"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("vX.Y.Z", r.stderr)

    def test_missing_file_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            r = run_script(
                {"TAG": "v0.1.2", "GH_REPO": "canact/canact", "DRY_RUN": "1"},
                cwd=Path(td),
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("leaving auto notes", r.stdout)

    def test_local_file_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            notes = root / "docs" / "releases" / "v0.1.2.md"
            notes.parent.mkdir(parents=True)
            notes.write_text("canact 0.1.2 notes\n", encoding="utf-8")
            r = run_script(
                {"TAG": "v0.1.2", "GH_REPO": "canact/canact", "DRY_RUN": "1"},
                cwd=root,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("DRY_RUN: would apply docs/releases/v0.1.2.md", r.stdout)
            self.assertIn("BYTES: 19", r.stdout)

    def test_empty_file_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            notes = root / "docs" / "releases" / "v0.1.2.md"
            notes.parent.mkdir(parents=True)
            notes.write_text("", encoding="utf-8")
            r = run_script(
                {"TAG": "v0.1.2", "GH_REPO": "canact/canact", "DRY_RUN": "1"},
                cwd=root,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("empty", r.stdout)


if __name__ == "__main__":
    unittest.main()
