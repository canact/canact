#!/usr/bin/env python3
"""Tests for scripts/apply-release-notes.sh."""

from __future__ import annotations

import os
import stat
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
    merged.pop("RELEASE_NOTES", None)
    merged.pop("RELEASE_NOTES_TAG", None)
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

    def test_missing_source_is_noop(self) -> None:
        r = run_script(
            {"TAG": "v0.1.2", "GH_REPO": "canact/canact", "DRY_RUN": "1"}
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("leaving auto notes", r.stdout)

    def test_notes_file_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            notes = Path(td) / "RELEASE_NOTES.md"
            notes.write_text("canact 0.1.2 notes\n", encoding="utf-8")
            r = run_script(
                {
                    "TAG": "v0.1.2",
                    "GH_REPO": "canact/canact",
                    "DRY_RUN": "1",
                    "NOTES_FILE": str(notes),
                }
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("DRY_RUN: would apply file:", r.stdout)
            self.assertIn("BYTES: 19", r.stdout)
            self.assertNotIn("would delete branch", r.stdout)

    def test_empty_file_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            notes = Path(td) / "RELEASE_NOTES.md"
            notes.write_text("", encoding="utf-8")
            r = run_script(
                {
                    "TAG": "v0.1.2",
                    "GH_REPO": "canact/canact",
                    "DRY_RUN": "1",
                    "NOTES_FILE": str(notes),
                }
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("leaving auto notes", r.stdout)

    def test_variable_requires_matching_tag(self) -> None:
        r = run_script(
            {
                "TAG": "v0.1.2",
                "GH_REPO": "canact/canact",
                "DRY_RUN": "1",
                "RELEASE_NOTES": "stale notes",
                "RELEASE_NOTES_TAG": "v0.1.1",
            }
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("leaving auto notes", r.stdout)

    def test_variable_applies_when_tag_matches(self) -> None:
        r = run_script(
            {
                "TAG": "v0.1.2",
                "GH_REPO": "canact/canact",
                "DRY_RUN": "1",
                "RELEASE_NOTES": "from var",
                "RELEASE_NOTES_TAG": "0.1.2",
            }
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Actions variable", r.stdout)
        self.assertIn("DRY_RUN: would apply variable to v0.1.2", r.stdout)
        self.assertNotIn("would delete branch", r.stdout)

    def test_branch_fetch_dry_run_deletes_branch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake_bin = Path(td) / "bin"
            fake_bin.mkdir()
            gh = fake_bin / "gh"
            gh.write_text(
                "#!/bin/bash\n"
                "if [ \"$1\" = api ] && [ \"$2\" != -X ]; then\n"
                "  printf '%s\\n' 'from branch'\n"
                "  exit 0\n"
                "fi\n"
                "exit 1\n",
                encoding="utf-8",
            )
            gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
            env_path = f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"
            r = run_script(
                {
                    "TAG": "v0.1.2",
                    "GH_REPO": "canact/canact",
                    "DRY_RUN": "1",
                    "GH_TOKEN": "test",
                    "PATH": env_path,
                }
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertIn("release-note-0.1.2", r.stdout)
            self.assertIn("DRY_RUN: would delete branch release-note-0.1.2", r.stdout)


if __name__ == "__main__":
    unittest.main()
