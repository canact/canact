#!/usr/bin/env python3
"""Tests for scripts/attach-release-signatures.sh dry-run selection."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "attach-release-signatures.sh"


def run_script(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=merged,
        check=False,
    )


class AttachReleaseSignaturesTests(unittest.TestCase):
    def test_dry_run_lists_assets_and_skips_signature_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts = Path(td) / "artifacts"
            artifacts.mkdir()
            (artifacts / "canact-x86_64-unknown-linux-gnu.tar.xz").write_bytes(b"bin")
            (artifacts / "canact-x86_64-unknown-linux-gnu.tar.xz.sha256").write_text(
                "deadbeef\n", encoding="utf-8"
            )
            (artifacts / "canact-x86_64-unknown-linux-gnu.tar.xz.sigstore.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (artifacts / "canact-x86_64-unknown-linux-gnu.tar.xz.intoto.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            (artifacts / "canact-x86_64-unknown-linux-gnu.tar.xz.sig").write_text(
                "sig\n", encoding="utf-8"
            )
            (artifacts / "canact-x86_64-unknown-linux-gnu.tar.xz.sigstore.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            # Relative ARTIFACTS must still resolve when cwd is the parent.
            r = subprocess.run(
                ["bash", str(SCRIPT)],
                capture_output=True,
                text=True,
                cwd=td,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "ARTIFACTS": "artifacts",
                    "TAG": "v0.1.1",
                    "REPO": "canact/canact",
                },
                check=False,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(f"from {artifacts.resolve()}", r.stdout)
            subjects = [
                line.removeprefix("SUBJECT: ")
                for line in r.stdout.splitlines()
                if line.startswith("SUBJECT: ")
            ]
            self.assertEqual(
                subjects,
                [
                    "canact-x86_64-unknown-linux-gnu.tar.xz",
                    "canact-x86_64-unknown-linux-gnu.tar.xz.sha256",
                ],
            )
            self.assertIn("DONE: dry-run", r.stdout)

    def test_rejects_empty_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            r = run_script(
                {
                    "DRY_RUN": "1",
                    "ARTIFACTS": td,
                    "TAG": "v0.1.1",
                    "REPO": "canact/canact",
                }
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("no signable assets", r.stderr)

    def test_rejects_bad_tag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            r = run_script(
                {
                    "DRY_RUN": "1",
                    "ARTIFACTS": td,
                    "TAG": "v0.1.0-stealth.1",
                    "REPO": "canact/canact",
                }
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("TAG must look like", r.stderr)


if __name__ == "__main__":
    unittest.main()
