#!/usr/bin/env python3
"""Tests for scripts/update-scoop-manifest.py"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "update-scoop-manifest.py"
HASH_X64 = "b194104e93904c82a9ffd30b5a6f9125b1ca41848b24c63353ec4bd775c332f1"


class UpdateScoopManifestTests(unittest.TestCase):
    def test_write_from_hash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "canact.json"
            r = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--version",
                    "0.1.1",
                    "--hash-x64",
                    HASH_X64,
                    "--output",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("wrote", r.stdout)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["version"], "0.1.1")
            self.assertEqual(data["bin"], "canact.exe")
            self.assertIn("v0.1.1", data["architecture"]["64bit"]["url"])
            self.assertIn(
                "canact-x86_64-pc-windows-msvc.zip",
                data["architecture"]["64bit"]["url"],
            )
            self.assertEqual(data["architecture"]["64bit"]["hash"], HASH_X64)
            self.assertNotIn("arm64", data["architecture"])
            self.assertIn("v$version", data["autoupdate"]["architecture"]["64bit"]["url"])

    def test_strips_v_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "canact.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--version",
                    "v0.1.1",
                    "--hash-x64",
                    HASH_X64,
                    "--output",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["version"], "0.1.1")
            self.assertIn("/v0.1.1/", data["architecture"]["64bit"]["url"])

    def test_artifacts_dir_and_check(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            art = root / "artifacts" / "nested"
            art.mkdir(parents=True)
            (art / "canact-x86_64-pc-windows-msvc.zip.sha256").write_text(
                f"{HASH_X64} *canact-x86_64-pc-windows-msvc.zip\n",
                encoding="utf-8",
            )
            out = root / "bucket" / "canact.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--version",
                    "v0.1.1",
                    "--artifacts-dir",
                    str(root / "artifacts"),
                    "--output",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            check = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--version",
                    "0.1.1",
                    "--artifacts-dir",
                    str(root / "artifacts"),
                    "--output",
                    str(out),
                    "--check",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("ok:", check.stdout)
            out.write_text("{}\n", encoding="utf-8")
            stale = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--version",
                    "0.1.1",
                    "--artifacts-dir",
                    str(root / "artifacts"),
                    "--output",
                    str(out),
                    "--check",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(stale.returncode, 0)
            self.assertIn("stale", stale.stderr)

    def test_missing_hash_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "artifacts").mkdir()
            out = root / "canact.json"
            r = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--version",
                    "0.1.1",
                    "--artifacts-dir",
                    str(root / "artifacts"),
                    "--output",
                    str(out),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("missing", r.stderr)
            self.assertFalse(out.exists())

    def test_requires_hash_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "canact.json"
            r = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--version",
                    "0.1.1",
                    "--output",
                    str(out),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("provide", r.stderr)


if __name__ == "__main__":
    unittest.main()
