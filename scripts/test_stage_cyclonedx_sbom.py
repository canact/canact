#!/usr/bin/env python3
"""Tests for scripts/stage-cyclonedx-sbom.sh without installing cargo-cyclonedx."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "stage-cyclonedx-sbom.sh"


def run_script(
    env: dict[str, str], cwd: str | None = None
) -> subprocess.CompletedProcess[str]:
    merged = {"PATH": os.environ.get("PATH", "")}
    merged.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=merged,
        check=False,
    )


class StageCyclonedxSbomTests(unittest.TestCase):
    def test_dry_run_prints_version_and_output_name(self) -> None:
        result = run_script({"DRY_RUN": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cargo install cargo-cyclonedx --version 0.5.9 --locked", result.stdout)
        self.assertIn(
            "cargo cyclonedx --format json --features cli --override-filename canact-sbom",
            result.stdout,
        )
        self.assertIn("canact-sbom.cdx.json", result.stdout)
        self.assertIn("DONE: dry-run", result.stdout)

    def test_one_planted_json_copies_bytes(self) -> None:
        payload = b'{"bomFormat":"CycloneDX","serial":"one"}\n'
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            search = root / "search"
            search.mkdir()
            (search / "canact-sbom.json").write_bytes(payload)
            (search / "notes.json").write_text("{}\n", encoding="utf-8")
            out = root / "canact-sbom.cdx.json"
            result = run_script(
                {
                    "COLLECT_ONLY": "1",
                    "SKIP_UPLOAD": "1",
                    "SEARCH_DIR": str(search),
                    "OUT": str(out),
                }
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(out.read_bytes(), payload)
            self.assertIn("DONE: staged", result.stdout)

    def test_same_file_is_left_in_place(self) -> None:
        payload = b'{"bomFormat":"CycloneDX"}\n'
        with tempfile.TemporaryDirectory() as td:
            planted = Path(td) / "canact-sbom.cdx.json"
            planted.write_bytes(payload)
            result = run_script(
                {
                    "COLLECT_ONLY": "1",
                    "SKIP_UPLOAD": "1",
                    "SEARCH_DIR": td,
                    "OUT": str(planted),
                }
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(planted.read_bytes(), payload)

    def test_zero_matches_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = run_script(
                {
                    "COLLECT_ONLY": "1",
                    "SKIP_UPLOAD": "1",
                    "SEARCH_DIR": td,
                    "OUT": str(Path(td) / "canact-sbom.cdx.json"),
                }
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("found 0", result.stderr)

    def test_two_matches_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "canact-sbom.json").write_text("{}\n", encoding="utf-8")
            (root / "canact-sbom.extra.json").write_text("{}\n", encoding="utf-8")
            result = run_script(
                {
                    "COLLECT_ONLY": "1",
                    "SKIP_UPLOAD": "1",
                    "SEARCH_DIR": td,
                    "OUT": str(root / "out.json"),
                }
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("found 2", result.stderr)

    def test_fixture_copies_bytes(self) -> None:
        payload = b'{"bomFormat":"CycloneDX","from":"fixture"}\n'
        with tempfile.TemporaryDirectory() as td:
            fixture = Path(td) / "fixture.json"
            fixture.write_bytes(payload)
            out = Path(td) / "canact-sbom.cdx.json"
            result = run_script(
                {
                    "FIXTURE": str(fixture),
                    "SKIP_UPLOAD": "1",
                    "OUT": str(out),
                }
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(out.read_bytes(), payload)

    def test_bad_tag_fails(self) -> None:
        result = run_script({"DRY_RUN": "1", "TAG": "v0.1.0-stealth.1"})
        self.assertEqual(result.returncode, 1)
        self.assertIn("TAG must look like vX.Y.Z", result.stderr)

    def test_upload_requires_tag_repo_and_token(self) -> None:
        payload = b"{}\n"
        with tempfile.TemporaryDirectory() as td:
            fixture = Path(td) / "fixture.json"
            fixture.write_bytes(payload)
            result = run_script(
                {
                    "FIXTURE": str(fixture),
                    "OUT": str(Path(td) / "canact-sbom.cdx.json"),
                }
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("TAG, REPO, and GH_TOKEN are required", result.stderr)


if __name__ == "__main__":
    unittest.main()
