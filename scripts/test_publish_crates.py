#!/usr/bin/env python3
"""Tests for scripts/publish-crates.sh."""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "publish-crates.sh"


def _write_exec(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def run_script(env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.pop("CARGO_REGISTRY_TOKEN", None)
    merged.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=merged,
        cwd=cwd,
        check=False,
    )


class PublishCratesTests(unittest.TestCase):
    def _tree(
        self, version: str = "0.1.2"
    ) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        (root / "Cargo.toml").write_text(
            f'[package]\nname = "canact"\nversion = "{version}"\n',
            encoding="utf-8",
        )
        return td, root

    def test_already_on_crates_io_skips_cargo(self) -> None:
        td, root = self._tree()
        with td:
            curl = root / "curl"
            cargo = root / "cargo"
            _write_exec(
                curl,
                """#!/bin/sh
out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-o" ]; then out="$2"; shift 2; continue; fi
  shift
done
[ -n "$out" ] && echo already >"$out"
echo 200
""",
            )
            _write_exec(cargo, "#!/bin/sh\necho RAN\nexit 99\n")
            r = run_script(
                {
                    "CURL": str(curl),
                    "CARGO": str(cargo),
                    "TAG": "v0.1.2",
                    "CARGO_REGISTRY_TOKEN": "cio-test",
                },
                root,
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertIn("already on crates.io", r.stdout)
            self.assertNotIn("RAN", r.stdout)

    def test_tag_must_match_manifest(self) -> None:
        td, root = self._tree("0.1.2")
        with td:
            r = run_script({"TAG": "v0.1.1"}, root)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("does not match", r.stderr)

    def test_missing_token_fails_when_unpublished(self) -> None:
        td, root = self._tree()
        with td:
            curl = root / "curl"
            _write_exec(curl, "#!/bin/sh\necho 404\n")
            r = run_script({"CURL": str(curl), "TAG": "v0.1.2"}, root)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("CARGO_REGISTRY_TOKEN", r.stderr)

    def test_dry_run_skips_cargo_when_unpublished(self) -> None:
        td, root = self._tree()
        with td:
            curl = root / "curl"
            cargo = root / "cargo"
            _write_exec(curl, "#!/bin/sh\necho 404\n")
            _write_exec(cargo, "#!/bin/sh\necho RAN\nexit 99\n")
            r = run_script(
                {
                    "CURL": str(curl),
                    "CARGO": str(cargo),
                    "TAG": "v0.1.2",
                    "DRY_RUN": "1",
                },
                root,
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertIn("DRY_RUN: would cargo publish", r.stdout)
            self.assertNotIn("RAN", r.stdout)

    def test_already_uploaded_is_success(self) -> None:
        td, root = self._tree()
        with td:
            curl = root / "curl"
            cargo = root / "cargo"
            _write_exec(curl, "#!/bin/sh\necho 404\n")
            _write_exec(
                cargo,
                "#!/bin/sh\necho 'error: crate canact@0.1.2 already uploaded'\nexit 1\n",
            )
            r = run_script(
                {
                    "CURL": str(curl),
                    "CARGO": str(cargo),
                    "TAG": "v0.1.2",
                    "CARGO_REGISTRY_TOKEN": "cio-test",
                },
                root,
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertIn("already uploaded", r.stdout)

    def test_cargo_error_is_failure(self) -> None:
        td, root = self._tree()
        with td:
            curl = root / "curl"
            cargo = root / "cargo"
            _write_exec(curl, "#!/bin/sh\necho 404\n")
            _write_exec(cargo, "#!/bin/sh\necho 'error: yanked'\nexit 7\n")
            r = run_script(
                {
                    "CURL": str(curl),
                    "CARGO": str(cargo),
                    "TAG": "v0.1.2",
                    "CARGO_REGISTRY_TOKEN": "cio-test",
                },
                root,
            )
            self.assertEqual(r.returncode, 7)
            self.assertIn("exited 7", r.stderr)

    def test_crates_io_outage_is_failure(self) -> None:
        td, root = self._tree()
        with td:
            curl = root / "curl"
            _write_exec(curl, "#!/bin/sh\necho 503\n")
            r = run_script({"CURL": str(curl), "TAG": "v0.1.2"}, root)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("HTTP 503", r.stderr)


if __name__ == "__main__":
    unittest.main()
