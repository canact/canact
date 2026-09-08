#!/usr/bin/env python3
"""Tests for scripts/rerun-cancelled-pr-checks.sh."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "rerun-cancelled-pr-checks.sh"


def run_script(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.pop("GITHUB_TOKEN", None)
    merged.pop("GITHUB_REPOSITORY", None)
    merged.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=merged,
        check=False,
    )


def write_gh(bin_dir: Path, log: Path, view: dict[str, object], heads: list[str]) -> None:
    gh = bin_dir / "gh"
    view_json = json.dumps(view)
    head_prints = "".join(f"  printf '%s\\n' '{h}'\n" for h in heads)
    gh.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' \"$*\" >> '{log}'\n"
        "if [ \"$1\" = pr ] && [ \"$2\" = list ]; then\n"
        f"{head_prints}"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = run ] && [ \"$2\" = view ]; then\n"
        f"  printf '%s\\n' '{view_json}'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = run ] && [ \"$2\" = rerun ]; then\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC)


class RerunCancelledTests(unittest.TestCase):
    def test_parses_details_url_and_reruns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            fake_bin = td_path / "bin"
            fake_bin.mkdir()
            log = td_path / "gh.log"
            write_gh(
                fake_bin,
                log,
                {
                    "conclusion": "cancelled",
                    "attempt": 1,
                    "headSha": "abc",
                    "status": "completed",
                },
                ["abc"],
            )
            r = run_script(
                {
                    "GH_REPO": "canact/canact",
                    "HEAD_SHA": "abc",
                    "DETAILS_URL": (
                        "https://github.com/canact/canact/actions/runs/34183810889/job/1"
                    ),
                    "GH_TOKEN": "test",
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
                }
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertIn("DONE: reran 34183810889", r.stdout)
            logged = log.read_text(encoding="utf-8")
            self.assertIn("run rerun 34183810889", logged)

    def test_dry_run_skips_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            fake_bin = td_path / "bin"
            fake_bin.mkdir()
            log = td_path / "gh.log"
            write_gh(
                fake_bin,
                log,
                {
                    "conclusion": "cancelled",
                    "attempt": 1,
                    "headSha": "abc",
                    "status": "completed",
                },
                ["abc"],
            )
            r = run_script(
                {
                    "GH_REPO": "canact/canact",
                    "HEAD_SHA": "abc",
                    "RUN_ID": "99",
                    "GH_TOKEN": "test",
                    "DRY_RUN": "1",
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
                }
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertIn("DRY_RUN: would gh run rerun 99", r.stdout)
            logged = log.read_text(encoding="utf-8")
            self.assertNotIn("run rerun 99", logged)

    def test_skips_when_sha_is_not_open_head(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            fake_bin = td_path / "bin"
            fake_bin.mkdir()
            log = td_path / "gh.log"
            write_gh(
                fake_bin,
                log,
                {
                    "conclusion": "cancelled",
                    "attempt": 1,
                    "headSha": "old",
                    "status": "completed",
                },
                ["new"],
            )
            r = run_script(
                {
                    "GH_REPO": "canact/canact",
                    "HEAD_SHA": "old",
                    "RUN_ID": "99",
                    "GH_TOKEN": "test",
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
                }
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertIn("not an open PR head", r.stdout)
            logged = log.read_text(encoding="utf-8")
            self.assertNotIn("run rerun", logged)

    def test_skips_second_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            fake_bin = td_path / "bin"
            fake_bin.mkdir()
            log = td_path / "gh.log"
            write_gh(
                fake_bin,
                log,
                {
                    "conclusion": "cancelled",
                    "attempt": 2,
                    "headSha": "abc",
                    "status": "completed",
                },
                ["abc"],
            )
            r = run_script(
                {
                    "GH_REPO": "canact/canact",
                    "HEAD_SHA": "abc",
                    "RUN_ID": "99",
                    "GH_TOKEN": "test",
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
                }
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertIn("attempt 2 >= 2", r.stdout)
            logged = log.read_text(encoding="utf-8")
            self.assertNotIn("run rerun", logged)

    def test_dispatch_without_run_id_is_noop(self) -> None:
        r = run_script(
            {
                "GH_REPO": "canact/canact",
                "HEAD_SHA": "abc",
                "GH_TOKEN": "test",
            }
        )
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("no RUN_ID", r.stdout)

    def test_skips_non_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            fake_bin = td_path / "bin"
            fake_bin.mkdir()
            log = td_path / "gh.log"
            write_gh(
                fake_bin,
                log,
                {
                    "conclusion": "success",
                    "attempt": 1,
                    "headSha": "abc",
                    "status": "completed",
                },
                ["abc"],
            )
            r = run_script(
                {
                    "GH_REPO": "canact/canact",
                    "HEAD_SHA": "abc",
                    "RUN_ID": "99",
                    "GH_TOKEN": "test",
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
                }
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertIn("conclusion success", r.stdout)
            logged = log.read_text(encoding="utf-8")
            self.assertNotIn("run rerun", logged)


if __name__ == "__main__":
    unittest.main()
