#!/usr/bin/env python3
"""Lock Recipe A and the cheap release-please split."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _on_block(text: str) -> str:
    start = text.index("\non:")
    rest = text[start + 1 :]
    end = rest.index("\njobs:")
    block = rest[:end]
    lines = []
    for line in block.splitlines():
        stripped = line.split("#", 1)[0].rstrip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


class WorkflowTriggerTests(unittest.TestCase):
    def test_ci_has_no_push_compile(self) -> None:
        on_block = _on_block((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
        self.assertIn("pull_request:", on_block)
        self.assertIn("workflow_dispatch:", on_block)
        self.assertNotIn("push:", on_block)
        self.assertNotIn("tags:", on_block)

    def test_release_is_tag_or_dispatch_only(self) -> None:
        on_block = _on_block((WORKFLOWS / "release.yml").read_text(encoding="utf-8"))
        self.assertNotIn("pull_request:", on_block)
        self.assertIn("workflow_dispatch:", on_block)
        self.assertIn("tags:", on_block)
        self.assertNotIn("branches:", on_block)

    def test_release_please_is_main_only(self) -> None:
        text = (WORKFLOWS / "release-please.yml").read_text(encoding="utf-8")
        on_block = _on_block(text)
        self.assertIn("push:", on_block)
        self.assertIn("branches: [main]", on_block)
        self.assertIn("workflow_dispatch:", on_block)
        self.assertNotIn("pull_request:", on_block)
        self.assertNotRegex(text, r"cargo (test|nextest|clippy|fuzz)")

    def test_auto_merge_skips_release_please_head(self) -> None:
        text = (WORKFLOWS / "auto-approve.yml").read_text(encoding="utf-8")
        self.assertIn("!startsWith(github.head_ref, 'release-please')", text)
        self.assertIn("autorelease: pending", text)

    def test_cheap_pr_status_checks_do_not_cancel(self) -> None:
        for name in ("pr-title.yml", "dco.yml"):
            text = (WORKFLOWS / name).read_text(encoding="utf-8")
            self.assertIn("cancel-in-progress: false", text, name)

    def test_rerun_cancelled_checks_is_check_run_only(self) -> None:
        text = (WORKFLOWS / "rerun-cancelled-pr-checks.yml").read_text(
            encoding="utf-8"
        )
        on_block = _on_block(text)
        self.assertIn("check_run:", on_block)
        self.assertIn("workflow_dispatch:", on_block)
        self.assertNotIn("push:", on_block)
        self.assertNotRegex(text, r"cargo (test|nextest|clippy|fuzz)")
        self.assertIn("cancel-in-progress: false", text)

    def test_apply_release_notes_is_dispatch_only(self) -> None:
        text = (WORKFLOWS / "apply-release-notes.yml").read_text(encoding="utf-8")
        on_block = _on_block(text)
        self.assertIn("workflow_dispatch:", on_block)
        self.assertNotIn("pull_request:", on_block)
        self.assertNotIn("push:", on_block)
        self.assertNotRegex(text, r"cargo (test|nextest|clippy|fuzz)")

    def test_rust_release_type_is_minor_pre_major(self) -> None:
        cfg = (ROOT / "release-please-config.json").read_text(encoding="utf-8")
        self.assertIn('"release-type": "rust"', cfg)
        self.assertIn('"bump-minor-pre-major": true', cfg)
        self.assertIn('"include-component-in-tag": false', cfg)
        self.assertNotIn("bump-patch-for-minor-pre-major", cfg)
        dist = (ROOT / "dist-workspace.toml").read_text(encoding="utf-8")
        self.assertIn('pr-run-mode = "skip"', dist)


if __name__ == "__main__":
    unittest.main()
