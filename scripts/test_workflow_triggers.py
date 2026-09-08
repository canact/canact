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

    def test_publish_crates_is_tag_or_dispatch(self) -> None:
        text = (WORKFLOWS / "publish-crates.yml").read_text(encoding="utf-8")
        on_block = _on_block(text)
        self.assertIn("workflow_dispatch:", on_block)
        self.assertIn("tags:", on_block)
        self.assertNotIn("pull_request:", on_block)
        self.assertNotIn("branches:", on_block)
        self.assertIn("id-token: write", text)
        self.assertIn("crates-io-auth-action", text)
        self.assertNotIn("CARGO_REGISTRY_TOKEN: ${{ secrets.", text)
        self.assertNotRegex(text, r"cargo (test|nextest|clippy|fuzz)")
        # Dispatch of an older tag must not run the wrapper from that
        # tree (v0.1.2 has no scripts/publish-crates.sh).
        self.assertIn("path: publisher", text)
        self.assertIn("path: crate", text)
        self.assertIn("ref: ${{ github.sha }}", text)
        self.assertIn("ref: ${{ inputs.tag || github.ref }}", text)
        self.assertIn("working-directory: crate", text)
        self.assertIn("bash ../publisher/scripts/publish-crates.sh", text)
        self.assertNotIn("run: bash scripts/publish-crates.sh", text)

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

    def test_required_jobs_stay_named_on_release_please(self) -> None:
        ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
        sec = (WORKFLOWS / "security.yml").read_text(encoding="utf-8")
        self.assertIn("name: Lint", ci)
        self.assertIn("name: Test", ci)
        self.assertIn("name: CodeQL (${{ matrix.language }})", sec)
        # Job-level skip would drop the required check name.
        lint = ci[ci.index("name: Lint") : ci.index("name: Test")]
        test = ci[ci.index("name: Test") : ci.index("name: Fuzz smoke")]
        self.assertNotIn("startsWith(github.head_ref, 'release-please')", lint.split("steps:")[0])
        self.assertNotIn("startsWith(github.head_ref, 'release-please')", test.split("steps:")[0])
        codeql = sec[sec.index("name: CodeQL") :]
        self.assertNotIn(
            "startsWith(github.head_ref, 'release-please')",
            codeql.split("steps:")[0],
        )

    def test_release_please_uses_cargo_check_not_full_matrix(self) -> None:
        ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("cargo check --locked --all-targets", ci)
        self.assertGreaterEqual(ci.count("Release-please check"), 2)
        for cmd in (
            "cargo clippy --locked --all-targets",
            "cargo nextest run --locked",
            "cargo test --locked --doc",
            "cargo doc --locked --no-deps --all-features",
        ):
            idx = ci.index(cmd)
            window = ci[max(0, idx - 400) : idx]
            self.assertIn("!startsWith(github.head_ref, 'release-please')", window, cmd)

    def test_codeql_standin_on_release_please(self) -> None:
        sec = (WORKFLOWS / "security.yml").read_text(encoding="utf-8")
        self.assertIn("startsWith(github.head_ref, 'release-please')", sec)
        for pin in (
            "github/codeql-action/init@",
            "github/codeql-action/analyze@",
        ):
            idx = sec.index(pin)
            window = sec[max(0, idx - 400) : idx]
            self.assertIn("!startsWith(github.head_ref, 'release-please')", window, pin)

    def test_fossa_push_and_pr_share_cargo_path_filter(self) -> None:
        on_block = _on_block((WORKFLOWS / "fossa.yml").read_text(encoding="utf-8"))
        self.assertIn("push:", on_block)
        self.assertIn("pull_request:", on_block)
        self.assertIn("workflow_dispatch:", on_block)
        push = on_block[on_block.index("push:") : on_block.index("pull_request:")]
        pr = on_block[on_block.index("pull_request:") : on_block.index("workflow_dispatch:")]
        for needle in (
            "Cargo.*",
            "src/**",
            "scripts/fossa-filter.py",
            "scripts/test_fossa_filter.py",
            ".github/workflows/fossa.yml",
        ):
            self.assertIn(needle, push, needle)
            self.assertIn(needle, pr, needle)


if __name__ == "__main__":
    unittest.main()
